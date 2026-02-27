"""
Experiment Logger for LatentMAS
================================
Directory structure:
  experiment_logs/<task>/<timestamp>/
    run_config.json
    summary.json
    results.json
    query_NNNN/
      agent_<idx>_<name>/
        trace.json
        kv_cache.parquet   (per-layer K/V as binary columns, gzip compressed)
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


class ExperimentLogger:
    """Saves config, per-agent traces + KV caches (parquet), and run summary."""

    def __init__(
        self,
        base_dir: str,
        task: str,
        args: Any = None,
        timestamp: Optional[str] = None,
    ):
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        self.run_dir = os.path.join(base_dir, task, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

        # Save run configuration
        if args is not None:
            config_path = os.path.join(self.run_dir, "run_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, ensure_ascii=False, default=str)

        self._query_counter = 0
        print(f"[ExperimentLogger] Logging to: {self.run_dir}")

    # ------------------------------------------------------------------
    # Query index management
    # ------------------------------------------------------------------

    def next_query_index(self) -> int:
        """Return the next query index and increment the counter."""
        idx = self._query_counter
        self._query_counter += 1
        return idx

    # ------------------------------------------------------------------
    # Per-agent saving
    # ------------------------------------------------------------------

    def save_agent_trace(
        self,
        query_idx: int,
        agent_idx: int,
        agent_name: str,
        trace: Dict,
    ) -> str:
        """Save agent trace (input/output/metadata) as JSON."""
        agent_dir = self._agent_dir(query_idx, agent_idx, agent_name)
        os.makedirs(agent_dir, exist_ok=True)

        trace_path = os.path.join(agent_dir, "trace.json")

        # Make a serialisable copy (drop raw tensors)
        serialisable = {}
        for k, v in trace.items():
            if isinstance(v, torch.Tensor):
                serialisable[k] = f"<Tensor shape={list(v.shape)} dtype={v.dtype}>"
            else:
                serialisable[k] = v

        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False, default=str)
        return trace_path

    def save_kv_cache(
        self,
        query_idx: int,
        agent_idx: int,
        agent_name: str,
        past_key_values,
        batch_idx: int = 0,
    ) -> Optional[str]:
        """
        Save KV cache as a single **Parquet** file (engine=pyarrow, gzip).

        Schema (one row per layer):
            layer_idx   : int32
            key         : binary   (raw float16 bytes)
            value       : binary   (raw float16 bytes)
            key_shape   : string   (JSON list, e.g. "[32, 210, 128]")
            value_shape : string   (JSON list)
            dtype       : string   ("float16")

        Tensors are converted bf16 → fp16 before serialisation because
        numpy does not natively support bfloat16.
        """
        if past_key_values is None:
            return None

        agent_dir = self._agent_dir(query_idx, agent_idx, agent_name)
        os.makedirs(agent_dir, exist_ok=True)

        legacy_kv = self._to_legacy(past_key_values)
        if legacy_kv is None or len(legacy_kv) == 0:
            return None

        rows: List[Dict] = []
        for layer_idx, layer_tensors in enumerate(legacy_kv):
            if not isinstance(layer_tensors, (tuple, list)) or len(layer_tensors) < 2:
                continue

            key_t = layer_tensors[0]   # [B, num_heads, seq_len, head_dim]
            val_t = layer_tensors[1]

            # Extract single batch item when batched
            if key_t.dim() == 4:
                key_t = key_t[batch_idx]    # [num_heads, seq_len, head_dim]
                val_t = val_t[batch_idx]

            # bf16 → fp16 → numpy → bytes
            key_np = key_t.detach().cpu().to(torch.float16).numpy()
            val_np = val_t.detach().cpu().to(torch.float16).numpy()

            rows.append({
                "layer_idx": np.int32(layer_idx),
                "key": key_np.tobytes(),
                "value": val_np.tobytes(),
                "key_shape": json.dumps(list(key_np.shape)),
                "value_shape": json.dumps(list(val_np.shape)),
                "dtype": "float16",
            })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        parquet_path = os.path.join(agent_dir, "kv_cache.parquet")
        df.to_parquet(parquet_path, engine="pyarrow", compression="gzip", index=False)
        return parquet_path

    # ------------------------------------------------------------------
    # Run-level saving
    # ------------------------------------------------------------------

    def save_results(self, preds: List[Dict]) -> str:
        """Save all per-query predictions (including agent traces)."""
        path = os.path.join(self.run_dir, "results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False, default=str)
        print(f"[ExperimentLogger] Saved {len(preds)} results -> {path}")
        return path

    def save_summary(self, summary: Dict) -> str:
        """Save final run summary (accuracy, timing, etc.)."""
        path = os.path.join(self.run_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"[ExperimentLogger] Saved summary -> {path}")
        return path

    # ------------------------------------------------------------------
    # Static helper to load a saved KV-cache parquet back to tensors
    # ------------------------------------------------------------------

    @staticmethod
    def load_kv_cache(parquet_path: str, device: str = "cpu") -> list:
        """
        Convenience: reload a kv_cache.parquet into a list of (key, value) tensors.

        Returns:
            list of (key_tensor, value_tensor) per layer
        """
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        layers = []
        for _, row in df.iterrows():
            shape = json.loads(row["key_shape"])
            key_np = np.frombuffer(row["key"], dtype=np.float16).reshape(shape)
            val_np = np.frombuffer(row["value"], dtype=np.float16).reshape(
                json.loads(row["value_shape"])
            )
            key_t = torch.from_numpy(key_np.copy()).to(device)
            val_t = torch.from_numpy(val_np.copy()).to(device)
            layers.append((key_t, val_t))
        return layers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _agent_dir(self, query_idx: int, agent_idx: int, agent_name: str) -> str:
        query_dir = os.path.join(self.run_dir, f"query_{query_idx:04d}")
        return os.path.join(query_dir, f"agent_{agent_idx}_{agent_name}")

    @staticmethod
    def _to_legacy(past_key_values):
        """Convert HF Cache objects to legacy tuple-of-tuples if needed."""
        if past_key_values is None:
            return None
        if Cache is not None and isinstance(past_key_values, Cache):
            try:
                return past_key_values.to_legacy_cache()
            except Exception:
                return None
        return past_key_values
