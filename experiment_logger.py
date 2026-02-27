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
    kv_caches/
      batch_0000.parquet   (queries 0–N, snappy compressed)
      batch_0001.parquet
      ...
"""

import gc
import json
import os
import threading
import queue
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


_DEFAULT_KV_FLUSH_INTERVAL = 30   # flush every N queries


class ExperimentLogger:
    """Saves config, per-agent traces + KV caches (parquet), and run summary."""

    def __init__(
        self,
        base_dir: str,
        task: str,
        args: Any = None,
        timestamp: Optional[str] = None,
        kv_flush_interval: int = _DEFAULT_KV_FLUSH_INTERVAL,
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

        # ------- Batched KV-cache writer -------
        self._kv_flush_interval = kv_flush_interval
        self._kv_buffer: List[Dict] = []       # serialised rows (CPU bytes)
        self._kv_pending_queries = 0            # queries accumulated since last flush
        self._kv_batch_counter = 0              # sequential file counter

        self._kv_queue: queue.Queue = queue.Queue()
        self._kv_thread = threading.Thread(
            target=self._kv_writer_loop, daemon=True, name="kv-cache-writer"
        )
        self._kv_thread.start()
        self._kv_files_written = 0
        self._kv_total_queries_saved = 0
        self._kv_lock = threading.Lock()

        print(f"[ExperimentLogger] Logging to: {self.run_dir}")
        print(f"[ExperimentLogger] KV-cache flush every {kv_flush_interval} queries")

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

    # ------------------------------------------------------------------
    # Legacy single-query KV-cache save  (kept for backward compat)
    # ------------------------------------------------------------------

    def save_kv_cache(
        self,
        query_idx: int,
        agent_idx: int,
        agent_name: str,
        past_key_values,
        batch_idx: int = 0,
    ) -> Optional[str]:
        """
        Save KV cache as a single Parquet file (snappy).
        Legacy method — prefer save_kv_cache_accumulated for batched saves.
        """
        if past_key_values is None:
            return None

        agent_dir = self._agent_dir(query_idx, agent_idx, agent_name)
        os.makedirs(agent_dir, exist_ok=True)

        rows = self._serialize_kv(past_key_values, batch_idx)
        if not rows:
            return None

        df = pd.DataFrame(rows)
        parquet_path = os.path.join(agent_dir, "kv_cache.parquet")
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
        return parquet_path

    # ------------------------------------------------------------------
    # Batched accumulated KV-cache save  (non-blocking)
    # ------------------------------------------------------------------

    def save_kv_cache_accumulated(
        self,
        query_idx: int,
        agent_names: List[str],
        past_key_values,
        batch_idx: int = 0,
    ) -> None:
        """
        **Non-blocking**: serialise KV tensors to CPU bytes, buffer them,
        and flush to disk every ``kv_flush_interval`` queries.

        One parquet file is written per flush batch (e.g. 50 queries),
        greatly reducing I/O frequency and memory pressure.
        """
        if past_key_values is None:
            return

        rows = self._serialize_kv(past_key_values, batch_idx)
        if not rows:
            return

        # Tag each row with query-level info
        agent_names_json = json.dumps(agent_names)
        for row in rows:
            row["query_idx"] = np.int32(query_idx)
            row["agent_names"] = agent_names_json

        self._kv_buffer.extend(rows)
        self._kv_pending_queries += 1

        # Auto-flush when we hit the interval
        if self._kv_pending_queries >= self._kv_flush_interval:
            self._enqueue_buffer()

    # ------------------------------------------------------------------
    # Internal: serialise KV tensors → list of dicts (CPU bytes)
    # ------------------------------------------------------------------

    def _serialize_kv(self, past_key_values, batch_idx: int = 0) -> List[Dict]:
        """Convert KV cache tensors to a list of row dicts with raw bytes."""
        legacy_kv = self._to_legacy(past_key_values)
        if legacy_kv is None or len(legacy_kv) == 0:
            return []

        rows: List[Dict] = []
        for layer_idx, layer_tensors in enumerate(legacy_kv):
            if not isinstance(layer_tensors, (tuple, list)) or len(layer_tensors) < 2:
                continue

            key_t = layer_tensors[0]
            val_t = layer_tensors[1]

            if key_t.dim() == 4:
                key_t = key_t[batch_idx]
                val_t = val_t[batch_idx]

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
        return rows

    # ------------------------------------------------------------------
    # Internal: move buffer → background queue
    # ------------------------------------------------------------------

    def _enqueue_buffer(self) -> None:
        """Move current buffer to the background writer and clear it."""
        if not self._kv_buffer:
            return

        job = {
            "batch_idx": self._kv_batch_counter,
            "rows": self._kv_buffer,
            "num_queries": self._kv_pending_queries,
        }
        self._kv_queue.put(job)

        # Reset buffer immediately → frees memory in main thread
        self._kv_buffer = []
        self._kv_pending_queries = 0
        self._kv_batch_counter += 1

        # Help the garbage collector release the old buffer promptly
        gc.collect()

    # ------------------------------------------------------------------
    # Background writer thread
    # ------------------------------------------------------------------

    def _kv_writer_loop(self) -> None:
        """Consume jobs from the queue and write parquet files."""
        while True:
            job = self._kv_queue.get()
            if job is None:                     # poison pill → exit
                self._kv_queue.task_done()
                break
            try:
                self._write_kv_batch(job)
            except Exception as e:
                print(f"[ExperimentLogger] KV-cache write error (batch {job['batch_idx']}): {e}")
            finally:
                self._kv_queue.task_done()

    def _write_kv_batch(self, job: Dict) -> None:
        """Write a batch of queries' KV caches into a single parquet file."""
        batch_idx = job["batch_idx"]
        rows = job["rows"]
        num_queries = job["num_queries"]

        cache_dir = os.path.join(self.run_dir, "kv_caches")
        os.makedirs(cache_dir, exist_ok=True)

        df = pd.DataFrame(rows)
        parquet_path = os.path.join(cache_dir, f"batch_{batch_idx:04d}.parquet")
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

        with self._kv_lock:
            self._kv_files_written += 1
            self._kv_total_queries_saved += num_queries

        # Free the rows memory as soon as possible
        del rows, df
        gc.collect()

        print(
            f"[ExperimentLogger] Wrote {parquet_path} "
            f"({num_queries} queries, batch {batch_idx})"
        )

    # ------------------------------------------------------------------
    # Public: flush & shutdown
    # ------------------------------------------------------------------

    def flush_kv_cache_queue(self) -> None:
        """
        Flush any remaining buffered KV caches and block until all
        background writes are done.  Call after the benchmark loop.
        """
        # Push whatever is left in the buffer
        self._enqueue_buffer()
        # Wait for the background thread to finish all pending jobs
        self._kv_queue.join()

        with self._kv_lock:
            files = self._kv_files_written
            queries = self._kv_total_queries_saved
        print(
            f"[ExperimentLogger] KV-cache queue flushed — "
            f"{files} batch files, {queries} queries total."
        )

    def shutdown(self) -> None:
        """Stop the background writer thread gracefully."""
        self._kv_queue.put(None)        # poison pill
        self._kv_thread.join(timeout=30)

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
    # Static helper to load saved KV caches back to tensors
    # ------------------------------------------------------------------

    @staticmethod
    def load_kv_cache(parquet_path: str, device: str = "cpu") -> list:
        """
        Reload a kv_cache parquet (single-query or batch) into tensors.

        For batch files, returns all rows as a flat list of (key, value).
        Use the ``query_idx`` column to group by query if present.
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

    @staticmethod
    def load_kv_cache_batch(parquet_path: str, device: str = "cpu") -> Dict[int, List]:
        """
        Load a batched KV-cache parquet and return a dict keyed by query_idx.

        Returns:
            {query_idx: [(key_tensor, value_tensor), ...per layer...], ...}
        """
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        result: Dict[int, List] = {}
        for _, row in df.iterrows():
            qidx = int(row["query_idx"])
            shape = json.loads(row["key_shape"])
            key_np = np.frombuffer(row["key"], dtype=np.float16).reshape(shape)
            val_np = np.frombuffer(row["value"], dtype=np.float16).reshape(
                json.loads(row["value_shape"])
            )
            key_t = torch.from_numpy(key_np.copy()).to(device)
            val_t = torch.from_numpy(val_np.copy()).to(device)
            result.setdefault(qidx, []).append((key_t, val_t))
        return result

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
