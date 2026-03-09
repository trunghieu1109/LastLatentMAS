"""Query-Aware AE Trainer: data collection + training + checkpoint management.

Checkpoint layout (mirrors ae_checkpoints/paired_hidden_state_cache/):
    <checkpoint_dir>/
        cache_metadata.json
        layer_{idx}_input.parquet       # [N, D] — last input token hidden states
        layer_{idx}_latent.parquet      # [N*latent_steps, D] — latent step hidden states
        ae_layer_{idx}.pt               # trained QueryAwareAutoencoder state_dict + config
        training_log_layer_{idx}.json   # per-epoch loss history

Usage from run.py:
    ae_trainer = AETrainer(method, args)
    ae_trainer.collect_and_train(dataset_iter)
"""

import copy
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .query_aware_ae import QueryAwareAutoencoder, total_loss


# ──────────────────────────────────────────────────────────────────────────────
# AELogger
# ──────────────────────────────────────────────────────────────────────────────

# ANSI colour helpers (no external dep)
_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "cyan":   "\033[36m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "red":    "\033[31m",
    "grey":   "\033[90m",
}


def _fmt(s: str, *codes: str) -> str:
    """Wrap string in ANSI codes."""
    prefix = "".join(_C.get(c, "") for c in codes)
    return f"{prefix}{s}{_C['reset']}"


class AELogger:
    """Lightweight logger for Query-Aware AE training.

    Outputs:
        - Console: coloured per-epoch lines (every ``print_every`` epochs).
        - CSV:  ``<log_dir>/layer_{idx}_training.csv``  (one row per epoch).
        - JSON summary: ``<log_dir>/layer_{idx}_summary.json`` (written at end).
        - Collect log: ``<log_dir>/collect_log.txt`` (sample-level events).

    Args:
        log_dir:     Directory to write files (usually same as checkpoint_dir).
        print_every: Print a console line every N epochs (default 10).
    """

    def __init__(self, log_dir: str, print_every: int = 10):
        self.log_dir = log_dir
        self.print_every = print_every
        os.makedirs(log_dir, exist_ok=True)

        self._run_start = time.time()
        self._csv_writers: Dict[int, csv.DictWriter] = {}
        self._csv_files: Dict[int, object] = {}   # file handles

        # Print run header
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print(_fmt(f"\n{'='*60}", "bold", "cyan"))
        print(_fmt(f"  AE Training started  {ts}", "bold", "cyan"))
        print(_fmt(f"  Log dir: {log_dir}", "cyan"))
        print(_fmt(f"{'='*60}", "bold", "cyan"))

    # ── Layer header ─────────────────────────────────────────────

    def begin_layer(
        self,
        signed_li: int,
        n_latent: int,
        n_input: int,
        epochs: int,
        bottleneck_dim: int,
        hidden_dim: int,
    ) -> None:
        """Print layer header and open CSV file."""
        print()
        print(_fmt(f"  Layer {signed_li}", "bold", "yellow") +
              _fmt(f"  |  {n_latent:,} latent samples  {n_input:,} input samples"
                   f"  D={hidden_dim}  Z={bottleneck_dim}  epochs={epochs}", "grey"))
        print(_fmt(f"  {'epoch':>6}  {'recon':>10}  {'sparse':>10}  "
                   f"{'attn':>10}  {'total':>10}  {'best':>10}  "
                   f"{'lr':>8}  pat", "grey"))
        print(_fmt("  " + "-" * 80, "grey"))

        # Open CSV
        csv_path = os.path.join(self.log_dir, f"layer_{signed_li}_training.csv")
        fh = open(csv_path, "w", newline="")
        writer = csv.DictWriter(
            fh,
            fieldnames=["epoch", "recon", "sparse", "attn", "total", "best", "lr", "patience"],
        )
        writer.writeheader()
        self._csv_writers[signed_li] = writer
        self._csv_files[signed_li] = fh

    # ── Per-epoch log ───────────────────────────────────────────

    def log_epoch(
        self,
        signed_li: int,
        epoch: int,
        total_epochs: int,
        avg: Dict[str, float],
        best_loss: float,
        lr: float,
        patience: int,
        max_patience: int,
        improved: bool,
    ) -> None:
        """Log one epoch: write CSV row and optionally print to console."""
        row = {
            "epoch": epoch + 1,
            "recon":   round(avg["recon"],   6),
            "sparse":  round(avg["sparse"],  6),
            "attn":    round(avg["attn"],     6),
            "total":   round(avg["total"],    6),
            "best":    round(best_loss,       6),
            "lr":      f"{lr:.2e}",
            "patience": patience,
        }
        writer = self._csv_writers.get(signed_li)
        if writer:
            writer.writerow(row)
            # Flush CSV every 10 epochs so data is safe even if training crashes
            if (epoch + 1) % 10 == 0:
                self._csv_files[signed_li].flush()

        if (epoch + 1) % self.print_every == 0:
            # Colour total green if improved, yellow otherwise
            total_col = "green" if improved else "yellow"
            pat_col = "red" if patience >= max_patience * 0.8 else "grey"

            line = (
                _fmt(f"  {epoch+1:>6}/{total_epochs}", "grey") +
                f"  {avg['recon']:>10.5f}"
                f"  {avg['sparse']:>10.4f}"
                f"  {avg['attn']:>10.4f}"
                + _fmt(f"  {avg['total']:>10.5f}", total_col) +
                _fmt(f"  {best_loss:>10.5f}", "green") +
                _fmt(f"  {lr:>8.2e}", "grey") +
                _fmt(f"  {patience}/{max_patience}", pat_col)
            )
            tqdm.write(line)

    # ── Layer end ───────────────────────────────────────────────

    def end_layer(
        self,
        signed_li: int,
        stop_epoch: int,
        total_epochs: int,
        best_loss: float,
        reason: str,         # "early_stop" | "completed"
        elapsed_sec: float,
        log_history: Dict[str, List[float]],
    ) -> None:
        """Print layer summary and write JSON summary file."""
        # Close CSV
        fh = self._csv_files.get(signed_li)
        if fh:
            fh.flush()
            fh.close()

        reason_str = (
            _fmt(f"early stop at epoch {stop_epoch}", "yellow")
            if reason == "early_stop"
            else _fmt(f"completed {stop_epoch} epochs", "green")
        )
        elapsed_str = _fmt(f"{elapsed_sec:.1f}s", "grey")
        print(_fmt("  " + "-" * 80, "grey"))
        print(
            _fmt(f"  Layer {signed_li} done", "bold", "green") +
            f"  best_loss={_fmt(f'{best_loss:.6f}', 'green')}"
            f"  {reason_str}  ({elapsed_str})"
        )

        # Write JSON summary
        summary = {
            "layer": signed_li,
            "stop_epoch": stop_epoch,
            "total_epochs": total_epochs,
            "best_loss": best_loss,
            "reason": reason,
            "elapsed_sec": round(elapsed_sec, 2),
            "final_losses": {
                k: round(v[-1], 6) for k, v in log_history.items() if v
            },
        }
        summary_path = os.path.join(self.log_dir, f"layer_{signed_li}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # ── Collect-phase logging ──────────────────────────────────

    def log_collect(self, msg: str) -> None:
        """Write a message to the collect log file and print it."""
        collect_log = os.path.join(self.log_dir, "collect_log.txt")
        ts = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with open(collect_log, "a") as f:
            f.write(line + "\n")
        tqdm.write(_fmt("  [collect] ", "cyan") + msg)

    # ── Run footer ──────────────────────────────────────────────

    def finish(self) -> None:
        """Print total elapsed time."""
        elapsed = time.time() - self._run_start
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        print()
        print(_fmt(f"{'='*60}", "bold", "cyan"))
        print(_fmt(f"  AE Training finished  total time: {h:02d}:{m:02d}:{s:02d}", "bold", "cyan"))
        print(_fmt(f"{'='*60}", "bold", "cyan") + "\n")



# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AEConfig:
    """Hyperparameters for the Query-Aware AE training."""

    # Architecture
    bottleneck_dim: int = 512          # Z — compressed representation size
    gate_hidden: int = 0               # 0 = auto (hidden_dim // 2)
    encoder_layers: int = 2
    decoder_layers: int = 3
    mlp_hidden: int = 2048

    # Training
    epochs: int = 150
    batch_size: int = 256
    lr: float = 1e-3
    lr_min: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 25                 # early stopping

    # Loss weights
    lambda_sparse: float = 0.01
    lambda_attn: float = 0.1
    jacobian_sample_rows: int = 64

    # Layer selection  (negative indices → e.g. -1 = last layer)
    target_layers: List[int] = field(default_factory=lambda: [-5, -4, -3, -2, -1])

    # Data collection
    non_judger_agents: List[str] = field(default_factory=lambda: ["Planner", "Critic", "Refiner"])
    collect_all_latent_steps: bool = True   # store every latent step; False → last step only

    # Misc
    device: str = "cuda"
    checkpoint_every: int = 50          # save parquet snapshots every N batches during collect


# ──────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ──────────────────────────────────────────────────────────────────────────────

def _layer_prefix(layer_idx: int) -> str:
    """Convert layer index to file-name prefix matching existing cache layout."""
    return f"layer_{layer_idx}"


def _parquet_path(cache_dir: str, layer_idx: int, kind: str) -> str:
    """Full path to parquet file.  kind ∈ {'input', 'latent'}."""
    return os.path.join(cache_dir, f"{_layer_prefix(layer_idx)}_{kind}.parquet")


def _ae_ckpt_path(checkpoint_dir: str, layer_idx: int) -> str:
    return os.path.join(checkpoint_dir, f"ae_layer_{layer_idx}.pt")


def _log_path(checkpoint_dir: str, layer_idx: int) -> str:
    return os.path.join(checkpoint_dir, f"training_log_layer_{layer_idx}.json")


def _meta_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "cache_metadata.json")


def _save_tensor_parquet(tensor: torch.Tensor, path: str) -> None:
    """Save [N, D] float tensor as parquet (appending if file exists)."""
    arr = tensor.float().cpu().numpy()
    df = pd.DataFrame(arr)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)


def _load_tensor_parquet(path: str, device: str = "cpu") -> torch.Tensor:
    """Load parquet → [N, D] float tensor."""
    df = pd.read_parquet(path)
    return torch.tensor(df.values, dtype=torch.float32, device=device)


def _has_ae_checkpoint(checkpoint_dir: str, layer_idx: int) -> bool:
    return os.path.isfile(_ae_ckpt_path(checkpoint_dir, layer_idx))


def _has_cache(cache_dir: str, layer_idx: int) -> bool:
    return (
        os.path.isfile(_parquet_path(cache_dir, layer_idx, "input")) and
        os.path.isfile(_parquet_path(cache_dir, layer_idx, "latent"))
    )


# ──────────────────────────────────────────────────────────────────────────────
# AETrainer
# ──────────────────────────────────────────────────────────────────────────────

class AETrainer:
    """Orchestrates data collection and per-layer AE training.

    Args:
        method: a LatentMASMethod / ComLatMAS instance whose ``hidden_states_record``
                is populated after each ``run_batch`` call.
        args:   argparse.Namespace with at least {model_name, latent_steps, device}.
        ae_cfg: AEConfig (uses defaults if None).
        checkpoint_dir: root directory for checkpoints and cache.
                        Defaults to ``ae_checkpoints/paired_hidden_state_cache``.
    """

    def __init__(
        self,
        method,
        args,
        ae_cfg: Optional[AEConfig] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.method = method
        self.args = args
        self.cfg = ae_cfg or AEConfig()
        self.device = self.cfg.device

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                project_root, "ae_checkpoints", "paired_hidden_state_cache"
            )
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Determine hidden_dim from model config
        hf_model = getattr(method.model, "HF_model", None) or getattr(method.model, "model", None)
        self.hidden_dim: int = hf_model.config.hidden_size
        self.num_total_layers: int = hf_model.config.num_hidden_layers + 1  # +1 for embedding

        # Resolve negative layer indices → absolute indices  (total_layers = num_hidden_layers+1)
        self.target_layers: List[int] = [
            li if li >= 0 else self.num_total_layers + li
            for li in self.cfg.target_layers
        ]
        # Keep track of the original signed index for file naming (matching existing cache)
        self._layer_signed = {abs_idx: signed for abs_idx, signed in
                              zip(self.target_layers, self.cfg.target_layers)}

        print(f"[AETrainer] hidden_dim={self.hidden_dim}, num_layers={self.num_total_layers}")
        print(f"[AETrainer] checkpoint_dir: {self.checkpoint_dir}")

        # Logger
        self.logger = AELogger(log_dir=self.checkpoint_dir)

        # Print per-layer status table at startup
        self._print_layer_status()

    # ── File-name helper ──────────────────────────────────────────────────

    def _signed(self, abs_layer: int) -> int:
        """Return the signed layer index used in file names."""
        return self._layer_signed.get(abs_layer, abs_layer)

    def _cache_sample_counts(self, signed_li: int) -> Dict[str, int]:
        """Return {'input': N, 'latent': N} for an existing cache, or {} if missing."""
        counts = {}
        for kind in ("input", "latent"):
            path = _parquet_path(self.checkpoint_dir, signed_li, kind)
            if os.path.isfile(path):
                try:
                    counts[kind] = len(pd.read_parquet(path, columns=[0]))
                except Exception:
                    counts[kind] = -1   # corrupt / unreadable
        return counts

    def _print_layer_status(self) -> None:
        """Print a status table: for each target layer show cache + AE state."""
        print()
        print(_fmt(f"  {'layer':>6}  {'data cache':^36}  {'AE model':^12}", "grey"))
        print(_fmt("  " + "-" * 60, "grey"))
        for abs_li in self.target_layers:
            signed_li = self._signed(abs_li)
            counts = self._cache_sample_counts(signed_li)
            has_ae = _has_ae_checkpoint(self.checkpoint_dir, signed_li)

            if "input" in counts and "latent" in counts:
                cache_str = _fmt(
                    f"OK  input={counts['input']:,}  latent={counts['latent']:,}",
                    "green",
                )
            else:
                missing = [k for k in ("input", "latent") if k not in counts]
                cache_str = _fmt(f"MISSING: {', '.join(missing)}", "red")

            ae_str = _fmt("trained", "green") if has_ae else _fmt("not yet", "yellow")
            print(f"  {signed_li:>6}  {cache_str:<46}  {ae_str}")
        print()

    # ── Data collection ───────────────────────────────────────────────────

    def _which_layers_need_cache(self) -> List[int]:
        return [
            li for li in self.target_layers
            if not _has_cache(self.checkpoint_dir, self._signed(li))
        ]

    def _which_layers_need_training(self) -> List[int]:
        return [
            li for li in self.target_layers
            if not _has_ae_checkpoint(self.checkpoint_dir, self._signed(li))
        ]

    def collect_hidden_states(self, dataset_items: List[dict]) -> None:
        """Run the MAS on dataset_items and record hidden states per layer.

        Layers whose cache files (input + latent parquet) already exist are
        automatically skipped — no MAS inference is run for them.
        """
        layers_needed = self._which_layers_need_cache()

        if not layers_needed:
            self.logger.log_collect(
                "All layer caches already exist — skipping data collection."
            )
            return

        # Report which layers will be collected vs skipped
        skipped = [
            self._signed(li) for li in self.target_layers
            if li not in layers_needed
        ]
        collect = [self._signed(li) for li in layers_needed]
        if skipped:
            self.logger.log_collect(
                f"Cache already exists for layers {skipped} — skipping those."
            )
        self.logger.log_collect(
            f"Will collect data for layers {collect} "
            f"over {len(dataset_items)} samples."
        )

        # Buffers: {abs_layer: {"input": list[Tensor[D]], "latent": list[Tensor[D]]}}
        buffers: Dict[int, Dict[str, List[torch.Tensor]]] = {
            li: {"input": [], "latent": []} for li in layers_needed
        }
        flush_every = self.cfg.checkpoint_every
        n_skipped = 0
        n_collected = 0

        # Skip the judger agent entirely: its text output is not needed during
        # hidden-state collection and running it wastes significant compute.
        had_skip_judger = getattr(self.method, "skip_judger", False)
        if hasattr(self.method, "skip_judger"):
            self.method.skip_judger = True
            self.logger.log_collect("skip_judger=True — Judger agent will be bypassed.")

        try:
            for item_idx, item in enumerate(tqdm(dataset_items, desc="Collecting")):
                try:
                    self.method.run_batch([item])
                    n_collected += 1
                except Exception as e:
                    n_skipped += 1
                    self.logger.log_collect(f"Error on sample {item_idx}: {e} — skip.")
                    continue

                # All entries are non-judger agents (judger is bypassed via skip_judger=True)
                for agent_rec in self.method.hidden_states_record:
                    last_input: torch.Tensor = agent_rec["last_input_token"]   # [B, L+1, D]
                    latent_steps_hs: List[torch.Tensor] = agent_rec["latent_steps"]
                    B = last_input.shape[0]

                    for abs_li in layers_needed:
                        inp_vec = last_input[:, abs_li, :].float().cpu()       # [B, D]
                        for b in range(B):
                            buffers[abs_li]["input"].append(inp_vec[b])

                        steps = latent_steps_hs if self.cfg.collect_all_latent_steps else latent_steps_hs[-1:]
                        for step_tensor in steps:                               # [B, L+1, D]
                            lat_vec = step_tensor[:, abs_li, :].float().cpu()  # [B, D]
                            for b in range(B):
                                buffers[abs_li]["latent"].append(lat_vec[b])

                # Periodic flush to parquet
                if (item_idx + 1) % flush_every == 0:
                    self._flush_buffers(buffers, append=True)
                    buffers = {li: {"input": [], "latent": []} for li in layers_needed}
                    self.logger.log_collect(
                        f"Flushed at sample {item_idx + 1} "
                        f"({n_collected} ok, {n_skipped} skipped)"
                    )

        finally:
            # Always restore skip_judger to its original value
            if hasattr(self.method, "skip_judger"):
                self.method.skip_judger = had_skip_judger

        # Final flush
        self._flush_buffers(buffers, append=True)
        self._save_metadata(dataset_items)
        self.logger.log_collect(
            f"Collection complete. {n_collected} samples collected, "
            f"{n_skipped} skipped."
        )

    def _flush_buffers(
        self,
        buffers: Dict[int, Dict[str, List]],
        append: bool = False,
    ) -> None:
        for abs_li, splits in buffers.items():
            signed_li = self._signed(abs_li)
            for kind in ("input", "latent"):
                vecs = splits[kind]
                if not vecs:
                    continue
                tensor = torch.stack(vecs, dim=0)          # [N, D]
                path = _parquet_path(self.checkpoint_dir, signed_li, kind)
                if append and os.path.exists(path):
                    _save_tensor_parquet(tensor, path)
                else:
                    arr = tensor.float().cpu().numpy()
                    pd.DataFrame(arr).to_parquet(path, index=False)

    def _save_metadata(self, dataset_items: List[dict]) -> None:
        samples_per_layer = {}
        for abs_li in self.target_layers:
            signed_li = self._signed(abs_li)
            lat_path = _parquet_path(self.checkpoint_dir, signed_li, "latent")
            if os.path.exists(lat_path):
                samples_per_layer[str(signed_li)] = len(pd.read_parquet(lat_path))

        meta = {
            "agent_names": self.cfg.non_judger_agents,
            "layer_indices": [self._signed(li) for li in self.target_layers],
            "samples_per_layer": samples_per_layer,
            "hidden_dim": self.hidden_dim,
            "latent_steps": self.args.latent_steps,
            "benchmark": getattr(self.args, "task", "unknown"),
            "model_name": self.args.model_name,
            "num_collect_samples": len(dataset_items),
            "cache_type": "paired",
        }
        with open(_meta_path(self.checkpoint_dir), "w") as f:
            json.dump(meta, f, indent=2)

    # ── Training ──────────────────────────────────────────────────────────

    def train_all_layers(self) -> None:
        """Train a QueryAwareAutoencoder for each target layer that is missing a checkpoint."""
        layers_to_train = self._which_layers_need_training()
        if not layers_to_train:
            print("[AETrainer] All AE checkpoints exist — skip training.")
            return

        print(f"[AETrainer] Training AE for {len(layers_to_train)} layers …")

        for abs_li in layers_to_train:
            signed_li = self._signed(abs_li)
            print(f"\n[AETrainer] === Layer {signed_li} (abs {abs_li}) ===")
            self._train_one_layer(abs_li, signed_li)

    def _train_one_layer(self, abs_li: int, signed_li: int) -> None:
        input_path = _parquet_path(self.checkpoint_dir, signed_li, "input")
        latent_path = _parquet_path(self.checkpoint_dir, signed_li, "latent")

        if not os.path.exists(input_path) or not os.path.exists(latent_path):
            raise FileNotFoundError(
                f"Cache files missing for layer {signed_li}. Run collect_hidden_states first."
            )

        # Load data
        H_input  = _load_tensor_parquet(input_path)   # [N_input, D]
        H_latent = _load_tensor_parquet(latent_path)   # [N_latent, D]

        # Pair up: each latent step row pairs with the corresponding input row
        # Input rows are repeated (one per agent sample × latent_steps)
        # We tile H_input to match H_latent length
        n_lat = len(H_latent)
        n_inp = len(H_input)
        repeat = math.ceil(n_lat / n_inp)
        H_input_tiled = H_input.repeat(repeat, 1)[:n_lat]  # [N_latent, D]

        # Normalize
        mean_lat = H_latent.mean(dim=0)
        std_lat  = H_latent.std(dim=0).clamp(min=1e-8)
        mean_inp = H_input_tiled.mean(dim=0)
        std_inp  = H_input_tiled.std(dim=0).clamp(min=1e-8)

        H_lat_norm = (H_latent      - mean_lat) / std_lat
        H_inp_norm = (H_input_tiled - mean_inp) / std_inp

        dataset = TensorDataset(
            H_lat_norm.to(self.device),
            H_inp_norm.to(self.device),
            H_latent.to(self.device),   # keep un-normalized for reconstruction target
        )
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        # Model
        model = QueryAwareAutoencoder(
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.cfg.bottleneck_dim,
            gate_hidden=self.cfg.gate_hidden or None,
            encoder_layers=self.cfg.encoder_layers,
            decoder_layers=self.cfg.decoder_layers,
            mlp_hidden=self.cfg.mlp_hidden,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.lr_min)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        log_history = {"recon": [], "sparse": [], "attn": [], "total": []}

        self.logger.begin_layer(
            signed_li=signed_li,
            n_latent=len(H_latent),
            n_input=len(H_input),
            epochs=self.cfg.epochs,
            bottleneck_dim=self.cfg.bottleneck_dim,
            hidden_dim=self.hidden_dim,
        )

        layer_start = time.time()
        stop_epoch = self.cfg.epochs
        stop_reason = "completed"

        iterator = tqdm(range(self.cfg.epochs), desc=f"AE layer {signed_li}", leave=False)
        for epoch in iterator:
            epoch_stats = {"recon": 0.0, "sparse": 0.0, "attn": 0.0, "total": 0.0}
            n_batches = 0

            for h_lat_n, h_inp_n, h_lat_orig in loader:
                optimizer.zero_grad()

                # Forward — inputs are already normalized
                h_rec, z, gate = model(h_lat_n, h_inp_n, normalize=False)

                # Reconstruction target = normalized latent (same space as h_rec)
                l_total, breakdown = total_loss(
                    h_lat_n, h_rec, z, gate, model.decoder,
                    lambda_sparse=self.cfg.lambda_sparse,
                    lambda_attn=self.cfg.lambda_attn,
                    jacobian_sample_rows=self.cfg.jacobian_sample_rows,
                )

                l_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                optimizer.step()

                for k, v in breakdown.items():
                    epoch_stats[k] += v
                n_batches += 1

            scheduler.step()

            avg = {k: v / n_batches for k, v in epoch_stats.items()}
            for k in log_history:
                log_history[k].append(avg[k])

            improved = avg["total"] < best_loss
            if improved:
                best_loss = avg["total"]
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            current_lr = optimizer.param_groups[0]["lr"]
            self.logger.log_epoch(
                signed_li=signed_li,
                epoch=epoch,
                total_epochs=self.cfg.epochs,
                avg=avg,
                best_loss=best_loss,
                lr=current_lr,
                patience=patience_counter,
                max_patience=self.cfg.patience,
                improved=improved,
            )

            if patience_counter >= self.cfg.patience:
                tqdm.write(
                    _fmt(f"  Early stop at epoch {epoch+1} "
                         f"(no improvement for {self.cfg.patience} epochs).",
                         "yellow")
                )
                stop_epoch = epoch + 1
                stop_reason = "early_stop"
                break

        # Restore best, embed norm stats
        if best_state is not None:
            model.load_state_dict(best_state)
        model.set_norm_stats(mean_lat, std_lat, mean_inp, std_inp)

        self._save_ae_checkpoint(model, signed_li, log_history)

        self.logger.end_layer(
            signed_li=signed_li,
            stop_epoch=stop_epoch,
            total_epochs=self.cfg.epochs,
            best_loss=best_loss,
            reason=stop_reason,
            elapsed_sec=time.time() - layer_start,
            log_history=log_history,
        )

    # ── Checkpoint I/O ────────────────────────────────────────────────────

    def _save_ae_checkpoint(
        self,
        model: QueryAwareAutoencoder,
        signed_li: int,
        log_history: dict,
    ) -> None:
        ckpt = {
            "state_dict": model.state_dict(),
            "config": {
                "hidden_dim":      model.hidden_dim,
                "bottleneck_dim":  model.bottleneck_dim,
                "gate_hidden":     model.feature_gate.gate_net[0].out_features,
                "encoder_layers":  self.cfg.encoder_layers,
                "decoder_layers":  self.cfg.decoder_layers,
                "mlp_hidden":      self.cfg.mlp_hidden,
            },
        }
        torch.save(ckpt, _ae_ckpt_path(self.checkpoint_dir, signed_li))
        with open(_log_path(self.checkpoint_dir, signed_li), "w") as f:
            json.dump(log_history, f, indent=2)

    def load_ae(self, layer_signed_idx: int) -> QueryAwareAutoencoder:
        """Load a trained AE for the given signed layer index."""
        path = _ae_ckpt_path(self.checkpoint_dir, layer_signed_idx)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No AE checkpoint at {path}")
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt["config"]
        model = QueryAwareAutoencoder(
            hidden_dim=cfg["hidden_dim"],
            bottleneck_dim=cfg["bottleneck_dim"],
            gate_hidden=cfg.get("gate_hidden"),
            encoder_layers=cfg["encoder_layers"],
            decoder_layers=cfg["decoder_layers"],
            mlp_hidden=cfg["mlp_hidden"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device).eval()
        return model

    def load_all_aes(self) -> Dict[int, QueryAwareAutoencoder]:
        """Load AE for every target layer.  Returns {signed_layer_idx: model}."""
        models = {}
        for abs_li in self.target_layers:
            signed_li = self._signed(abs_li)
            models[signed_li] = self.load_ae(signed_li)
        return models

    # ── Main entry point ──────────────────────────────────────────────────

    def collect_and_train(self, dataset_items: List[dict]) -> None:
        """Full pipeline: collect missing caches, then train missing AEs."""
        self.collect_hidden_states(dataset_items)
        self.train_all_layers()
        self.logger.finish()


# ── math import needed in _train_one_layer ────────────────────────────────────
import math
