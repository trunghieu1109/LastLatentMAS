"""
WCT Parameter Builder
=====================
Precompute WCT (Whitening-Coloring Transform) parameters for each layer,
aligning hidden states from layer_i → target_layer (last layer).

Saves one .npz file per layer containing: W_total, mean_source, mean_target.
Also saves a combined index file for easy loading during inference.

Usage:
    python3 build_wct_params.py \
        --data_dir latent_data/medqa \
        --out_dir wct_params \
        --target_layer 36

    # Use all available layers
    python3 build_wct_params.py \
        --data_dir latent_data/medqa \
        --out_dir wct_params

    # Specify layers
    python3 build_wct_params.py \
        --data_dir latent_data/medqa \
        --out_dir wct_params \
        --source_layers 0,5,10,18,25,34
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_parquet_robust(path: str) -> np.ndarray:
    import pyarrow.parquet as pq
    import pyarrow as pa
    table = pq.ParquetFile(path).read()
    numeric_indices = [
        i for i, f in enumerate(table.schema)
        if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)
    ]
    col_names = [table.schema.field(i).name for i in numeric_indices]
    unique_names = set(col_names)
    if len(unique_names) < len(col_names):
        n_unique = len(unique_names)
        arr = np.column_stack([
            table.column(numeric_indices[i]).to_numpy(zero_copy_only=False)
            for i in range(n_unique)
        ])
    else:
        arr = np.column_stack([
            table.column(i).to_numpy(zero_copy_only=False)
            for i in numeric_indices
        ])
    return arr.astype(np.float32)


def load_latent(data_dir: str, layer_idx: int) -> np.ndarray:
    """Load latent hidden states for a layer. Returns [N, D]."""
    path = os.path.join(data_dir, f"layer_{layer_idx}_latent.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return _load_parquet_robust(path)


# ──────────────────────────────────────────────────────────────────────────────
# WCT fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_wct(H_source: np.ndarray, H_target: np.ndarray, eps: float = 1e-5) -> dict:
    """Fit WCT parameters from source to target distribution.

    Returns dict with:
        W_total:     [D, D]  — combined whiten+color matrix
        mean_source: [D]     — mean of source distribution
        mean_target: [D]     — mean of target distribution

    Usage at inference:
        h_aligned = (h - mean_source) @ W_total + mean_target
    """
    assert H_source.shape[1] == H_target.shape[1], "Dimension mismatch"
    D = H_source.shape[1]

    mean_source = H_source.mean(axis=0)
    mean_target = H_target.mean(axis=0)

    H_s_c = H_source - mean_source
    H_t_c = H_target - mean_target

    Cov_s = (H_s_c.T @ H_s_c) / len(H_source) + eps * np.eye(D, dtype=np.float32)
    Cov_t = (H_t_c.T @ H_t_c) / len(H_target) + eps * np.eye(D, dtype=np.float32)

    # Eigendecompose
    eigvals_s, eigvecs_s = np.linalg.eigh(Cov_s)
    eigvals_t, eigvecs_t = np.linalg.eigh(Cov_t)

    # Whitening: E_s @ Λ_s^{-1/2} @ E_s^T
    D_inv_sqrt_s = np.diag(1.0 / np.sqrt(np.maximum(eigvals_s, eps)))
    W_whiten = eigvecs_s @ D_inv_sqrt_s @ eigvecs_s.T

    # Coloring: E_t @ Λ_t^{1/2} @ E_t^T
    D_sqrt_t = np.diag(np.sqrt(np.maximum(eigvals_t, eps)))
    W_color = eigvecs_t @ D_sqrt_t @ eigvecs_t.T

    W_total = (W_whiten @ W_color).astype(np.float32)

    return {
        "W_total": W_total,
        "mean_source": mean_source.astype(np.float32),
        "mean_target": mean_target.astype(np.float32),
    }


def apply_wct(h: np.ndarray, params: dict) -> np.ndarray:
    """Apply precomputed WCT. h can be [D] or [N, D]."""
    return (h - params["mean_source"]) @ params["W_total"] + params["mean_target"]


# ──────────────────────────────────────────────────────────────────────────────
# Loader for inference
# ──────────────────────────────────────────────────────────────────────────────

class WCTAligner:
    """Load and apply precomputed WCT parameters.

    Usage:
        aligner = WCTAligner.load("wct_params/")
        h_aligned = aligner.transform(h_layer_10, source_layer=10)

        # Or with torch tensors:
        h_aligned = aligner.transform_torch(h_tensor, source_layer=10)
    """

    def __init__(self, params_dir: str):
        self.params_dir = params_dir
        self.params = {}  # layer_idx → {W_total, mean_source, mean_target}
        self.target_layer = None
        self.hidden_dim = None

        # Load index
        index_path = os.path.join(params_dir, "wct_index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.index = json.load(f)
            self.target_layer = self.index["target_layer"]
            self.hidden_dim = self.index["hidden_dim"]

    @classmethod
    def load(cls, params_dir: str):
        return cls(params_dir)

    def _ensure_loaded(self, layer_idx: int):
        """Lazy-load WCT params for a layer."""
        if layer_idx in self.params:
            return
        path = os.path.join(self.params_dir, f"wct_layer_{layer_idx}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No WCT params for layer {layer_idx}. "
                f"Available: {self.available_layers()}"
            )
        data = np.load(path)
        self.params[layer_idx] = {
            "W_total": data["W_total"],
            "mean_source": data["mean_source"],
            "mean_target": data["mean_target"],
        }

    def available_layers(self) -> list:
        """List layers with precomputed WCT params."""
        if hasattr(self, "index"):
            return self.index.get("source_layers", [])
        import glob, re
        files = glob.glob(os.path.join(self.params_dir, "wct_layer_*.npz"))
        layers = []
        for f in files:
            m = re.search(r"wct_layer_(\d+)\.npz", os.path.basename(f))
            if m:
                layers.append(int(m.group(1)))
        return sorted(layers)

    def transform(self, h: np.ndarray, source_layer: int) -> np.ndarray:
        """Apply WCT alignment. h: [D] or [N, D] numpy array."""
        self._ensure_loaded(source_layer)
        p = self.params[source_layer]
        return (h - p["mean_source"]) @ p["W_total"] + p["mean_target"]

    def transform_torch(self, h_tensor, source_layer: int):
        """Apply WCT alignment to a torch tensor. Returns torch tensor."""
        import torch
        self._ensure_loaded(source_layer)
        p = self.params[source_layer]
        device = h_tensor.device
        dtype = h_tensor.dtype
        W = torch.from_numpy(p["W_total"]).to(device=device, dtype=dtype)
        m_s = torch.from_numpy(p["mean_source"]).to(device=device, dtype=dtype)
        m_t = torch.from_numpy(p["mean_target"]).to(device=device, dtype=dtype)
        return (h_tensor - m_s) @ W + m_t


# ──────────────────────────────────────────────────────────────────────────────
# Main: Build WCT params
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Precompute WCT parameters for all layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory with parquet files")
    parser.add_argument("--out_dir", default="wct_params",
                        help="Output directory for WCT parameters")
    parser.add_argument("--target_layer", type=int, default=None,
                        help="Target layer (default: last)")
    parser.add_argument("--source_layers", type=str, default="all",
                        help="Comma-separated layers or 'all'")
    parser.add_argument("--eps", type=float, default=1e-5,
                        help="Regularization for eigenvalues")

    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(ROOT, out_dir)

    # Load metadata
    meta_path = os.path.join(data_dir, "cache_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    all_layers = sorted(meta.get("target_layers", []))
    hidden_dim = meta.get("hidden_dim", None)
    target_layer = args.target_layer or max(all_layers)

    if args.source_layers == "all":
        source_layers = [l for l in all_layers if l != target_layer]
    else:
        source_layers = [int(x.strip()) for x in args.source_layers.split(",")]
        source_layers = [l for l in source_layers if l != target_layer]

    print(f"Data dir:     {data_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Target layer: {target_layer}")
    print(f"Source layers: {len(source_layers)} layers → {source_layers}")
    print(f"Hidden dim:   {hidden_dim}")
    print(f"Eps:          {args.eps}")
    print()

    # Load target layer
    print(f"Loading target layer {target_layer}...")
    t0 = time.time()
    H_target = load_latent(data_dir, target_layer)
    print(f"  Shape: {H_target.shape}  ({time.time()-t0:.1f}s)")
    if hidden_dim is None:
        hidden_dim = H_target.shape[1]

    os.makedirs(out_dir, exist_ok=True)
    fitted_layers = []

    for i, src_layer in enumerate(source_layers):
        print(f"\n[{i+1}/{len(source_layers)}] Layer {src_layer} → Layer {target_layer}")

        # Load source
        t0 = time.time()
        H_source = load_latent(data_dir, src_layer)
        print(f"  Loaded: {H_source.shape}  ({time.time()-t0:.1f}s)")

        # Ensure same N
        n = min(len(H_source), len(H_target))
        H_source = H_source[:n]
        H_target_use = H_target[:n]
        print(f"  Using {n} vectors for fitting (N/D = {n/hidden_dim:.1f})")

        # Fit WCT
        t0 = time.time()
        params = fit_wct(H_source, H_target_use, eps=args.eps)
        fit_time = time.time() - t0
        print(f"  WCT fitted in {fit_time:.1f}s")

        # Quick validation
        H_aligned = apply_wct(H_source[:100], params)
        norms_aligned = np.linalg.norm(H_aligned, axis=1)
        norms_target = np.linalg.norm(H_target_use[:100], axis=1)
        print(f"  Sanity check (first 100 vectors):")
        print(f"    Aligned norm: {norms_aligned.mean():.1f} ± {norms_aligned.std():.1f}")
        print(f"    Target norm:  {norms_target.mean():.1f} ± {norms_target.std():.1f}")

        # Save
        out_path = os.path.join(out_dir, f"wct_layer_{src_layer}.npz")
        np.savez_compressed(
            out_path,
            W_total=params["W_total"],
            mean_source=params["mean_source"],
            mean_target=params["mean_target"],
        )
        fsize = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  → Saved: {out_path} ({fsize:.1f} MB)")
        fitted_layers.append(src_layer)

    # Save index
    index = {
        "target_layer": target_layer,
        "source_layers": sorted(fitted_layers),
        "hidden_dim": hidden_dim,
        "eps": args.eps,
        "data_dir": data_dir,
        "model_name": meta.get("model_name", ""),
        "n_samples_used": n,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    index_path = os.path.join(out_dir, "wct_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n→ Index saved: {index_path}")

    # Print loading example
    print(f"\n{'='*60}")
    print("Usage in inference:")
    print(f"{'='*60}")
    print(f"""
from build_wct_params import WCTAligner

# Load (lazy — only loads params when first used)
aligner = WCTAligner.load("{args.out_dir}")

# With numpy
h_aligned = aligner.transform(h_layer_10, source_layer=10)

# With torch
h_aligned = aligner.transform_torch(h_tensor, source_layer=10)

# Check available layers
print(aligner.available_layers())
""")

    total_size = sum(
        os.path.getsize(os.path.join(out_dir, f"wct_layer_{l}.npz"))
        for l in fitted_layers
    ) / (1024 * 1024)
    print(f"✅ Done. {len(fitted_layers)} layers, total {total_size:.1f} MB in {out_dir}/")


if __name__ == "__main__":
    main()
