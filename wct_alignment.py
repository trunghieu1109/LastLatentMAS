"""
Whitening-Coloring Transform (WCT) for Hidden State Space Alignment
====================================================================
Pipeline: h_latent_i → AE_i(h_latent_i, h_input_i) → WCT → space of layer L

Usage:
    # AE + WCT for selected layers
    python3 wct_alignment.py --data_dir latent_data/medqa \
        --ae_checkpoint_dir ae_checkpoints/paired_hidden_state_cache \
        --source_layers 0,5,10,18,25,34

    # All layers
    python3 wct_alignment.py --data_dir latent_data/medqa \
        --ae_checkpoint_dir ae_checkpoints/paired_hidden_state_cache \
        --source_layers all
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ──────────────────────────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────────────────────────

COLOR_RAW       = "#2B8C8C"  # teal   — raw latent (before AE)
COLOR_AE        = "#5B7EC8"  # blue   — after AE only
COLOR_AE_WCT    = "#D4A030"  # amber  — after AE + WCT
COLOR_TARGET    = "#E06050"  # coral  — target layer (actual last layer)


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor":   "#F7F7F7",
        "axes.edgecolor":   "#CCCCCC",
        "axes.labelcolor":  "#333333",
        "xtick.color":      "#555555",
        "ytick.color":      "#555555",
        "text.color":       "#222222",
        "grid.color":       "#E0E0E0",
        "grid.linewidth":   0.6,
        "font.family":      "sans-serif",
        "font.size":        10,
    })
    return plt


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


def load_layer_data(data_dir: str, layer_idx: int, n_agents: int = 3, latent_steps: int = 50):
    """Load both input and latent hidden states for a layer, with IDs."""
    H_input  = _load_parquet_robust(os.path.join(data_dir, f"layer_{layer_idx}_input.parquet"))
    H_latent = _load_parquet_robust(os.path.join(data_dir, f"layer_{layer_idx}_latent.parquet"))

    n_inp = len(H_input)
    n_lat = len(H_latent)
    n_q = n_inp // n_agents

    # Build query/agent IDs
    q_ids_inp = np.repeat(np.arange(n_q), n_agents)
    a_ids_inp = np.tile(np.arange(n_agents), n_q)

    latent_per_query = n_agents * latent_steps
    q_ids_lat = np.repeat(np.arange(n_q), latent_per_query)
    a_ids_lat = np.tile(np.repeat(np.arange(n_agents), latent_steps), n_q)

    return {
        "H_input": H_input, "H_latent": H_latent,
        "q_ids_inp": q_ids_inp, "a_ids_inp": a_ids_inp,
        "q_ids_lat": q_ids_lat, "a_ids_lat": a_ids_lat,
        "n_q": n_q,
    }


# ──────────────────────────────────────────────────────────────────────────────
# AE helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_ae_for_layer(ae_checkpoint_dir: str, signed_layer: int, device: str = "cpu"):
    """Load a QueryAwareAutoencoder for one layer."""
    from methods.query_aware_ae import QueryAwareAutoencoder
    import torch

    ckpt_path = os.path.join(ae_checkpoint_dir, f"ae_layer_{signed_layer}.pt")
    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ae = QueryAwareAutoencoder(
        hidden_dim     = cfg["hidden_dim"],
        bottleneck_dim = cfg["bottleneck_dim"],
        gate_hidden    = cfg.get("gate_hidden"),
        encoder_layers = cfg.get("encoder_layers", 2),
        decoder_layers = cfg.get("decoder_layers", 3),
        mlp_hidden     = cfg.get("mlp_hidden", 2048),
    ).to(device)
    sd_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
    ae.load_state_dict(ckpt[sd_key])
    ae.eval()
    return ae


def apply_ae(ae, H_latent, H_input, q_ids_lat, a_ids_lat, q_ids_inp, a_ids_inp, device="cpu"):
    """Apply AE reconstruction. Returns H_ae [N_lat, D]."""
    import torch

    H_rec = np.zeros_like(H_latent)

    # Build lookup: (query_id, agent_id) → input row index
    inp_lookup = {}
    for i in range(len(q_ids_inp)):
        key = (int(q_ids_inp[i]), int(a_ids_inp[i]))
        inp_lookup[key] = i

    batch_size = 256
    n = len(H_latent)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_lat = H_latent[start:end]
        batch_q = q_ids_lat[start:end]
        batch_a = a_ids_lat[start:end]

        inp_indices = [inp_lookup.get((int(q), int(a)), 0) for q, a in zip(batch_q, batch_a)]
        batch_inp = H_input[inp_indices]

        with torch.no_grad():
            h_lat = torch.from_numpy(batch_lat).float().to(device)
            h_inp = torch.from_numpy(batch_inp).float().to(device)
            h_rec, _z, gate = ae(h_lat, h_inp, normalize=True)
            if ae._has_norm:
                h_rec = h_rec * ae._norm_std_latent.to(device) + ae._norm_mean_latent.to(device)
            h_rec = (gate * h_rec).cpu().numpy()

        H_rec[start:end] = h_rec

    return H_rec.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# WCT Core
# ──────────────────────────────────────────────────────────────────────────────

class WCT:
    """Whitening-Coloring Transform for space alignment."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.mean_source = None
        self.mean_target = None
        self.W_total = None

    def fit(self, H_source: np.ndarray, H_target: np.ndarray):
        assert H_source.shape[1] == H_target.shape[1], "Dimension mismatch"
        D = H_source.shape[1]
        self.mean_source = H_source.mean(axis=0)
        self.mean_target = H_target.mean(axis=0)

        H_s_c = H_source - self.mean_source
        H_t_c = H_target - self.mean_target

        Cov_s = (H_s_c.T @ H_s_c) / len(H_source) + self.eps * np.eye(D, dtype=np.float32)
        Cov_t = (H_t_c.T @ H_t_c) / len(H_target) + self.eps * np.eye(D, dtype=np.float32)

        eigvals_s, eigvecs_s = np.linalg.eigh(Cov_s)
        eigvals_t, eigvecs_t = np.linalg.eigh(Cov_t)

        D_inv_sqrt_s = np.diag(1.0 / np.sqrt(np.maximum(eigvals_s, self.eps)))
        W_whiten = eigvecs_s @ D_inv_sqrt_s @ eigvecs_s.T

        D_sqrt_t = np.diag(np.sqrt(np.maximum(eigvals_t, self.eps)))
        W_color = eigvecs_t @ D_sqrt_t @ eigvecs_t.T

        self.W_total = (W_whiten @ W_color).astype(np.float32)
        return self

    def transform(self, H: np.ndarray) -> np.ndarray:
        assert self.W_total is not None, "Must call fit() first"
        return (H - self.mean_source) @ self.W_total + self.mean_target


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(H_a: np.ndarray, H_b: np.ndarray) -> dict:
    """Compute alignment metrics between two arrays."""
    norms_a = np.linalg.norm(H_a, axis=1, keepdims=True).clip(1e-8)
    norms_b = np.linalg.norm(H_b, axis=1, keepdims=True).clip(1e-8)
    cos_sim = ((H_a / norms_a) * (H_b / norms_b)).sum(axis=1)
    l2_dist = np.linalg.norm(H_a - H_b, axis=1)
    norm_ratio = norms_a.ravel() / norms_b.ravel()
    mean_diff = np.linalg.norm(H_a.mean(axis=0) - H_b.mean(axis=0))

    return {
        "cosine_sim_mean": float(cos_sim.mean()),
        "cosine_sim_std":  float(cos_sim.std()),
        "l2_dist_mean":    float(l2_dist.mean()),
        "norm_ratio_mean": float(norm_ratio.mean()),
        "mean_diff":       float(mean_diff),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def pca_project(*arrays):
    from sklearn.decomposition import PCA
    combined = np.vstack(arrays)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)
    result, offset = [], 0
    for arr in arrays:
        n = len(arr)
        result.append(proj[offset:offset + n])
        offset += n
    return result, pca.explained_variance_ratio_


def plot_single_layer(
    plt,
    H_raw: np.ndarray,
    H_ae: np.ndarray,
    H_ae_wct: np.ndarray,
    H_target: np.ndarray,
    source_layer: int,
    target_layer: int,
    metrics_ae: dict,
    metrics_ae_wct: dict,
    out_dir: str,
    max_points: int = 2000,
):
    """Plot 4-panel comparison: Raw, AE-only, AE+WCT, Overlay."""
    import matplotlib.gridspec as gridspec

    # Subsample
    n = len(H_raw)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        H_raw = H_raw[idx]
        H_ae = H_ae[idx]
        H_ae_wct = H_ae_wct[idx]
        H_target = H_target[idx]

    # Joint PCA over all four
    [proj_raw, proj_ae, proj_wct, proj_tgt], var_ratio = pca_project(
        H_raw, H_ae, H_ae_wct, H_target
    )

    fig = plt.figure(figsize=(28, 7))
    fig.suptitle(
        f"AE + WCT Pipeline: Layer {source_layer} → Layer {target_layer}  |  "
        f"D={H_raw.shape[1]}  |  {len(H_raw)} vectors",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.28)

    # ── Panel 1: Raw latent vs Target ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(proj_raw[:, 0], proj_raw[:, 1],
                c=COLOR_RAW, marker="s", s=10, alpha=0.35,
                edgecolors="none", label=f"Raw latent (L{source_layer})")
    ax1.scatter(proj_tgt[:, 0], proj_tgt[:, 1],
                c=COLOR_TARGET, marker="D", s=10, alpha=0.35,
                edgecolors="none", label=f"Target (L{target_layer})")
    ax1.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax1.set_title("Raw Latent vs Target", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, facecolor="white", edgecolor="#CCC", framealpha=0.9)

    # ── Panel 2: AE-only vs Target ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(proj_ae[:, 0], proj_ae[:, 1],
                c=COLOR_AE, marker="o", s=10, alpha=0.35,
                edgecolors="none", label=f"AE-filtered (L{source_layer})")
    ax2.scatter(proj_tgt[:, 0], proj_tgt[:, 1],
                c=COLOR_TARGET, marker="D", s=10, alpha=0.35,
                edgecolors="none", label=f"Target (L{target_layer})")
    ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax2.set_title("AE-filtered vs Target", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, facecolor="white", edgecolor="#CCC", framealpha=0.9)

    info_ae = (
        f"cos: {metrics_ae['cosine_sim_mean']:.3f}±{metrics_ae['cosine_sim_std']:.3f}\n"
        f"L2: {metrics_ae['l2_dist_mean']:.1f}\n"
        f"norm: {metrics_ae['norm_ratio_mean']:.3f}"
    )
    ax2.text(0.02, 0.98, info_ae, transform=ax2.transAxes, va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#CCC"))

    # ── Panel 3: AE+WCT vs Target ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(proj_wct[:, 0], proj_wct[:, 1],
                c=COLOR_AE_WCT, marker="^", s=10, alpha=0.35,
                edgecolors="none", label=f"AE+WCT (L{source_layer}→L{target_layer})")
    ax3.scatter(proj_tgt[:, 0], proj_tgt[:, 1],
                c=COLOR_TARGET, marker="D", s=10, alpha=0.35,
                edgecolors="none", label=f"Target (L{target_layer})")
    ax3.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax3.set_title("AE + WCT vs Target", fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7, facecolor="white", edgecolor="#CCC", framealpha=0.9)

    info_wct = (
        f"cos: {metrics_ae_wct['cosine_sim_mean']:.3f}±{metrics_ae_wct['cosine_sim_std']:.3f}\n"
        f"L2: {metrics_ae_wct['l2_dist_mean']:.1f}\n"
        f"norm: {metrics_ae_wct['norm_ratio_mean']:.3f}"
    )
    ax3.text(0.02, 0.98, info_wct, transform=ax3.transAxes, va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#CCC"))

    # ── Panel 4: All overlaid ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(proj_raw[:, 0], proj_raw[:, 1],
                c=COLOR_RAW, marker="s", s=8, alpha=0.20,
                edgecolors="none", label="Raw latent")
    ax4.scatter(proj_ae[:, 0], proj_ae[:, 1],
                c=COLOR_AE, marker="o", s=8, alpha=0.20,
                edgecolors="none", label="AE-filtered")
    ax4.scatter(proj_wct[:, 0], proj_wct[:, 1],
                c=COLOR_AE_WCT, marker="^", s=8, alpha=0.30,
                edgecolors="none", label="AE+WCT")
    ax4.scatter(proj_tgt[:, 0], proj_tgt[:, 1],
                c=COLOR_TARGET, marker="D", s=8, alpha=0.30,
                edgecolors="none", label="Target")
    ax4.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax4.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax4.set_title("All Overlaid", fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=7, facecolor="white", edgecolor="#CCC", framealpha=0.9,
               loc="lower right")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ae_wct_L{source_layer}_to_L{target_layer}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {out_path}")


def plot_summary(plt, all_metrics, target_layer, out_dir):
    """Summary comparing AE-only vs AE+WCT across layers."""
    import matplotlib.gridspec as gridspec

    layers = sorted(all_metrics.keys())
    n = len(layers)
    x = np.arange(n)
    w = 0.35
    labels = [f"L{l}" for l in layers]

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(
        f"AE vs AE+WCT  |  Target: Layer {target_layer}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30)

    # Cosine sim
    ax1 = fig.add_subplot(gs[0, 0])
    ae_cos  = [all_metrics[l]["ae"]["cosine_sim_mean"] for l in layers]
    wct_cos = [all_metrics[l]["ae_wct"]["cosine_sim_mean"] for l in layers]
    ax1.bar(x - w/2, ae_cos, w, color=COLOR_AE, alpha=0.7, label="AE only")
    ax1.bar(x + w/2, wct_cos, w, color=COLOR_AE_WCT, alpha=0.7, label="AE + WCT")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7, rotation=45)
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Cos Sim vs Target", fontweight="bold", fontsize=10)
    ax1.axhline(1.0, color="#888", ls="--", lw=0.8, alpha=0.5)
    ax1.legend(fontsize=8); ax1.grid(True, axis="y", alpha=0.3)

    # Norm ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ae_norm  = [all_metrics[l]["ae"]["norm_ratio_mean"] for l in layers]
    wct_norm = [all_metrics[l]["ae_wct"]["norm_ratio_mean"] for l in layers]
    ax2.bar(x - w/2, ae_norm, w, color=COLOR_AE, alpha=0.7, label="AE only")
    ax2.bar(x + w/2, wct_norm, w, color=COLOR_AE_WCT, alpha=0.7, label="AE + WCT")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7, rotation=45)
    ax2.set_ylabel("Norm Ratio (aligned / target)")
    ax2.set_title("Norm Ratio", fontweight="bold", fontsize=10)
    ax2.axhline(1.0, color="#888", ls="--", lw=0.8, alpha=0.5)
    ax2.legend(fontsize=8); ax2.grid(True, axis="y", alpha=0.3)

    # L2 distance
    ax3 = fig.add_subplot(gs[0, 2])
    ae_l2  = [all_metrics[l]["ae"]["l2_dist_mean"] for l in layers]
    wct_l2 = [all_metrics[l]["ae_wct"]["l2_dist_mean"] for l in layers]
    ax3.bar(x - w/2, ae_l2, w, color=COLOR_AE, alpha=0.7, label="AE only")
    ax3.bar(x + w/2, wct_l2, w, color=COLOR_AE_WCT, alpha=0.7, label="AE + WCT")
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=7, rotation=45)
    ax3.set_ylabel("L2 Distance")
    ax3.set_title("L2 Dist vs Target", fontweight="bold", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ae_wct_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AE + WCT alignment pipeline")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ae_checkpoint_dir", required=True)
    parser.add_argument("--source_layers", type=str, default="0,10,18,25,34",
                        help="Comma-separated layers or 'all'")
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--out_dir", default="wct_analysis")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)
    ae_dir = args.ae_checkpoint_dir
    if not os.path.isabs(ae_dir):
        ae_dir = os.path.join(ROOT, ae_dir)

    # Load metadata
    meta_path = os.path.join(data_dir, "cache_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    all_layers = meta.get("target_layers", [])
    n_total_layers = meta.get("num_total_layers", max(all_layers) + 1)
    latent_steps = meta.get("latent_steps", 50)
    n_agents = len(meta.get("agent_names", ["A", "B", "C"]))

    target_layer = args.target_layer or max(all_layers)
    print(f"Target layer: {target_layer}")
    print(f"Model: {meta.get('model_name', '?')}, D={meta.get('hidden_dim', '?')}")
    print(f"Agents: {n_agents}, Latent steps: {latent_steps}\n")

    # Parse source layers
    if args.source_layers == "all":
        source_layers = [l for l in all_layers if l != target_layer]
    else:
        source_layers = [int(x.strip()) for x in args.source_layers.split(",")]
        source_layers = [l for l in source_layers if l != target_layer]

    # Discover AE checkpoints (signed layer mapping)
    ae_files = glob.glob(os.path.join(ae_dir, "ae_layer_*.pt"))
    ae_signed_map = {}  # abs_layer → signed_layer
    for f in ae_files:
        m = re.search(r"ae_layer_([-]?\d+)\.pt", os.path.basename(f))
        if m:
            signed = int(m.group(1))
            abs_idx = signed if signed >= 0 else n_total_layers + signed
            ae_signed_map[abs_idx] = signed

    print(f"AE checkpoints found for {len(ae_signed_map)} layers")
    print(f"Source layers: {source_layers}\n")

    # Load target layer data (latent vectors)
    print(f"Loading target layer {target_layer}...")
    tgt_data = load_layer_data(data_dir, target_layer, n_agents, latent_steps)
    H_target_all = tgt_data["H_latent"]
    print(f"  Target latent shape: {H_target_all.shape}")

    plt = _setup_mpl()
    out_dir = os.path.join(ROOT, args.out_dir)
    all_summary = {}

    for src_layer in source_layers:
        print(f"\n{'='*60}")
        print(f"Layer {src_layer} → Layer {target_layer}")
        print(f"{'='*60}")

        # Check AE availability
        if src_layer not in ae_signed_map:
            print(f"  ⚠ No AE checkpoint for layer {src_layer}, skipping")
            continue

        signed_layer = ae_signed_map[src_layer]

        # Load source data
        src_data = load_layer_data(data_dir, src_layer, n_agents, latent_steps)
        H_latent_src = src_data["H_latent"]
        H_input_src  = src_data["H_input"]
        print(f"  Source latent: {H_latent_src.shape}, input: {H_input_src.shape}")

        # Load AE
        print(f"  Loading AE (signed layer {signed_layer})...")
        ae = load_ae_for_layer(ae_dir, signed_layer, args.device)
        if ae is None:
            print(f"  ⚠ Failed to load AE, skipping")
            continue

        # Apply AE reconstruction
        print(f"  Applying AE reconstruction...")
        H_ae = apply_ae(
            ae, H_latent_src, H_input_src,
            src_data["q_ids_lat"], src_data["a_ids_lat"],
            src_data["q_ids_inp"], src_data["a_ids_inp"],
            device=args.device,
        )
        print(f"  AE output shape: {H_ae.shape}")

        # Ensure same N for source and target
        n_rows = min(len(H_ae), len(H_target_all))
        H_ae_use = H_ae[:n_rows]
        H_raw_use = H_latent_src[:n_rows]
        H_tgt_use = H_target_all[:n_rows]

        # Split train / eval
        n_train = int(n_rows * args.train_ratio)
        H_ae_train  = H_ae_use[:n_train]
        H_tgt_train = H_tgt_use[:n_train]
        H_ae_eval   = H_ae_use[n_train:]
        H_raw_eval  = H_raw_use[n_train:]
        H_tgt_eval  = H_tgt_use[n_train:]

        print(f"  Train: {n_train}, Eval: {len(H_ae_eval)}")

        # Fit WCT on AE-filtered data → target layer
        print(f"  Fitting WCT (AE output → target)...")
        wct = WCT(eps=1e-5)
        wct.fit(H_ae_train, H_tgt_train)

        # Transform eval
        H_ae_wct_eval = wct.transform(H_ae_eval)

        # Metrics: AE-only vs target
        metrics_ae = compute_metrics(H_ae_eval, H_tgt_eval)
        # Metrics: AE+WCT vs target
        metrics_ae_wct = compute_metrics(H_ae_wct_eval, H_tgt_eval)

        print(f"  AE-only  → target: cos={metrics_ae['cosine_sim_mean']:.4f}, "
              f"norm={metrics_ae['norm_ratio_mean']:.3f}, "
              f"L2={metrics_ae['l2_dist_mean']:.1f}")
        print(f"  AE+WCT   → target: cos={metrics_ae_wct['cosine_sim_mean']:.4f}, "
              f"norm={metrics_ae_wct['norm_ratio_mean']:.3f}, "
              f"L2={metrics_ae_wct['l2_dist_mean']:.1f}")

        all_summary[src_layer] = {"ae": metrics_ae, "ae_wct": metrics_ae_wct}

        # Plot
        plot_single_layer(
            plt, H_raw_eval, H_ae_eval, H_ae_wct_eval, H_tgt_eval,
            src_layer, target_layer, metrics_ae, metrics_ae_wct,
            out_dir, max_points=args.max_points,
        )

        # Free AE memory
        del ae

    # Summary plot
    if len(all_summary) >= 2:
        print(f"\nGenerating summary...")
        plot_summary(plt, all_summary, target_layer, out_dir)

    print(f"\n✅ Done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
