"""
AE Analysis Demo
================
Phân tích tình trạng Query-Aware Autoencoder sau khi train:

Panels (per layer):
  1. Distribution: h_latent (input) vs h_rec (reconstruction) — histogram overlay
  2. Reconstruction error heatmap — per-feature mean squared error
  3. Gate distribution — histogram of gate values (should peak near 0 and 1)
  4. Gate activation rate — fraction of features with gate > 0.5 per sample
  5. Bottleneck (z) sparsity — distribution of |z_i|
  6. PCA projection — h_latent vs h_rec in 2D

Extra panels:
  7. Training loss curves — recon, sparse, attn, total per epoch
  8. Cross-layer comparison — best loss and gate sparsity per layer

Usage:
    python analyze_ae.py [--layer -1] [--checkpoint_dir ae_checkpoints/...] [--n_samples 2000]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — saves PNG files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from methods.query_aware_ae import QueryAwareAutoencoder


# ── colour palette ───────────────────────────────────────────────────────────
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
GREY   = "#8C8C8C"

plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#181C24",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "#DDD",
    "xtick.color":      "#AAA",
    "ytick.color":      "#AAA",
    "text.color":       "#EEE",
    "grid.color":       "#2A2E38",
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
    "font.size":        9,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet(path: str, n: int = -1) -> torch.Tensor:
    df = pd.read_parquet(path)
    arr = df.values.astype(np.float32)
    if n > 0:
        arr = arr[:n]
    return torch.from_numpy(arr)


def _load_ae(checkpoint_dir: str, layer_idx: int, device: str = "cpu") -> QueryAwareAutoencoder:
    ckpt_path = os.path.join(checkpoint_dir, f"ae_layer_{layer_idx}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No AE checkpoint found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = QueryAwareAutoencoder(
        hidden_dim     = cfg["hidden_dim"],
        bottleneck_dim = cfg["bottleneck_dim"],
        gate_hidden    = cfg.get("gate_hidden"),
        encoder_layers = cfg.get("encoder_layers", 2),
        decoder_layers = cfg.get("decoder_layers", 3),
        mlp_hidden     = cfg.get("mlp_hidden", 2048),
    ).to(device)
    sd_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[sd_key])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Analysis for a single layer
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyse_layer(
    data_dir: str,
    checkpoint_dir: str,
    layer_idx: int,
    n_samples: int = 2000,
    device: str = "cpu",
    out_dir: str = "ae_analysis",
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Layer {layer_idx}")
    print(f"{'='*60}")

    # ── load data (parquet from data_dir) ──────────────────────────────
    lat_path = os.path.join(data_dir, f"layer_{layer_idx}_latent.parquet")
    inp_path = os.path.join(data_dir, f"layer_{layer_idx}_input.parquet")

    H_latent = _load_parquet(lat_path, n_samples).to(device)  # [N, D]
    H_input  = _load_parquet(inp_path, n_samples).to(device)  # [M, D]

    N = len(H_latent)
    repeat = math.ceil(N / len(H_input))
    H_input = H_input.repeat(repeat, 1)[:N]

    print(f"  Loaded: {N:,} latent samples, D={H_latent.shape[1]}")

    # ── load model (AE from checkpoint_dir) ────────────────────────────
    model = _load_ae(checkpoint_dir, layer_idx, device)

    # ── normalize (same as training) ───────────────────────────────────
    mean_lat = H_latent.mean(0)
    std_lat  = H_latent.std(0).clamp(min=1e-8)
    mean_inp = H_input.mean(0)
    std_inp  = H_input.std(0).clamp(min=1e-8)

    H_lat_n = (H_latent - mean_lat) / std_lat
    H_inp_n = (H_input  - mean_inp) / std_inp

    # ── batch forward (avoid OOM) ──────────────────────────────────────
    BS = 512
    H_rec_list, Z_list, Gate_list = [], [], []
    for i in range(0, N, BS):
        h_rec, z, gate = model(H_lat_n[i:i+BS], H_inp_n[i:i+BS], normalize=False)
        H_rec_list.append(h_rec.cpu())
        Z_list.append(z.cpu())
        Gate_list.append(gate.cpu())

    H_rec  = torch.cat(H_rec_list)   # [N, D]  — normalized space
    Z      = torch.cat(Z_list)        # [N, bottleneck_dim]
    Gate   = torch.cat(Gate_list)     # [N, D]

    H_lat_cpu = H_lat_n.cpu()        # normalized (same space as H_rec)
    H_inp_cpu = H_inp_n.cpu()

    # ── scalar stats ───────────────────────────────────────────────────
    mse_per_feat = (H_rec - H_lat_cpu).pow(2).mean(0)          # [D]
    mse_total    = mse_per_feat.mean().item()
    cosine_sim   = torch.nn.functional.cosine_similarity(H_rec, H_lat_cpu, dim=1).mean().item()
    gate_mean    = Gate.mean().item()
    gate_sparse  = (Gate > 0.5).float().mean().item()           # fraction of "on" gates
    z_l1         = Z.abs().mean().item()

    print(f"  MSE (recon):      {mse_total:.5f}")
    print(f"  Cosine similarity:{cosine_sim:.4f}")
    print(f"  Gate mean:         {gate_mean:.4f}")
    print(f"  Gate > 0.5 frac:  {gate_sparse:.4f}")
    print(f"  |z| mean:          {z_l1:.4f}")

    # ── load training CSV (from data_dir, where logger writes it) ─────
    csv_path = os.path.join(data_dir, f"layer_{layer_idx}_training.csv")
    loss_df  = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    # ─────────────────────────────────────────────────────────────────
    # Plotting
    # ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"QAAE — Layer {layer_idx}  |  N={N:,}  D={H_latent.shape[1]}  Z={Z.shape[1]}",
                 fontsize=13, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Panel 1: Value distribution h_latent vs h_rec ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    vals_lat = H_lat_cpu.numpy().ravel()
    vals_rec = H_rec.numpy().ravel()
    bins = np.linspace(
        min(np.percentile(vals_lat, 1), np.percentile(vals_rec, 1)),
        max(np.percentile(vals_lat, 99), np.percentile(vals_rec, 99)),
        80
    )
    ax1.hist(vals_lat, bins=bins, alpha=0.65, color=BLUE,   label="h_latent (input)",       density=True)
    ax1.hist(vals_rec, bins=bins, alpha=0.65, color=ORANGE, label="h_rec (reconstruction)", density=True)
    ax1.set_title("Value Distribution", fontweight="bold")
    ax1.set_xlabel("Activation value (normalised)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=7)
    ax1.grid(True)
    ax1.text(0.97, 0.97, f"MSE={mse_total:.4f}\ncos={cosine_sim:.3f}",
             transform=ax1.transAxes, va="top", ha="right", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.8))

    # ── Panel 2: Per-feature MSE heatmap (top-D features) ──────────────
    ax2 = fig.add_subplot(gs[0, 1])
    D = mse_per_feat.shape[0]
    stride = max(1, D // 256)
    mse_plot = mse_per_feat[::stride].numpy()
    feat_idx = np.arange(len(mse_plot)) * stride
    ax2.bar(feat_idx, mse_plot, width=stride*0.9, color=RED, alpha=0.75)
    ax2.set_title("Per-Feature MSE", fontweight="bold")
    ax2.set_xlabel("Feature index")
    ax2.set_ylabel("MSE")
    ax2.grid(True, axis="y")
    worst_k = 5
    worst_feats = mse_per_feat.topk(worst_k).indices.numpy()
    ax2.text(0.97, 0.97,
             f"Top-{worst_k} worst:\n" + ", ".join(str(f) for f in worst_feats),
             transform=ax2.transAxes, va="top", ha="right", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.8))

    # ── Panel 3: Gate value distribution ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    gate_vals = Gate.numpy().ravel()
    ax3.hist(gate_vals, bins=60, color=GREEN, alpha=0.8, density=True, edgecolor="none")
    ax3.set_title("Gate Distribution", fontweight="bold")
    ax3.set_xlabel("Gate value (0=off, 1=on)")
    ax3.set_ylabel("Density")
    ax3.axvline(0.5, color=RED, ls="--", lw=1.2, label="threshold=0.5")
    ax3.legend(fontsize=7)
    ax3.grid(True)
    ax3.text(0.5, 0.97,
             f"mean={gate_mean:.3f}\n>0.5 frac={gate_sparse:.3f}",
             transform=ax3.transAxes, va="top", ha="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.8))

    # ── Panel 4: Per-sample gate activation rate ────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    per_sample_on = (Gate > 0.5).float().mean(1).numpy()   # [N]
    ax4.hist(per_sample_on, bins=40, color="#A77BCA", alpha=0.85, density=True)
    ax4.set_title("Gate Activation Rate per Sample", fontweight="bold")
    ax4.set_xlabel("Fraction of features with gate > 0.5")
    ax4.set_ylabel("Density")
    ax4.grid(True)

    # ── Panel 5: Bottleneck |z| distribution ───────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    z_vals = Z.numpy().ravel()
    ax5.hist(z_vals, bins=60, color="#4DBCD4", alpha=0.85, density=True, edgecolor="none")
    ax5.set_title("Bottleneck |z| Distribution", fontweight="bold")
    ax5.set_xlabel("z value")
    ax5.set_ylabel("Density")
    ax5.axvline(0, color=GREY, ls="--", lw=1)
    ax5.grid(True)
    ax5.text(0.97, 0.97, f"mean |z|={z_l1:.4f}",
             transform=ax5.transAxes, va="top", ha="right", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.8))

    # ── Panel 6: PCA projection of h_latent vs h_rec ───────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    try:
        from sklearn.decomposition import PCA
        n_pca = min(1000, N)
        idx   = np.random.choice(N, n_pca, replace=False)
        both  = np.vstack([H_lat_cpu[idx].numpy(), H_rec[idx].numpy()])
        pca   = PCA(n_components=2)
        proj  = pca.fit_transform(both)
        proj_lat = proj[:n_pca]
        proj_rec = proj[n_pca:]
        ax6.scatter(proj_lat[:, 0], proj_lat[:, 1], s=6, alpha=0.4, color=BLUE,   label="h_latent")
        ax6.scatter(proj_rec[:, 0], proj_rec[:, 1], s=6, alpha=0.4, color=ORANGE, label="h_rec")
        ax6.set_title("PCA (2D) — h_latent vs h_rec", fontweight="bold")
        ax6.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax6.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax6.legend(fontsize=7, markerscale=2)
        ax6.grid(True)
    except ImportError:
        ax6.text(0.5, 0.5, "scikit-learn not installed\n(pip install scikit-learn)",
                 ha="center", va="center", transform=ax6.transAxes, fontsize=9)
        ax6.set_title("PCA — unavailable", fontweight="bold")

    # ── Panel 7: Training loss curves ──────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :2])
    if loss_df is not None:
        epochs = loss_df["epoch"].values
        ax7.plot(epochs, loss_df["total"],  color=BLUE,   lw=1.5, label="total")
        ax7.plot(epochs, loss_df["recon"],  color=ORANGE, lw=1.2, label="recon",  ls="--")
        ax7.plot(epochs, loss_df["sparse"], color=GREEN,  lw=1.2, label="sparse", ls=":")
        ax7.plot(epochs, loss_df["attn"],   color=RED,    lw=1.2, label="attn",   ls="-.")
        if "best" in loss_df.columns:
            ax7.plot(epochs, loss_df["best"], color=GREY, lw=1.0, ls="--", label="best", alpha=0.5)
        ax7.set_title("Training Loss Curves", fontweight="bold")
        ax7.set_xlabel("Epoch")
        ax7.set_ylabel("Loss")
        ax7.legend(fontsize=7, ncol=3)
        ax7.grid(True)
    else:
        ax7.text(0.5, 0.5, "No training CSV found", ha="center", va="center",
                 transform=ax7.transAxes)
        ax7.set_title("Training Loss Curves", fontweight="bold")

    # ── Panel 8: Std of (h_latent - h_rec) per feature ─────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    std_err = (H_rec - H_lat_cpu).std(0).numpy()
    feat_idx2 = np.arange(0, D, stride)
    ax8.plot(feat_idx2, std_err[::stride], color="#FF6B9D", lw=0.8, alpha=0.8)
    ax8.fill_between(feat_idx2, 0, std_err[::stride], color="#FF6B9D", alpha=0.25)
    ax8.set_title("Std of Reconstruction Error per Feature", fontweight="bold")
    ax8.set_xlabel("Feature index")
    ax8.set_ylabel("Std(h_rec - h_latent)")
    ax8.grid(True)

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"layer_{layer_idx}_analysis.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {out_path}")

    return {
        "layer": layer_idx,
        "mse": mse_total,
        "cosine_sim": cosine_sim,
        "gate_mean": gate_mean,
        "gate_sparse_rate": gate_sparse,
        "z_l1": z_l1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-layer summary plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_layer_summary(summaries: list, out_dir: str) -> None:
    if len(summaries) < 2:
        return

    layers = [s["layer"] for s in summaries]
    x = np.arange(len(layers))
    labels = [str(l) for l in layers]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Cross-Layer Summary", fontsize=12, fontweight="bold")

    metrics = [
        ("mse",              "MSE (recon)",         RED),
        ("cosine_sim",       "Cosine Similarity",   GREEN),
        ("gate_sparse_rate", "Gate >0.5 Fraction",  BLUE),
        ("z_l1",             "|z| L1 (sparsity)",   ORANGE),
    ]
    for ax, (key, title, color) in zip(axes, metrics):
        vals = [s[key] for s in summaries]
        ax.bar(x, vals, color=color, alpha=0.8, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.grid(True, axis="y")
        for xi, v in zip(x, vals):
            ax.text(xi, v * 1.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "cross_layer_summary.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ Cross-layer summary saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse trained QAAE checkpoints")
    parser.add_argument("--data_dir", default="latent_data/gsm8k",
                        help="Directory with parquet data files (hidden states).")
    parser.add_argument("--checkpoint_dir", default="ae_checkpoints/paired_hidden_state_cache",
                        help="Directory with AE model checkpoints (.pt files).")
    parser.add_argument("--layers", default="-5,-4,-3,-2,-1",
                        help="Comma-separated layer indices to analyse (default: all 5)")
    parser.add_argument("--n_samples", type=int, default=45000,
                        help="Max samples to load per layer (default: 45000)")
    parser.add_argument("--device", default="cpu",
                        help="torch device (cpu / cuda)")
    parser.add_argument("--out_dir", default="ae_analysis",
                        help="Output directory for PNG files")
    args = parser.parse_args()

    data_dir       = os.path.join(ROOT, args.data_dir)
    checkpoint_dir = os.path.join(ROOT, args.checkpoint_dir)
    out_dir        = os.path.join(ROOT, args.out_dir)
    layers         = [int(l.strip()) for l in args.layers.split(",")]

    print(f"Data dir       : {data_dir}")
    print(f"Checkpoint dir : {checkpoint_dir}")
    print(f"Output dir     : {out_dir}")
    print(f"Layers         : {layers}")
    print(f"Max samples    : {args.n_samples}")
    print(f"Device         : {args.device}")

    summaries = []
    for layer_idx in layers:
        try:
            s = analyse_layer(
                data_dir       = data_dir,
                checkpoint_dir = checkpoint_dir,
                layer_idx      = layer_idx,
                n_samples      = args.n_samples,
                device         = args.device,
                out_dir        = out_dir,
            )
            summaries.append(s)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    plot_cross_layer_summary(summaries, out_dir)

    # Print final table
    print(f"\n{'─'*65}")
    print(f"  {'Layer':>6}  {'MSE':>8}  {'CosSim':>8}  {'Gate>0.5':>9}  {'|z|L1':>7}")
    print(f"{'─'*65}")
    for s in summaries:
        print(f"  {s['layer']:>6}  {s['mse']:>8.5f}  {s['cosine_sim']:>8.4f}"
              f"  {s['gate_sparse_rate']:>9.4f}  {s['z_l1']:>7.4f}")
    print(f"{'─'*65}\n")


if __name__ == "__main__":
    main()
