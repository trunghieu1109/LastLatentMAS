"""
Hidden States Visualizer
========================
PCA scatter-plot of collected hidden states, colour-coded by query.

Panels per layer:
  1. PCA 2D — input (●) vs latent (▲), colour = query index
  2. Per-query trajectory: input → latent centroid arrows
  3. Norms distribution per query

Optionally applies AE reconstruction & gating to show filtered latent states
alongside the raw latent states.

Usage:
    # Basic: visualize input vs latent at one layer
    python visualize_hidden_states.py --data_dir latent_data/medqa --layer 36

    # Multiple layers
    python visualize_hidden_states.py --data_dir latent_data/medqa --layers 32,33,34,35,36

    # All layers
    python visualize_hidden_states.py --data_dir latent_data/medqa --layers all --n_queries 15

    # With AE reconstruction overlay
    python visualize_hidden_states.py --data_dir latent_data/medqa --layer 36 \
        --ae_checkpoint_dir ae_checkpoints/paired_hidden_state_cache \
        --ae_layers "-5,-4,-3,-2,-1"

    # Interactive mode with layer slider
    python visualize_hidden_states.py --data_dir latent_data/medqa --interactive
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
# Dark theme
# ──────────────────────────────────────────────────────────────────────────────

def _setup_mpl(interactive: bool = False):
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
        "font.size":        10,
    })
    return plt


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def discover_layers(data_dir: str):
    """Return sorted list of layer indices found in data_dir."""
    pattern = os.path.join(data_dir, "layer_*_input.parquet")
    layers = []
    for f in glob.glob(pattern):
        m = re.search(r"layer_(-?\d+)_input\.parquet", os.path.basename(f))
        if m:
            layers.append(int(m.group(1)))
    return sorted(layers)


def load_metadata(data_dir: str) -> dict:
    """Load cache_metadata.json if it exists."""
    meta_path = os.path.join(data_dir, "cache_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _load_parquet_robust(path: str) -> np.ndarray:
    """Load parquet → numpy float32, handling duplicate/metadata columns."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    pf = pq.ParquetFile(path)
    table = pf.read()

    # Keep only numeric (float/int) columns — drop __fragment_index etc.
    numeric_indices = [
        i for i, f in enumerate(table.schema)
        if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)
    ]

    col_names = [table.schema.field(i).name for i in numeric_indices]
    unique_names = set(col_names)

    if len(unique_names) < len(col_names):
        # Duplicate columns: take first unique set
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


def load_layer_vectors(
    data_dir: str,
    layer_idx: int,
    n_queries: int,
    n_agents: int,
    latent_steps: int = 0,
) -> dict:
    """Load input + latent parquet for one layer and assign per-query labels.

    Data format (from collect_only):
        input:  total_queries × n_agents rows  (one per agent per query)
        latent: total_queries × n_agents × latent_steps rows

    Returns dict with keys:
        H_input, H_latent, query_ids_input, query_ids_latent,
        agent_ids_input, agent_ids_latent,
        n_queries_actual, n_agents, latent_steps_actual, hidden_dim
    """
    inp_path = os.path.join(data_dir, f"layer_{layer_idx}_input.parquet")
    lat_path = os.path.join(data_dir, f"layer_{layer_idx}_latent.parquet")

    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Missing: {inp_path}")
    if not os.path.exists(lat_path):
        raise FileNotFoundError(f"Missing: {lat_path}")

    H_input  = _load_parquet_robust(inp_path)
    H_latent = _load_parquet_robust(lat_path)

    total_input_rows = len(H_input)
    total_latent_rows = len(H_latent)
    total_queries = total_input_rows // n_agents

    # Infer latent_steps from data
    if latent_steps <= 0 and total_queries > 0:
        latent_per_query = total_latent_rows // total_queries
        latent_steps_actual = latent_per_query // n_agents
    else:
        latent_steps_actual = latent_steps

    latent_per_query = n_agents * latent_steps_actual

    n_q = min(n_queries, total_queries)
    if n_q <= 0:
        raise ValueError(f"Layer {layer_idx}: only {total_queries} queries found")

    H_inp_sub = H_input[:n_q * n_agents]
    H_lat_sub = H_latent[:n_q * latent_per_query]

    # Query IDs
    q_ids_inp = np.repeat(np.arange(n_q), n_agents)
    q_ids_lat = np.repeat(np.arange(n_q), latent_per_query)

    # Agent IDs
    a_ids_inp = np.tile(np.arange(n_agents), n_q)
    a_ids_lat = np.tile(
        np.repeat(np.arange(n_agents), latent_steps_actual),
        n_q,
    )

    return {
        "H_input":             H_inp_sub,
        "H_latent":            H_lat_sub,
        "query_ids_input":     q_ids_inp,
        "query_ids_latent":    q_ids_lat,
        "agent_ids_input":     a_ids_inp,
        "agent_ids_latent":    a_ids_lat,
        "n_queries_actual":    n_q,
        "n_agents":            n_agents,
        "latent_steps_actual": latent_steps_actual,
        "hidden_dim":          H_inp_sub.shape[1],
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


def apply_ae_reconstruction(
    ae,
    H_latent: np.ndarray,
    H_input_ref: np.ndarray,
    query_ids_latent: np.ndarray,
    agent_ids_latent: np.ndarray,
    query_ids_input: np.ndarray,
    agent_ids_input: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Apply AE reconstruction + gating to latent vectors.

    For each latent vector, pair it with its corresponding input vector
    (same query, same agent) and run through the AE.

    Returns H_reconstructed: same shape as H_latent.
    """
    import torch

    H_rec = np.zeros_like(H_latent)

    # Build lookup: (query_id, agent_id) → input row index
    inp_lookup = {}
    for i in range(len(query_ids_input)):
        key = (int(query_ids_input[i]), int(agent_ids_input[i]))
        inp_lookup[key] = i

    batch_size = 256
    n = len(H_latent)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_lat = H_latent[start:end]
        batch_q = query_ids_latent[start:end]
        batch_a = agent_ids_latent[start:end]

        inp_indices = []
        for q, a in zip(batch_q, batch_a):
            idx = inp_lookup.get((int(q), int(a)), 0)
            inp_indices.append(idx)

        batch_inp = H_input_ref[inp_indices]

        with torch.no_grad():
            h_lat = torch.from_numpy(batch_lat).float().to(device)
            h_inp = torch.from_numpy(batch_inp).float().to(device)
            h_rec, _z, gate = ae(h_lat, h_inp, normalize=True)
            # Denormalize h_rec back to original scale before gating
            if ae._has_norm:
                h_rec = h_rec * ae._norm_std_latent.to(device) + ae._norm_mean_latent.to(device)
            h_rec = (gate * h_rec).cpu().numpy()

        H_rec[start:end] = h_rec

    return H_rec.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────────────────────────────────────

def pca_project(*arrays):
    """Joint PCA → 2D over multiple arrays.
    Returns list of projected arrays + explained_var_ratio.
    """
    from sklearn.decomposition import PCA
    combined = np.vstack(arrays)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)

    result = []
    offset = 0
    for arr in arrays:
        n = len(arr)
        result.append(proj[offset:offset + n])
        offset += n

    return result, pca.explained_variance_ratio_


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

_QUERY_COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFD93D", "#6BCB77", "#4D96FF",
    "#FF8C32", "#AD68FF", "#FF4C8B", "#00D2D3", "#FFC75F",
    "#A5E887", "#FF6F91", "#6FDFDF", "#FFB3BA", "#C9B1FF",
    "#FFF176", "#80CBC4", "#F48FB1", "#81D4FA", "#A5D6A7",
]


def get_query_color(q_idx: int, n_queries: int):
    if n_queries <= len(_QUERY_COLORS):
        return _QUERY_COLORS[q_idx % len(_QUERY_COLORS)]
    import matplotlib.cm as cm
    return cm.tab20(q_idx / max(1, n_queries - 1))


def plot_layer(
    plt,
    data: dict,
    layer_idx: int,
    out_dir: str = None,
    show: bool = False,
    H_ae: np.ndarray = None,
    qi_ae: np.ndarray = None,
):
    """Generate a figure for one layer.

    If H_ae is provided, adds a 4th panel showing AE-reconstructed states.
    Panels:
      1. PCA scatter (input ● vs latent ▲ [vs AE-filtered ★])
      2. Per-query centroid trajectories (arrows)
      3. L2 norm distributions
      4. (if AE) PCA scatter: latent vs AE-filtered only
    """
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    H_inp   = data["H_input"]
    H_lat   = data["H_latent"]
    qi_inp  = data["query_ids_input"]
    qi_lat  = data["query_ids_latent"]
    n_q     = data["n_queries_actual"]
    D       = data["hidden_dim"]
    has_ae  = H_ae is not None

    n_panels = 4 if has_ae else 3
    ratios   = [2, 1.2, 1, 2] if has_ae else [2, 1.2, 1]
    figw     = 30 if has_ae else 22

    # PCA
    if has_ae:
        projs, var_ratio = pca_project(H_inp, H_lat, H_ae)
        proj_inp, proj_lat, proj_ae = projs
    else:
        projs, var_ratio = pca_project(H_inp, H_lat)
        proj_inp, proj_lat = projs
        proj_ae = None

    fig = plt.figure(figsize=(figw, 8))
    ae_tag = " + AE" if has_ae else ""
    fig.suptitle(
        f"Hidden States — Layer {layer_idx}{ae_tag}  |  {n_q} queries  |  "
        f"D={D}  |  {len(H_inp)} input  {len(H_lat)} latent vectors",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(1, n_panels, figure=fig, width_ratios=ratios, wspace=0.32)

    # ── Panel 1: PCA scatter ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for q in range(n_q):
        c = get_query_color(q, n_q)
        mask_inp = qi_inp == q
        ax1.scatter(proj_inp[mask_inp, 0], proj_inp[mask_inp, 1],
                    c=c, marker="o", s=50, alpha=0.85,
                    edgecolors="white", linewidths=0.3, zorder=3)
        mask_lat = qi_lat == q
        ax1.scatter(proj_lat[mask_lat, 0], proj_lat[mask_lat, 1],
                    c=c, marker="^", s=18, alpha=0.35,
                    edgecolors="none", zorder=2)
        if has_ae and qi_ae is not None:
            mask_ae = qi_ae == q
            ax1.scatter(proj_ae[mask_ae, 0], proj_ae[mask_ae, 1],
                        c=c, marker="*", s=30, alpha=0.6,
                        edgecolors="none", zorder=4)

    ax1.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax1.set_title("PCA Projection", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCC",
               markersize=8, label="Input (last input token)", linestyle="None"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#CCC",
               markersize=8, label="Latent (thinking steps)", linestyle="None"),
    ]
    if has_ae:
        legend_elements.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD93D",
                   markersize=10, label="AE-filtered (★)", linestyle="None"))
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=8,
               facecolor="#1A1E28", edgecolor="#333")

    # ── Panel 2: Centroid arrows ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for q in range(n_q):
        c = get_query_color(q, n_q)
        mask_inp = qi_inp == q
        mask_lat = qi_lat == q
        centroid_inp = proj_inp[mask_inp].mean(axis=0)
        centroid_lat = proj_lat[mask_lat].mean(axis=0)

        ax2.scatter(centroid_inp[0], centroid_inp[1],
                    c=c, marker="o", s=100, edgecolors="white",
                    linewidths=0.8, zorder=3)
        ax2.annotate("", xy=(centroid_lat[0], centroid_lat[1]),
                     xytext=(centroid_inp[0], centroid_inp[1]),
                     arrowprops=dict(arrowstyle="->", color=c, lw=1.5, alpha=0.7),
                     zorder=2)

        if has_ae and qi_ae is not None:
            mask_ae = qi_ae == q
            centroid_ae = proj_ae[mask_ae].mean(axis=0)
            ax2.scatter(centroid_ae[0], centroid_ae[1],
                        c=c, marker="*", s=120, edgecolors="white",
                        linewidths=0.5, zorder=4)
            ax2.annotate("", xy=(centroid_ae[0], centroid_ae[1]),
                         xytext=(centroid_lat[0], centroid_lat[1]),
                         arrowprops=dict(arrowstyle="->", color=c, lw=1.2,
                                         alpha=0.5, linestyle="dashed"),
                         zorder=2)

        ax2.text(centroid_inp[0], centroid_inp[1], f" Q{q}",
                 fontsize=7, color=c, alpha=0.9, va="center")

    ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    title2 = "Trajectories: input → latent"
    if has_ae:
        title2 += " → AE"
    ax2.set_title(title2, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Norm distribution ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    norms_inp = np.linalg.norm(H_inp, axis=1)
    norms_lat = np.linalg.norm(H_lat, axis=1)
    inp_means = [norms_inp[qi_inp == q].mean() for q in range(n_q)]
    lat_means = [norms_lat[qi_lat == q].mean() for q in range(n_q)]

    x = np.arange(n_q)
    n_bars = 3 if has_ae else 2
    w = 0.8 / n_bars

    ax3.bar(x - w, inp_means, w, color="#4ECDC4", alpha=0.8, label="Input ‖h‖")
    ax3.bar(x,     lat_means, w, color="#FF6B6B", alpha=0.8, label="Latent ‖h‖")

    if has_ae:
        norms_ae = np.linalg.norm(H_ae, axis=1)
        ae_means = [norms_ae[qi_ae == q].mean() for q in range(n_q)]
        ax3.bar(x + w, ae_means, w, color="#FFD93D", alpha=0.8, label="AE ‖h‖")

    ax3.set_xlabel("Query")
    ax3.set_ylabel("Mean L2 norm")
    ax3.set_title("Hidden State Norms", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Q{q}" for q in range(n_q)], fontsize=7,
                        rotation=45 if n_q > 10 else 0)
    ax3.legend(fontsize=8, facecolor="#1A1E28", edgecolor="#333")
    ax3.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: AE vs Latent only ────────────────────────────────────────
    if has_ae:
        ax4 = fig.add_subplot(gs[0, 3])
        [proj_lat2, proj_ae2], var2 = pca_project(H_lat, H_ae)

        for q in range(n_q):
            c = get_query_color(q, n_q)
            mask_lat = qi_lat == q
            mask_ae = qi_ae == q
            ax4.scatter(proj_lat2[mask_lat, 0], proj_lat2[mask_lat, 1],
                        c=c, marker="^", s=18, alpha=0.4,
                        edgecolors="none", zorder=2)
            ax4.scatter(proj_ae2[mask_ae, 0], proj_ae2[mask_ae, 1],
                        c=c, marker="*", s=30, alpha=0.7,
                        edgecolors="none", zorder=3)

            if mask_lat.sum() > 0 and mask_ae.sum() > 0:
                c_lat = proj_lat2[mask_lat].mean(axis=0)
                c_ae  = proj_ae2[mask_ae].mean(axis=0)
                ax4.annotate("", xy=c_ae, xytext=c_lat,
                             arrowprops=dict(arrowstyle="->", color=c,
                                             lw=1.2, alpha=0.6), zorder=1)

        ax4.set_xlabel(f"PC1 ({var2[0]*100:.1f}%)")
        ax4.set_ylabel(f"PC2 ({var2[1]*100:.1f}%)")
        ax4.set_title("Latent (▲) vs AE-filtered (★)", fontweight="bold")
        ax4.grid(True, alpha=0.3)
        legend4 = [
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#CCC",
                   markersize=8, label="Latent (raw)", linestyle="None"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD93D",
                   markersize=10, label="AE-filtered", linestyle="None"),
        ]
        ax4.legend(handles=legend4, loc="lower right", fontsize=8,
                   facecolor="#1A1E28", edgecolor="#333")

    # Info box
    info_text = (
        f"Queries: {n_q}\n"
        f"Input vectors: {len(H_inp)}\n"
        f"Latent vectors: {len(H_lat)}\n"
        f"Hidden dim: {D}\n"
        f"PCA var: {(var_ratio[0]+var_ratio[1])*100:.1f}%"
    )
    if has_ae:
        info_text += f"\nAE vectors: {len(H_ae)}"
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, va="top",
             ha="left", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0F1117",
                       alpha=0.85, edgecolor="#333"))

    plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        suffix = "_ae" if has_ae else ""
        out_path = os.path.join(out_dir, f"hidden_states_layer_{layer_idx}{suffix}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Interactive mode
# ──────────────────────────────────────────────────────────────────────────────

def interactive_viewer(data_dir, n_queries, n_agents, layers, latent_steps=0):
    """Launch interactive matplotlib window with a layer slider."""
    plt = _setup_mpl(interactive=True)
    from matplotlib.widgets import Slider
    from matplotlib.lines import Line2D

    cache = {}

    def _get_data(layer_idx):
        if layer_idx not in cache:
            d = load_layer_vectors(data_dir, layer_idx, n_queries, n_agents, latent_steps)
            projs, var_ratio = pca_project(d["H_input"], d["H_latent"])
            proj_inp, proj_lat = projs
            cache[layer_idx] = (proj_inp, proj_lat,
                                d["query_ids_input"], d["query_ids_latent"],
                                d["n_queries_actual"], var_ratio, d["hidden_dim"])
        return cache[layer_idx]

    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.03], facecolor="#1A1E28")
    slider = Slider(ax_slider, "Layer", 0, len(layers) - 1,
                    valinit=0, valstep=1, color="#4ECDC4")
    slider.valtext.set_color("#EEE")

    def update(val):
        idx = int(slider.val)
        layer_idx = layers[idx]
        slider.valtext.set_text(f"Layer {layer_idx}")
        proj_inp, proj_lat, qi_inp, qi_lat, n_q, var_ratio, D = _get_data(layer_idx)
        ax.cla()
        ax.set_facecolor("#181C24")
        for q in range(n_q):
            c = get_query_color(q, n_q)
            mask_inp = qi_inp == q
            mask_lat = qi_lat == q
            ax.scatter(proj_inp[mask_inp, 0], proj_inp[mask_inp, 1],
                       c=c, marker="o", s=60, alpha=0.85,
                       edgecolors="white", linewidths=0.4, zorder=3)
            ax.scatter(proj_lat[mask_lat, 0], proj_lat[mask_lat, 1],
                       c=c, marker="^", s=20, alpha=0.35,
                       edgecolors="none", zorder=2)
            if mask_inp.sum() > 0 and mask_lat.sum() > 0:
                c_inp = proj_inp[mask_inp].mean(axis=0)
                c_lat = proj_lat[mask_lat].mean(axis=0)
                ax.annotate("", xy=c_lat, xytext=c_inp,
                            arrowprops=dict(arrowstyle="->", color=c,
                                            lw=1.2, alpha=0.5), zorder=1)
                ax.text(c_inp[0], c_inp[1], f" Q{q}", fontsize=7,
                        color=c, alpha=0.8, va="center")
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
        ax.set_title(f"Hidden States PCA — Layer {layer_idx}  |  {n_q} queries  |  D={D}",
                     fontweight="bold", fontsize=13)
        ax.grid(True, alpha=0.3)
        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCC",
                   markersize=8, label="Input (●)", linestyle="None"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#CCC",
                   markersize=8, label="Latent (▲)", linestyle="None"),
        ]
        ax.legend(handles=legend_elems, loc="lower right", fontsize=9,
                  facecolor="#1A1E28", edgecolor="#333")
        info = f"Queries: {n_q}\nPCA var: {(var_ratio[0]+var_ratio[1])*100:.1f}%"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0F1117",
                          alpha=0.85, edgecolor="#333"))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Parse layer specification
# ──────────────────────────────────────────────────────────────────────────────

def parse_layers(spec: str, available: list) -> list:
    """Parse layer spec: 'all', '0-10', '32,33,34,35,36', or single int."""
    spec = spec.strip()
    if spec.lower() == "all":
        return available
    m = re.match(r"^(-?\d+)-(-?\d+)$", spec)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return [l for l in available if lo <= l <= hi]
    try:
        requested = [int(x.strip()) for x in spec.split(",")]
        return [l for l in requested if l in available]
    except ValueError:
        pass
    return available


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PCA visualization of hidden states per query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data_dir latent_data/medqa --layer 36
  %(prog)s --data_dir latent_data/medqa --layers 32,33,34,35,36 --n_queries 15
  %(prog)s --data_dir latent_data/medqa --layers all

  # With AE overlay:
  %(prog)s --data_dir latent_data/medqa --layer 36 \\
      --ae_checkpoint_dir ae_checkpoints/paired_hidden_state_cache \\
      --ae_layers "-5,-4,-3,-2,-1"
        """,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory with parquet files (e.g., latent_data/medqa)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Single layer to visualize")
    parser.add_argument("--layers", default=None,
                        help="Layer spec: 'all', '0-10', '32,33,34,35,36'")
    parser.add_argument("--n_queries", type=int, default=20,
                        help="Number of queries to display (default: 20)")
    parser.add_argument("--n_agents", type=int, default=3,
                        help="Number of non-judger agents")
    parser.add_argument("--out_dir", default="ae_analysis",
                        help="Output directory for PNGs (default: ae_analysis)")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive viewer with layer slider")

    # ── AE reconstruction args ─────────────────────────────────────────────
    parser.add_argument("--ae_checkpoint_dir", type=str, default="",
                        help="AE checkpoint directory (enables AE overlay)")
    parser.add_argument("--ae_layers", type=str, default="",
                        help='Signed layer indices for AE, e.g. "-5,-4,-3,-2,-1"')
    parser.add_argument("--ae_device", type=str, default="cpu",
                        help="Device for AE inference (default: cpu)")

    # Fix: argparse chokes on --ae_layers -5,... (thinks -5 is a flag).
    # Pre-process argv to merge: --ae_layers  -5,...  →  --ae_layers=-5,...
    argv = sys.argv[1:]
    fixed_argv = []
    i = 0
    while i < len(argv):
        if argv[i] == "--ae_layers" and i + 1 < len(argv):
            fixed_argv.append(f"--ae_layers={argv[i + 1]}")
            i += 2
        else:
            fixed_argv.append(argv[i])
            i += 1

    args = parser.parse_args(fixed_argv)

    # Resolve paths
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)

    available = discover_layers(data_dir)
    if not available:
        print(f"❌ No parquet files found in {data_dir}")
        sys.exit(1)

    print(f"Data dir: {data_dir}")
    print(f"Available layers ({len(available)}): {available}")

    # Load metadata
    meta = load_metadata(data_dir)
    n_agents = args.n_agents
    agent_names = meta.get("agent_names", [])
    if agent_names:
        n_agents = len(agent_names)
    latent_steps = meta.get("latent_steps", 0)
    num_total_layers = meta.get("num_total_layers", max(available) + 1)
    print(f"Agents ({n_agents}): {agent_names or '(default)'}")
    print(f"Latent steps (from metadata): {latent_steps}")

    # Determine target layers
    if args.layer is not None:
        target_layers = [args.layer] if args.layer in available else []
        if not target_layers:
            closest = min(available, key=lambda x: abs(x - args.layer))
            print(f"⚠ Layer {args.layer} not found, using closest: {closest}")
            target_layers = [closest]
    elif args.layers is not None:
        target_layers = parse_layers(args.layers, available)
    else:
        target_layers = available[:1]

    if not target_layers:
        print("❌ No matching layers found")
        sys.exit(1)

    print(f"Target layers: {target_layers}")
    print(f"Queries to show: {args.n_queries}")

    # ── Parse AE configuration ──────────────────────────────────────────────
    ae_map = {}  # abs_layer → (AE model, signed_layer)
    ae_ckpt_dir = args.ae_checkpoint_dir
    if not ae_ckpt_dir:
        ae_ckpt_dir = os.path.join(ROOT, "ae_checkpoints", "paired_hidden_state_cache")

    if args.ae_layers and os.path.isdir(ae_ckpt_dir):
        ae_signed_layers = [int(x.strip()) for x in args.ae_layers.split(",")]
        print(f"\nLoading AE models from: {ae_ckpt_dir}")
        for sl in ae_signed_layers:
            abs_l = num_total_layers + sl if sl < 0 else sl
            ae = load_ae_for_layer(ae_ckpt_dir, sl, args.ae_device)
            if ae is not None:
                ae_map[abs_l] = (ae, sl)
                print(f"  Loaded AE for signed layer {sl} (abs={abs_l})")
            else:
                print(f"  ⚠ No AE checkpoint for layer {sl}")

    # ── Interactive mode ────────────────────────────────────────────────────
    if args.interactive:
        print("\n🎮 Launching interactive viewer...")
        interactive_viewer(data_dir, args.n_queries, n_agents, target_layers,
                           latent_steps)
        return

    # ── Static PNG mode ─────────────────────────────────────────────────────
    plt = _setup_mpl(interactive=False)
    out_dir = os.path.join(ROOT, args.out_dir)
    print(f"Output dir: {out_dir}\n")

    for layer_idx in target_layers:
        try:
            print(f"Layer {layer_idx}:")
            data = load_layer_vectors(data_dir, layer_idx, args.n_queries,
                                      n_agents, latent_steps)
            print(f"  Loaded: {len(data['H_input'])} input, "
                  f"{len(data['H_latent'])} latent vectors "
                  f"({data['n_queries_actual']} queries, "
                  f"{data['latent_steps_actual']} steps/agent)")

            # AE reconstruction if available
            H_ae = None
            qi_ae = None
            if layer_idx in ae_map:
                ae_model, signed_l = ae_map[layer_idx]
                print(f"  Applying AE reconstruction (signed layer {signed_l})...")
                H_ae = apply_ae_reconstruction(
                    ae_model,
                    data["H_latent"],
                    data["H_input"],
                    data["query_ids_latent"],
                    data["agent_ids_latent"],
                    data["query_ids_input"],
                    data["agent_ids_input"],
                    device=args.ae_device,
                )
                qi_ae = data["query_ids_latent"]
                print(f"  AE vectors: {len(H_ae)}")

            plot_layer(plt, data, layer_idx, out_dir=out_dir, show=False,
                       H_ae=H_ae, qi_ae=qi_ae)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [SKIP] {e}")

    print(f"\n✅ Done. PNGs saved to {out_dir}/")


if __name__ == "__main__":
    main()
