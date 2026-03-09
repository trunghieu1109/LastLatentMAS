"""Query-Aware Autoencoder for extracting latent thoughts from hidden states.

Architecture (per layer):
    Input:
        h_latent  : [B, D]  — hidden state of one latent step token at this layer
        h_input   : [B, D]  — hidden state of last input token at this layer

    Pipeline:
        1. FeatureGate   : gate = sigmoid(W([h_latent ∥ h_input])) → [B, D]
                           Each gate_i ∈ (0,1) measures how much feature i of h_latent
                           is relevant given h_input (the query context).
                           attended = gate ⊙ h_latent  (feature-wise scaling)

        2. Encoder       : attended → z [B, bottleneck_dim]     (compress)

        3. Decoder       : z → h_rec [B, D]                    (reconstruct)

    Losses:
        L_recon   : Attention-Weighted MSE                       (reconstruction)
                    = mean_i( gate_i * (h_rec_i - h_latent_i)² )
                    → High-gate features must be reconstructed precisely.
                    → Low-gate features can be lossy (model doesn't care).

        L_sparse  : Stochastic Jacobian L1 on decoder            (sparsity)
                    = E[||J_decoder||_1]
                    → Encourages independence of bottleneck features.

        L_attn    : Binary-entropy of gate values                 (peaked gate)
                    = -mean( gate_i * log(gate_i) + (1-gate_i) * log(1-gate_i) )
                    → Pushes gate toward 0 or 1 (crisp selection), not diffuse 0.5.
                    → High-importance features get gate ≈ 1; irrelevant get gate ≈ 0.

        L_total   : L_recon + λ_sparse * L_sparse + λ_attn * L_attn
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Feature Gate  (replaces old CrossAttention)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureGate(nn.Module):
    """Produces a per-feature importance gate [B, D] from latent + input states.

    gate_i = sigmoid( W_g([h_latent ∥ h_input]) )_i  ∈ (0, 1)

    Interpretation:
        gate_i ≈ 1 → feature i of h_latent is relevant given the query h_input
        gate_i ≈ 0 → feature i carries little query-relevant information

    The attended output is: attended = gate ⊙ h_latent
    (feature-wise scaling: relevant features pass through, irrelevant are suppressed)

    Args:
        hidden_dim: D — dimension of hidden states.
        gate_hidden: width of the gating MLP hidden layer (default D//2).
    """

    def __init__(self, hidden_dim: int, gate_hidden: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        gate_hidden = gate_hidden or (hidden_dim // 2)

        # Maps concat(h_latent [D], h_input [D]) → gate [D]
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(gate_hidden, hidden_dim),
            nn.Sigmoid(),   # output ∈ (0, 1) per feature
        )

    def forward(
        self,
        h_latent: torch.Tensor,   # [B, D]
        h_input: torch.Tensor,    # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            attended : [B, D]  — gate-weighted latent state
            gate     : [B, D]  — per-feature importance weights ∈ (0, 1)
        """
        gate = self.gate_net(torch.cat([h_latent, h_input], dim=-1))  # [B, D]
        attended = gate * h_latent                                      # [B, D]
        return attended, gate


# ──────────────────────────────────────────────────────────────────────────────
# MLP block helper
# ──────────────────────────────────────────────────────────────────────────────

def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Query-Aware Autoencoder (one per layer)
# ──────────────────────────────────────────────────────────────────────────────

class QueryAwareAutoencoder(nn.Module):
    """Per-layer Query-Aware Autoencoder with attention-weighted reconstruction.

    Args:
        hidden_dim:     D  — dimension of the model's hidden states.
        bottleneck_dim: Z  — compressed latent dimension (Z ≤ D).
        gate_hidden:    width of the gating MLP hidden layer (default D//2).
        encoder_layers: depth of the encoder MLP.
        decoder_layers: depth of the decoder MLP.
        mlp_hidden:     width of encoder/decoder MLP layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        gate_hidden: int | None = None,
        encoder_layers: int = 2,
        decoder_layers: int = 3,
        mlp_hidden: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # 1. Feature Gate: produces per-feature gate [B, D]
        self.feature_gate = FeatureGate(hidden_dim, gate_hidden=gate_hidden)

        # 2. Layer-norm after gating (residual: attended + h_latent)
        self.norm = nn.LayerNorm(hidden_dim)

        # 3. Encoder: gated_ctx → bottleneck
        self.encoder = _build_mlp(hidden_dim, bottleneck_dim, mlp_hidden, encoder_layers)

        # 4. Decoder: bottleneck → hidden_dim
        self.decoder = _build_mlp(bottleneck_dim, hidden_dim, mlp_hidden, decoder_layers)

        # Normalization stats (set after training — persisted in state_dict)
        self.register_buffer("_norm_mean_latent", torch.zeros(hidden_dim))
        self.register_buffer("_norm_std_latent",  torch.ones(hidden_dim))
        self.register_buffer("_norm_mean_input",  torch.zeros(hidden_dim))
        self.register_buffer("_norm_std_input",   torch.ones(hidden_dim))
        self.register_buffer("_has_norm", torch.tensor(False))

    # ── Normalization ──────────────────────────────────────────────────────

    def set_norm_stats(
        self,
        mean_latent: torch.Tensor,
        std_latent: torch.Tensor,
        mean_input: torch.Tensor,
        std_input: torch.Tensor,
    ) -> None:
        """Persist normalization statistics for inference-time auto-normalization."""
        self._norm_mean_latent.copy_(mean_latent.detach())
        self._norm_std_latent.copy_(std_latent.detach().clamp(min=1e-8))
        self._norm_mean_input.copy_(mean_input.detach())
        self._norm_std_input.copy_(std_input.detach().clamp(min=1e-8))
        self._has_norm.fill_(True)

    def _normalize(
        self, h_latent: torch.Tensor, h_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._has_norm:
            h_latent = (h_latent - self._norm_mean_latent.to(h_latent.device)) / self._norm_std_latent.to(h_latent.device)
            h_input  = (h_input  - self._norm_mean_input.to(h_input.device))   / self._norm_std_input.to(h_input.device)
        return h_latent, h_input

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        h_latent: torch.Tensor,  # [B, D]
        h_input: torch.Tensor,   # [B, D]
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            h_latent : hidden state of a latent step at this layer.
            h_input  : hidden state of last input token at this layer.
            normalize: apply stored norm stats (use True at inference time).

        Returns:
            h_rec  : [B, D]  — reconstructed latent hidden state.
            z      : [B, Z]  — bottleneck representation.
            gate   : [B, D]  — per-feature importance gate ∈ (0, 1).
        """
        if normalize:
            h_latent, h_input = self._normalize(h_latent, h_input)

        # 1. Feature gate + residual norm
        attended, gate = self.feature_gate(h_latent, h_input)  # [B, D], [B, D]
        h_ctx = self.norm(h_latent + attended)                  # residual

        # 2. Encode → bottleneck
        z = self.encoder(h_ctx)                                 # [B, Z]

        # 3. Decode → reconstruction
        h_rec = self.decoder(z)                                 # [B, D]

        return h_rec, z, gate

    def encode(
        self,
        h_latent: torch.Tensor,
        h_input: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Encode to bottleneck (no decode)."""
        if normalize:
            h_latent, h_input = self._normalize(h_latent, h_input)
        attended, _ = self.feature_gate(h_latent, h_input)
        h_ctx = self.norm(h_latent + attended)
        return self.encoder(h_ctx)


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def weighted_reconstruction_loss(
    h_orig: torch.Tensor,
    h_rec: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Attention-weighted MSE reconstruction loss.

    Features with high gate values must be reconstructed precisely;
    low-gate features are allowed to be lossy.

    Formula:
        L_recon = mean_{b,i}( gate_{b,i} * (h_rec_{b,i} - h_orig_{b,i})^2 )

    This is equivalent to a weighted MSE where weights come from the gate.
    We normalize by mean(gate) so the loss scale doesn't collapse when gates
    are globally small, keeping it comparable to a standard MSE.

    Args:
        h_orig : [B, D]  — original latent hidden states.
        h_rec  : [B, D]  — reconstructed latent hidden states.
        gate   : [B, D]  — per-feature importance weights ∈ (0, 1).

    Returns:
        Scalar weighted reconstruction loss.
    """
    sq_err = (h_rec - h_orig).pow(2)             # [B, D]
    weighted = gate * sq_err                      # [B, D]
    # Normalize by mean(gate) so loss scale is stable
    gate_mean = gate.mean().clamp(min=1e-6)
    return weighted.mean() / gate_mean


def gate_binary_entropy_loss(gate: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Binary entropy of gate values to encourage crisp 0/1 decisions.

    Formula:
        L_attn = -mean_{b,i}( gate_i * log(gate_i + ε)
                             + (1 - gate_i) * log(1 - gate_i + ε) )

    Minimizing this loss pushes each gate_i toward 0 or 1:
        - gate_i → 1  : feature i is strongly selected (must be reconstructed)
        - gate_i → 0  : feature i is discarded (irrelevant to query)
        - gate_i = 0.5: maximum entropy (penalized most, forced away from this)

    This is intentionally different from softmax-entropy which peaks at
    uniform distribution.  Binary entropy peaks at 0.5 and is zero at 0 or 1.

    Args:
        gate : [B, D]  — per-feature gate values in (0, 1).
        eps  : small constant for numerical stability.

    Returns:
        Scalar binary entropy loss (non-negative; lower = crisper gate).
    """
    h = -(gate * torch.log(gate + eps) + (1.0 - gate) * torch.log(1.0 - gate + eps))
    return h.mean()


def sparsity_loss(z: torch.Tensor) -> torch.Tensor:
    """Simple L1 sparsity on bottleneck codes — fast and effective.

    Formula:
        L_sparse = mean( |z_i| )

    Pushes bottleneck activations toward zero (sparse representation).
    Much faster than the Jacobian approach: O(B×Z) vs O(B×D×Z).

    Args:
        z : [B, Z] bottleneck representations.

    Returns:
        Scalar L1 mean (differentiable).
    """
    return z.abs().mean()


def sparsity_jacobian_loss(
    decoder: nn.Module,
    z: torch.Tensor,
    num_sample_rows: int = 64,
) -> torch.Tensor:
    """Stochastic Jacobian L1 on decoder to encourage sparse decoding.

    NOTE: This is kept for reference but is very slow for large D.
    Use sparsity_loss(z) instead for practical training.

    Computes E[||J_{decoder}||_1] by sampling output rows.
    Promotes independence between bottleneck features and reconstructed dimensions.

    Args:
        decoder:         the decoder MLP.
        z:               [B, Z] bottleneck representations (from encoder).
        num_sample_rows: number of output rows to sample for efficiency.

    Returns:
        Scalar L1 Jacobian estimate (differentiable w.r.t. decoder params).
    """
    try:
        return _jacobian_l1_vmap(decoder, z, num_sample_rows)
    except Exception:
        return _jacobian_l1_loop(decoder, z, num_sample_rows)


def _jacobian_l1_vmap(decoder: nn.Module, z: torch.Tensor, num_rows: int) -> torch.Tensor:
    from torch.func import jacrev, vmap

    Z = z.detach()
    B, n_z = Z.shape

    def single_decode(zi):
        return decoder(zi.unsqueeze(0)).squeeze(0)

    jac_fn = vmap(jacrev(single_decode))

    with torch.no_grad():
        n_h = decoder(Z[:1]).shape[1]

    max_elements = 32 * 1024 * 1024
    sub_batch = max(1, min(B, max_elements // max(n_h * n_z, 1)))
    row_idx = torch.randint(0, n_h, (num_rows,), device=Z.device)

    abs_sum = torch.tensor(0.0, device=Z.device)
    for i in range(0, B, sub_batch):
        Z_sub = Z[i:i + sub_batch]
        J_sub = jac_fn(Z_sub)               # [sub, n_h, n_z]
        J_sampled = J_sub[:, row_idx, :]    # [sub, num_rows, n_z]
        abs_sum = abs_sum + J_sampled.abs().sum()

    actual_rows = min(num_rows, n_h)
    return abs_sum / (actual_rows * B * n_z)


def _jacobian_l1_loop(decoder: nn.Module, z: torch.Tensor, num_rows: int) -> torch.Tensor:
    # Use a fresh z copy with grad — detached from the main encoder graph
    # so that inplace ops on g_out don't corrupt the outer computation graph.
    Z = z.detach().requires_grad_(True)
    h_rec = decoder(Z)                      # [B, n_h]
    n_h = h_rec.shape[1]
    row_idx = torch.randint(0, n_h, (num_rows,), device=Z.device)

    rows = []
    for ri in row_idx:
        # Create a fresh zero tensor each iteration — no link to h_rec whatsoever
        g_out = torch.zeros(h_rec.shape, dtype=h_rec.dtype, device=h_rec.device)
        g_out[:, ri] = 1.0
        grads = torch.autograd.grad(
            h_rec, Z, grad_outputs=g_out,
            create_graph=True, retain_graph=True,
        )[0]                                # [B, n_z]
        rows.append(grads)

    J = torch.stack(rows, dim=0)           # [num_rows, B, n_z]
    return J.abs().mean()


def total_loss(
    h_orig: torch.Tensor,
    h_rec: torch.Tensor,
    z: torch.Tensor,
    gate: torch.Tensor,
    decoder: nn.Module,
    *,
    lambda_sparse: float = 0.01,
    lambda_attn: float = 0.1,
    jacobian_sample_rows: int = 64,
) -> Tuple[torch.Tensor, dict]:
    """Compute the combined training loss.

    Loss summary:
        L_recon   = weighted_MSE( gate, h_orig, h_rec )
                    High-gate features → high reconstruction penalty.
                    Low-gate features  → low reconstruction penalty.
        L_sparse  = mean(|z|)   [L1 on bottleneck — fast sparsity]
        L_attn    = binary_entropy( gate )
                    Pushes gate toward {0,1} — crisp feature selection.
        L_total   = L_recon + λ_sparse * L_sparse + λ_attn * L_attn
    """
    l_recon  = weighted_reconstruction_loss(h_orig, h_rec, gate)
    l_sparse = sparsity_loss(z)                # fast O(B×Z) L1, not Jacobian
    l_attn   = gate_binary_entropy_loss(gate)

    l_total = l_recon + lambda_sparse * l_sparse + lambda_attn * l_attn

    breakdown = {
        "recon":  l_recon.item(),
        "sparse": l_sparse.item(),
        "attn":   l_attn.item(),
        "total":  l_total.item(),
    }
    return l_total, breakdown
