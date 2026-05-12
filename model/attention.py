"""
Causal self-attention with Rotary Position Embeddings (RoPE).

RoPE encodes position by rotating query and key vectors before the dot
product. The rotation angle depends on the token's position and the
dimension pair, so the dot product Q·K naturally captures relative
position. Only Q and K are rotated — V is left unchanged.

The rotation is implemented with real arithmetic (rotate_half) rather
than complex multiplication because MPS does not support complex tensors.
Both formulations are mathematically equivalent.
"""

import torch
import torch.nn as nn

from model.config import ModelConfig


def precompute_rope_freqs(
    head_dim: int,
    context_length: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine tables for RoPE.

    For each dimension pair i, the frequency is θ_i = 1 / (theta^(2i/head_dim)).
    For each position m, the rotation angle is m × θ_i.

    Args:
        head_dim: dimension of each attention head (must be even).
        context_length: maximum sequence length to precompute for.
        theta: base for the geometric frequency progression (default 10000,
               matching the original RoPE paper and LLaMA).

    Returns:
        cos, sin: each of shape (context_length, head_dim), float32.
        The first and second halves of the last dimension are identical —
        this supports the rotate_half convention used in apply_rope.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Inverse frequencies: one per dimension pair, shape (head_dim//2,)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Position indices, shape (context_length,)
    positions = torch.arange(context_length).float()

    # Outer product: angle[m, i] = m × θ_i, shape (context_length, head_dim//2)
    angles = torch.outer(positions, inv_freq)

    # Duplicate for both halves so the shape matches head_dim
    angles = torch.cat([angles, angles], dim=-1)  # (context_length, head_dim)

    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rearrange x for the RoPE rotation: swap and negate the two halves.

    For x = [x1, x2] (each half of the last dimension), returns [-x2, x1].
    This implements the 2D rotation matrix applied blockwise across pairs.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    The rotation for a token at position m is:
        x_rotated = x * cos(m·θ) + rotate_half(x) * sin(m·θ)

    This is the real-arithmetic equivalent of multiplying complex-valued
    pairs by e^(i·m·θ). Only Q and K are rotated; call this before computing
    attention scores, not on V.

    Args:
        q, k: shape (batch, n_heads, seq_len, head_dim).
        cos, sin: shape (seq_len, head_dim) — already sliced to current seq_len.

    Returns:
        Rotated q and k, same shapes as inputs.
    """
    # Broadcast cos/sin over batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE positional encoding.

    Each token attends only to itself and earlier tokens (causal masking).
    Positions are encoded via rotations applied to Q and K before scoring.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5  # 1/sqrt(head_dim) for score scaling

        # Q, K, V input projections — no bias, following modern convention
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Output projection mixes information across heads
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular boolean, registered as a non-learned buffer.
        # True = allowed to attend, False = masked out (set to -inf before softmax).
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length,
                                  dtype=torch.bool)),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch, seq_len, d_model).
            cos, sin: RoPE tables sliced to seq_len, shape (seq_len, head_dim).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, T, _ = x.shape

        # Project input to queries, keys, values
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into separate heads: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K — V carries content, not position
        q, k = apply_rope(q, k, cos, sin)

        # Scaled dot-product attention scores: (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Zero out attention to future positions by setting their scores to -inf
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        # Softmax over the key dimension gives attention weights
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Weighted sum of values: (B, n_heads, T, head_dim)
        out = weights @ v

        # Merge heads back: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.out_proj(out)
