"""
Transformer block: causal self-attention + SwiGLU MLP, both with pre-norm.

Pre-norm applies layer normalization to the input before each sub-layer,
then adds the sub-layer output back via a residual connection. This is
more stable than post-norm, especially for small models and long training.

SwiGLU MLP replaces the standard activation(W₁x)W₂ design with a learned
gate: SiLU(W_gate·x) ⊙ (W₁·x), projected down by W₂. The gate learns
which neurons to activate for each input rather than using a fixed
mathematical criterion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import CausalSelfAttention
from model.config import ModelConfig


class SwiGLUMLP(nn.Module):
    """
    Feed-forward network using the SwiGLU activation.

    Uses three weight matrices instead of two: W_gate and W1 project up
    in parallel, their outputs are combined via a gated product, and W2
    projects back down. Hidden dimension is 8/3 × d_model (≈2.67×) to
    keep total parameters equivalent to a standard 4× expansion FFN.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.ffn_hidden_dim

        # Gate projection: produces the learned activation mask
        self.w_gate = nn.Linear(config.d_model, hidden, bias=False)
        # Value projection: produces the content to be gated
        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        # Down projection: maps gated hidden state back to d_model
        self.w2 = nn.Linear(hidden, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU gate controls how much of each hidden neuron passes through
        gate = F.silu(self.w_gate(x))
        # Element-wise product applies the gate to the value projection
        hidden = gate * self.w1(x)
        return self.dropout(self.w2(hidden))


class TransformerBlock(nn.Module):
    """
    One transformer block: pre-norm attention + pre-norm SwiGLU MLP.

    The residual connections allow gradients to flow directly from output
    to input, enabling training of deep networks without vanishing gradients.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Layer norms applied before (not after) each sub-layer
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            cos, sin: RoPE tables sliced to seq_len, passed through to attention.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Attention sub-layer with residual
        x = x + self.attn(self.norm1(x), cos, sin)
        # MLP sub-layer with residual
        x = x + self.mlp(self.norm2(x))
        return x
