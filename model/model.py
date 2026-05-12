"""
GPT-style decoder-only transformer for story text generation.

Architecture choices:
  - RoPE positional encoding (no learned positional embedding table)
  - Pre-norm (layer norm before each sub-layer)
  - SwiGLU MLP activation
  - Weight tying: LM head shares the token embedding matrix
  - Causal self-attention (each token attends only to prior tokens)

Weight initialization follows GPT-2: normal(0, 0.02) for all weights,
with residual projections scaled down by 1/sqrt(2 × n_layers) to prevent
the residual stream from growing too large in magnitude as depth increases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import precompute_rope_freqs
from model.block import TransformerBlock
from model.config import ModelConfig


class GPT(nn.Module):
    """
    Decoder-only transformer language model.

    Takes a sequence of token IDs and returns logits over the vocabulary
    at each position. When targets are provided, also returns the
    cross-entropy loss (used during training).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings — no positional embedding; RoPE handles position
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm before the LM head
        self.norm_f = nn.LayerNorm(config.d_model)

        # LM head projects d_model → vocab_size to produce next-token logits.
        # Weight-tied with token_emb: the same matrix used to embed tokens on
        # input is reused to score tokens on output.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE cos/sin tables and register as non-learned buffers
        # so they are moved to the correct device with the model
        cos, sin = precompute_rope_freqs(config.head_dim, config.context_length)
        self.register_buffer("rope_cos", cos)  # (context_length, head_dim)
        self.register_buffer("rope_sin", sin)  # (context_length, head_dim)

        # Apply weight initialization to all submodules
        self.apply(self._init_weights)

        # Residual projections receive an additional scale-down by 1/sqrt(2*n_layers).
        # These are the layers whose outputs feed directly into residual streams:
        # the attention output projection and the MLP down projection.
        # Scaling prevents the residual stream magnitude from growing with depth.
        residual_std = 0.02 / (2 * config.n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.w2.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding weights to normal(0, 0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: token ID sequence, shape (batch, seq_len).
            targets: next-token targets for loss computation, shape (batch, seq_len).
                     If None, no loss is returned.

        Returns:
            logits: shape (batch, seq_len, vocab_size).
            loss: scalar cross-entropy loss, or None if targets not provided.
        """
        B, T = idx.shape
        assert T <= self.config.context_length, (
            f"Input length {T} exceeds context_length {self.config.context_length}"
        )

        # Embed tokens
        x = self.token_emb(idx)  # (B, T, d_model)

        # Slice RoPE tables to the current sequence length
        cos = self.rope_cos[:T]  # (T, head_dim)
        sin = self.rope_sin[:T]  # (T, head_dim)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin)

        # Final layer norm
        x = self.norm_f(x)  # (B, T, d_model)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for F.cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
