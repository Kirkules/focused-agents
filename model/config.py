"""Hyperparameter configuration for the story writer model."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 1024
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 10
    # Dropout rate; 0.0 for pre-training, small value (e.g. 0.1) for fine-tuning
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        """
        Hidden dimension for the SwiGLU FFN.

        Target: parameter parity with a standard 4× GELU FFN.
        A standard FFN has 2 matrices of size d_model × 4*d_model.
        SwiGLU has 3 matrices of size d_model × hidden, so:
            3 × d_model × hidden = 2 × d_model × 4*d_model
            hidden = 8/3 × d_model ≈ 2.67 × d_model
        Rounded up to the nearest multiple of 64 for hardware efficiency.
        """
        raw = int(8 * self.d_model / 3)
        return ((raw + 63) // 64) * 64
