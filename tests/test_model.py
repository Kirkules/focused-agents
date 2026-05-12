"""
Tests for the GPT model architecture.

Covers output shapes, parameter count, weight tying, causal masking
(both structural and behavioral), RoPE properties, SwiGLU hidden dim,
loss computation, and MPS availability.
"""

import torch
import pytest

from model.config import ModelConfig
from model.model import GPT
from model.attention import precompute_rope_freqs, rotate_half, apply_rope


# Small config used across most tests to keep them fast
@pytest.fixture
def small_config():
    return ModelConfig(vocab_size=256, context_length=32, d_model=64, n_heads=4, n_layers=2)


@pytest.fixture
def small_model(small_config):
    return GPT(small_config)


# --- Config ---

def test_head_dim(small_config):
    # head_dim should be d_model / n_heads
    assert small_config.head_dim == 64 // 4 == 16


def test_ffn_hidden_dim_is_approximate_swiglu_target(small_config):
    # ffn_hidden_dim should be close to 8/3 * d_model and a multiple of 64
    expected_raw = 8 * small_config.d_model / 3
    hidden = small_config.ffn_hidden_dim
    assert hidden % 64 == 0
    assert abs(hidden - expected_raw) < 64  # within one rounding step


def test_default_config_parameter_count():
    # Default config should land near 10M parameters
    model = GPT(ModelConfig())
    n = model.count_parameters()
    assert 8_000_000 < n < 12_000_000, f"Expected ~10M params, got {n:,}"


# --- Weight tying ---

def test_weight_tying(small_model):
    # LM head and token embedding must share the exact same parameter tensor
    assert small_model.lm_head.weight is small_model.token_emb.weight


# --- Output shapes ---

def test_output_logits_shape(small_model, small_config):
    # Forward pass should return (batch, seq_len, vocab_size) logits
    batch, seq_len = 2, 16
    idx = torch.randint(0, small_config.vocab_size, (batch, seq_len))
    logits, loss = small_model(idx)
    assert logits.shape == (batch, seq_len, small_config.vocab_size)
    assert loss is None


def test_output_loss_is_scalar_when_targets_provided(small_model, small_config):
    # Providing targets should return a scalar loss
    batch, seq_len = 2, 16
    idx = torch.randint(0, small_config.vocab_size, (batch, seq_len))
    targets = torch.randint(0, small_config.vocab_size, (batch, seq_len))
    logits, loss = small_model(idx, targets=targets)
    assert loss is not None
    assert loss.shape == ()  # scalar
    assert loss.item() > 0


def test_sequence_length_one(small_model, small_config):
    # Model should handle a single-token sequence without error
    idx = torch.randint(0, small_config.vocab_size, (1, 1))
    logits, _ = small_model(idx)
    assert logits.shape == (1, 1, small_config.vocab_size)


def test_exceeding_context_length_raises(small_model, small_config):
    # Inputs longer than context_length should raise an AssertionError
    idx = torch.randint(0, small_config.vocab_size, (1, small_config.context_length + 1))
    with pytest.raises(AssertionError):
        small_model(idx)


# --- Causal masking ---

def test_causal_mask_is_lower_triangular(small_model, small_config):
    # The causal mask stored in each attention module should be lower-triangular
    mask = small_model.blocks[0].attn.causal_mask
    T = small_config.context_length
    expected = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert torch.equal(mask, expected)


def test_causal_masking_is_enforced(small_model, small_config):
    """
    Behavioral test: changing a future token must not affect logits at
    earlier positions. This verifies the causal mask is actually applied,
    not just stored.
    """
    small_model.eval()
    seq_len = 8
    idx = torch.randint(0, small_config.vocab_size, (1, seq_len))

    with torch.no_grad():
        logits_original, _ = small_model(idx)

    # Modify the token at position 5; positions 0-4 should be unaffected
    idx_modified = idx.clone()
    idx_modified[0, 5] = (idx[0, 5] + 1) % small_config.vocab_size

    with torch.no_grad():
        logits_modified, _ = small_model(idx_modified)

    # Positions before the change: logits must be identical
    assert torch.allclose(logits_original[0, :5], logits_modified[0, :5]), \
        "Causal masking failed: earlier positions were affected by a future token change"

    # Position at the change: logits may differ
    assert not torch.allclose(logits_original[0, 5], logits_modified[0, 5]), \
        "Expected logits at the changed position to differ"


# --- RoPE ---

def test_rope_freqs_shape():
    # precompute_rope_freqs should return (context_length, head_dim) tensors
    head_dim, context_length = 16, 32
    cos, sin = precompute_rope_freqs(head_dim, context_length)
    assert cos.shape == (context_length, head_dim)
    assert sin.shape == (context_length, head_dim)


def test_rope_freqs_are_unit_amplitude():
    # cos²(θ) + sin²(θ) = 1 should hold for all entries
    cos, sin = precompute_rope_freqs(16, 32)
    assert torch.allclose(cos ** 2 + sin ** 2, torch.ones_like(cos), atol=1e-6)


def test_rope_buffers_registered(small_model, small_config):
    # rope_cos and rope_sin should be registered as buffers (not parameters)
    buffer_names = {name for name, _ in small_model.named_buffers()}
    assert "rope_cos" in buffer_names
    assert "rope_sin" in buffer_names


def test_rotate_half_shape_preserved():
    # rotate_half should not change the tensor shape
    x = torch.randn(2, 4, 8, 16)
    assert rotate_half(x).shape == x.shape


def test_apply_rope_shape_preserved():
    # apply_rope should return tensors with the same shape as inputs
    B, H, T, D = 2, 4, 8, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    cos, sin = precompute_rope_freqs(D, T)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_apply_rope_changes_values():
    # RoPE should actually modify Q and K (not be a no-op)
    B, H, T, D = 1, 1, 4, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    cos, sin = precompute_rope_freqs(D, T)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    # At position 0 cos=1, sin=0 so rotation is identity — check a later position
    assert not torch.allclose(q[0, 0, 1], q_rot[0, 0, 1])


# --- MPS ---

def test_model_runs_on_mps_if_available(small_config):
    # If MPS is available, the model should run on it without error
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")
    model = GPT(small_config).to("mps")
    idx = torch.randint(0, small_config.vocab_size, (1, 8)).to("mps")
    logits, _ = model(idx)
    assert logits.device.type == "mps"
