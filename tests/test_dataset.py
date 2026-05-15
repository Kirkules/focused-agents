"""
Tests for TokenDataset.

The dataset reads a flat binary token file and produces (input, target) pairs
where target is input shifted right by one position. This shift is the
conceptual core of autoregressive language model training: at every position
the model's job is to predict the next token.
"""

import numpy as np
import torch
import pytest

from train.dataset import TokenDataset


def _write_tokens(path, tokens: list[int]) -> None:
    """Write a list of token IDs to a binary file as uint16."""
    np.array(tokens, dtype=np.uint16).tofile(str(path))


# --- Length ---

def test_dataset_length(tmp_path):
    # With N tokens and context_length C, there are N - C valid samples.
    # Each sample needs C+1 consecutive tokens (C input + 1 target overhang).
    _write_tokens(tmp_path / "tok.bin", list(range(100)))
    ds = TokenDataset(str(tmp_path / "tok.bin"), context_length=10)
    assert len(ds) == 100 - 10


def test_dataset_length_minimum(tmp_path):
    # Exactly context_length + 1 tokens yields one sample
    _write_tokens(tmp_path / "tok.bin", list(range(11)))
    ds = TokenDataset(str(tmp_path / "tok.bin"), context_length=10)
    assert len(ds) == 1


# --- Shift relationship ---

def test_target_is_input_shifted_by_one(tmp_path):
    """
    The defining property of the dataset: target[j] == input[j+1] for all
    interior positions. This test is the clearest expression of what
    'next-token prediction' means at the data level.

    Also verifies sample[0] specifically so the absolute token values are
    checked, not just the relative shift.
    """
    tokens = list(range(32))
    _write_tokens(tmp_path / "tok.bin", tokens)
    ds = TokenDataset(str(tmp_path / "tok.bin"), context_length=8)

    x, y = ds[0]
    # First sample: input = 0..7, target = 1..8
    assert torch.equal(x, torch.arange(0, 8, dtype=torch.long))
    assert torch.equal(y, torch.arange(1, 9, dtype=torch.long))

    # For any sample: the interior shift must hold (x[1:] == y[:-1])
    x, y = ds[5]
    assert torch.equal(x[1:], y[:-1]), \
        "Interior shift violated: x[j+1] != y[j] for some j"


def test_last_valid_index_does_not_go_out_of_bounds(tmp_path):
    # The last index (len - 1) should not read past the end of the file
    n = 50
    _write_tokens(tmp_path / "tok.bin", list(range(n)))
    context_length = 10
    ds = TokenDataset(str(tmp_path / "tok.bin"), context_length=context_length)
    x, y = ds[len(ds) - 1]
    assert x.shape == (context_length,)
    assert y.shape == (context_length,)


# --- Output types and shapes ---

def test_output_tensors_are_long(tmp_path):
    # Embedding lookup requires torch.long (int64) indices
    _write_tokens(tmp_path / "tok.bin", list(range(20)))
    ds = TokenDataset(str(tmp_path / "tok.bin"), context_length=8)
    x, y = ds[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_output_shapes(tmp_path):
    context_length = 16
    _write_tokens(tmp_path / "tok.bin", list(range(64)))
    ds = TokenDataset(str(tmp_path / "tok.bin"), context_length=context_length)
    x, y = ds[0]
    assert x.shape == (context_length,)
    assert y.shape == (context_length,)


# --- Memory mapping ---

def test_dataset_uses_memory_map_not_full_load(tmp_path, monkeypatch):
    """
    The dataset should memory-map the file rather than loading it fully into
    RAM. This matters for large corpora that don't fit in memory.
    Verified by checking that np.memmap is used (not np.fromfile).
    """
    calls = []
    real_memmap = np.memmap

    def tracking_memmap(*args, **kwargs):
        calls.append(args[0])
        return real_memmap(*args, **kwargs)

    monkeypatch.setattr(np, "memmap", tracking_memmap)
    _write_tokens(tmp_path / "tok.bin", list(range(20)))
    TokenDataset(str(tmp_path / "tok.bin"), context_length=8)
    assert len(calls) == 1, "Expected exactly one np.memmap call"
