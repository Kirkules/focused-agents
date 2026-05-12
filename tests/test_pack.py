"""Tests for unit-level shuffle, packing, and train/val splitting."""

import os
import tempfile

import numpy as np
import pytest

from pipeline.pack import pack_units, CONTEXT_LENGTH, _tokenize_and_pack
from pipeline.tokenizer import train_tokenizer
from pipeline.tokens import BOT, EOT


def _sample_units(n: int = 40) -> list[str]:
    """Generate units with varied content to make shuffle ordering detectable."""
    # Each unit uses a different anchor word so different orderings produce
    # different token streams (needed to test seed-dependent shuffle)
    anchors = ["fox", "dog", "cat", "bird", "fish", "wolf", "bear", "deer", "frog", "hawk"]
    return [
        f"{BOT} " + f"the {anchors[i % len(anchors)]} runs across the field " * 30 + f" {EOT}"
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    """Train a small tokenizer once for all pack tests."""
    save_dir = str(tmp_path_factory.mktemp("tok"))
    units = _sample_units(60)
    return train_tokenizer(iter(units), save_dir, vocab_size=256)


def test_packed_windows_are_exact_context_length(tokenizer, tmp_path):
    # Every window in the output file must be exactly context_length tokens
    context_length = 64
    units = _sample_units(40)
    counts = pack_units(units, tokenizer, str(tmp_path), context_length=context_length)

    train_path = os.path.join(str(tmp_path), "train.bin")
    arr = np.fromfile(train_path, dtype=np.uint16)

    assert len(arr) % context_length == 0
    assert len(arr) == counts["train_tokens"]


def test_val_split_is_nonzero(tokenizer, tmp_path):
    # Validation set must contain at least one complete window.
    # Use context_length=64 so val units (4 of 40) produce enough tokens to fill a window.
    context_length = 64
    units = _sample_units(40)
    counts = pack_units(units, tokenizer, str(tmp_path), context_length=context_length, val_fraction=0.1)
    assert counts["val_tokens"] > 0


def test_train_and_val_token_counts_match_files(tokenizer, tmp_path):
    # Reported token counts should match actual file sizes
    units = _sample_units(40)
    counts = pack_units(units, tokenizer, str(tmp_path))

    train_arr = np.fromfile(os.path.join(str(tmp_path), "train.bin"), dtype=np.uint16)
    val_arr = np.fromfile(os.path.join(str(tmp_path), "val.bin"), dtype=np.uint16)

    assert len(train_arr) == counts["train_tokens"]
    assert len(val_arr) == counts["val_tokens"]


def test_different_seeds_produce_different_orderings(tokenizer, tmp_path):
    # Two runs with different seeds should produce different train files
    units = _sample_units(40)

    dir_a = os.path.join(str(tmp_path), "a")
    dir_b = os.path.join(str(tmp_path), "b")

    pack_units(units, tokenizer, dir_a, seed=1)
    pack_units(units, tokenizer, dir_b, seed=2)

    arr_a = np.fromfile(os.path.join(dir_a, "train.bin"), dtype=np.uint16)
    arr_b = np.fromfile(os.path.join(dir_b, "train.bin"), dtype=np.uint16)

    # Same total token count but different ordering
    assert len(arr_a) == len(arr_b)
    assert not np.array_equal(arr_a, arr_b)


def test_same_seed_is_reproducible(tokenizer, tmp_path):
    # Two runs with the same seed should produce identical files
    units = _sample_units(40)

    dir_a = os.path.join(str(tmp_path), "a")
    dir_b = os.path.join(str(tmp_path), "b")

    pack_units(units, tokenizer, dir_a, seed=99)
    pack_units(units, tokenizer, dir_b, seed=99)

    arr_a = np.fromfile(os.path.join(dir_a, "train.bin"), dtype=np.uint16)
    arr_b = np.fromfile(os.path.join(dir_b, "train.bin"), dtype=np.uint16)

    assert np.array_equal(arr_a, arr_b)


def test_packing_integrity_special_tokens(tokenizer, tmp_path):
    # Every BOT token ID should be matched by a subsequent EOT within the stream.
    # This verifies that units are not interleaved across boundaries.
    units = _sample_units(20)
    pack_units(units, tokenizer, str(tmp_path))

    bot_id = tokenizer.encode(BOT).ids[0]
    eot_id = tokenizer.encode(EOT).ids[0]

    arr = np.fromfile(os.path.join(str(tmp_path), "train.bin"), dtype=np.uint16)
    tokens = arr.tolist()

    # Walk through and verify BOT is always followed eventually by EOT
    # before another BOT appears (or end of stream)
    inside_unit = False
    for tok in tokens:
        if tok == bot_id:
            # Should not encounter BOT while already inside a unit
            assert not inside_unit, "BOT encountered before previous unit closed with EOT"
            inside_unit = True
        elif tok == eot_id:
            inside_unit = False
    # Stream may end mid-unit (tail truncation is expected), so no assertion at end


def test_output_files_are_created(tokenizer, tmp_path):
    # Both train.bin and val.bin must exist after a successful pack run
    units = _sample_units(20)
    pack_units(units, tokenizer, str(tmp_path))
    assert os.path.exists(os.path.join(str(tmp_path), "train.bin"))
    assert os.path.exists(os.path.join(str(tmp_path), "val.bin"))
