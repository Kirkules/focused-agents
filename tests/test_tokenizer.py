"""Tests for BPE tokenizer training and special token handling."""

import os
import tempfile

import pytest

from pipeline.tokenizer import train_tokenizer, load_tokenizer
from pipeline.tokens import BOT, EOT, SPECIAL_TOKENS


def _sample_units() -> list[str]:
    """Small corpus sufficient to train a minimal tokenizer."""
    return [
        f"{BOT} Once upon a time there was a small dragon who lived in a cave. {EOT}",
        f"{BOT} The knight rode through the dark forest seeking the lost crown. {EOT}",
        f"{BOT} Chapter 1 The Arrival\nShe stepped off the train into cold morning air. {EOT}",
        f"{BOT} Chapter 2 The Discovery\nBehind the door lay a room full of old books. {EOT}",
    ] * 20  # repeat to give BPE enough data to find merges


@pytest.fixture
def trained_tokenizer(tmp_path):
    """Train a small tokenizer and return it along with its save directory."""
    tokenizer = train_tokenizer(iter(_sample_units()), str(tmp_path), vocab_size=256)
    return tokenizer, str(tmp_path)


def test_special_tokens_are_single_ids(trained_tokenizer):
    # Each special token must encode to exactly one token ID
    tokenizer, _ = trained_tokenizer
    for token in SPECIAL_TOKENS:
        encoded = tokenizer.encode(token)
        assert len(encoded.ids) == 1, (
            f"Special token {token!r} encoded to {len(encoded.ids)} tokens, expected 1"
        )


def test_special_tokens_vocabulary_round_trip(trained_tokenizer):
    # Each special token's ID must map back to the original string via the vocabulary.
    # We check id_to_token rather than decode() because the ByteLevel decoder strips
    # special tokens from decoded output — that is expected behavior.
    tokenizer, _ = trained_tokenizer
    for token in SPECIAL_TOKENS:
        ids = tokenizer.encode(token).ids
        assert len(ids) == 1
        recovered = tokenizer.id_to_token(ids[0])
        assert recovered == token, f"Vocabulary round-trip failed for {token!r}: got {recovered!r}"


def test_bot_and_eot_have_distinct_ids(trained_tokenizer):
    # BOT and EOT must be different tokens
    tokenizer, _ = trained_tokenizer
    bot_id = tokenizer.encode(BOT).ids[0]
    eot_id = tokenizer.encode(EOT).ids[0]
    assert bot_id != eot_id


def test_vocab_size_does_not_exceed_requested(trained_tokenizer):
    # Actual vocabulary size should not exceed the requested size
    tokenizer, _ = trained_tokenizer
    assert tokenizer.get_vocab_size() <= 256


def test_tokenizer_persisted_and_loadable(trained_tokenizer):
    # A tokenizer saved to disk should load and produce identical output
    tokenizer, save_dir = trained_tokenizer
    loaded = load_tokenizer(save_dir)

    sample = f"{BOT} The dragon flew over the mountains. {EOT}"
    original_ids = tokenizer.encode(sample).ids
    loaded_ids = loaded.encode(sample).ids
    assert original_ids == loaded_ids


def test_load_tokenizer_raises_on_missing_file(tmp_path):
    # Attempting to load from a directory with no tokenizer should raise
    with pytest.raises(FileNotFoundError):
        load_tokenizer(str(tmp_path))


def test_regular_text_encodes_without_error(trained_tokenizer):
    # Arbitrary prose should encode and decode without raising
    tokenizer, _ = trained_tokenizer
    text = f"{BOT} She walked slowly down the cobblestone street. {EOT}"
    ids = tokenizer.encode(text).ids
    assert len(ids) > 0
    decoded = tokenizer.decode(ids)
    assert isinstance(decoded, str)
