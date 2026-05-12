"""
Train and load a BPE tokenizer for the story model.

Special tokens are registered before BPE training so they are never
decomposed into sub-tokens. Byte-level pre-tokenization ensures all
UTF-8 text is representable without unknown tokens.
"""

import os
from collections.abc import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from pipeline.tokens import SPECIAL_TOKENS

# Default vocabulary size. Kept small relative to GPT-2 (50k) because
# the embedding table is proportionally expensive at ~10M total parameters.
# 8192 tokens at d_model=512 costs 4M params — about 40% of the budget.
DEFAULT_VOCAB_SIZE = 8192

_TOKENIZER_FILENAME = "tokenizer.json"


def train_tokenizer(
    unit_iterator: Iterator[str],
    save_dir: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> Tokenizer:
    """
    Train a BPE tokenizer on the provided text units and save it to disk.

    Special tokens are pre-registered so they are treated as atomic units
    and never split by BPE merges. Byte-level pre-tokenization handles all
    Unicode without requiring an explicit unknown token.

    Args:
        unit_iterator: iterator of wrapped text strings (BOT ... EOT).
        save_dir: directory where tokenizer.json will be written.
        vocab_size: total vocabulary size including special tokens.

    Returns:
        the trained Tokenizer, ready for encoding and decoding.
    """
    tokenizer = Tokenizer(BPE(unk_token=None))

    # Byte-level ensures every possible byte is representable as a token
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        # Merges for pairs seen fewer than min_frequency times are not learned
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(unit_iterator, trainer=trainer)

    # Persist tokenizer to disk so it can be reloaded without retraining
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, _TOKENIZER_FILENAME))

    return tokenizer


def load_tokenizer(save_dir: str) -> Tokenizer:
    """
    Load a previously trained tokenizer from disk.

    Args:
        save_dir: directory containing tokenizer.json.

    Returns:
        the loaded Tokenizer object.
    """
    path = os.path.join(save_dir, _TOKENIZER_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No tokenizer found at {path}")
    return Tokenizer.from_file(path)
