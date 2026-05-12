"""
Shuffle, tokenize, pack, and split document units into training examples.

The shuffle happens at the unit level (before tokenization) so that
within-unit contiguity is preserved. The train/val split is also at the
unit level to prevent data leakage from the same source document appearing
in both sets.

Output: flat binary numpy uint16 arrays (train.bin, val.bin), one token
per element. The training loop reads fixed-length chunks from these files.
"""

import os
import random

import numpy as np
from tokenizers import Tokenizer

# Size of each packed training example in tokens.
CONTEXT_LENGTH = 1024

# Fraction of units held out for validation.
VAL_FRACTION = 0.05

# uint16 supports vocabulary sizes up to 65535, sufficient for our 8192-token vocab.
_TOKEN_DTYPE = np.uint16

_TRAIN_FILENAME = "train.bin"
_VAL_FILENAME = "val.bin"


def pack_units(
    units: list[str],
    tokenizer: Tokenizer,
    output_dir: str,
    context_length: int = CONTEXT_LENGTH,
    val_fraction: float = VAL_FRACTION,
    seed: int = 42,
) -> dict[str, int]:
    """
    Tokenize, shuffle, pack, and split units into train and val sets.

    Units are shuffled at the document level, then concatenated into a
    long token stream and chunked into context_length-sized windows. The
    tail tokens that do not fill a complete window are discarded.

    Note: all units are held in memory simultaneously. For very large
    datasets this may need to be replaced with a disk-based shuffle.

    Args:
        units: list of all wrapped text strings from all sources.
        tokenizer: trained tokenizer used to encode each unit.
        output_dir: directory where train.bin and val.bin are written.
        context_length: number of tokens per training example.
        val_fraction: fraction of units to hold out for validation.
        seed: random seed for the unit-level shuffle and split.

    Returns:
        dict mapping "train_tokens" and "val_tokens" to their counts.
    """
    rng = random.Random(seed)

    # Shuffle at unit level before any tokenization
    shuffled = units[:]
    rng.shuffle(shuffled)

    # Split at the unit level so no source document spans both sets
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_units = shuffled[:n_val]
    train_units = shuffled[n_val:]

    os.makedirs(output_dir, exist_ok=True)

    # Tokenize, pack, and write each split
    train_path = os.path.join(output_dir, _TRAIN_FILENAME)
    val_path = os.path.join(output_dir, _VAL_FILENAME)

    train_count = _tokenize_and_pack(train_units, tokenizer, context_length, train_path)
    val_count = _tokenize_and_pack(val_units, tokenizer, context_length, val_path)

    return {"train_tokens": train_count, "val_tokens": val_count}


def _tokenize_and_pack(
    units: list[str],
    tokenizer: Tokenizer,
    context_length: int,
    output_path: str,
) -> int:
    """
    Encode units, concatenate tokens, chunk into windows, write to disk.

    Units are encoded one at a time and appended to a flat list. The list
    is then truncated to a multiple of context_length and saved as uint16.

    Args:
        units: text strings to encode.
        tokenizer: tokenizer for encoding.
        context_length: size of each output window in tokens.
        output_path: file path for the binary output.

    Returns:
        total number of tokens written to disk.
    """
    # Encode all units and concatenate into one long token stream
    all_tokens: list[int] = []
    for unit in units:
        encoded = tokenizer.encode(unit)
        all_tokens.extend(encoded.ids)

    # Discard the tail that doesn't fill a complete context window
    n_complete = (len(all_tokens) // context_length) * context_length
    all_tokens = all_tokens[:n_complete]

    # Write as a flat binary array of uint16 token IDs
    arr = np.array(all_tokens, dtype=_TOKEN_DTYPE)
    arr.tofile(output_path)

    return n_complete
