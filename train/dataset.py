"""
PyTorch Dataset for flat binary token files.

Reads a uint16 binary file via memory mapping and yields (input, target) pairs
for autoregressive language model training. The target is the input shifted
right by one position: target[j] == input[j+1].
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """
    Dataset over a flat binary file of uint16 token IDs.

    Each sample is a (input, target) pair of length context_length.
    With N tokens and context length C, there are N - C valid samples:
    sample i covers tokens[i : i+C] (input) and tokens[i+1 : i+C+1] (target).

    The file is memory-mapped, so only the accessed pages are loaded into RAM.
    This is essential for large corpora that exceed available memory.

    Args:
        path: path to the flat binary uint16 token file.
        context_length: number of tokens per input/target sequence.
    """

    def __init__(self, path: str, context_length: int) -> None:
        self.context_length = context_length
        # Memory-map as read-only uint16 — no full load into RAM
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        # Each sample needs context_length + 1 consecutive tokens
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Slice the window; copy forces materialization from the mmap view
        chunk = self.tokens[idx : idx + self.context_length + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
