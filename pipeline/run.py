"""
End-to-end training data pipeline runner.

Downloads and preprocesses Gutenberg and TinyStories, trains a BPE
tokenizer on the combined corpus, then packs everything into binary
training files.

Usage:
    python -m pipeline.run [options]

The tokenizer is trained first (pass 1 over the data), then units are
collected into memory for the unit-level shuffle (pass 2). This requires
two passes over the data but keeps the shuffle correct.
"""

import argparse
import itertools

from pipeline.gutenberg import gutenberg_units
from pipeline.tinystories import tinystories_units
from pipeline.tokenizer import train_tokenizer
from pipeline.pack import pack_units, CONTEXT_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the training data pipeline.")
    parser.add_argument(
        "--max-books", type=int, default=200,
        help="Max Gutenberg books to process (default: 200)",
    )
    parser.add_argument(
        "--max-stories", type=int, default=500_000,
        help="Max TinyStories stories to process (default: 500000)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=8192,
        help="BPE vocabulary size (default: 8192)",
    )
    parser.add_argument(
        "--context-length", type=int, default=CONTEXT_LENGTH,
        help=f"Tokens per training example (default: {CONTEXT_LENGTH})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Root directory for tokenizer and packed data (default: data/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffle and split (default: 42)",
    )
    args = parser.parse_args()

    tokenizer_dir = f"{args.output_dir}/tokenizer"
    packed_dir = f"{args.output_dir}/packed"

    # --- Pass 1: train tokenizer on combined corpus ---
    # Both iterators are chained so the tokenizer sees the full vocabulary
    print(f"Training tokenizer (vocab_size={args.vocab_size})...")
    gut_iter = gutenberg_units(max_books=args.max_books)
    ts_iter = tinystories_units(max_stories=args.max_stories)
    tokenizer = train_tokenizer(
        itertools.chain(gut_iter, ts_iter),
        tokenizer_dir,
        args.vocab_size,
    )
    print(f"Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")

    # --- Pass 2: collect units into memory for unit-level shuffle ---
    # Iterators are exhausted after pass 1, so we regenerate them
    print("Collecting units for packing...")
    gut_units = list(gutenberg_units(max_books=args.max_books))
    ts_units = list(tinystories_units(max_stories=args.max_stories))
    all_units = gut_units + ts_units
    print(
        f"Collected {len(all_units)} units "
        f"({len(gut_units)} chapters, {len(ts_units)} stories)"
    )

    # --- Pack into fixed-length training examples ---
    print("Packing units into training examples...")
    counts = pack_units(
        all_units,
        tokenizer,
        packed_dir,
        context_length=args.context_length,
        seed=args.seed,
    )
    print(
        f"Done.\n"
        f"  Train: {counts['train_tokens']:,} tokens → {packed_dir}/train.bin\n"
        f"  Val:   {counts['val_tokens']:,} tokens → {packed_dir}/val.bin"
    )


if __name__ == "__main__":
    main()
