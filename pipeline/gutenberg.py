"""
Preprocess Project Gutenberg texts for model training.

Loads books from the deepmind/pg19 HuggingFace dataset — English books
published before 1919, mostly fiction. Each book is split into chapters;
each chapter becomes one wrapped training unit.
"""

import re
from collections.abc import Iterator

from datasets import load_dataset

from pipeline.tokens import BOT, EOT

# Matches chapter or part headings at the start of a line, e.g.:
# "Chapter 1", "CHAPTER IV", "Part 2", "PART III."
_CHAPTER_RE = re.compile(
    r'^\s*(chapter|part)\s+(\d+|[ivxlcdm]+)[\s\.\:]',
    re.IGNORECASE | re.MULTILINE,
)

# Chapters shorter than this character count are skipped.
# Filters out table-of-contents entries and stub sections.
_MIN_CHAPTER_CHARS = 200


def clean_text(text: str) -> str:
    """
    Remove Project Gutenberg boilerplate and normalize whitespace.

    Strips content before the START marker and after the END marker
    if either is present. Collapses runs of 3+ blank lines to two.
    """
    # Remove header (everything up to and including the START marker)
    start = re.search(r'\*{3}\s*START OF .+?\*{3}', text, re.IGNORECASE)
    if start:
        text = text[start.end():]

    # Remove footer (everything from the END marker onward)
    end = re.search(r'\*{3}\s*END OF .+?\*{3}', text, re.IGNORECASE)
    if end:
        text = text[:end.start()]

    # Collapse runs of blank lines to at most two
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def split_chapters(text: str) -> list[str]:
    """
    Split a book's text into chapters using heading detection.

    Finds lines that match the chapter/part heading pattern and uses them
    as split boundaries. If no headings are found, returns the whole text
    as a single unit (or an empty list if below the minimum length).
    """
    boundaries = [m.start() for m in _CHAPTER_RE.finditer(text)]

    if not boundaries:
        # No chapter structure found — treat whole book as one unit
        return [text] if len(text) >= _MIN_CHAPTER_CHARS else []

    # Extract the text slice between each pair of consecutive boundaries
    chapters = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chapter = text[start:end].strip()
        if len(chapter) >= _MIN_CHAPTER_CHARS:
            chapters.append(chapter)

    return chapters


def gutenberg_units(max_books: int | None = None) -> Iterator[str]:
    """
    Yield wrapped training units (chapters) from the pg19 dataset.

    Streams the dataset to avoid downloading all 37GB upfront. Each
    yielded string is one chapter wrapped with BOT and EOT tokens.

    Args:
        max_books: cap on the number of source books to process.
                   None means process all books in the dataset.
    """
    # Streaming avoids downloading the full dataset before processing
    dataset = load_dataset("deepmind/pg19", split="train", streaming=True)

    for book_idx, book in enumerate(dataset):
        if max_books is not None and book_idx >= max_books:
            break

        # pg19 texts are mostly pre-cleaned but apply cleaning defensively
        text = clean_text(book["text"])
        chapters = split_chapters(text)

        for chapter in chapters:
            yield f"{BOT} {chapter} {EOT}"
