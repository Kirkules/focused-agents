"""Tests for Gutenberg preprocessing: cleaning and chapter splitting."""

import pytest

from pipeline.gutenberg import clean_text, split_chapters, gutenberg_units
from pipeline.tokens import BOT, EOT


# --- clean_text ---

def test_clean_text_strips_gutenberg_header():
    # Header boilerplate before the START marker should be removed
    raw = "Some header junk\n*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\nActual content."
    result = clean_text(raw)
    assert "header junk" not in result
    assert "Actual content." in result


def test_clean_text_strips_gutenberg_footer():
    # Footer boilerplate after the END marker should be removed
    raw = "Story text.\n*** END OF THE PROJECT GUTENBERG EBOOK FOO ***\nLicense blurb."
    result = clean_text(raw)
    assert "Story text." in result
    assert "License blurb." not in result


def test_clean_text_no_markers_returns_full_text():
    # Text without markers is returned as-is (pg19 texts may be pre-cleaned)
    raw = "Just plain prose with no markers."
    assert clean_text(raw) == raw


def test_clean_text_collapses_excess_blank_lines():
    # Three or more consecutive blank lines should collapse to two
    raw = "Para one.\n\n\n\nPara two."
    result = clean_text(raw)
    assert "\n\n\n" not in result
    assert "Para one." in result
    assert "Para two." in result


# --- split_chapters ---

def test_split_chapters_detects_chapter_headings():
    # One heading per chapter; content long enough to pass the minimum length filter
    text = (
        "Chapter 1 The Beginning\n" + "Some prose here. " * 20 + "\n\n"
        "Chapter 2 The Middle\n" + "More prose here. " * 20 + "\n\n"
    )
    chapters = split_chapters(text)
    assert len(chapters) == 2


def test_split_chapters_detects_roman_numeral_headings():
    # Roman numeral chapter headings should be recognized
    text = ("Chapter IV The Storm\n" + "a" * 200 + "\n") * 3
    chapters = split_chapters(text)
    assert len(chapters) == 3


def test_split_chapters_detects_part_headings():
    # "Part N" headings should also split the text
    text = ("Part 1 Opening\n" + "a" * 200 + "\n") * 2
    chapters = split_chapters(text)
    assert len(chapters) == 2


def test_split_chapters_no_headings_returns_single_unit():
    # Books without chapter headings come back as one unit
    text = "a" * 300
    chapters = split_chapters(text)
    assert len(chapters) == 1
    assert chapters[0] == text


def test_split_chapters_filters_short_chapters():
    # Chapters below the minimum character count are dropped
    text = "Chapter 1 Tiny\nShort.\n\nChapter 2 Substantial\n" + "a" * 300
    chapters = split_chapters(text)
    # Chapter 1 is too short; only Chapter 2 should survive
    assert len(chapters) == 1
    assert "Substantial" in chapters[0]


def test_split_chapters_empty_text_returns_empty():
    assert split_chapters("") == []


def test_split_chapters_short_text_no_headings_returns_empty():
    # Text below the minimum length with no headings should not be returned
    assert split_chapters("Too short.") == []


# --- gutenberg_units (using synthetic data via monkeypatching) ---

def _make_fake_dataset(books: list[dict]):
    """Return an iterable that mimics a HuggingFace streaming dataset."""
    return books


def test_gutenberg_units_wraps_with_special_tokens(monkeypatch):
    # Each yielded unit should begin with BOT and end with EOT
    fake_books = [{"text": "Chapter 1 Opening\n" + "a" * 300}]
    monkeypatch.setattr(
        "pipeline.gutenberg.load_dataset",
        lambda *a, **kw: _make_fake_dataset(fake_books),
    )
    units = list(gutenberg_units())
    assert len(units) > 0
    for unit in units:
        assert unit.startswith(BOT)
        assert unit.endswith(EOT)


def test_gutenberg_units_respects_max_books(monkeypatch):
    # max_books should limit the number of source books processed
    fake_books = [{"text": "Chapter 1\n" + "a" * 300}] * 10
    monkeypatch.setattr(
        "pipeline.gutenberg.load_dataset",
        lambda *a, **kw: _make_fake_dataset(fake_books),
    )
    units_limited = list(gutenberg_units(max_books=2))
    units_all = list(gutenberg_units(max_books=10))
    assert len(units_limited) < len(units_all)
