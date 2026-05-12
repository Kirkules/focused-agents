"""Tests for TinyStories preprocessing."""

from pipeline.tinystories import tinystories_units
from pipeline.tokens import BOT, EOT


def _make_fake_dataset(stories: list[dict]):
    """Return an iterable that mimics a HuggingFace streaming dataset."""
    return stories


def test_tinystories_units_wraps_with_special_tokens(monkeypatch):
    # Each yielded unit should begin with BOT and end with EOT
    fake = [{"text": "Once upon a time there was a little girl who loved to read books every single day without fail or exception."}]
    monkeypatch.setattr(
        "pipeline.tinystories.load_dataset",
        lambda *a, **kw: _make_fake_dataset(fake),
    )
    units = list(tinystories_units())
    assert len(units) == 1
    assert units[0].startswith(BOT)
    assert units[0].endswith(EOT)


def test_tinystories_units_filters_short_stories(monkeypatch):
    # Stories below the minimum word count should be skipped
    fake = [
        {"text": "Too short."},
        {"text": " ".join(["word"] * 25)},  # above minimum
    ]
    monkeypatch.setattr(
        "pipeline.tinystories.load_dataset",
        lambda *a, **kw: _make_fake_dataset(fake),
    )
    units = list(tinystories_units())
    assert len(units) == 1


def test_tinystories_units_respects_max_stories(monkeypatch):
    # max_stories should limit the number of stories yielded
    fake = [{"text": " ".join(["word"] * 30)}] * 10
    monkeypatch.setattr(
        "pipeline.tinystories.load_dataset",
        lambda *a, **kw: _make_fake_dataset(fake),
    )
    units = list(tinystories_units(max_stories=3))
    assert len(units) == 3


def test_tinystories_units_strips_whitespace(monkeypatch):
    # Leading/trailing whitespace in story text should be stripped
    fake = [{"text": "  A story with leading spaces and many more extra words spread across the whole sentence here to pass the word minimum.  "}]
    monkeypatch.setattr(
        "pipeline.tinystories.load_dataset",
        lambda *a, **kw: _make_fake_dataset(fake),
    )
    units = list(tinystories_units())
    assert len(units) == 1
    # The content inside the tokens should not have leading/trailing whitespace
    inner = units[0][len(BOT):- len(EOT)].strip()
    assert not inner.startswith(" ")
    assert not inner.endswith(" ")
