"""
Preprocess TinyStories for model training.

Loads from roneneldan/TinyStories on HuggingFace. Each story is already
a complete short narrative and becomes one wrapped training unit directly.
"""

from collections.abc import Iterator

from datasets import load_dataset

from pipeline.tokens import BOT, EOT

# Stories shorter than this word count are skipped.
# Guards against empty or near-empty entries in the dataset.
_MIN_WORDS = 20


def tinystories_units(max_stories: int | None = None) -> Iterator[str]:
    """
    Yield wrapped training units (stories) from the TinyStories dataset.

    Streams the dataset. Each yielded string is one complete story wrapped
    with BOT and EOT tokens.

    Args:
        max_stories: cap on the number of stories to yield.
                     None means yield all stories in the dataset.
    """
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    for story_idx, entry in enumerate(dataset):
        if max_stories is not None and story_idx >= max_stories:
            break

        story = entry["text"].strip()

        # Skip stories that are too short to be meaningful training signal
        if len(story.split()) < _MIN_WORDS:
            continue

        yield f"{BOT} {story} {EOT}"
