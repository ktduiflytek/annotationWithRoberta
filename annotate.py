"""Annotate raw sentences from ``sentences.txt`` using the trained model."""

from __future__ import annotations

import logging
from pathlib import Path

from annotation_with_roberta.inference import AutoAnnotator

LOGGER = logging.getLogger("annotation_with_roberta.annotate")

SENTENCES_FILE = Path("sentences.txt")
MODEL_DIR = Path("model")
SEGMENTS_FILE = Path("data/segments.xlsx")
SLOT_VALUE_MAP = MODEL_DIR / "slot_value_map.json"
MAX_LENGTH = 256


def _iter_sentences(path: Path):
    if not path.exists():
        LOGGER.error("Sentences file %s not found", path)
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sentence = line.strip()
            if sentence:
                yield sentence


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    slot_value_override = SLOT_VALUE_MAP if SLOT_VALUE_MAP.exists() else None
    annotator = AutoAnnotator(
        model_dir=MODEL_DIR,
        segments_file=SEGMENTS_FILE,
        slot_value_map_path=slot_value_override,
        max_length=MAX_LENGTH,
    )

    for sentence in _iter_sentences(SENTENCES_FILE):
        if not sentence:
            continue
        print(annotator.annotate(sentence))


if __name__ == "__main__":
    main()
