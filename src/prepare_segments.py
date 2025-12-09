"""Preprocess the ``segments.xlsx`` catalogue into JSON artifacts."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from annotation_with_roberta.data import (
    SlotMetadata,
    ensure_processed_datasets,
    read_segments_metadata,
    save_label_catalogue_json,
    save_metadata_json,
)

LOGGER = logging.getLogger("annotation_with_roberta.prepare_segments")

SEGMENTS_XLSX = PROJECT_ROOT / "data/segments.xlsx"
SEGMENTS_NEW_XLSX = PROJECT_ROOT / "data/segments_new.xlsx"
SEGMENTS_NEW_ALT_XLSX = PROJECT_ROOT / "data/segmnet_new.xlsx"
RAW_METADATA_JSON = PROJECT_ROOT / "data/segments_metadata.json"
CATALOGUE_JSON = PROJECT_ROOT / "data/segments_catalogue.json"
TRAIN_TEXT = PROJECT_ROOT / "data/train.txt"
DEV_TEXT = PROJECT_ROOT / "data/dev.txt"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
TRAIN_CONLL = PROCESSED_DIR / "train.conll"
DEV_CONLL = PROCESSED_DIR / "dev.conll"
LABEL2ID_JSON = PROCESSED_DIR / "label2id.json"
ID2LABEL_JSON = PROCESSED_DIR / "id2label.json"


def _merge_metadata(
    *,
    previous: Mapping[str, SlotMetadata],
    current: Mapping[str, SlotMetadata],
) -> Tuple[Dict[str, SlotMetadata], Tuple[str, ...], Tuple[str, ...]]:
    """Merge previous and current metadata while marking removals as deprecated."""

    merged: Dict[str, SlotMetadata] = {}
    added: Tuple[str, ...] = tuple(sorted(set(current) - set(previous)))
    removed: Tuple[str, ...] = tuple(sorted(set(previous) - set(current)))

    for label, slot in current.items():
        prior = previous.get(label)
        if prior:
            merged[label] = replace(slot, deprecated=slot.deprecated or prior.deprecated)
        else:
            merged[label] = slot

    for label in removed:
        prior = previous[label]
        merged[label] = replace(prior, deprecated=True)

    return merged, added, removed


def _load_metadata() -> Tuple[Mapping[str, SlotMetadata], Path]:
    """Load metadata, optionally merging with a newly uploaded spreadsheet."""

    previous_source = SEGMENTS_XLSX if SEGMENTS_XLSX.exists() else None
    new_source = None
    for candidate in (SEGMENTS_NEW_XLSX, SEGMENTS_NEW_ALT_XLSX):
        if candidate.exists():
            new_source = candidate
            break

    if new_source is None and previous_source is None:
        LOGGER.error(
            "Segments spreadsheet not found at %s (or *_new.xlsx variants)", SEGMENTS_XLSX
        )
        raise SystemExit(1)

    if new_source is None:
        LOGGER.info("Reading label definitions from %s", SEGMENTS_XLSX)
        return read_segments_metadata(SEGMENTS_XLSX), SEGMENTS_XLSX

    LOGGER.info("Detected new segments spreadsheet at %s", new_source)
    current_metadata = read_segments_metadata(new_source)
    previous_metadata = read_segments_metadata(previous_source) if previous_source else {}

    merged_metadata, added, removed = _merge_metadata(
        previous=previous_metadata, current=current_metadata
    )

    if added:
        LOGGER.info("New labels added (%d): %s", len(added), ", ".join(added))
    if removed:
        LOGGER.info(
            "Labels removed from spreadsheet will be marked deprecated (%d): %s",
            len(removed),
            ", ".join(removed),
        )

    return merged_metadata, new_source


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    metadata, segments_source = _load_metadata()

    LOGGER.info("Saving raw metadata to %s", RAW_METADATA_JSON)
    save_metadata_json(metadata, RAW_METADATA_JSON)

    LOGGER.info("Saving label catalogue to %s", CATALOGUE_JSON)
    save_label_catalogue_json(metadata, CATALOGUE_JSON)

    LOGGER.info("Regenerating processed datasets and label maps under %s", PROCESSED_DIR)
    ensure_processed_datasets(
        metadata=metadata,
        train_text_file=TRAIN_TEXT,
        train_output_file=TRAIN_CONLL,
        label_map_file=LABEL2ID_JSON,
        id_map_file=ID2LABEL_JSON,
        segments_source=segments_source,
        eval_text_file=DEV_TEXT,
        eval_output_file=DEV_CONLL,
    )

    LOGGER.info("Finished preprocessing spreadsheet")


if __name__ == "__main__":
    main()
