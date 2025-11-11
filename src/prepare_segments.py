"""Preprocess the ``segments.xlsx`` catalogue into JSON artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from annotation_with_roberta.data import (
    ensure_processed_datasets,
    read_segments_metadata,
    save_label_catalogue_json,
    save_metadata_json,
)

LOGGER = logging.getLogger("annotation_with_roberta.prepare_segments")

SEGMENTS_XLSX = PROJECT_ROOT / "data/segments.xlsx"
RAW_METADATA_JSON = PROJECT_ROOT / "data/segments_metadata.json"
CATALOGUE_JSON = PROJECT_ROOT / "data/segments_catalogue.json"
TRAIN_TEXT = PROJECT_ROOT / "data/train.txt"
DEV_TEXT = PROJECT_ROOT / "data/dev.txt"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
TRAIN_CONLL = PROCESSED_DIR / "train.conll"
DEV_CONLL = PROCESSED_DIR / "dev.conll"
LABEL2ID_JSON = PROCESSED_DIR / "label2id.json"
ID2LABEL_JSON = PROCESSED_DIR / "id2label.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    if not SEGMENTS_XLSX.exists():
        LOGGER.error("Segments spreadsheet not found at %s", SEGMENTS_XLSX)
        raise SystemExit(1)

    LOGGER.info("Reading label definitions from %s", SEGMENTS_XLSX)
    metadata = read_segments_metadata(SEGMENTS_XLSX)

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
        segments_source=SEGMENTS_XLSX,
        eval_text_file=DEV_TEXT,
        eval_output_file=DEV_CONLL,
    )

    LOGGER.info("Finished preprocessing spreadsheet")


if __name__ == "__main__":
    main()
