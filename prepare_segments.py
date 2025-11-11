"""Preprocess the ``segments.xlsx`` catalogue into JSON artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

from annotation_with_roberta.data import (
    read_segments_metadata,
    save_label_catalogue_json,
    save_metadata_json,
)

LOGGER = logging.getLogger("annotation_with_roberta.prepare_segments")

SEGMENTS_XLSX = Path("data/segments.xlsx")
RAW_METADATA_JSON = Path("data/segments_metadata.json")
CATALOGUE_JSON = Path("data/segments_catalogue.json")


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

    LOGGER.info("Finished preprocessing spreadsheet")


if __name__ == "__main__":
    main()
