"""Data loading helpers for the multilingual slot annotation project."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from zipfile import ZipFile
from xml.etree import ElementTree as ET

LOGGER = logging.getLogger(__name__)

SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass(frozen=True)
class SlotMetadata:
    """Metadata describing a single slot/label."""

    name: str
    limited: bool
    values: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "limited": self.limited,
            "values": list(self.values),
        }


def _column_index(cell_reference: str) -> int:
    letters = ''.join(filter(str.isalpha, cell_reference))
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord('A') + 1)
    return index - 1


def _load_shared_strings(zfile: ZipFile) -> List[str]:
    try:
        xml_bytes = zfile.read('xl/sharedStrings.xml')
    except KeyError:
        return []
    root = ET.fromstring(xml_bytes)
    strings: List[str] = []
    for si in root.findall(SPREADSHEET_NS + 'si'):
        text_parts: List[str] = []
        for text_node in si.findall('.//' + SPREADSHEET_NS + 't'):
            if text_node.text:
                text_parts.append(text_node.text)
        strings.append(''.join(text_parts))
    return strings


def _read_sheet_rows(path: Path) -> Iterable[List[str]]:
    with ZipFile(path) as zfile:
        shared_strings = _load_shared_strings(zfile)
        sheet_bytes = zfile.read('xl/worksheets/sheet1.xml')
        root = ET.fromstring(sheet_bytes)
        sheet_data = root.find(SPREADSHEET_NS + 'sheetData')
        if sheet_data is None:
            return []
        for row in sheet_data.findall(SPREADSHEET_NS + 'row'):
            cells: MutableMapping[int, str] = {}
            for cell in row.findall(SPREADSHEET_NS + 'c'):
                ref = cell.get('r')
                if ref is None:
                    continue
                idx = _column_index(ref)
                value = ''
                cell_type = cell.get('t')
                if cell_type == 's':
                    v_elem = cell.find(SPREADSHEET_NS + 'v')
                    if v_elem is not None and v_elem.text is not None:
                        string_idx = int(v_elem.text)
                        if 0 <= string_idx < len(shared_strings):
                            value = shared_strings[string_idx]
                elif cell_type == 'inlineStr':
                    t_elem = cell.find(SPREADSHEET_NS + 'is')
                    if t_elem is not None:
                        text_nodes = t_elem.findall('.//' + SPREADSHEET_NS + 't')
                        value = ''.join(tn.text or '' for tn in text_nodes)
                else:
                    v_elem = cell.find(SPREADSHEET_NS + 'v')
                    if v_elem is not None and v_elem.text is not None:
                        value = v_elem.text
                cells[idx] = value.strip()
            if not cells:
                yield []
                continue
            max_index = max(cells)
            row_values = [''] * (max_index + 1)
            for idx, value in cells.items():
                row_values[idx] = value
            yield row_values


def read_segments_metadata(path: Path) -> Dict[str, SlotMetadata]:
    """Parse the ``segments.xlsx`` file and return metadata for each label."""

    rows = list(_read_sheet_rows(path))
    if not rows:
        raise ValueError(f"No data found in spreadsheet: {path}")
    metadata: Dict[str, SlotMetadata] = {}
    for row in rows[1:]:  # Skip header
        if not row or not row[0].strip():
            continue
        label = row[0].strip()
        slot_type = row[2].strip() if len(row) > 2 else ''
        limited = slot_type == '有限槽'
        raw_values = row[3].strip() if len(row) > 3 else ''
        values: Tuple[str, ...] = tuple(v.strip() for v in raw_values.split('|') if v.strip())
        metadata[label] = SlotMetadata(name=label, limited=limited, values=values)
    return metadata


def read_conll_dataset(path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """Read a CoNLL-style token classification dataset."""

    sentences: List[List[str]] = []
    labels: List[List[str]] = []
    current_tokens: List[str] = []
    current_labels: List[str] = []

    with path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\n')
            if not line.strip():
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue
            parts = line.split('\t')
            if len(parts) == 1:
                parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line in {path}: {line!r}")
            token = parts[0]
            label = parts[-1]
            current_tokens.append(token)
            current_labels.append(label)
    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)
    return sentences, labels


def validate_labels(
    label_sequences: Sequence[Sequence[str]],
    metadata: Mapping[str, SlotMetadata],
) -> None:
    """Ensure that every annotated label is defined in the metadata."""

    allowed = set(metadata.keys())
    missing: Dict[str, int] = {}
    for sequence in label_sequences:
        for label in sequence:
            if label == 'O':
                continue
            base = label.split('-', 1)[1] if '-' in label else label
            if base not in allowed:
                missing[base] = missing.get(base, 0) + 1
    if missing:
        details = ', '.join(f"{name} ({count})" for name, count in sorted(missing.items()))
        raise ValueError(f"Found labels not present in segments metadata: {details}")


SPAN_PATTERN = re.compile(r"\[(?P<label>[^:\]]+):(?P<body>[^\]]+)\]")


def build_slot_value_map(
    path: Path,
    metadata: Mapping[str, SlotMetadata],
) -> Dict[str, Dict[str, str]]:
    """Build a mapping from surface form to canonical slot value.

    Parameters
    ----------
    path:
        Path to the annotated training text (e.g. ``train.txt``).
    metadata:
        Mapping with slot metadata so we can validate limited slots.
    """

    value_map: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        LOGGER.warning("Annotated text file %s does not exist; no slot value map created", path)
        return value_map

    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            for match in SPAN_PATTERN.finditer(line):
                label = match.group('label').strip()
                body = match.group('body').strip()
                if label not in metadata:
                    LOGGER.debug("Skipping unknown label %s on line %d", label, line_no)
                    continue
                surface = body
                value: Optional[str] = None
                if '##' in body:
                    surface, value = body.split('##', 1)
                surface = ' '.join(surface.split())
                if not surface:
                    continue
                slot_meta = metadata[label]
                if value is not None:
                    value = value.strip()
                if slot_meta.limited:
                    if not value:
                        # Maybe the surface form already matches an allowed value
                        normalized_surface = surface.lower()
                        value_candidates = {
                            candidate.lower(): candidate for candidate in slot_meta.values
                        }
                        value = value_candidates.get(normalized_surface)
                    if not value:
                        LOGGER.debug(
                            "Skipping limited slot %s with missing value on line %d", label, line_no
                        )
                        continue
                    if value not in slot_meta.values:
                        LOGGER.debug(
                            "Skipping value %s for slot %s because it is not in the allowed list",
                            value,
                            label,
                        )
                        continue
                elif value:
                    # Keep the training-provided mapping for unlimited slots as well.
                    pass
                if value is None:
                    continue
                value_map.setdefault(label, {})
                normalized_surface = surface.lower()
                if normalized_surface not in value_map[label]:
                    value_map[label][normalized_surface] = value
    return value_map


def save_metadata_json(
    metadata: Mapping[str, SlotMetadata],
    destination: Path,
) -> None:
    """Persist the raw label metadata to JSON for later reuse."""

    destination.write_text(
        json.dumps(
            {name: slot.to_dict() for name, slot in sorted(metadata.items())},
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


def build_label_catalogue(metadata: Mapping[str, SlotMetadata]) -> Dict[str, object]:
    """Split labels into normal slots and limited slots with their values."""

    normal_labels = sorted(slot.name for slot in metadata.values() if not slot.limited)
    limited_labels = {
        name: list(slot.values)
        for name, slot in sorted(metadata.items())
        if slot.limited
    }
    return {"normal": normal_labels, "limited": limited_labels}


def save_label_catalogue_json(
    metadata: Mapping[str, SlotMetadata], destination: Path
) -> None:
    """Persist the separated normal/limited label catalogue to JSON."""

    destination.write_text(
        json.dumps(build_label_catalogue(metadata), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def save_value_map_json(value_map: Mapping[str, Mapping[str, str]], destination: Path) -> None:
    destination.write_text(
        json.dumps(
            {label: mapping for label, mapping in sorted(value_map.items())},
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
