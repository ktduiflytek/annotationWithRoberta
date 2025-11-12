"""Data loading helpers for the multilingual slot annotation project."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from zipfile import ZipFile
from xml.etree import ElementTree as ET

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
ANNOTATION_ERROR_LOG = LOG_DIR / "annotation_errors.log"

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


class AnnotationParseError(ValueError):
    """Raised when the annotated text cannot be converted into BIO labels."""

    def __init__(self, message: str, *, line: Optional[str] = None, line_no: Optional[int] = None):
        if line is not None and line_no is not None:
            message = f"Line {line_no}: {message} :: {line.strip()}"
        super().__init__(message)


def _log_annotation_error(message: str) -> None:
    """Persist annotation parsing issues so they can be reviewed later."""

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        with ANNOTATION_ERROR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except OSError as exc:  # pragma: no cover - best effort logging
        LOGGER.error("Failed to write annotation error log: %s (%s)", message, exc)


def _split_surface_and_value(body: str) -> Tuple[str, Optional[str]]:
    if "##" not in body:
        return body, None
    surface, value = body.split("##", 1)
    return surface, value


def _normalise_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def parse_annotated_line(
    line: str,
    metadata: Mapping[str, SlotMetadata],
    *,
    line_no: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Convert a bracket-annotated sentence into BIO tokens and labels."""

    stripped = line.strip()
    if not stripped:
        return [], []

    tokens: List[str] = []
    labels: List[str] = []
    cursor = 0
    for match in SPAN_PATTERN.finditer(line):
        start, end = match.span()
        if cursor < start:
            leftover = _normalise_whitespace(line[cursor:start])
            if leftover:
                raise AnnotationParseError(
                    "Found text outside of annotated spans",
                    line=line,
                    line_no=line_no,
                )
        cursor = end

        label = match.group("label").strip()
        if label not in metadata:
            raise AnnotationParseError(
                f"Unknown label '{label}' not present in segments metadata",
                line=line,
                line_no=line_no,
            )
        body = match.group("body").strip()
        surface, value = _split_surface_and_value(body)
        surface = _normalise_whitespace(surface)
        if not surface:
            continue

        slot_meta = metadata[label]
        if slot_meta.limited:
            resolved_value = value.strip() if value else ""
            if not resolved_value:
                candidates = {candidate.lower(): candidate for candidate in slot_meta.values}
                resolved_value = candidates.get(surface.lower(), "")
            if not resolved_value:
                raise AnnotationParseError(
                    f"Limited slot '{label}' is missing a valid value",
                    line=line,
                    line_no=line_no,
                )
            if resolved_value not in slot_meta.values:
                raise AnnotationParseError(
                    f"Value '{resolved_value}' is not allowed for limited slot '{label}'",
                    line=line,
                    line_no=line_no,
                )

        span_tokens = [token for token in surface.split(" ") if token]
        if not span_tokens:
            continue
        tokens.extend(span_tokens)
        labels.append(f"B-{label}")
        labels.extend(f"I-{label}" for _ in span_tokens[1:])

    if cursor < len(line):
        trailing = _normalise_whitespace(line[cursor:])
        if trailing:
            raise AnnotationParseError(
                "Found text outside of annotated spans",
                line=line,
                line_no=line_no,
            )

    return tokens, labels


def parse_annotated_corpus(
    path: Path,
    metadata: Mapping[str, SlotMetadata],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Parse an annotated text file into token/label sequences."""

    sentences: List[List[str]] = []
    labels: List[List[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                tokens, token_labels = parse_annotated_line(line, metadata, line_no=line_no)
            except AnnotationParseError as error:
                error_message = f"{path} :: {error}"
                LOGGER.warning(
                    "Skipping annotated line %d in %s due to parse error: %s",
                    line_no,
                    path,
                    error,
                )
                _log_annotation_error(error_message)
                continue
            if tokens:
                sentences.append(tokens)
                labels.append(token_labels)
    if not sentences:
        raise AnnotationParseError(f"No annotated examples found in {path}")
    return sentences, labels


def write_conll_dataset(
    tokens: Sequence[Sequence[str]],
    labels: Sequence[Sequence[str]],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for token_seq, label_seq in zip(tokens, labels):
            for token, label in zip(token_seq, label_seq):
                handle.write(f"{token}\t{label}\n")
            handle.write("\n")


def build_label_map(sequences: Sequence[Sequence[str]]) -> Dict[str, int]:
    labels: Set[str] = set()
    for sequence in sequences:
        labels.update(sequence)
    if not labels:
        raise ValueError("Cannot build label map from empty sequences")
    return {label: idx for idx, label in enumerate(sorted(labels))}


def save_label_map_json(label_map: Mapping[str, int], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({key: int(value) for key, value in label_map.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_id_label_map_json(id_map: Mapping[int, str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({str(key): value for key, value in id_map.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_label_map_json(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(label): int(index) for label, index in raw.items()}


def _needs_rebuild(target: Path, dependencies: Iterable[Path]) -> bool:
    if not target.exists():
        return True
    target_mtime = target.stat().st_mtime
    for dependency in dependencies:
        if not dependency:
            continue
        if dependency.exists() and dependency.stat().st_mtime > target_mtime:
            return True
    return False


def ensure_processed_datasets(
    *,
    metadata: Mapping[str, SlotMetadata],
    train_text_file: Optional[Path],
    train_output_file: Path,
    label_map_file: Path,
    id_map_file: Optional[Path] = None,
    segments_source: Optional[Path] = None,
    eval_text_file: Optional[Path] = None,
    eval_output_file: Optional[Path] = None,
) -> None:
    """Rebuild the processed CoNLL datasets when their sources change."""

    dependencies: List[Path] = [p for p in [segments_source] if p is not None]

    train_dependencies = list(dependencies)
    if train_text_file is not None:
        train_dependencies.append(train_text_file)
    rebuild_train = _needs_rebuild(train_output_file, train_dependencies)

    if rebuild_train:
        if train_text_file is None or not train_text_file.exists():
            raise FileNotFoundError(
                "Annotated training text is required to rebuild processed data, but none was provided",
            )
        LOGGER.info("Rebuilding processed training data from %s", train_text_file)
        train_tokens, train_labels = parse_annotated_corpus(train_text_file, metadata)
        write_conll_dataset(train_tokens, train_labels, train_output_file)
    else:
        _, train_labels = read_conll_dataset(train_output_file)

    label_sequences: List[Sequence[str]] = []
    if rebuild_train:
        label_sequences.extend(train_labels)
    else:
        label_sequences.extend(read_conll_dataset(train_output_file)[1])

    if eval_output_file is not None:
        eval_dependencies = list(dependencies)
        if eval_text_file is not None:
            eval_dependencies.append(eval_text_file)
        rebuild_eval = _needs_rebuild(eval_output_file, eval_dependencies)
        if rebuild_eval:
            if eval_text_file is None or not eval_text_file.exists():
                LOGGER.warning(
                    "Skipping dev set rebuild because annotated text %s is missing",
                    eval_text_file,
                )
            else:
                LOGGER.info("Rebuilding processed dev data from %s", eval_text_file)
                eval_tokens, eval_labels = parse_annotated_corpus(eval_text_file, metadata)
                write_conll_dataset(eval_tokens, eval_labels, eval_output_file)
                label_sequences.extend(eval_labels)
        elif eval_output_file.exists():
            label_sequences.extend(read_conll_dataset(eval_output_file)[1])
    else:
        rebuild_eval = False

    label_dependencies = [train_output_file]
    if eval_output_file is not None and eval_output_file.exists():
        label_dependencies.append(eval_output_file)
    rebuild_label_map = _needs_rebuild(label_map_file, label_dependencies)
    if rebuild_label_map:
        LOGGER.info("Rebuilding label2id map at %s", label_map_file)
        if not label_sequences:
            label_sequences.extend(read_conll_dataset(train_output_file)[1])
            if eval_output_file is not None and eval_output_file.exists():
                label_sequences.extend(read_conll_dataset(eval_output_file)[1])
        label_map = build_label_map(label_sequences)
        save_label_map_json(label_map, label_map_file)
        if id_map_file is not None:
            id_map = {idx: label for label, idx in label_map.items()}
            save_id_label_map_json(id_map, id_map_file)
    elif id_map_file is not None and not id_map_file.exists():
        LOGGER.info("Writing id2label map to %s", id_map_file)
        label_map = load_label_map_json(label_map_file)
        id_map = {idx: label for label, idx in label_map.items()}
        save_id_label_map_json(id_map, id_map_file)


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
