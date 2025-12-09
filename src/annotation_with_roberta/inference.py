"""Inference helpers for the multilingual slot annotation model."""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

try:
    from .data import SlotMetadata, read_segments_metadata
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys

    package_root = Path(__file__).resolve().parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from data import SlotMetadata, read_segments_metadata

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SENTENCES_FILE = PROJECT_ROOT / "data/sentences.txt"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_SEGMENTS_FILE = PROJECT_ROOT / "data/segments.xlsx"
DEFAULT_SLOT_VALUE_MAP = DEFAULT_MODEL_DIR / "slot_value_map.json"


@dataclass
class AnnotatedSpan:
    label: str
    words: List[str]

    def text(self) -> str:
        return ' '.join(self.words)


class AutoAnnotator:
    """Annotate sentences with the trained XLM-RoBERTa model."""

    def __init__(
        self,
        model_dir: Path,
        segments_file: Path,
        *,
        slot_value_map_path: Optional[Path] = None,
        device: Optional[str] = None,
        max_length: int = 256,
    ) -> None:
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.id2label: Dict[int, str] = {
            int(idx): label for idx, label in self.model.config.id2label.items()
        }
        self.label2id: Dict[str, int] = {
            label: idx for idx, label in self.id2label.items()
        }

        self.metadata = read_segments_metadata(segments_file)
        self.fallback_label = 'other' if 'other' in self.metadata else None

        if slot_value_map_path is None:
            slot_value_map_path = model_dir / 'slot_value_map.json'
        if slot_value_map_path.exists():
            self.slot_value_map = json.loads(slot_value_map_path.read_text(encoding='utf-8'))
        else:
            LOGGER.warning("Slot value map %s not found; limited slots may miss values", slot_value_map_path)
            self.slot_value_map = {}
        self.max_length = max_length

    def annotate(self, sentence: str) -> str:
        words = sentence.strip().split()
        if not words:
            return ''
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoding.word_ids()
        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        word_labels: List[str] = []
        for word_index in range(len(words)):
            token_indices = [i for i, wid in enumerate(word_ids) if wid == word_index]
            if not token_indices:
                word_labels.append('O')
                continue
            label_id = predictions[token_indices[0]]
            word_labels.append(self.id2label[int(label_id)])

        spans = self._merge_spans(words, word_labels)
        return ' '.join(self._format_span(span) for span in spans)

    def _merge_spans(self, words: Sequence[str], labels: Sequence[str]) -> List[AnnotatedSpan]:
        spans: List[AnnotatedSpan] = []
        current_label: Optional[str] = None
        current_words: List[str] = []
        for word, label in zip(words, labels):
            base_label = label.split('-', 1)[1] if '-' in label else label
            prefix = label.split('-', 1)[0] if '-' in label else 'B'
            if label == 'O' or base_label not in self.metadata:
                if self.fallback_label is None:
                    if current_label is not None and current_words:
                        spans.append(AnnotatedSpan(current_label, current_words))
                    current_label = None
                    current_words = []
                    continue
                base_label = self.fallback_label
                prefix = 'B'
            if current_label is None or base_label != current_label or prefix == 'B':
                if current_label is not None and current_words:
                    spans.append(AnnotatedSpan(current_label, current_words))
                current_label = base_label
                current_words = [word]
            else:
                current_words.append(word)
        if current_label is not None and current_words:
            spans.append(AnnotatedSpan(current_label, current_words))
        return spans

    def _format_span(self, span: AnnotatedSpan) -> str:
        surface = span.text()
        slot_meta = self.metadata.get(span.label)
        if slot_meta is None:
            LOGGER.warning("Unknown label %s encountered during formatting", span.label)
            return f"[{span.label}:{surface}]"
        value = self._resolve_value(span.label, surface, slot_meta)
        if value:
            return f"[{span.label}:{surface}##{value}]"
        return f"[{span.label}:{surface}]"

    def _resolve_value(
        self,
        label: str,
        surface: str,
        slot_meta: SlotMetadata,
    ) -> Optional[str]:
        normalized_surface = ' '.join(surface.split()).lower()
        label_value_map = self.slot_value_map.get(label, {})
        if normalized_surface in label_value_map:
            return label_value_map[normalized_surface]
        # Fallback: look for direct value match ignoring case
        for candidate in slot_meta.values:
            if candidate.lower() == normalized_surface:
                return candidate
        if slot_meta.limited:
            allowed_lower = [candidate.lower() for candidate in slot_meta.values]
            choice_lookup = dict(zip(allowed_lower, slot_meta.values))
            best_match: Optional[str] = None
            best_score = -1.0
            for candidate in allowed_lower:
                score = difflib.SequenceMatcher(None, normalized_surface, candidate).ratio()
                if score > best_score:
                    best_score = score
                    best_match = candidate
            if best_match is not None:
                resolved = choice_lookup[best_match]
                LOGGER.debug(
                    "Resolved limited slot %s value '%s' to closest allowed option '%s'",
                    label,
                    surface,
                    resolved,
                )
                return resolved
            if slot_meta.values:
                return slot_meta.values[0]
        return None


def _iter_sentences(path: Path) -> Iterator[str]:
    if not path.exists():
        LOGGER.error("Sentences file %s not found", path)
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sentence = line.strip()
            if sentence:
                yield sentence


def annotate_sentences(
    annotator: AutoAnnotator,
    sentences: Iterable[str],
) -> Iterator[str]:
    for sentence in sentences:
        if not sentence:
            continue
        yield annotator.annotate(sentence)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    slot_value_override = DEFAULT_SLOT_VALUE_MAP if DEFAULT_SLOT_VALUE_MAP.exists() else None
    annotator = AutoAnnotator(
        model_dir=DEFAULT_MODEL_DIR,
        segments_file=DEFAULT_SEGMENTS_FILE,
        slot_value_map_path=slot_value_override,
    )

    for line in annotate_sentences(annotator, _iter_sentences(DEFAULT_SENTENCES_FILE)):
        print(line)


if __name__ == "__main__":
    main()
