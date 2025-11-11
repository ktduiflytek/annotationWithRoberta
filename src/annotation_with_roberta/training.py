"""Model training pipeline for multilingual slot annotation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from datasets import Dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import (
    build_slot_value_map,
    read_conll_dataset,
    read_segments_metadata,
    save_label_catalogue_json,
    save_metadata_json,
    save_value_map_json,
    validate_labels,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingConfig:
    model_name: str
    train_file: Path
    eval_file: Optional[Path]
    segments_file: Path
    label_map_file: Path
    output_dir: Path
    train_text_file: Optional[Path]
    max_length: int = 256
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    num_train_epochs: float = 5.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    label_all_tokens: bool = False
    seed: int = 13


def _load_label_map(path: Path) -> Dict[str, int]:
    with path.open('r', encoding='utf-8') as handle:
        raw = json.load(handle)
    return {str(label): int(idx) for label, idx in raw.items()}


def _prepare_dataset(
    tokens: Sequence[Sequence[str]],
    labels: Sequence[Sequence[str]],
) -> Dataset:
    return Dataset.from_dict({"tokens": list(tokens), "labels": list(labels)})


def _tokenize_and_align(
    examples: Mapping[str, Sequence[Sequence[str]]],
    tokenizer: AutoTokenizer,
    label2id: Mapping[str, int],
    *,
    label_all_tokens: bool,
    max_length: int,
) -> Mapping[str, List[List[int]]]:
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    all_labels: List[List[int]] = []
    for batch_index, label_sequence in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids: List[int] = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label2id[label_sequence[word_id]])
            else:
                if not label_all_tokens:
                    label_ids.append(-100)
                else:
                    label_str = label_sequence[word_id]
                    if label_str.startswith("B-"):
                        continuation = "I-" + label_str[2:]
                        label_ids.append(label2id.get(continuation, label2id[label_str]))
                    else:
                        label_ids.append(label2id[label_str])
            previous_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def _build_datasets(
    tokenizer: AutoTokenizer,
    label2id: Mapping[str, int],
    *,
    train_tokens: Sequence[Sequence[str]],
    train_labels: Sequence[Sequence[str]],
    eval_tokens: Optional[Sequence[Sequence[str]]] = None,
    eval_labels: Optional[Sequence[Sequence[str]]] = None,
    label_all_tokens: bool,
    max_length: int,
) -> Dict[str, Dataset]:
    datasets: Dict[str, Dataset] = {}
    train_dataset = _prepare_dataset(train_tokens, train_labels)
    train_dataset = train_dataset.map(
        lambda examples: _tokenize_and_align(
            examples,
            tokenizer,
            label2id,
            label_all_tokens=label_all_tokens,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=["tokens", "labels"],
    )
    train_dataset.set_format(type="torch")
    datasets["train"] = train_dataset

    if eval_tokens is not None and eval_labels is not None:
        eval_dataset = _prepare_dataset(eval_tokens, eval_labels)
        eval_dataset = eval_dataset.map(
            lambda examples: _tokenize_and_align(
                examples,
                tokenizer,
                label2id,
                label_all_tokens=label_all_tokens,
                max_length=max_length,
            ),
            batched=True,
            remove_columns=["tokens", "labels"],
        )
        eval_dataset.set_format(type="torch")
        datasets["eval"] = eval_dataset
    return datasets


def _compute_metrics_builder(id2label: Mapping[int, str]):
    def compute_metrics(predictions_and_labels):
        logits, labels = predictions_and_labels
        predictions = np.argmax(logits, axis=2)
        true_labels: List[List[str]] = []
        true_predictions: List[List[str]] = []
        for prediction, label_sequence in zip(predictions, labels):
            example_labels: List[str] = []
            example_predictions: List[str] = []
            for pred_id, label_id in zip(prediction, label_sequence):
                if label_id == -100:
                    continue
                example_labels.append(id2label[int(label_id)])
                example_predictions.append(id2label[int(pred_id)])
            true_labels.append(example_labels)
            true_predictions.append(example_predictions)
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    return compute_metrics


def train_model(config: TrainingConfig) -> None:
    """Train the multilingual slot annotation model."""

    config.output_dir.mkdir(parents=True, exist_ok=True)

    label2id = _load_label_map(config.label_map_file)
    id2label = {idx: label for label, idx in label2id.items()}

    segments = read_segments_metadata(config.segments_file)
    train_tokens, train_labels = read_conll_dataset(config.train_file)
    validate_labels(train_labels, segments)

    eval_tokens: Optional[Sequence[Sequence[str]]] = None
    eval_labels: Optional[Sequence[Sequence[str]]] = None
    if config.eval_file:
        eval_tokens, eval_labels = read_conll_dataset(config.eval_file)
        validate_labels(eval_labels, segments)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    datasets = _build_datasets(
        tokenizer,
        label2id,
        train_tokens=train_tokens,
        train_labels=train_labels,
        eval_tokens=eval_tokens,
        eval_labels=eval_labels,
        label_all_tokens=config.label_all_tokens,
        max_length=config.max_length,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    set_seed(config.seed)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        evaluation_strategy="epoch" if "eval" in datasets else "no",
        save_strategy="epoch" if "eval" in datasets else "no",
        load_best_model_at_end="eval" in datasets,
        metric_for_best_model="f1",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        report_to=["none"],
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("eval"),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics_builder(id2label),
    )

    LOGGER.info("Starting training with %d training examples", len(train_tokens))
    trainer.train()

    if "eval" in datasets:
        LOGGER.info("Evaluating best model on validation set")
        trainer.evaluate()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metadata_path = config.output_dir / "label_metadata.json"
    save_metadata_json(segments, metadata_path)

    catalogue_path = config.output_dir / "label_catalogue.json"
    save_label_catalogue_json(segments, catalogue_path)

    train_text_file = config.train_text_file or config.train_file.with_suffix('.txt')
    value_map = build_slot_value_map(train_text_file, segments)
    value_map_path = config.output_dir / "slot_value_map.json"
    save_value_map_json(value_map, value_map_path)

    LOGGER.info("Artifacts saved to %s", config.output_dir)


def default_training_config() -> TrainingConfig:
    """Build a :class:`TrainingConfig` populated with sensible defaults."""

    train_file = PROJECT_ROOT / "data/processed/train.conll"
    eval_file = PROJECT_ROOT / "data/processed/dev.conll"
    segments_file = PROJECT_ROOT / "data/segments.xlsx"
    label_map_file = PROJECT_ROOT / "data/processed/label2id.json"
    train_text_file = PROJECT_ROOT / "data/train.txt"
    output_dir = PROJECT_ROOT / "model"

    return TrainingConfig(
        model_name="xlm-roberta-base",
        train_file=train_file,
        eval_file=eval_file if eval_file.exists() else None,
        segments_file=segments_file,
        label_map_file=label_map_file,
        train_text_file=train_text_file if train_text_file.exists() else None,
        output_dir=output_dir,
        max_length=256,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=500,
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        label_all_tokens=True,
        seed=42,
    )


def main() -> None:
    """Train the model using the baked-in defaults."""

    logging.basicConfig(level=logging.INFO)
    config = default_training_config()
    LOGGER.info("Starting training with configuration: %s", config)
    train_model(config)


if __name__ == "__main__":
    main()
