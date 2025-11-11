"""Command-line entry point for training the multilingual annotation model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from annotation_with_roberta.training import TrainingConfig, train_model

LOGGER = logging.getLogger("annotation_with_roberta.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="xlm-roberta-base", help="Pretrained checkpoint")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.conll"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/processed/dev.conll"))
    parser.add_argument("--segments-file", type=Path, default=Path("data/segments.xlsx"))
    parser.add_argument(
        "--label-map-file", type=Path, default=Path("data/processed/label2id.json"),
        help="JSON file mapping labels to integer ids",
    )
    parser.add_argument(
        "--train-text-file", type=Path, default=Path("data/train.txt"),
        help="Annotated training text used to build the slot value map",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--label-all-tokens",
        action="store_true",
        help="Propagate labels to all subword tokens instead of masking them",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation and checkpoint selection on the validation split",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    eval_file = None if args.no_eval else args.eval_file

    config = TrainingConfig(
        model_name=args.model_name,
        train_file=args.train_file,
        eval_file=eval_file,
        segments_file=args.segments_file,
        label_map_file=args.label_map_file,
        output_dir=args.output_dir,
        train_text_file=args.train_text_file,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        label_all_tokens=args.label_all_tokens,
        seed=args.seed,
    )

    train_model(config)


if __name__ == "__main__":
    main()
