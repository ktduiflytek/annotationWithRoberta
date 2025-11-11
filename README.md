# Multilingual Slot Annotation with XLM-RoBERTa

This repository trains an XLM-RoBERTa token-classification model that labels every word in a sentence with one of the pre-defined slot names from [`data/segments.xlsx`](data/segments.xlsx). Limited slots (column `C` marked `有限槽`) also require a canonical value from column `D`, which is appended to the annotated output as `##取值`.

## Environment

Install the required dependencies (PyTorch, 🤗 Transformers, Datasets, and SeqEval). Example using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets seqeval
```

All project modules live inside `src/`, so add it to your `PYTHONPATH` when running the scripts:

```bash
export PYTHONPATH=src
```

## Training

Use the `train.py` script to fine-tune `xlm-roberta-base` on the processed CoNLL data:

```bash
python train.py \
  --model-name xlm-roberta-base \
  --train-file data/processed/train.conll \
  --eval-file data/processed/dev.conll \
  --segments-file data/segments.xlsx \
  --label-map-file data/processed/label2id.json \
  --train-text-file data/train.txt \
  --output-dir model
```

Key features:

* Validates that every label in the dataset is defined in `segments.xlsx`.
* Aligns labels with subword tokens and trains with the Hugging Face `Trainer` API.
* Saves the fine-tuned model, tokenizer, label metadata, and a `slot_value_map.json` file derived from the annotated training text so that limited slots can be rendered with the proper `##取值` values.

Use `--no-eval` to skip validation, adjust learning rate or batch sizes via CLI flags, and enable `--label-all-tokens` if you prefer to propagate labels across subword pieces.

## Annotation

After training, generate annotations for raw sentences with `annotate.py`:

```bash
python annotate.py \
  --model-dir model \
  --segments-file data/segments.xlsx \
  "Åben siden for bruger manual nu"
```

The script loads the trained model, enforces the slot/value constraints from the spreadsheet, and outputs bracketed annotations such as:

```
[open:Åben] [page:siden] [prep:for] [userManual:bruger manual] [now:nu]
```

If the model predicts a label that is not listed in `segments.xlsx`, or if it returns `O`, the token is emitted with the fallback label `other` so that no word is left unlabeled. Limited slots that lack a known value trigger a warning.

You can also stream sentences from standard input:

```bash
echo "Sluk for apple carplay nu" | python annotate.py --model-dir model --segments-file data/segments.xlsx
```

## Project Structure

```
.
├── annotate.py                  # CLI for inference
├── train.py                     # CLI for fine-tuning XLM-RoBERTa
├── src/annotation_with_roberta/
│   ├── __init__.py              # Lightweight package entry point
│   ├── data.py                  # Spreadsheet & dataset utilities
│   ├── inference.py             # AutoAnnotator class for predictions
│   └── training.py              # Training loop built on Hugging Face Trainer
└── data/
    ├── segments.xlsx            # Label catalogue and slot values
    ├── train.txt / dev.txt      # Annotated training data
    └── processed/*.conll/json   # Token/label data used for training
```

This setup lets you fine-tune once and reuse the saved artifacts to annotate utterances in any of the 26 supported languages.
