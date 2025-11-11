# Multilingual Slot Annotation with XLM-RoBERTa

This repository trains an XLM-RoBERTa token-classification model that labels every word in a sentence with one of the pre-defined slot names from [`data/segments.xlsx`](data/segments.xlsx). Limited slots (column `C` marked `有限槽`) also require a canonical value from column `D`, which is appended to the annotated output as `##取值`.

The project is organised as a three-stage pipeline:

1. **Spreadsheet preprocessing** – convert the Excel catalogue into JSON metadata so labels and their limited values can be validated quickly.
2. **Model training** – fine-tune XLM-RoBERTa on the token/label dataset while enforcing the spreadsheet catalogue.
3. **Sentence annotation** – read raw sentences from `sentences.txt` and emit fully-labelled sequences using the trained model.

## Environment

Install the required dependencies (PyTorch, 🤗 Transformers, Datasets, and SeqEval). Example using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All project modules live inside `src/`, so add it to your `PYTHONPATH` when running the scripts:

```bash
export PYTHONPATH=src
```

## Pipeline

Follow these steps to reproduce the full workflow.

### 1. Preprocess the spreadsheet

`prepare_segments.py` reads the Excel file with helpers from [`src/annotation_with_roberta/data.py`](src/annotation_with_roberta/data.py) and writes two JSON artifacts:

* `data/segments_metadata.json` – the raw rows from the spreadsheet with all supporting columns.
* `data/segments_catalogue.json` – a compact catalogue that splits normal labels from limited (`有限槽`) labels and maps each limited label to its allowed `取值` values.

Run the script whenever `segments.xlsx` changes:

```bash
python prepare_segments.py
```

### 2. Train the model

`train.py` fine-tunes `xlm-roberta-base` (configurable) on the processed dataset and persists the model plus helpful metadata (tokenizer, label maps, and `slot_value_map.json`).

```bash
export PYTHONPATH=src
python train.py \
  --model-name xlm-roberta-base \
  --train-file data/processed/train.conll \
  --eval-file data/processed/dev.conll \
  --segments-file data/segments.xlsx \
  --label-map-file data/processed/label2id.json \
  --train-text-file data/train.txt \
  --output-dir model
```

Use `--no-eval` to skip validation, adjust optimisation hyperparameters through the CLI flags, and enable `--label-all-tokens` if you prefer to propagate labels across subword pieces. All paths default to the locations shown above, so you can omit flags when using the standard layout.

### 3. Annotate sentences

`annotate.py` loads the trained model through the [`AutoAnnotator` in src/annotation_with_roberta/inference.py](src/annotation_with_roberta/inference.py) and annotates every sentence in `sentences.txt`. Each line should contain one sentence to annotate, with blank lines ignored.

```bash
python annotate.py
```

The script prints bracketed annotations such as:

```
[open:Åben] [page:siden] [prep:for] [userManual:bruger manual] [now:nu]
```

If the model predicts a label that is not listed in `segments.xlsx`, or if it returns `O`, the token is emitted with the fallback label `other` so that no word is left unlabeled. Limited slots that lack a known value trigger a warning so you can update the slot value map or underlying data.

## Project Structure

```
.
├── annotate.py                  # Reads sentences.txt and prints annotations with the trained model
├── prepare_segments.py          # Converts segments.xlsx into JSON metadata/catalogue
├── train.py                     # Fine-tunes XLM-RoBERTa using the processed dataset
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
