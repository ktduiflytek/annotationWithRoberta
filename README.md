# Multilingual Slot Annotation with XLM-RoBERTa

This repository trains an XLM-RoBERTa token-classification model that labels each token span—covering individual words or multi-word phrases—in a sentence with one of the pre-defined slot names from [`data/segments.xlsx`](data/segments.xlsx). Limited slots (column `C` marked `有限槽`) also require a canonical value from column `D`, which is appended to the annotated output as `##取值`.

The project is organised as a three-stage pipeline. Run the steps in order the first time you set up a workspace, then re-run the pieces that correspond to the artefacts you change:

1. **Spreadsheet preprocessing** – convert the Excel catalogue into JSON metadata so labels and their limited values can be validated quickly.
2. **Model training** – fine-tune XLM-RoBERTa on the token/label dataset while enforcing the spreadsheet catalogue.
3. **Sentence annotation** – read raw sentences from `data/sentences.txt` and emit fully labelled sequences using the trained model.

### Quickstart

```bash
# 1) Regenerate JSON metadata from the spreadsheet (rerun whenever segments.xlsx changes)
python src/prepare_segments.py

# 2) Fine-tune the model; processed/*.conll and label2id.json are rebuilt automatically
python src/annotation_with_roberta/training.py

# 3) Label every sentence listed in data/sentences.txt
python src/annotation_with_roberta/inference.py
```

## Environment

Install the required dependencies (PyTorch, 🤗 Transformers, Datasets, and SeqEval). Example using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All project code lives under `src/`, so run each step with `python path/to/script.py` from the repository root.

## Pipeline

Follow these steps to reproduce the full workflow.

### 1. Preprocess the spreadsheet

`src/prepare_segments.py` is the gatekeeper for the spreadsheet. It performs all of the following:

* loads `data/segments.xlsx` with the XML-level parsers defined in [`src/annotation_with_roberta/data.py`](src/annotation_with_roberta/data.py) so no external Excel reader is required,
* validates that every row with a label name in column **A** also records whether it is a limited slot (`有限槽`) in column **C**,
* expands the pipe-delimited list of canonical values in column **D** into Python tuples, and
* persists two JSON artifacts that downstream steps reuse:
  * `data/segments_metadata.json` – a faithful JSON rendering of the spreadsheet so you can audit every label, whether it is limited, and its allowed values.
  * `data/segments_catalogue.json` – a compact summary that separates normal labels from limited labels and only keeps the value lists. This format loads quickly inside training and inference.

Run the script whenever `segments.xlsx` changes so both JSON files stay synchronised with the authoritative spreadsheet. If you skip this step after editing the spreadsheet, the later stages will warn that the metadata is stale.

```bash
python src/prepare_segments.py
```

### 2. Train the model

`src/annotation_with_roberta/training.py` fine-tunes `xlm-roberta-base` using the JSON artefacts above plus the token/label data under `data/processed/`. Deleting the processed artefacts is safe—if those files are missing or older than your spreadsheet / annotated text sources, the script automatically regenerates them from `data/train.txt` (and `data/dev.txt` when available) before training starts. When you run the module directly it will:

1. Load the default configuration returned by `default_training_config()` (see the next subsection for every knob) and log it for traceability.
2. Rebuild the processed `.conll` files and `label2id.json` when the annotated text or spreadsheet changes so the training data always reflects the latest inputs.
3. Read `data/processed/train.conll` (and `data/processed/dev.conll` if present), ensuring every label exists in the spreadsheet metadata.
4. Tokenise the examples with the XLM-R tokenizer while aligning the token-level labels, optionally propagating labels to subwords.
5. Train the model with Hugging Face `Trainer`, evaluate on the dev split if available, and write all checkpoints to `model/`.
6. Export the tokenizer, spreadsheet metadata, label catalogue, and `slot_value_map.json` (a surface→取值 lookup generated from the annotated training text) so inference can resolve limited slots.

```bash
python src/annotation_with_roberta/training.py
```

#### Training parameters you may want to change later

The `TrainingConfig` dataclass defines all tunable parameters. Key ones for this project include:

| Parameter | Default | Why it matters |
|-----------|---------|----------------|
| `model_name` | `xlm-roberta-base` | Switch to a larger variant (e.g. `xlm-roberta-large`) for higher accuracy at the cost of GPU memory and slower inference. |
| `train_file` / `eval_file` | `data/processed/train.conll` / `dev.conll` | Point these at new dataset splits when you expand or replace the corpus. Missing eval data disables evaluation and best-model selection. |
| `train_text_file` / `eval_text_file` | `data/train.txt` / `dev.txt` | Override when the bracket-annotated source files live elsewhere. These drive the automatic `.conll`/`label2id.json` regeneration performed before training. |
| `segments_file` | `data/segments.xlsx` | Update only if the catalogue moves. Out-of-sync spreadsheets will cause validation failures. |
| `label_map_file` | `data/processed/label2id.json` | Must match the dataset labelling scheme. Rebuilding it incorrectly will misalign labels and degrade performance. |
| `output_dir` | `model/` | Change to version your experiments or keep multiple checkpoints. Ensure inference points at the matching directory. |
| `max_length` | `256` tokens | Increasing captures longer sentences but uses more memory and may require reducing batch size. Decreasing can truncate long utterances, losing context. |
| `learning_rate` | `3e-5` | Higher values speed up convergence but risk instability; lower values are safer but need more epochs. |
| `weight_decay` | `0.01` | Regularises the model; set to `0` if you observe underfitting. |
| `warmup_steps` | `500` | Helps stabilise early training. Too few warmup steps can cause spikes in loss; too many delays learning. |
| `num_train_epochs` | `8` | More epochs can improve accuracy when data is scarce, but overfitting appears as worsening dev metrics. |
| `per_device_train_batch_size` | `16` | Raise for faster training if memory allows; lower when you encounter out-of-memory errors. |
| `gradient_accumulation_steps` | `2` | Compensates for small batches. Increasing mimics larger effective batch size, but training slows down. |
| `label_all_tokens` | `True` | Keeps labels on subword pieces. Turning it off removes subword supervision and can hurt languages with many compound words. |
| `seed` | `42` | Controls reproducibility. Changing it may yield slightly different metrics due to random initialisation. |

Whenever you adjust a parameter, keep an eye on the evaluation metrics logged by the trainer. Sudden drops in F1 usually indicate either label/metadata mismatches or overly aggressive learning rates.

### 3. Annotate sentences

`src/annotation_with_roberta/inference.py` exposes the [`AutoAnnotator` class](src/annotation_with_roberta/inference.py) and provides an entry point that reads `data/sentences.txt`. Each line should contain one sentence to annotate, with blank lines ignored.

```bash
python src/annotation_with_roberta/inference.py
```

The script prints bracketed annotations such as:

```
[open:Åben] [page:siden] [prep:for] [userManual:bruger manual] [now:nu]
```

If the model predicts a label that is not listed in `segments.xlsx`, or if it returns `O`, the token is emitted with the fallback label `other` so that no span is left unlabeled. Limited slots that lack a known value trigger a warning so you can update the slot value map or underlying data.

#### Inference parameters worth tuning

The `AutoAnnotator` constructor provides these extension points:

| Parameter | Default | Impact |
|-----------|---------|--------|
| `model_dir` | `model/` | Point this at a different checkpoint when you iterate on training. Incorrect directories raise loading errors. |
| `segments_file` | `data/segments.xlsx` | Must match the labels the model was trained on. Swapping spreadsheets without retraining introduces unknown-label fallbacks. |
| `slot_value_map_path` | `model/slot_value_map.json` | Override when experimenting with custom value mappings. Missing or stale maps will produce warnings and omit the `##取值` suffix. |
| `device` | auto-detected (`cuda` if available) | Force `"cpu"` when GPU memory is scarce, trading latency for compatibility. |
| `max_length` | `256` | Should mirror the training configuration. Lowering it can truncate long sentences before annotation; raising it increases memory use. |

Because inference mirrors the training tokenisation, any mismatch between tokenizer settings (e.g. `max_length`) or label catalogues will manifest as truncated outputs or increased use of the fallback label.

## Project Structure

```
.
├── data/
│   ├── segments.xlsx               # Authoritative label catalogue with limited-slot values
│   ├── sentences.txt               # Sentences queued for annotation
│   ├── train.txt / dev.txt         # Human-annotated examples that drive the slot value map
│   └── processed/
│       ├── train.conll, dev.conll  # Auto-generated token/label sequences derived from train.txt / dev.txt
│       └── label2id.json           # Auto-generated label→index map aligned with the dataset
├── model/
│   ├── config.json, pytorch_model.bin, tokenizer/  # Standard Hugging Face checkpoint assets
│   ├── label_catalogue.json        # Saved by the trainer for quick lookup at inference time
│   ├── label_metadata.json         # Spreadsheet metadata snapshot bundled with the model
│   └── slot_value_map.json         # Surface→取值 mappings resolved from the training text
└── src/
    ├── prepare_segments.py         # Spreadsheet CLI that powers the JSON artefacts under data/
    └── annotation_with_roberta/
        ├── __init__.py             # Adds src/ to sys.path and re-exports package utilities
        ├── data.py                 # XML spreadsheet reader, dataset validators, and value-map builders
        ├── inference.py            # AutoAnnotator implementation plus CLI for batch labelling
        └── training.py             # Training config defaults, Hugging Face Trainer loop, and artifact export
```

### When to revisit each file

* **`src/prepare_segments.py`** – Update when the Excel layout changes or when you need to regenerate JSON artefacts after new labels are added.
* **`src/annotation_with_roberta/data.py`** – Extend when the spreadsheet gains new columns or when you want additional validation (e.g. enforcing casing). The functions here underpin both training and inference, so bugs propagate widely.
* **`src/annotation_with_roberta/training.py`** – Modify to adjust hyperparameters, introduce callbacks, or wire in new evaluation strategies. Monitor dev metrics to understand the effect of each change.
* **`src/annotation_with_roberta/inference.py`** – Tweak when you need custom post-processing, alternative output formats, or batched inference for throughput.

This setup lets you fine-tune once and reuse the saved artifacts to annotate utterances in any of the 26 supported languages. When experimentation starts, adjust the parameters highlighted above and observe how the outputs shift—for example, higher learning rates typically speed up training but can cause noisy annotations, while richer slot value maps reduce fallback warnings by covering more surface forms.
