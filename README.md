# SGMoE

**Semantics-Guided Mixture of Experts for Multi-class Disease Classification Based on Radiology Reports**

This repository implements the **complete SGMoE model** for user-provided text classification data. Its semantic encoder, semantic-center prior, learned gate, class-group experts, dynamic expert loss and assignment-based gating loss are always present. There are no named-task presets, baseline selectors or ablation modes.

Labels and expert count are not fixed in source code. Labels are read from your training records, and semantic preparation generates expert groups for those labels. The main model uses a LoRA-adapted Hugging Face text encoder; the semantic encoder is independently trained and then frozen. Feature dimensions are inferred from the actual encoders.

## Install

Use Python 3.10 or newer. Install the PyTorch build appropriate for your CPU/CUDA environment, then run from the repository root:

```bash
python -m pip install -r requirements.txt
# Optional editable installation, including command-line entry points:
python -m pip install -e .
```

The equivalent installed commands are `sgmoe-prepare`, `sgmoe-train` and `sgmoe-predict`. Dependencies are not a locked historical experiment environment.

## Provide your data

Use UTF-8 CSV or JSONL files. A labeled CSV has a text column and a single class label:

```csv
id,text,label
example-1,"Example text assigned to the first class.",Class_A
example-2,"Another text assigned to the second class.",Class_B
```

These rows illustrate the format; no dataset is bundled. Column names are configurable. Class labels can be arbitrary strings or numeric labels, which are read as strings. There are no filename-label conventions. Unknown validation/test labels, empty text, duplicate IDs within a split and empty inputs cause errors instead of being silently skipped.

Set `data.labels` to `null` to infer the ordered label list from training data, or provide your own ordered list. The training split must cover every configured class. Preparation and training support single-label classification, with at least two classes. See [input format](docs/DATA.md).

## Configure paths and encoder

Copy [configs/example.json](configs/example.json) to `configs/local.json` and edit:

- `data`: training/validation files, optional test file, and column names.
- `encoder`: local encoder directory or Hugging Face repository ID, token length and LoRA module names.
- `semantic`: artifact directory, semantic-pretraining epochs and clustering search range.
- `training`: output directory and optimization settings.

Relative filesystem paths are resolved against the configuration file. Explicit model paths should begin with `./` or `../`; an identifier such as `organization/model-name` is passed to Hugging Face. Models must expose `last_hidden_state` and `hidden_size` or `d_model`. Adjust `lora_target_modules` to your encoder's module names. Set `global_attention=true` for an encoder that requires a global attention mask.

## Prepare, train, predict

```bash
# Train the semantic encoder and generate centers/expert groups:
python prepare.py --config configs/local.json --device cuda:0

# Validate data and generated semantic files without loading model weights:
python train.py --config configs/local.json --dry-run

# Train the full SGMoE:
python train.py --config configs/local.json --device cuda:0

# Predict on an unlabeled CSV/JSONL containing the text column:
python predict.py --checkpoint outputs/best.pt \
  --input path/to/unlabeled.csv --output path/to/predictions.csv --device cuda:0
```

Use `--device cpu` explicitly for small CPU runs. `prepare.py --dry-run` validates the preparation inputs before training. `test_file` may be `null`; inference does not require labels in its input. When moving a trained model, pass `predict.py --config <new-local-config>` to relocate the encoder and the same semantic artifacts; incompatible groups are rejected.

Preparation uses training reports to build centers and expert groups; validation selects the semantic encoder. Test reports are never consulted during preparation. Reconstructed artifacts are linked by SHA256 hashes. Missing or changed semantic artifacts fail explicitly; the model never silently switches off its semantic prior.

The semantic prior is derived from distances to frozen-encoder centers. The gate combines it with learned weights using a fixed arithmetic mean and normalization. Expert assignments can overlap, and each expert's head covers only its assigned classes. Training combines classification cross-entropy, assignment-based gating loss and the quadratically decaying expert-group loss. See [architecture](docs/MODEL.md).

## Outputs and continuation

Preparation creates `semantic_encoder.pt`, `centers.npy` and `groups.json` in your artifact directory. Main training writes `best.pt`, `last.pt`, `manifest.json`, `history.json`, `metrics.json`, optional test predictions, and a verified `completed.json` marker. Validation macro F1 selects the best epoch; both macro and weighted F1 are reported.

```bash
python prepare.py --config configs/local.json --device cuda:0 --resume
python train.py --config configs/local.json --device cuda:0 --resume
```

Completed matching runs are verified and skipped. Interrupted runs restore model, optimizer, scheduler, early-stopping and RNG states from the last checkpoint. Existing mismatched, missing or corrupt checkpoint state is preserved and causes an error. Changes in input files, configuration or source require a new output/artifact directory. An interrupted epoch restarts from its beginning. Bitwise equivalence across different hardware/software is not guaranteed.

## Python API

```python
from sgmoe.config import load_config
from sgmoe.model import SGMoE

config = load_config("configs/local.json")
labels = ["Class_A", "Class_B"]  # same ordered labels as groups.json
model = SGMoE(config, labels)
# outputs = model(input_ids=..., attention_mask=...)
```

`TextDataset` also accepts in-memory records for custom data adapters. No model or data registry needs to be edited to introduce another classification task.

## Tests and release scope

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests use synthetic encoders and inputs, covering complete-model mathematics, gradients, arbitrary labels, preparation, training, prediction and protected continuation. The code does not distribute data or pretrained/trained weights. This general implementation preserves the full model's components but does not claim automatic reproduction of historical manuscript numbers. Real-data GPU training and online dependency installation need validation in your environment.

## License

Code is licensed under [MIT](LICENSE). Data and pretrained-model permissions remain with their respective providers. Formal manuscript citation metadata will be added when available.
