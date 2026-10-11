# Generic input format

No datasets or named-task registries are included. The input API is the same for any single-label text classification task.

CSV files have configurable `text`, `label` and optional `id` columns. JSONL files contain one object per line with the same fields. Numeric labels are converted to strings; labels can also be arbitrary names. Specify an ordered list with `data.labels`, or infer it lexicographically from training data by leaving that field `null`.

```json
{"id": "example-1", "text": "An example text.", "label": "Class_A"}
```

Unlabeled prediction input needs only `text`; `id` is optional and generated from row position when absent. IDs must be unique within each file. Text must be nonempty UTF-8 text. Each labeled record has exactly one label. Unknown validation/test classes and empty inputs are rejected. A configuration's training records must cover every configured class.

The semantic encoder uses training and validation data; semantic centers use only training features. The test split is optional and is consulted only for final evaluation of the best main-model checkpoint. The code does not generate train/validation/test splits; create and record them beforehand, keeping records from the same patient/document group together where applicable.

The silhouette-based clustering stage needs at least three training samples and valid distinct feature clusters. Configure the cluster search range for your task. A semantic cluster and an expert are not the same: the grouping rule can merge single-class clusters, so the resulting expert count is inferred rather than directly supplied.

All filesystem paths in configuration are relative to the configuration file unless absolute. CSV and JSONL sources, weights, generated local configurations and prediction outputs are excluded by `.gitignore` and the release archive allowlist.
