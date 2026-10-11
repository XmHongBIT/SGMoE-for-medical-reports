# Public release scope

Prepared on 2026-10-11. This source release contains only the complete SGMoE implementation, generic input/configuration adapters, semantic preparation, training, prediction and relevant tests. It contains no task-specific presets, comparison models, ablation entry points, report files, checkpoints or original experiment results.

The maintained full-model equations are retained: independently trained frozen semantic features, Euclidean-distance center prior, learned concatenated-feature gate, fixed arithmetic-mean fusion with normalization, overlapping class-group experts, classification cross-entropy, assignment-based gating BCE and quadratic expert-group loss decay. Feature widths and class/expert counts are inferred rather than fixed.

Genericization adds CSV/JSONL input, arbitrary ordered labels, strict artifact compatibility checks, a semantic encoder kept in evaluation mode during main training, and atomic checkpoints with epoch-boundary RNG restoration. These changes do not constitute a claim of numerical identity with every earlier script or historical result.

## Validation

Local CPU tests use synthetic encoders/tokenizers and generated temporary records. They verify:

- Arbitrary and nonconsecutive labels, custom CSV/JSONL columns, Unicode text, explicit label order, unknown-label rejection and empty/malformed-input errors.
- Complete-model gating fusion, class routing, gradients, frozen semantic evaluation mode, multiple feature widths, artifact hash checks and the original expert-group loss equation.
- Semantic preparation reading training/validation records while ignoring the test path.
- End-to-end preparation, full-model training and unlabeled prediction.
- Verified completion skipping, protected missing/corrupt state, and input/configuration identity checks.
- A simulated first-epoch interruption whose resumed CPU model matches an uninterrupted synthetic run.
- Seeded clustering and class-group construction.

The focused suite passed **33 tests**. Its environment used Python 3.12 and CPU execution; pretrained encoder and LoRA loading were substituted, so actual Hugging Face/PEFT loading and real-data GPU training remain unverified locally. Cross-hardware bitwise reproducibility is not promised. Dependency declarations are not a locked historical training environment. GitHub Actions is configured but has not run remotely.

The source archive is built from an explicit Git file allowlist. It excludes training/preparation outputs, data, weights, local configuration, caches and prior research drafts. The active experiment checkout is kept separately from this public release.
