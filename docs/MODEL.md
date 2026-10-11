# Full SGMoE components

The model always has two encoder branches. The main encoder is LoRA-adapted; the semantic encoder is pretrained on labeled training text, selected with validation macro F1, frozen, and kept in evaluation mode during main training. Their feature widths are inferred from their configurations.

## Semantic groups

Preparation fits seeded KMeans (`n_init=10`) to training semantic features. The requested cluster range is clipped by the number of training samples; silhouette score selects the count, breaking ties toward the smaller count. Full silhouette computation can be costly on large inputs.

Class i is retained in cluster j when its fraction of that class's training examples reaches `1 / number_of_clusters`. If denoising removes a class everywhere, it is restored to its most populated cluster. Multi-class clusters receive individual experts; single-class clusters are merged into one expert. Expert groups may overlap. All training classes must remain covered.

`groups.json` records the ordered labels, expert classes, center-to-expert mapping and SHA256 hashes of `semantic_encoder.pt` and `centers.npy`. The model loads the semantic encoder strictly and checks the center width. The mapping can omit a center discarded during denoising; its prior mass is not assigned to an expert, and the final fused weights are normalized.

## Gating and class routing

For a sample, center weights are `softmax(-EuclideanDistance(semantic_feature, centers))`. The expert prior sums weights over centers mapped to that expert. A learned gate receives concatenated main/semantic features, with layer normalization and MLP layers. The two weight vectors are averaged and normalized before mixing expert logits.

Each expert uses an MLP with a local classification head sized for its assigned classes. Local logits are placed into the global class space; other entries are zero, following the maintained implementation. The final logits are the gate-weighted sum of expert logits. Class coverage and the ordered label vocabulary are checked explicitly.

## Training objective

```text
L = CE(final_logits, label)
    + gating_loss_weight * BCE(gate_logits, assigned_expert_targets)
    + (1 - epoch / max(total_epochs - 1, 1))^2 * expert_group_CE
```

Targets mark every expert covering a sample's true class. The expert-group term combines logits only from those relevant experts and normalizes their gate weights. The dynamic term starts at one and reaches zero at the last epoch for multi-epoch training. Classification and gating losses remain active.

Main-model selection uses validation macro F1 over the complete configured label vocabulary. Accuracy, macro F1 and weighted F1 are reported. Test records are evaluated only after loading the best validation checkpoint.

This public implementation generalizes input handling and encoder dimensions while retaining the maintained full-model equations. It does not establish numerical equivalence to all earlier training scripts or manuscript results; evaluation requires your own data, artifacts and environment.
