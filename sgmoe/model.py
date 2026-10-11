"""The complete SGMoE: frozen semantic encoder, prior, learned gate and experts."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sgmoe.encoders import load_encoder, feature_width, pooled_features
from sgmoe.io import file_hash


def load_artifacts(directory, labels):
    root = Path(directory)
    metadata = json.loads((root / 'groups.json').read_text(encoding='utf-8'))
    if metadata['labels'] != labels:
        raise ValueError('Semantic artifacts use a different ordered label list.')
    for name in ['semantic_encoder.pt', 'centers.npy']:
        if file_hash(root / name) != metadata['sha256'][name]:
            raise ValueError(f'Semantic artifact changed or is inconsistent: {name}')
    centers = np.load(root / 'centers.npy', allow_pickle=False)
    categories = metadata['expert_classes']
    if not categories or any(not values or len(values) != len(set(values)) for values in categories):
        raise ValueError('Each expert needs nonempty distinct class assignments.')
    if {value for values in categories for value in values} != set(labels):
        raise ValueError('Expert groups must cover exactly the configured classes.')
    if centers.ndim != 2 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError('Semantic centers must be a finite, nonempty matrix.')
    mapping = {int(k): int(v) for k, v in metadata['center_to_expert'].items()}
    if not mapping or any(k < 0 or k >= len(centers) or v < 0 or v >= len(categories) for k, v in mapping.items()):
        raise ValueError('Invalid center-to-expert mapping.')
    metadata['center_to_expert'] = mapping
    return metadata, torch.tensor(centers, dtype=torch.float32)


class SGMoE(nn.Module):
    def __init__(self, config, labels, artifact_bundle=None):
        super().__init__()
        metadata, centers = artifact_bundle or load_artifacts(config['semantic']['artifact_dir'], labels)
        self.labels = labels
        lookup = {label: index for index, label in enumerate(labels)}
        self.expert_classes = [list(map(lookup.__getitem__, values)) for values in metadata['expert_classes']]
        self.num_experts = len(self.expert_classes)
        self.shared_encoder = load_encoder(config['encoder'], lora=True)
        self.semantic_encoder = load_encoder(config['encoder'], lora=False)
        checkpoint = torch.load(Path(config['semantic']['artifact_dir'])/'semantic_encoder.pt',
                                map_location='cpu', weights_only=False)
        self.semantic_encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=True)
        self.semantic_encoder.requires_grad_(False).eval()
        shared_width = feature_width(self.shared_encoder)
        semantic_width = feature_width(self.semantic_encoder)
        if centers.shape[1] != semantic_width:
            raise ValueError('Semantic center width does not match the frozen encoder.')
        self.register_buffer('centers', centers)
        mapping = torch.full((len(centers),), -1, dtype=torch.long)
        for center, expert in metadata['center_to_expert'].items():
            mapping[int(center)] = int(expert)
        self.register_buffer('center_to_expert', mapping)
        dropout = config['training']['dropout']
        self.dropout = nn.Dropout(dropout)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(shared_width, shared_width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(shared_width, len(values))) for values in self.expert_classes])
        gate_width = shared_width + semantic_width
        self.gate_norm = nn.LayerNorm(gate_width)
        self.gate = nn.Sequential(nn.Linear(gate_width, gate_width), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(gate_width, shared_width), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(shared_width, self.num_experts))

    def train(self, mode=True):
        super().train(mode)
        self.semantic_encoder.eval()
        return self

    def forward(self, **inputs):
        shared = self.dropout(pooled_features(self.shared_encoder, inputs))
        with torch.no_grad():
            semantic = pooled_features(self.semantic_encoder, inputs)
            center_weights = torch.softmax(-torch.cdist(semantic, self.centers), dim=-1)
        gate_logits = self.gate(self.dropout(self.gate_norm(torch.cat([shared, semantic], dim=-1))))
        learned = torch.softmax(gate_logits, dim=-1)
        prior = torch.zeros_like(learned)
        valid = self.center_to_expert >= 0
        prior.scatter_add_(1, self.center_to_expert[valid].unsqueeze(0).expand(len(shared), -1),
                           center_weights[:, valid])
        weights = (prior + learned) / 2
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        experts = []
        for head, classes in zip(self.experts, self.expert_classes):
            local = head(shared)
            full = torch.zeros(len(shared), len(self.labels), device=shared.device, dtype=local.dtype)
            full[:, classes] = local
            experts.append(full)
        expert_logits = torch.stack(experts, dim=1)
        return {'logits': (weights.unsqueeze(-1)*expert_logits).sum(dim=1),
                'gate_logits': gate_logits, 'gate_weights': weights,
                'learned_weights': learned, 'prior_weights': prior, 'expert_logits': expert_logits}
