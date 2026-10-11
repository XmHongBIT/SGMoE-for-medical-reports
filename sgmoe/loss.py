"""Classification, class-assignment gating, and dynamic expert-group losses."""
import torch
import torch.nn.functional as F


def training_loss(outputs, labels, expert_classes, epoch, total_epochs, gating_weight):
    classification = F.cross_entropy(outputs['logits'], labels)
    targets = torch.zeros_like(outputs['gate_logits'])
    assignments = torch.zeros(len(expert_classes), outputs['logits'].shape[-1],
                              device=labels.device, dtype=torch.bool)
    for expert, classes in enumerate(expert_classes):
        assignments[expert, classes] = True
    relevant = assignments[:, labels].T
    if not relevant.any(dim=1).all():
        raise ValueError('A training class has no assigned expert.')
    targets[relevant] = 1
    gate_loss = F.binary_cross_entropy_with_logits(outputs['gate_logits'], targets)
    masked_weights = outputs['gate_weights'] * relevant
    combined = (masked_weights.unsqueeze(-1)*outputs['expert_logits']).sum(dim=1)
    combined = combined / masked_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    expert_loss = F.cross_entropy(combined, labels)
    progress = min(max(epoch / max(total_epochs-1, 1), 0), 1)
    alpha = (1-progress)**2
    total = classification + gating_weight*gate_loss + alpha*expert_loss
    return total, {'classification': float(classification.detach()), 'gating': float(gate_loss.detach()),
                   'expert': float(expert_loss.detach()), 'expert_weight': alpha}
