"""Train semantic features, cluster training records, and build expert groups."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from sgmoe.clustering import fit_kmeans_with_silhouette, build_expert_assignment
from sgmoe.data import make_loader, model_inputs
from sgmoe.encoders import load_encoder, load_tokenizer, pooled_features, feature_width
from sgmoe.io import (
    save_json, atomic_checkpoint, file_hash, digest, set_seed,
    resolve_device, rng_state, restore_rng,
)
from sgmoe.training import load_training_inputs, lr_scheduler, source_hashes, classification_metrics


class SemanticClassifier(nn.Module):
    def __init__(self, config, classes):
        super().__init__()
        self.encoder = load_encoder(config['encoder'], lora=False)
        self.classifier = nn.Linear(feature_width(self.encoder), classes)

    def forward(self, **inputs):
        return self.classifier(pooled_features(self.encoder, inputs))


def prepare(config, device_name='cuda:0', resume=False, dry_run=False):
    # Test reports must never be consulted when constructing semantic groups.
    preparation_config = {**config, 'data': {**config['data'], 'test_file': None}}
    records, labels = load_training_inputs(preparation_config)
    semantic, settings = config['semantic'], config['training']
    root = Path(semantic['artifact_dir'])
    identity = {'data_sha256': {key: file_hash(config['data'][key]) for key in ['train_file','validation_file']},
                'labels': labels, 'encoder': config['encoder'], 'semantic': semantic,
                'training': {k:v for k,v in settings.items() if k not in ['output_dir','epochs','gating_loss_weight']},
                'source_sha256': source_hashes()}
    signature = digest(identity)
    if dry_run:
        print(json.dumps({'labels': labels, 'training_samples': len(records['train']),
                          'cluster_range': [semantic['min_clusters'], semantic['max_clusters']],
                          'artifact_dir': str(root)}, ensure_ascii=False, indent=2))
        return
    manifest = root/'preparation_manifest.json'
    if root.exists() and any(root.iterdir()):
        if not manifest.is_file() or json.loads(manifest.read_text(encoding='utf-8'))['signature'] != signature:
            raise ValueError('Artifact directory contains a different or unverified preparation run.')
        if (root/'groups.json').is_file():
            from sgmoe.model import load_artifacts
            load_artifacts(root, labels)
            print(f'Already prepared: {root}')
            return
        if not resume:
            raise FileExistsError('Incomplete semantic preparation exists; use --resume.')
        if not (root/'semantic_last.pt').is_file():
            raise FileNotFoundError('Semantic resume checkpoint is missing; preserving files.')
    device = resolve_device(device_name)
    set_seed(settings['seed'])
    tokenizer = load_tokenizer(config['encoder'])
    loaders = {split: make_loader(rows, tokenizer, config, labels, split=='train') for split,rows in records.items()}
    model = SemanticClassifier(config, len(labels)).to(device)
    optimizer = AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = lr_scheduler(optimizer, len(loaders['train'])*semantic['epochs'], settings['warmup_steps'])
    start, best, stale, done = 0, -1., 0, False
    if resume and (root/'semantic_last.pt').is_file():
        try:
            state = torch.load(root/'semantic_last.pt', map_location=device, weights_only=False)
            if state['signature'] != signature:
                raise ValueError('Semantic resume identity mismatch.')
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            start, best, stale, done = state['epoch']+1, state['best_score'], state['stale_epochs'], state['done']
            restore_rng(state['rng'])
        except Exception as exc:
            raise RuntimeError('Cannot restore semantic checkpoint; preserving files.') from exc
    root.mkdir(parents=True, exist_ok=True)
    save_json({'signature': signature, **identity}, manifest)
    if not (root/'semantic_last.pt').exists():
        atomic_checkpoint({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                           'scheduler_state_dict': scheduler.state_dict(), 'epoch': -1,
                           'best_score': best, 'stale_epochs': stale, 'done': False,
                           'rng': rng_state(), 'signature': signature}, root/'semantic_last.pt')
    for epoch in range(start, semantic['epochs']) if not done else []:
        model.train()
        for batch in tqdm(loaders['train'], desc=f'Semantic epoch {epoch+1}/{semantic["epochs"]}'):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(**model_inputs(batch, device)), batch['label'].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings['max_grad_norm'])
            optimizer.step()
            scheduler.step()
        model.eval()
        actual, predictions = [], []
        with torch.no_grad():
            for batch in loaders['validation']:
                predictions.extend(model(**model_inputs(batch, device)).argmax(-1).cpu().tolist())
                actual.extend(batch['label'].tolist())
        validation = classification_metrics(actual, predictions, labels)
        improved = validation['macro_f1'] > best
        best = max(best, validation['macro_f1'])
        stale = 0 if improved else stale+1
        if improved:
            atomic_checkpoint({'encoder_state_dict': model.encoder.state_dict(), 'labels': labels,
                               'encoder': config['encoder'], 'validation': validation, 'epoch': epoch,
                               'signature': signature}, root/'semantic_encoder.pt')
        done = stale >= settings['early_stopping_patience'] or epoch+1 == semantic['epochs']
        atomic_checkpoint({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                           'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch,
                           'best_score': best, 'stale_epochs': stale, 'done': done,
                           'rng': rng_state(), 'signature': signature}, root/'semantic_last.pt')
        if done:
            break
    checkpoint = torch.load(root/'semantic_encoder.pt', map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.encoder.eval()
    training_loader = make_loader(records['train'], tokenizer, config, labels, False)
    vectors, class_names = [], []
    with torch.no_grad():
        for batch in training_loader:
            vectors.append(pooled_features(model.encoder, model_inputs(batch, device)).cpu().numpy())
            class_names.extend(labels[int(value)] for value in batch['label'])
    vectors = np.concatenate(vectors)
    count, kmeans, scores = fit_kmeans_with_silhouette(vectors, semantic['min_clusters'], semantic['max_clusters'], settings['seed'])
    assignment = build_expert_assignment(class_names, kmeans.labels_, count)
    temporary = root/'centers.npy.tmp'
    with temporary.open('wb') as handle:
        np.save(handle, kmeans.cluster_centers_.astype(np.float32), allow_pickle=False)
    temporary.replace(root/'centers.npy')
    metadata = {'labels': labels, 'expert_classes': list(assignment.expert_categories.values()),
                'center_to_expert': assignment.center_to_expert_mapping,
                'feature_width': int(vectors.shape[1]), 'num_clusters': count,
                'silhouette_scores': scores, 'training_samples': len(class_names),
                'preparation_signature': signature,
                'sha256': {name: file_hash(root/name) for name in ['semantic_encoder.pt','centers.npy']}}
    save_json(metadata, root/'groups.json')
    print(f'Semantic artifacts prepared: {root}; experts={len(metadata["expert_classes"])}')
