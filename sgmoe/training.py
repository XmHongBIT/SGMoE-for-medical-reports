"""Train the complete SGMoE with validation selection and protected continuation."""
import csv
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from sgmoe.data import read_records, resolve_labels, make_loader, model_inputs
from sgmoe.encoders import load_tokenizer
from sgmoe.io import (
    save_json, atomic_checkpoint, file_hash, digest, set_seed,
    resolve_device, rng_state, restore_rng,
)
from sgmoe.loss import training_loss
from sgmoe.model import SGMoE, load_artifacts


def source_hashes():
    return {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob('*.py'))}


def lr_scheduler(optimizer, total_steps, warmup):
    def factor(step):
        if step < warmup:
            return step / max(warmup, 1)
        return max(0., (total_steps-step)/max(total_steps-warmup, 1))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def classification_metrics(actual, predicted, labels):
    from sklearn.metrics import accuracy_score, f1_score
    indices = list(range(len(labels)))
    return {'accuracy': float(accuracy_score(actual, predicted)),
            'macro_f1': float(f1_score(actual, predicted, labels=indices, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(actual, predicted, labels=indices, average='weighted', zero_division=0))}


def evaluate(model, loader, device, labels):
    model.eval()
    actual, predicted, rows = [], [], []
    loss_sum, samples = 0., 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(**model_inputs(batch, device))
            targets = batch['label'].to(device)
            loss_sum += float(torch.nn.functional.cross_entropy(outputs['logits'], targets, reduction='sum'))
            probabilities = torch.softmax(outputs['logits'], dim=-1).cpu()
            choices = probabilities.argmax(dim=-1)
            samples += len(targets)
            actual.extend(targets.cpu().tolist())
            predicted.extend(choices.tolist())
            for index, sample_id in enumerate(batch['id']):
                rows.append({'id': sample_id, 'label': labels[int(targets[index])],
                             'prediction': labels[int(choices[index])],
                             'confidence': float(probabilities[index, choices[index]]),
                             'probabilities': json.dumps(dict(zip(labels, probabilities[index].tolist())), ensure_ascii=False)})
    metrics = classification_metrics(actual, predicted, labels)
    metrics['loss'] = loss_sum/samples
    return metrics, rows


def save_predictions(rows, filename):
    path = Path(filename)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_training_inputs(config):
    records = {split: read_records(config['data'][field], config['data'])
               for split, field in [('train','train_file'), ('validation','validation_file')]}
    if config['data']['test_file']:
        records['test'] = read_records(config['data']['test_file'], config['data'])
    labels = resolve_labels(records['train'], config['data']['labels'])
    for split, rows in records.items():
        unknown = {row['label'] for row in rows} - set(labels)
        if unknown:
            raise ValueError(f'{split} contains labels absent from the training label set: {sorted(unknown)}')
    return records, labels


def train(config, device_name='cuda:0', resume=False, dry_run=False):
    records, labels = load_training_inputs(config)
    artifact_bundle = load_artifacts(config['semantic']['artifact_dir'], labels)
    identity = {'config': config, 'labels': labels, 'source_sha256': source_hashes(),
                'input_sha256': {field: file_hash(path) for field, path in config['data'].items()
                                 if field in ['train_file','validation_file','test_file'] and path},
                'artifacts': artifact_bundle[0]}
    signature = digest(identity)
    if dry_run:
        print(json.dumps({'labels': labels, 'num_experts': len(artifact_bundle[0]['expert_classes']),
                          'split_sizes': {k: len(v) for k,v in records.items()},
                          'output_dir': config['training']['output_dir']}, ensure_ascii=False, indent=2))
        return
    root = Path(config['training']['output_dir'])
    manifest = root/'manifest.json'
    completed = root/'completed.json'
    if root.exists() and any(root.iterdir()):
        if not manifest.is_file() or json.loads(manifest.read_text(encoding='utf-8'))['signature'] != signature:
            raise ValueError('Output directory contains a different or unverified run; use a new output_dir.')
        if completed.is_file():
            marker = json.loads(completed.read_text(encoding='utf-8'))
            if marker['signature'] != signature or any(file_hash(root/name) != value for name,value in marker['sha256'].items()):
                raise ValueError('Completed outputs are changed or inconsistent; preserving them.')
            print(f'Already complete: {root}')
            return
        if not resume:
            raise FileExistsError('Incomplete run exists; use --resume or a new output_dir.')
        if not (root/'last.pt').is_file():
            raise FileNotFoundError('Resume checkpoint is missing; preserving all existing files.')
    device = resolve_device(device_name)
    set_seed(config['training']['seed'])
    tokenizer = load_tokenizer(config['encoder'])
    loaders = {split: make_loader(rows, tokenizer, config, labels, split=='train') for split,rows in records.items()}
    model = SGMoE(config, labels, artifact_bundle).to(device)
    settings = config['training']
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = lr_scheduler(optimizer, len(loaders['train'])*settings['epochs'], settings['warmup_steps'])
    start, best, stale, history, training_done = 0, -1., 0, [], False
    if resume and (root/'last.pt').is_file():
        try:
            checkpoint = torch.load(root/'last.pt', map_location=device, weights_only=False)
            if checkpoint['signature'] != signature:
                raise ValueError('Resume identity mismatch.')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start, best, stale = checkpoint['epoch']+1, checkpoint['best_score'], checkpoint['stale_epochs']
            history, training_done = checkpoint['history'], checkpoint['training_done']
            restore_rng(checkpoint['rng'])
        except Exception as exc:
            raise RuntimeError('Cannot restore checkpoint; preserving files instead of starting fresh.') from exc
    root.mkdir(parents=True, exist_ok=True)
    save_json({'signature': signature, **identity}, manifest)
    if not (root/'last.pt').exists():
        atomic_checkpoint({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                           'scheduler_state_dict': scheduler.state_dict(), 'epoch': -1,
                           'best_score': best, 'stale_epochs': stale, 'history': history,
                           'training_done': False, 'rng': rng_state(), 'signature': signature}, root/'last.pt')
    for epoch in range(start, settings['epochs']) if not training_done else []:
        model.train()
        loss_total = 0.
        for batch in tqdm(loaders['train'], desc=f'Epoch {epoch+1}/{settings["epochs"]}'):
            targets = batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**model_inputs(batch, device))
            loss, parts = training_loss(outputs, targets, model.expert_classes, epoch,
                                        settings['epochs'], settings['gating_loss_weight'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            loss_total += float(loss.detach())*len(targets)
        validation, _ = evaluate(model, loaders['validation'], device, labels)
        history.append({'epoch': epoch+1, 'training_loss': loss_total/len(records['train']),
                        'validation': validation, 'last_batch_loss_components': parts})
        improved = validation['macro_f1'] > best
        best = max(best, validation['macro_f1'])
        stale = 0 if improved else stale+1
        if improved:
            atomic_checkpoint({'model_state_dict': model.state_dict(), 'labels': labels,
                               'config': config, 'epoch': epoch, 'validation': validation,
                               'signature': signature, 'artifacts': artifact_bundle[0]}, root/'best.pt')
        training_done = stale >= settings['early_stopping_patience'] or epoch+1 == settings['epochs']
        atomic_checkpoint({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                           'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch,
                           'best_score': best, 'stale_epochs': stale, 'history': history,
                           'training_done': training_done, 'rng': rng_state(), 'signature': signature}, root/'last.pt')
        save_json(history, root/'history.json')
        print(f'Epoch {epoch+1}: validation macro F1={validation["macro_f1"]:.4f}')
        if training_done:
            break
    save_json(history, root/'history.json')
    checkpoint = torch.load(root/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    generated = ['best.pt', 'history.json']
    metrics = {'validation_at_best_epoch': checkpoint['validation'], 'best_epoch': checkpoint['epoch']+1}
    if 'test' in loaders:
        metrics['test'], predictions = evaluate(model, loaders['test'], device, labels)
        save_predictions(predictions, root/'predictions.csv')
        generated.append('predictions.csv')
    save_json(metrics, root/'metrics.json')
    generated.append('metrics.json')
    save_json({'signature': signature, 'sha256': {name: file_hash(root/name) for name in generated}}, completed)
    print(f'Finished: {root}')
