"""CSV/JSONL input adapters with arbitrary string labels and strict validation."""
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


def read_records(filename, config, labeled=True):
    path = Path(filename)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    else:
        raise ValueError('Text input must be a CSV or JSONL file.')
    records = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Row {index + 1} must be an object.')
        text = row.get(config['text_column'])
        label = row.get(config['label_column'])
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Missing/empty text in row {index + 1}.')
        if labeled and (label is None or isinstance(label, (list, dict, bool)) or not str(label).strip()):
            raise ValueError(f'Missing/invalid single label in row {index + 1}.')
        sample_id = str(row.get(config['id_column'], index))
        if sample_id in seen:
            raise ValueError(f'Duplicate sample ID in row {index + 1}.')
        seen.add(sample_id)
        records.append({'text': text.strip(), 'label': str(label) if labeled else None, 'id': sample_id})
    if not records:
        raise ValueError(f'Input is empty: {path}')
    return records


def resolve_labels(training_records, configured=None):
    observed = {row['label'] for row in training_records}
    labels = list(configured) if configured is not None else sorted(observed)
    if len(labels) < 2 or len(set(labels)) != len(labels) or set(labels) != observed:
        raise ValueError('Training records must cover every configured label and at least two classes.')
    return labels


class TextDataset(Dataset):
    """Also accepts in-memory records for application-specific adapters."""
    def __init__(self, records, tokenizer, encoder_config, labels=None):
        self.records = records
        self.tokenizer = tokenizer
        self.config = encoder_config
        self.label_to_index = None if labels is None else {label: i for i, label in enumerate(labels)}
        if self.label_to_index is not None:
            unknown = {row['label'] for row in records} - set(self.label_to_index)
            if unknown:
                raise ValueError(f'Unknown class labels: {sorted(unknown)}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        tokens = self.tokenizer(row['text'], max_length=self.config['max_length'],
                                padding='max_length', truncation=True, return_tensors='pt')
        batch = {key: tokens[key].squeeze(0) for key in ['input_ids', 'attention_mask']}
        batch['id'] = row['id']
        if self.label_to_index is not None:
            batch['label'] = torch.tensor(self.label_to_index[row['label']], dtype=torch.long)
        if self.config['global_attention']:
            mask = torch.zeros_like(batch['input_ids'])
            valid = torch.nonzero(batch['attention_mask'], as_tuple=True)[0]
            if len(valid):
                mask[valid[0]] = 1
            batch['global_attention_mask'] = mask
        return batch


def make_loader(records, tokenizer, config, labels, shuffle=False):
    return DataLoader(TextDataset(records, tokenizer, config['encoder'], labels),
                      batch_size=config['training']['batch_size'], shuffle=shuffle,
                      num_workers=config['training']['num_workers'])


def model_inputs(batch, device=None):
    inputs = {key: batch[key] for key in ['input_ids', 'attention_mask', 'global_attention_mask'] if key in batch}
    return inputs if device is None else {key: value.to(device) for key, value in inputs.items()}
