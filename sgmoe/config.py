"""One configuration for arbitrary labeled text and the complete SGMoE model."""
from copy import deepcopy
import json
import math
from pathlib import Path

DEFAULTS = {
    'data': {'train_file': None, 'validation_file': None, 'test_file': None,
             'text_column': 'text', 'label_column': 'label', 'id_column': 'id', 'labels': None},
    'encoder': {'model_name_or_path': None, 'max_length': 400, 'global_attention': False,
                'lora_target_modules': ['query', 'value'], 'lora_rank': 16,
                'lora_alpha': 32, 'lora_dropout': 0.1},
    'semantic': {'artifact_dir': 'artifacts', 'epochs': 60, 'min_clusters': 2, 'max_clusters': 20},
    'training': {'output_dir': 'outputs', 'epochs': 100, 'batch_size': 32,
                 'learning_rate': 2e-5, 'weight_decay': 0.01, 'warmup_steps': 100,
                 'dropout': 0.2, 'gating_loss_weight': 0.5, 'max_grad_norm': 1.0,
                 'early_stopping_patience': 30, 'num_workers': 0, 'seed': 42},
}


def load_config(filename):
    source = Path(filename).expanduser().resolve()
    payload = json.loads(source.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict) or set(payload) - set(DEFAULTS):
        raise ValueError('Configuration sections must be data, encoder, semantic, and training.')
    config = deepcopy(DEFAULTS)
    for section, values in payload.items():
        if not isinstance(values, dict) or set(values) - set(config[section]):
            raise ValueError(f'Unknown configuration settings in {section}.')
        config[section].update(values)
    required = [('data', 'train_file'), ('data', 'validation_file'), ('encoder', 'model_name_or_path')]
    for section, field in required:
        if not isinstance(config[section][field], str) or not config[section][field].strip():
            raise ValueError(f'{section}.{field} is required.')
    for section, fields in [('data', ['train_file', 'validation_file', 'test_file']),
                            ('semantic', ['artifact_dir']), ('training', ['output_dir'])]:
        for field in fields:
            value = config[section][field]
            if value is not None:
                path = Path(value).expanduser()
                config[section][field] = str((source.parent / path).resolve())
    model = config['encoder']['model_name_or_path']
    # An explicit relative path is local; a repository identifier remains an ID.
    if model.startswith(('.', '~', '/', '\\')) or Path(model).is_absolute():
        config['encoder']['model_name_or_path'] = str((source.parent / Path(model).expanduser()).resolve())
    validate_config(config)
    return config


def validate_config(config):
    for section, fields in [('encoder', ['max_length', 'lora_rank', 'lora_alpha']),
                            ('semantic', ['epochs', 'min_clusters', 'max_clusters']),
                            ('training', ['epochs', 'batch_size', 'early_stopping_patience'])]:
        for field in fields:
            value = config[section][field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{section}.{field} must be a positive integer.')
    for field in ['num_workers', 'warmup_steps']:
        value = config['training'][field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'training.{field} must be a nonnegative integer.')
    seed = config['training']['seed']
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('training.seed must be an integer in [0, 2**32).')
    for field in ['learning_rate', 'gating_loss_weight', 'max_grad_norm']:
        value = config['training'][field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'training.{field} must be finite and positive.')
    decay = config['training']['weight_decay']
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:
        raise ValueError('training.weight_decay must be finite and nonnegative.')
    for section, field in [('encoder', 'lora_dropout'), ('training', 'dropout')]:
        value = config[section][field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ValueError(f'{section}.{field} must be in [0, 1).')
    if config['semantic']['min_clusters'] < 2 or config['semantic']['max_clusters'] < config['semantic']['min_clusters']:
        raise ValueError('Require 2 <= min_clusters <= max_clusters.')
    if not isinstance(config['encoder']['global_attention'], bool):
        raise ValueError('encoder.global_attention must be boolean.')
    targets = config['encoder']['lora_target_modules']
    if not isinstance(targets, list) or not targets or any(not isinstance(t, str) or not t for t in targets):
        raise ValueError('lora_target_modules must be a nonempty list of module names.')
    labels = config['data']['labels']
    if labels is not None and (not isinstance(labels, list) or len(labels) < 2 or
                               any(not isinstance(v, str) or not v for v in labels) or len(set(labels)) != len(labels)):
        raise ValueError('data.labels must be null or a distinct list of at least two strings.')
    for field in ['text_column', 'label_column', 'id_column']:
        if not isinstance(config['data'][field], str) or not config['data'][field]:
            raise ValueError(f'data.{field} must be a nonempty string.')
