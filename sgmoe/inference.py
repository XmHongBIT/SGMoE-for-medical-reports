"""Apply a saved full SGMoE model to arbitrary unlabeled text records."""
from pathlib import Path
import json

import torch

from sgmoe.data import read_records, make_loader, model_inputs
from sgmoe.encoders import load_tokenizer
from sgmoe.io import resolve_device, digest
from sgmoe.model import SGMoE, load_artifacts
from sgmoe.training import save_predictions


def predict(checkpoint_file, input_file, output_file, device_name='cuda:0', path_config=None):
    output = Path(output_file)
    if output.exists():
        raise FileExistsError('Prediction output already exists; choose a new path.')
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    config, labels = checkpoint['config'], checkpoint['labels']
    if path_config is not None:
        config['encoder']['model_name_or_path'] = path_config['encoder']['model_name_or_path']
        config['semantic']['artifact_dir'] = path_config['semantic']['artifact_dir']
    artifacts = load_artifacts(config['semantic']['artifact_dir'], labels)
    if digest(artifacts[0]) != digest(checkpoint['artifacts']):
        raise ValueError('Inference artifacts do not match the trained expert assignment.')
    records = read_records(input_file, config['data'], labeled=False)
    tokenizer = load_tokenizer(config['encoder'])
    model = SGMoE(config, labels, artifacts).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    loader = make_loader(records, tokenizer, config, None)
    rows = []
    with torch.no_grad():
        for batch in loader:
            probabilities = model(**model_inputs(batch, device))['logits'].softmax(-1).cpu()
            for sample_id, values in zip(batch['id'], probabilities):
                choice = int(values.argmax())
                rows.append({'id': sample_id, 'prediction': labels[choice], 'confidence': float(values[choice]),
                             'probabilities': json.dumps(dict(zip(labels, values.tolist())), ensure_ascii=False)})
    output.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(rows, output)
    return rows
