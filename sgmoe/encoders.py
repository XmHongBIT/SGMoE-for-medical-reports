"""Hugging Face encoder loading, shared across preparation and the full model."""
import torch


def load_tokenizer(config):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config['model_name_or_path'])


def load_encoder(config, lora=False):
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(config['model_name_or_path'])
    if lora:
        from peft import LoraConfig, get_peft_model
        encoder = get_peft_model(encoder, LoraConfig(
            r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            target_modules=config['lora_target_modules'], lora_dropout=config['lora_dropout'], bias='none'))
    return encoder


def feature_width(encoder):
    width = getattr(encoder.config, 'hidden_size', None) or getattr(encoder.config, 'd_model', None)
    if not width:
        raise ValueError('Text encoder must expose hidden_size or d_model.')
    return int(width)


def pooled_features(encoder, inputs):
    outputs = encoder(**inputs)
    if not hasattr(outputs, 'last_hidden_state'):
        raise ValueError('Text encoder must expose last_hidden_state.')
    # First non-padding token supports either left or right padding.
    positions = inputs['attention_mask'].long().argmax(dim=1)
    return outputs.last_hidden_state[torch.arange(len(positions), device=positions.device), positions].float()
