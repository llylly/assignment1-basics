from dataclasses import dataclass, field, asdict
from typing import Literal
import tyro
import os
import sys
import yaml
import json
import time
import numpy as np
import torch
import wandb
from tqdm import tqdm

from cs336_basics.ltrain_utils import load_checkpoint, save_checkpoint, dict_to_dataclass, LGetBatch
from cs336_basics.lmodeling import LTransformerLM
from cs336_basics.lopt import LAdamW, LCosineLR, LCrossEntropy, LGradientClipping
from cs336_basics.ltokenizer import LTokenizer

@dataclass
class MainConfig:
    model_config: str
    """yaml file of model configs"""
    model_path: str | None = None
    """path to a .pth; if none, has to be filled by args"""
    tokenizer_vocab_path: str | None = None
    """txt file of vocabulary; if none, has to be filled by args"""
    tokenizer_merges_path: str | None = None
    """text file of merges; if none, has to be filled by args"""
    prompt: str = 'Hello'
    output_path: str | None = None
    """whether to output generated response to a file"""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    special_token: str = '<|endoftext|>'
    device: str = 'cuda'
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16'

def generate(model: LTransformerLM, prompt_tokens: str, tokenizer: LTokenizer,
             max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 1.0, device: str = 'cuda'):
    token_list = [tokenizer.encode(prompt_tokens)]
    eof = tokenizer.encode(tokenizer.special_tokens[0])[0] # '<|endoftext|>'
    x = torch.tensor(token_list, dtype=torch.long, device=device)
    print(prompt_tokens, end='')
    ys = []
    now_new_tokens = 0
    with torch.no_grad():
        while True:
            if now_new_tokens >= max_new_tokens or x.shape[-1] > model.layers[0].attn.max_seq_len:
                print(f'exceeds length: now new tokens = {now_new_tokens}, now ctx len = {x.shape[-1]}')
            y = model(x)
            pred_y = y.argmax(dim=-1)[:, -1:]
            x = torch.concat([x, pred_y], dim=1)
            ys.append(pred_y)
            pred_y = pred_y.item()
            if pred_y == eof:
                break
            print(tokenizer.vocab[pred_y].decode('utf-8'), end='')
    return torch.tensor(ys)

if __name__ == '__main__':
    if '--config_path' in sys.argv:
        idx = sys.argv.index('--config_path')
        with open(sys.argv[idx + 1]) as f:
            defaults = yaml.safe_load(f)
        sys.argv.pop(idx)
        sys.argv.pop(idx)
        config = tyro.cli(MainConfig, default=dict_to_dataclass(MainConfig, defaults))
    else:
        config = tyro.cli(MainConfig)
    
    # load tokenizer
    tokenizer = LTokenizer.from_files(config.tokenizer_vocab_path, config.tokenizer_merges_path, [config.special_token])

    # construct model and optimizer
    with open(config.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float}[config.dtype]
    model_config |= {'device': config.device, 'dtype': dtype}
    model = LTransformerLM(**model_config)
    
    # load model
    assert config.model_path is not None
    with open(config.model_path, 'rb') as f:
        states = torch.load(f)
    model.load_state_dict(states['model'])

    ret = generate(model, config.prompt, tokenizer,
                   config.max_new_tokens, config.temperature, config.top_p, config.device)
    
    print(ret)

