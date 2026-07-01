from dataclasses import dataclass, field, asdict
from typing import Literal, List
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
from cs336_basics.lmodeling import LTransformerLM, LSoftmax
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
    prompts: List[str] = field(default_factory=list)
    output_path: str | None = None
    """whether to output generated response to a file"""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    special_token: str = '<|endoftext|>'
    device: str = 'cuda'
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16'

def sampler(y: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0): # y: [B, V] # output [B, 1]
    if temperature <= 1e-10:
        # greedy
        return y.argmax(dim=-1).unsqueeze(dim=-1)
    else:
        # print(y, y.shape)
        y_temped = LSoftmax(y / temperature, dim=-1)
        # print(y_temped)
        y_argsort = torch.argsort(y_temped, dim=-1, descending=True)
        # print(y_argsort, torch.max(y_argsort), torch.min(y_argsort), y_temped.shape)
        y_sorted = torch.gather(y_temped, 1, y_argsort)
        # print(y_sorted)
        y_cumsum = y_sorted.cumsum(dim=-1)
        # print(y_cumsum)
        y_kept = torch.where(y_cumsum <= top_p, y_sorted, torch.zeros_like(y_sorted))
        y_kept = y_kept / y_kept.sum(dim=-1, keepdim=True)
        y_raw_sampled = torch.multinomial(y_kept, num_samples=1)
        # print(y_raw_sampled)
        y_sampled = torch.gather(y_argsort, 1, y_raw_sampled)
        return y_sampled

def generate(model: LTransformerLM, prompts: list[str], tokenizer: LTokenizer,
             max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 1.0, device: str = 'cuda', pad_token_id = -100):
    token_list = [tokenizer.encode(pp) for pp in prompts]
    max_token_len = max([len(item) for item in token_list])
    padded_token_list = [[pad_token_id] * (max_token_len - len(item)) + item for item in token_list]
    x = torch.tensor(padded_token_list, dtype=torch.long, device=device)
    eof = tokenizer.encode(tokenizer.special_tokens[0])[0] # '<|endoftext|>'
    print(prompts)
    ys = [[] for _ in prompts]
    now_new_tokens = 0

    tot_seq = x.shape[0]
    not_finished = torch.ones((tot_seq,), dtype=torch.bool, device=device)
    kv_cache: dict | None = None

    with torch.no_grad():
        while True:
            if now_new_tokens >= max_new_tokens or x.shape[-1] > model.layers[0].attn.max_seq_len:
                print(f'exceeds length: now new tokens = {now_new_tokens}, now ctx len = {x.shape[-1]}')
                break
            y, kv_cache = model.batch_generate(x, kv_cache=kv_cache, pad_token_id=pad_token_id)
            last_y = y[:, -1]
            pred_y = sampler(last_y, temperature, top_p)
            x = torch.concat([x, pred_y], dim=1)
            now_new_tokens += 1

            ii = 0
            for i in range(len(prompts)):
                if not_finished[i]:
                    ys[i].append(pred_y[ii].item())
                    ii += 1
            
            new_finished = (pred_y == eof).view(-1)
            # print(new_finished)
            kv_cache = [(k[~new_finished], v[~new_finished]) for k,v in kv_cache]
            x = x[~new_finished]
            not_finished[not_finished.clone()] = ~new_finished

            print(tokenizer.vocab[pred_y[0].item()].decode('utf-8'), end='', flush=True) # just for showing the first sentence to demonstrate the progress

            if not torch.any(not_finished):
                break
    
    text = []
    for prompt, response_lst in zip(prompts, ys):
        try:
            eofidx = response_lst.index(eof)
            response = tokenizer.decode(response_lst[:eofidx])
        except ValueError:
            response = tokenizer.decode(response_lst)
        text.append(response)
    return token_list, ys, text

"""
Example Usage:
uv run python cs336_basics/linference.py --config_path cs336_basics/configs/gen_config_ts_small.yaml --model_path models/ts_small_corrected_20260625_18
1530/step_65999.pth --prompts "You're so beautiful!" "Bob and Alice are playing a game" --temperature 0.0
"""

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

    prompt_tokenized, response_tokenized, response_text = generate(model, config.prompts, tokenizer,
                                                                   config.max_new_tokens, config.temperature, config.top_p, config.device)
    print('=' * 10)
    print(('*' * 10 + '\n').join([text + '|' + response for text, response in zip(config.prompts, response_text)]))
    print('=' * 10)

