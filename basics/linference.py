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

from basics.ltrain_utils import load_checkpoint, save_checkpoint, dict_to_dataclass, LGetBatch
from basics.lmodeling import LTransformerLM, LSoftmax
from basics.lopt import LAdamW, LCosineLR, LCrossEntropy, LGradientClipping
from basics.ltokenizer import LTokenizer

from basics.lmodeling_olmo import LOlmo2TransformerLM

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


@dataclass
class LCompletion:
    prompt: str | None
    text: str
    token_ids: list[int]
    finish_reason: str | None # "stop" / "length" / "content_filter"

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

def generate(model: LTransformerLM | LOlmo2TransformerLM, prompts: list[str], tokenizer: LTokenizer,
             max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 1.0, device: str = 'cuda', 
             pad_token_id = -100, extra_stop_tokens: list[str] | None = None, include_stop_str_in_output=False, verbose=True) -> List[LCompletion]:
    
    # tokenize and left padding
    token_list = [tokenizer.encode(pp) for pp in prompts]
    max_token_len = max([len(item) for item in token_list])
    padded_token_list = [[pad_token_id] * (max_token_len - len(item)) + item for item in token_list]
    x = torch.tensor(padded_token_list, dtype=torch.long, device=device)
    token_positions = (torch.cumsum(x != pad_token_id, dim=1) - 1).clamp(min=0).to(x.device)

    if verbose: print(prompts)
    ys = [[] for _ in prompts]
    now_new_tokens = 0

    input_len = x.shape[-1]
    tot_seq = x.shape[0]
    not_finished = torch.ones((tot_seq,), dtype=torch.bool, device=device)
    kv_cache: dict | None = None

    if extra_stop_tokens is None:
        extra_stop_tokens = []
    stop_token_ids = [tokenizer.encode(item)[0] for item in extra_stop_tokens if len(tokenizer.encode(item)) == 1]
    eof = tokenizer.encode(tokenizer.eos_token)[0] if tokenizer.eos_token else None # '<|endoftext|>'
    if eof:
        stop_token_ids = torch.tensor([eof] + stop_token_ids, device=device).view(1, -1)
        extra_stop_tokens.append(tokenizer.eos_token)
    
    with torch.no_grad():
        while True:
            max_seq_len: int = model.max_seq_len if isinstance(model, LTransformerLM) else model.max_seq_len
            if now_new_tokens >= max_new_tokens or input_len + now_new_tokens > max_seq_len:
                if verbose: print(f'!!! exceeds length: now new tokens = {now_new_tokens}, now ctx len = {x.shape[-1]}')
                break
            y, kv_cache = model.batch_generate(x, kv_cache=kv_cache, pad_token_id=pad_token_id, token_positions=token_positions if kv_cache else None, kv_cache_sliced_to=input_len + now_new_tokens - 1)
            last_y = y[:, -1]
            pred_y = sampler(last_y, temperature, top_p)
            token_positions = token_positions[:, -1:] + torch.cumsum(pred_y != pad_token_id, dim=1)
            x = pred_y # no concatenate any more, fully rely on kv_cache to store previous x's
            now_new_tokens += 1

            new_finished = torch.any(pred_y == stop_token_ids, dim=1).to(x.device).view(-1)

            ii = 0
            for i in range(len(prompts)):
                if not_finished[i]:
                    ys[i].append(pred_y[ii].item())
                    try:
                        now_decoded = tokenizer.decode(ys[i])
                        if any([now_decoded.find(extra_stop_token) != -1 for extra_stop_token in extra_stop_tokens]):
                            new_finished[ii] = True
                    except UnicodeDecodeError:
                        pass
                    ii += 1
            
            if any(new_finished):
                kv_cache = [(k[~new_finished], v[~new_finished]) for k,v in kv_cache]
                x = x[~new_finished]
                token_positions = token_positions[~new_finished]
                not_finished[not_finished.clone()] = ~new_finished

            if verbose: 
                try:
                    print(tokenizer.vocab[pred_y[0].item()].decode('utf-8'), end='', flush=True) # just for showing the first sentence to demonstrate the progress
                except UnicodeDecodeError:
                    pass # For non-English multi-byte char, this can happen

            if not torch.any(not_finished):
                break
    
    stop_reasons = ['length' if not_finished[i] else 'stop' for i in range(len(prompts))]
    ret = []

    for prompt, response_lst, stop_reason in zip(prompts, ys, stop_reasons):
        if include_stop_str_in_output:
            fin_response_lst = response_lst
        else:
            fin_response_lst = response_lst
            for i in range(len(response_lst)):
                if response_lst[i] in stop_token_ids:
                    fin_response_lst = response_lst[:i]
                    break
        response = tokenizer.decode(fin_response_lst)
        ret.append(LCompletion(prompt, response, fin_response_lst, stop_reason))
    return ret

"""
Example Usage:
uv run python basics/linference.py --config_path configs/gen_config_ts_small.yaml --model_path models/ts_small_20260701_001346/step_65999.pth --prompts "You're so beautiful!" "Bob and Alice are playing a game" --temperature 1.0
uv run python basics/linference.py --config_path configs/gen_config_ts_tiny.yaml --model_path models/ts_tiny_20260701_175105/step_65999.pth --prompts "You're so beautiful!" "Bob and Alice are playing a game" --temperature 0.0
uv run python basics/linference.py --config_path configs/gen_config_owt_small.yaml --model_path models/owt_small_20260702_035910/step_164999.pth --prompts "You're so beautiful!" "Bob and Alice are playing a game" --temperature 1.0
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

    completions = generate(model, config.prompts, tokenizer,
                           config.max_new_tokens, config.temperature, config.top_p, config.device)
    print('=' * 10)
    print(('*' * 10 + '\n').join([text + '|' + completion.text for text, completion in zip(config.prompts, completions)]))
    print('=' * 10)

