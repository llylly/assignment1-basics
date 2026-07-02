"""
    The script profiles the execution time of our model given hyperparameters
"""
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
import timeit
from datetime import datetime
from tqdm import tqdm

from cs336_basics.lmodeling import LTransformerLM
from cs336_basics.lopt import LAdamW, LCrossEntropy


@dataclass
class MainConfig:
    model_config: str
    """yaml file of model configs"""
    num_layers: int | None = None
    """maybe overwrite"""
    d_model: int | None = None
    """maybe overwrite"""
    d_ff: int | None = None
    """maybe overwrite"""
    batch_size: int = 4 # default asgn 2 req.
    seq_len: int = 512 # default asgn 2 req.
    modes: List[str] = field(default_factory=list) # O1 = forward, O2 = forward + backward, O3 = forward + backward + optimization
    warmup_steps: int = 32
    test_steps: int = 64
    memory_tracing: bool = False
    device: str = 'cuda'
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16'

"""
Example Usage:
uv run python cs336_systems/lprofile.py --model_config cs336_basics/configs/models/gpt2_tiny.yaml --modes O1 O2 O3  
uv run python cs336_systems/lprofile.py --model_config cs336_basics/configs/models/gpt2_small.yaml --modes O1 O2 O3  
uv run python cs336_systems/lprofile.py --model_config cs336_basics/configs/models/gpt2_medium.yaml --modes O1 O2 O3  
uv run python cs336_systems/lprofile.py --model_config cs336_basics/configs/models/gpt2_large.yaml --modes O1 O2 O3  
uv run python cs336_systems/lprofile.py --model_config cs336_basics/configs/models/gpt2_xl.yaml --modes O1 O2 O3  

uv run nsys profile python cs336_systems/lprofile.py --model_config cs336_basics/configs/models/gpt2_small.yaml --modes O
uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --cudabacktrace=all --python-backtrace=cuda python cs336_systems/lprofile.py --model_config cs336_basics/configs/models/gpt2_small.yaml --modes O1
"""
if __name__ == '__main__':
    config = tyro.cli(MainConfig)

    # construct model and optimizer
    with open(config.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float}[config.dtype]
    model_config |= {'device': config.device, 'dtype': dtype}
    if config.num_layers is not None:
        model_config['num_layers'] = config.num_layers # overwrite
    if config.d_model is not None:
        model_config['d_model'] = config.d_model # overwrite
    if config.d_ff is not None:
        model_config['d_ff'] = config.d_ff # overwrite
    
    if config.memory_tracing:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    model = LTransformerLM(**model_config)
    if 'O3' in config.modes:
        optimizer = LAdamW(model.parameters(), 1e-4, (0.9, 0.99), 0.1)
    model.resource_count(batch_size=config.batch_size, seq_len=config.seq_len)

    # _ = input('Continue?')

    print('Warmup...')
    for i in range(config.warmup_steps):
        with torch.no_grad():
            x = torch.randint(0, model_config['vocab_size'], (config.batch_size, config.seq_len), device=config.device, dtype=torch.long)
            output = model(x)

    def pack_hook(t):
        # shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
        # print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
        return t

    def unpack_hook(t):
        # shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
        # print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
        return t

    print('Main...')
    results = dict()
    for mode in config.modes:
        def main():
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                if mode == 'O1':
                    with torch.no_grad():
                        x = torch.randint(0, model_config['vocab_size'], (config.batch_size, config.seq_len), device=config.device, dtype=torch.long)
                        y_pred = model(x)
                elif mode == 'O2' or mode == 'O3':
                    x = torch.randint(0, model_config['vocab_size'], (config.batch_size, config.seq_len), device=config.device, dtype=torch.long)
                    y = torch.randint(0, model_config['vocab_size'], (config.batch_size, config.seq_len), device=config.device, dtype=torch.long)
                    y_pred = model(x)
                    loss = LCrossEntropy(y_pred, y)
                    if mode == 'O3':
                        optimizer.zero_grad()
                    loss.backward()
                    if mode == 'O3':
                        optimizer.step()
                    # print(loss.item())
            torch.cuda.synchronize()

        t = timeit.timeit(main, number=config.test_steps)
        print(f'{mode} Time is {t / config.test_steps:10.4f} s over {config.test_steps} trials') 
        results[mode] = t / config.test_steps

    if config.memory_tracing:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        print('memory tracing saved to memory_snapshot.pickle')
    
        torch.cuda.memory._record_memory_history(enabled=None)

    peak = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    print(f"Peak memory: {peak:.2f} MB")
    print(f"Peak reserved memory: {peak_reserved:.2f} MB")
    # print(torch.cuda.memory_summary())
