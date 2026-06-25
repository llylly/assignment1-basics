import os
import typing
import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
import numpy as np
from numpy import typing as npt

def LGetBatch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    datalen = len(dataset)
    s_idxes = np.random.randint(0, datalen - context_length, batch_size)
    e_idxes = s_idxes + context_length
    xs = torch.tensor(np.stack([dataset[s_idx: e_idx] for s_idx, e_idx in zip(s_idxes, e_idxes)], axis=0)).to(device=device)
    ys = torch.tensor(np.stack([dataset[s_idx+1: e_idx+1] for s_idx, e_idx in zip(s_idxes, e_idxes)], axis=0)).to(device=device)
    return xs, ys

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_states = model.state_dict()
    opt_states = optimizer.state_dict()
    all_states = {
        'model': model_states,
        'opt': opt_states,
        'iteration': iteration
    }
    torch.save(all_states, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    dict = torch.load(src)
    model.load_state_dict(dict['model'])
    optimizer.load_state_dict(dict['opt'])
    return dict['iteration']

def dict_to_dataclass(cls, d):
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if dataclasses.is_dataclass(fields[k].type) and isinstance(v, dict):
            kwargs[k] = dict_to_dataclass(fields[k].type, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)

if __name__ == '__main__':
    from cs336_basics.lmodeling import LTransformerLM
    from cs336_basics.lmodel_configs import gpt2_medium_model_config
    config = gpt2_medium_model_config | {'device': 'cuda', 'dtype': torch.bfloat16}
    lm = LTransformerLM(**config)
    from cs336_basics.lopt import LAdamW
    optimizer = LAdamW(lm.parameters(), 0.001, (0.9, 0.95), 0.01)
    # save_checkpoint(lm, optimizer, 10, 'tmp/tmp.pt')
