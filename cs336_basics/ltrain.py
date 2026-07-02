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
from datetime import datetime
from tqdm import tqdm

from cs336_basics.ltrain_utils import load_checkpoint, save_checkpoint, dict_to_dataclass, LGetBatch
from cs336_basics.lmodeling import LTransformerLM
from cs336_basics.lopt import LAdamW, LSGD, LCosineLR, LCrossEntropy, LGradientClipping

@dataclass
class TrainerConfig:
    batch_size: int
    tot_steps: int
    warmup_steps: int
    cooldown_steps: int
    learning_rate: float
    cooldown_learning_rate: float
    seqlen: int
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.1
    accum_steps: int = 1
    gradient_clipping: float | None = 3.0
    opt_type: Literal['adam', 'sgd'] = 'adam'

@dataclass
class MainConfig:
    trainer: TrainerConfig
    model_config: str
    """yaml file of model configs"""
    data: str
    """tokenized one-dimensional numpy array data for training"""
    save_path: str
    """path to save model and opt, should be a folder"""
    val_data: str | None = None
    """tokenized one-dimensional numpy array data for validation"""
    device: str = 'cuda'
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16'
    resume_path: str | None = None
    run_name: str | None = ''
    """run_name is appended to both save_path and wandb"""
    val_step: int = 1000
    save_step: int = 1000

"""
Example Usage:
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_small.yaml
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_small.yaml --trainer.learning_rate 0.0005 --run_name lr_5e-4
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_owt_small.yaml --val_step 5000 --save_step 5000
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_owt_medium.yaml --val_step 5000 --save_step 5000

Ablations on tiny training config:

uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.learning_rate 0.0001 --trainer.batch_size 32 --trainer.seqlen 256 --trainer.tot_steps 66000 --run_name lr_1e-4_bs_32_len_256_step_66k
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.learning_rate 0.0001 --trainer.batch_size 64 --trainer.seqlen 256 --trainer.tot_steps 33000 --run_name lr_1e-4_bs_64_len_256_step_33k
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.learning_rate 0.0003 --trainer.batch_size 32 --trainer.seqlen 256 --trainer.tot_steps 66000 --run_name lr_3e-4_bs_32_len_256_step_66k
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.learning_rate 0.0001 --trainer.batch_size 32 --trainer.seqlen 512 --trainer.tot_steps 33000 --run_name lr_1e-4_bs_32_len_512_step_33k
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.learning_rate 0.0005 --trainer.batch_size 32 --trainer.seqlen 256 --trainer.tot_steps 66000 --run_name lr_5e-4_bs_32_len_256_step_66k
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.learning_rate 0.0010 --trainer.batch_size 32 --trainer.seqlen 256 --trainer.tot_steps 66000 --run_name lr_1e-3_bs_32_len_256_step_66k 
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.learning_rate 0.0015 --trainer.batch_size 32 --trainer.seqlen 256 --trainer.tot_steps 66000 --run_name lr_1.5e-3_bs_32_len_256_step_66k
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.learning_rate 0.0010 --trainer.batch_size 32 --trainer.seqlen 256 --trainer.tot_steps 66000 --trainer.warmup_steps 5000 --trainer.cooldown_steps 5000 --run_name lr_1e-3_bs_32_len_256_step_66k_warmup_5k_cool_5k [* best]
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.learning_rate 0.0010 --trainer.batch_size 32 --trainer.seqlen 256 --trainer.tot_steps 66000 --trainer.warmup_steps 5000 --trainer.cooldown_steps 5000 --trainer.weight_decay 0.2 --run_name lr_1e-3_bs_32_len_256_step_66k_warmup_5k_cool_5k_wd_0.2 

Ablations on tiny arch config (from best training config):

uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --model_config cs336_basics/configs/models/gpt2_tiny_no_rms_norm.yaml --run_name no_rms_norm
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --model_config cs336_basics/configs/models/gpt2_tiny_post_norm.yaml --run_name post_norm
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --model_config cs336_basics/configs/models/gpt2_tiny_nope.yaml --run_name nope
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --model_config cs336_basics/configs/models/gpt2_tiny_silu.yaml --run_name silu
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml # correction after fixing adamw bug

Still search for best lr for SGD: [ongoing]
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.opt_type sgd --run_name sgd # need much larger lr, such as 0.01
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_tiny.yaml --trainer.opt_type sgd --trainer.learning_rate 0.03 --run_name sgd_lr_3e-2 

ML system optimization group: [ongoing]
uv run python cs336_basics/ltrain.py --config_path cs336_basics/configs/main_config_ts_small.yaml --save_path models/speedopt_ts_small
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
    nowtime = datetime.now().strftime('_%Y%m%d_%H%M%S')
    original_save_path = config.save_path
    if config.save_path.endswith('/'):
        config.save_path = config.save_path[:-1]
    if config.run_name:
        config.save_path += '_' + config.run_name
    config.save_path += nowtime
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    with open(os.path.join(config.save_path, 'configs.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(json.dumps(asdict(config), indent=2))

    # construct model and optimizer
    with open(config.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float}[config.dtype]
    model_config |= {'device': config.device, 'dtype': dtype}
    model = LTransformerLM(**model_config)
    if config.trainer.opt_type == 'adam':
        optimizer = LAdamW(model.parameters(), config.trainer.learning_rate, (config.trainer.beta1, config.trainer.beta2), config.trainer.weight_decay)
    elif config.trainer.opt_type == 'sgd':
        optimizer = LSGD(model.parameters(), config.trainer.learning_rate, config.trainer.beta1, config.trainer.weight_decay)

    if config.resume_path:
        start_iter = load_checkpoint(config.resume_path, model, optimizer)
    else:
        start_iter = 0

    # load dataset
    dataset = np.load(config.data, 'r')
    if config.val_data is not None:
        val_dataset = np.load(config.val_data, 'r')
    else:
        val_dataset = None
    
    stats = {
        'stat_model_param': model.count_parameters()[0],
        'stat_model_non_embed_param': model.count_parameters()[1],
        'stat_dataset_len': len(dataset),
        'stat_batch_token': config.trainer.batch_size * config.trainer.seqlen,
        'stat_epochs': config.trainer.batch_size * config.trainer.seqlen * config.trainer.tot_steps / len(dataset)
    }
    print(stats)

    # wandb
    wandb.init(project=('LLLM/' + original_save_path).replace('/', '|'), name=config.save_path.replace('/', '|'), config=asdict(config) | stats, dir=os.path.join(config.save_path, 'wandb_logs'))

    stime = time.time()
    tot_token_trained = 0
    
    for now_step in tqdm(range(start_iter, config.trainer.tot_steps), desc='training'):
        now_lr = LCosineLR(now_step, config.trainer.learning_rate, config.trainer.cooldown_learning_rate, config.trainer.warmup_steps, config.trainer.tot_steps - config.trainer.cooldown_steps)
        x, y = LGetBatch(dataset, config.trainer.batch_size, config.trainer.seqlen, config.device)
        x = x.type(torch.long)
        y = y.type(torch.long)
        y_pred = model(x)
        loss = LCrossEntropy(y_pred, y)
        print(now_step, 'train loss =', loss.item())
        # update learning rate according to cosine scheduler
        for param_group in optimizer.param_groups:
            param_group['lr'] = now_lr
        if now_step % config.trainer.accum_steps == 0:
            optimizer.zero_grad()
        loss.backward()
        if now_step % config.trainer.accum_steps == 0:
            if config.trainer.gradient_clipping is not None:
                grad_norm = LGradientClipping(model.parameters(), config.trainer.gradient_clipping)
            optimizer.step()

        train_loss_item = loss.item()
        tot_token_trained += stats['stat_batch_token']
        train_info = {'train/loss': train_loss_item,
                    'train/lr': now_lr,
                    'train/grad_norm': grad_norm,
                    'train/tokens_per_sec': (now_step - start_iter) * stats['stat_batch_token'] / (time.time() - stime),
                    'train/token_trained': tot_token_trained}
        print(','.join(f'{k}: {v:.2f}' for k, v in train_info.items()))
        wandb.log(train_info, step=now_step)
        
        if val_dataset is not None and (now_step % config.val_step == 0 or now_step == config.trainer.tot_steps - 1):
            # evaluating on validation data
            chunk_size = config.trainer.batch_size * config.trainer.seqlen
            tot_val_batch = 0
            tot_val_loss = 0.
            for s_idx in tqdm(range(0, len(val_dataset), chunk_size), desc='validation'):
                if s_idx + chunk_size + 1 > len(val_dataset):
                    break
                x = torch.tensor(dataset[s_idx: s_idx + chunk_size], dtype=torch.long, device=config.device).view((config.trainer.batch_size, config.trainer.seqlen))
                y = torch.tensor(dataset[s_idx+1: s_idx + chunk_size + 1], dtype=torch.long, device=config.device).view((config.trainer.batch_size, config.trainer.seqlen))
                with torch.no_grad():
                    y_pred = model(x)
                    loss = LCrossEntropy(y_pred, y).item()
                tot_val_loss = tot_val_loss * tot_val_batch / (tot_val_batch + 1) + loss / (tot_val_batch + 1)
                tot_val_batch += 1
            print('val loss @ step', now_step, '=', tot_val_loss)
            wandb.log({'val/loss': tot_val_loss}, step=now_step)
        
        if now_step % config.save_step == 0 or now_step == config.trainer.tot_steps - 1:
            print('saving...')
            save_checkpoint(model, optimizer, now_step, os.path.join(config.save_path, f'step_{now_step}.pth'))
        
        if now_step == config.trainer.tot_steps - 1:
            with open(os.path.join(config.save_path, 'final.log'), 'w') as f:
                json.dump({
                    'val_loss': tot_val_loss,
                    'last_train_loss': train_loss_item,
                    'time_elased': time.time() - stime
                }, f, indent=2)

    print('Done!')
