"""
    Entry script for pretrain
"""

from typing import Callable, Literal
from dataclasses import dataclass, field, asdict
from typing import Literal
import tyro
import os
import sys
import yaml
import json
import time
import numpy as np
import wandb
from datetime import datetime
from tqdm import tqdm

import torch
import transformers

from basics.ltokenizer import LTokenizer
from basics.ltrain import LGradientClipping
from basics.lopt import LSGD, LAdamW
from basics.ltrain_utils import dict_to_dataclass, load_checkpoint
from basics.lmodeling import LTransformerLM
from alignment import vllm_utils
from alignment.lrl_utils import *


@dataclass
class RLTrainerConfig:
    batch_size: int
    """n_different_prompts = batch_size / group_size"""
    tot_steps: int
    learning_rate: float
    group_size: float
    gradient_accumulation_steps: int = 32

    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    gradient_clipping: float | None = 1.0
    opt_type: Literal['adam', 'sgd'] = 'adam'

    rollout_temperature: float = 1.0
    """Use rollout batch size = train batch size"""
    rollout_max_new_tokens: int = 512

    max_seqlen_clipping: int | None = None
    """If none, clip to model's context length"""

@dataclass
class RLConfig:
    trainer: RLTrainerConfig
    base_model_ckpt: str
    """Huggingface-style or LLLM-style base model dir, contain all information such as weights, tokenizer, and the architecture str"""
    base_model_format: Literal['olmo_hf', 'olmo_lllm']
    # """Now only support Olmo models """
    task_name: Literal['gsm8k']
    """use the task_name to locate grader method and test method"""
    save_path: str
    """path to save LLLM-style model and opt, should be a folder"""
    train_data: str
    """jsonl format training data"""
    val_data: str
    """jsonl format valiation data"""
    device: str = 'cuda:0'
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16'
    resume_path: str | None = None
    """if exists, it can overwrite base_model_ckpt"""
    tokenizer_path: str | None = None
    # """if base_model_format == 'olmo_lllm', need to specify tokenizer_path"""
    model_config: str | None = None
    # """if base_model_format == 'olmo_lllm', need to specify model_config"""
    inference_backend: Literal['vllm', 'lllm'] = 'vllm'
    inference_device: str='cuda:1'
    inference_vllm_server_no_host: bool = False # in this case, assume the host is already there
    inference_vllm_ip: str = '127.0.0.1'
    inference_vllm_port: int = 8080
    inference_vllm_seed: int = 42
    inference_vllm_dummy_model_path: str | None = None
    # if base_model_format == 'olmo_lllm', we need a dummy hf model path to launch vllm inference engine
    run_name: str | None = ''
    """run_name is appended to both save_path and wandb"""
    val_step: int = 10
    rollout_save_step: int = 10
    ckpt_save_step: int = 100
    debug: bool = False 
    """if debug, dump more statistics (act norm, grad norm per layer) on wandb"""


def grpo_train_step(
        model: transformers.PreTrainedModel | LTransformerLM,
        tokenizer: transformers.PreTrainedTokenizer | LTokenizer,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None,
        reward_fn: Callable[[str, str], dict[str, float]],
        repeated_prompts: list[str],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        # Reward normalization
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
        # Importance reweighting and clipping
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        # Loss normalization
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    macro_batch_size = len(repeated_prompts)
    num_prompts = macro_batch_size // group_size
    micro_batch_size = (macro_batch_size - 1) // gradient_accumulation_steps + 1

    # stage 1: grading
    raw_rewardses, raw_rewards_metadatas = [], []
    for i in range(0, macro_batch_size, group_size):
        raw_rewards, raw_rewards_metadata = compute_rollout_rewards(reward_fn, rollout_responses[i: i+group_size], repeated_ground_truths[i: i+group_size])
        raw_rewardses.append(raw_rewards)
        raw_rewards_metadatas.append(raw_rewards_metadata)
    raw_rewards = torch.concat(raw_rewardses)

    # stage 2: compute advantage
    advantages, advantage_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)

    # stage 3: retokenize
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids, labels, response_masks = tokenized['input_ids'].to(model.device), tokenized['labels'].to(model.device), tokenized['response_mask'].to(model.device)

    # stage 4: within the microbatch, compute logits and update the model
    batch_loss = torch.tensor(0., dtype=model.dtype, device=model.device)
    mean_token_entropy = torch.tensor(0., dtype=model.dtype, device=model.device)
    for i in range(0, macro_batch_size, micro_batch_size):
        actual_batch_size = min(micro_batch_size, macro_batch_size - i)
        model_ret = get_response_log_probs(model, input_ids[i: i+micro_batch_size], labels[i: i+micro_batch_size], return_token_entropy=True)
        log_probs, token_entropy = model_ret['log_probs'], model_ret['token_entropy'] # [B,L] and [B]
        token_level_loss, token_level_loss_metadata = compute_policy_gradient_loss(advantages[i: i+micro_batch_size], log_probs, importance_reweighting_method, old_log_probs, cliprange, response_masks[i: i+micro_batch_size])
        microbatch_loss = aggregate_loss_across_microbatch_sequence(token_level_loss, response_masks[i: i+micro_batch_size], loss_normalization, normalization_constant)
        # calibration
        microbatch_loss = microbatch_loss * actual_batch_size / macro_batch_size
        microbatch_loss.backward()

        with torch.no_grad():
            batch_loss = batch_loss + microbatch_loss
            mean_token_entropy += ((token_entropy * response_masks[i: i+micro_batch_size]).sum(dim=-1) / response_masks[i: i+micro_batch_size].sum(dim=-1)).sum() * actual_batch_size / macro_batch_size

    # stage 5: grad norm clipping & update weights for the mini-batch
    grad_norm = LGradientClipping(model.parameters(), max_grad_norm)
    optimizer.step()

    # stage 6: zero grad optimizer to prepare
    optimizer.zero_grad()

    # stage 7: prepare metadata
    raw_rewards_metadata_agg = {}
    for k in raw_rewards_metadatas[0]:
        raw_rewards_metadata_agg[k] = sum([item[k] for item in raw_rewards_metadatas]) / len(raw_rewards_metadatas)
    advantage_metadata_agg = {k: sum(v) / len(v) for k, v in advantage_metadata.items()}
    metadata = {
        'num_prompts': num_prompts,
        'micro_batch_size': micro_batch_size,
        'macro_batch_size': macro_batch_size,
        'token_entropy': mean_token_entropy,
        'grad_norm': grad_norm
    } | raw_rewards_metadata_agg | advantage_metadata_agg

    return batch_loss, metadata

if __name__ == '__main__':
    if '--config_path' in sys.argv:
        idx = sys.argv.index('--config_path')
        with open(sys.argv[idx + 1]) as f:
            defaults = yaml.safe_load(f)
        sys.argv.pop(idx)
        sys.argv.pop(idx)
        config = tyro.cli(RLConfig, default=dict_to_dataclass(RLConfig, defaults))
    else:
        config = tyro.cli(RLConfig)
    
    # config parameter validation
    if config.base_model_format == 'olmo_lllm':
        assert config.inference_vllm_dummy_model_path is not None, "if base_model_format == 'olmo_lllm', we need a dummy hf model path to launch vllm inference engine"
        assert config.tokenizer_path is not None, "if base_model_format == 'olmo_lllm', need to specify tokenizer_path"
        assert config.model_config is not None, "if base_model_format == 'olmo_lllm', need to specify model_config"
    assert config.inference_backend == 'vllm', 'backend engine from our own lllm is not supported yet until I know how to dynamically update engine weights'
    assert config.inference_device.startswith('cuda:'), 'VLLM inference device should be of the format cuda:X'

    nowtime = datetime.now().strftime('_%Y%m%d_%H%M%S')
    original_save_path = config.save_path
    if config.save_path.endswith('/'):
        config.save_path = config.save_path[:-1]
    if config.run_name:
        config.save_path += '_' + config.run_name
    config.save_path += nowtime
    # if not os.path.exists(config.save_path):
    #     os.makedirs(config.save_path)
    # with open(os.path.join(config.save_path, 'configs.json'), 'w') as f:
    #     json.dump(asdict(config), f, indent=2)
    print(json.dumps(asdict(config), indent=2))

    # consturct model and optimizer
    print('Loading model and optimizer...')
    if config.base_model_format == 'olmo_lllm':
        with open(config.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
        dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float}[config.dtype]
        model_config |= {'device': config.device, 'dtype': dtype, 'require_kv_cache': False}
        model = LTransformerLM(**model_config)
        tokenizer = LTokenizer.from_hf_tokenizers(config.tokenizer_path)
        load_checkpoint(config.base_model_ckpt, model, None, no_optimizer_load=True)
    elif config.base_model_format == 'olmo_hf':
        from basics.lmodeling_olmo import from_pretrained
        model, tokenizer = from_pretrained(config.base_model_ckpt, config.dtype, config.device, flash_attn=True)

    if config.trainer.opt_type == 'adam':
        optimizer = LAdamW(model.parameters(), config.trainer.learning_rate, (config.trainer.beta1, config.trainer.beta2), config.trainer.weight_decay)
    elif config.trainer.opt_type == 'sgd':
        optimizer = LSGD(model.parameters(), config.trainer.learning_rate, config.trainer.beta1, config.trainer.weight_decay)

    if config.resume_path:
        start_step = load_checkpoint(config.resume_path, model, optimizer)

    config.trainer.max_seqlen_clipping = model.max_seq_len

    # launch vllm server, then replace dummy parameter with the real one
    print('Setting up vllm server...')
    if config.inference_backend == 'vllm':
        if config.base_model_format == 'olmo_lllm':
            init_vllm_model_path = config.inference_vllm_dummy_model_path
        else:
            init_vllm_model_path = config.base_model_ckpt
        inference_device_no = int(config.inference_device[5:])
        rank_offset = inference_device_no - int(config.device[5:])
        vllm_base_url = f'http://{config.inference_vllm_ip}:{config.inference_vllm_port}'
        weight_format_converter = None
        if config.base_model_format in ['olmo_lllm', 'olmo_hf']:
            weight_format_converter = lrl_utils.lolmo2_to_vllm_ckpt_converter

        inference_serv_proc = vllm_utils.start_server(init_vllm_model_path, config.inference_vllm_ip, config.inference_vllm_port, config.dtype, inference_device_no, config.inference_vllm_seed, "auto", 'INFO')
        vllm_utils.wait_for_server(vllm_base_url, inference_serv_proc, 300) # wait for 300s
        print('Inference server launched...\nNow attempt to sync up weights')

        weight_sync_group = vllm_utils.init_weight_sync(vllm_base_url, config.device, rank_offset)
        vllm_utils.sync_policy_weights(model, vllm_base_url, weight_sync_group, weight_format_converter)
    else:
        raise NotImplementedError

    # main loop
    # TODO
