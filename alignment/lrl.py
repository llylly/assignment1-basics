"""
    Entry script for pretrain
"""

from typing import Callable, Literal
from dataclasses import dataclass, field, asdict
from typing import Literal
from pathlib import Path
import tyro
import os
import sys
import resource
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
from basics.ltrain import LGradientClipping, save_checkpoint
from basics.lopt import LSGD, LAdamW
from basics.ltrain_utils import dict_to_dataclass, load_checkpoint, get_git_hash
from basics.lmodeling import LTransformerLM
from basics.linference import generate, LCompletion
from alignment.benchmarks.lbenchmarks import *
from alignment import vllm_utils
from alignment.lrl_utils import *


@dataclass
class RLTrainerConfig:
    batch_size: int
    """n_different_prompts = batch_size / group_size"""
    tot_steps: int
    learning_rate: float
    group_size: int
    gradient_accumulation_steps: int = 32

    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    gradient_clipping: float | None = 1.0
    opt_type: Literal['adam', 'sgd'] = 'adam'

    rollout_temperature: float = 1.0
    """Use rollout batch size = train batch size"""
    rollout_max_new_tokens: int = 512
    rollout_stop_words: list[str] = field(default_factory=lambda: ['</answer>'])

    rollout_batch_size: int | None = None
    """If none, equal to batch_size, otherwise whould be a divisor of batch_size -> maybe not needed, now this is lifted"""
    max_seqlen_clipping: int | None = None
    """If none, clip to model's context length"""

    # Reward normalization
    baseline: Literal["mean", "none"] = "mean"
    advantage_eps: float = 1e-6
    advantage_normalizer: Literal["std", "none", "mean"] = "std"
    # Importance reweighting and clipping
    importance_reweighting_method: Literal['none', 'noclip', 'grpo', 'gspo', 'cispo', 'dapo'] = "none"
    cliprange: float | None = None
    clip_higher_range: float | None = None # only used for dapo
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence"
    normalization_constant: int | None = None

    # async stale steps
    async_rl_maxstep: int = 0 # 0 means fully online, each step 

@dataclass
class RLConfig:
    trainer: RLTrainerConfig
    base_model_ckpt: str
    """Huggingface-style or LLLM-style base model dir, contain all information such as weights, tokenizer, and the architecture str"""
    base_model_format: Literal['olmo_hf', 'lllm']
    # """Now only support Olmo models """
    task_name: Literal['gsm8k']
    """use the task_name to locate templated formulator method, grader method, and test method"""
    prompt_template: str
    """path to prompt template which will be loaded and passed to task-specific assembling method"""
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
    # """if base_model_format == 'lllm', need to specify tokenizer_path"""
    model_config: str | None = None
    # """if base_model_format == 'lllm', need to specify model_config"""
    inference_backend: Literal['vllm', 'lllm', 'sft'] = 'vllm'
    inference_device: str='cuda:1'
    inference_vllm_server_no_host: bool = False # in this case, assume the host is already there
    inference_vllm_ip: str = '127.0.0.1'
    inference_vllm_port: int = 8080
    inference_vllm_seed: int | None = None #42
    inference_vllm_dummy_model_path: str | None = None
    # if base_model_format == 'vllm', we need a dummy hf model path to launch vllm inference engine
    inference_vllm_gpu_memory_utilization: float = 0.8
    inference_sft_field_name: str | None = None # if it is sft, need to know the field name to extract response
    inference_sft_run_lllm_eval: bool = True # if it is sft, whether to use our lllm inference to evaluate
    run_name: str | None = ''
    """run_name is appended to both save_path and wandb"""
    val_step: int = 20
    rollout_save_step: int = 10
    ckpt_save_step: int = 50
    logging_level: Literal["ERROR", "WARNING", "INFO"] = "INFO" # mainly used by vllm
    debug: bool = False 
    """if debug, dump more statistics (act norm, grad norm per layer) on wandb"""
    git_hash: str = ''
    """Keep track of commit id, will auto populate"""


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
        importance_reweighting_method: Literal['none', 'noclip', 'grpo', 'gspo', 'cispo', 'dapo'] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        clip_higher_range: float | None = None, # only used for dapo
        # Loss normalization
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None,
        response_token_ids: list | None = None, # if provided, no need to retokenized response to ensure inference-training consistency
        max_seqlen_clipping: int | None = None,
        is_sft: bool = False # we view sft as a special mode that corresponds to response_token from instruction set, all reward = 1, and group_size = 1
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    macro_batch_size = len(repeated_prompts)
    num_prompts = macro_batch_size // group_size
    micro_batch_size = (macro_batch_size - 1) // gradient_accumulation_steps + 1

    # stage 1: grading
    raw_rewardses, raw_rewards_metadatas = [], []
    if is_sft:
        raw_rewards = torch.ones((len(rollout_responses),), dtype=torch.float)
        raw_rewards_metadatas.append({'reward/mean': 1.})
    else:
        for i in range(0, macro_batch_size, group_size):
            raw_rewards, raw_rewards_metadata = compute_rollout_rewards(reward_fn, rollout_responses[i: i+group_size], repeated_ground_truths[i: i+group_size])
            raw_rewardses.append(raw_rewards)
            raw_rewards_metadatas.append(raw_rewards_metadata)
        raw_rewards = torch.concat(raw_rewardses)

    # stage 2: compute advantage
    advantages, advantage_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)

    # stage 3: retokenize
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer, response_token_ids)
    input_ids, labels, response_masks = tokenized['input_ids'].to(model.device), tokenized['labels'].to(model.device), tokenized['response_mask'].to(model.device)
    if max_seqlen_clipping is not None:
        # trim
        input_ids, labels, response_masks = input_ids[:, :max_seqlen_clipping], labels[:, :max_seqlen_clipping], response_masks[:, :max_seqlen_clipping]

    # stage 4pre: prune zero advantaged sequences
    a_prune_eps = 0.00001
    actual_macro_batch_size = ((advantages >= a_prune_eps).sum() + (advantages <= -a_prune_eps).sum()).item()

    # stage 4: within the microbatch, compute logits and update the model
    token_level_loss_metadatas = []
    batch_loss = torch.tensor(0., dtype=model.dtype, device=model.device)
    mean_token_entropy = torch.tensor(0., dtype=model.dtype, device=model.device)
    for i in tqdm(range(0, macro_batch_size, micro_batch_size)):
        seq_keep_mask = (advantages[i: i+micro_batch_size] >= a_prune_eps) | (advantages[i: i+micro_batch_size] <= -a_prune_eps)
        actual_batch_size = seq_keep_mask.sum()
        if actual_batch_size == 0: # all zero
            continue
        model_ret = get_response_log_probs(model, input_ids[i: i+micro_batch_size][seq_keep_mask], labels[i: i+micro_batch_size][seq_keep_mask], return_token_entropy=True)
        log_probs, token_entropy = model_ret['log_probs'], model_ret['token_entropy'] # [B,L] and [B]
        token_level_loss, token_level_loss_metadata = compute_policy_gradient_loss(advantages[i: i+micro_batch_size][seq_keep_mask], log_probs, importance_reweighting_method, old_log_probs[i: i+micro_batch_size][seq_keep_mask] if old_log_probs is not None else None, cliprange, response_masks[i: i+micro_batch_size][seq_keep_mask], clip_higher_range)
        microbatch_loss = aggregate_loss_across_microbatch_sequence(token_level_loss, response_masks[i: i+micro_batch_size][seq_keep_mask], loss_normalization, normalization_constant)
        # calibration
        if loss_normalization == 'sequence':
            microbatch_loss = microbatch_loss * actual_batch_size / actual_macro_batch_size
        elif loss_normalization == 'constant':
            pass
        else:
            raise NotImplementedError
        microbatch_loss.backward()

        with torch.no_grad():
            batch_loss = batch_loss + microbatch_loss
            mean_token_entropy += ((token_entropy * response_masks[i: i+micro_batch_size][seq_keep_mask]).sum(dim=-1) / response_masks[i: i+micro_batch_size][seq_keep_mask].sum(dim=-1)).sum() / actual_macro_batch_size
        token_level_loss_metadatas.append(token_level_loss_metadata)

    # stage 5: grad norm clipping & update weights for the mini-batch
    grad_norm = LGradientClipping(model.parameters(), max_grad_norm)
    optimizer.step()

    # stage 6: zero grad optimizer to prepare
    optimizer.zero_grad()

    # stage 7: prepare metadata
    raw_rewards_metadata_agg = {}
    for k in raw_rewards_metadatas[0]:
        raw_rewards_metadata_agg[k] = sum([item[k] for item in raw_rewards_metadatas]) / len(raw_rewards_metadatas)
    advantage_metadata_agg = {k: (sum(v) / len(v)) if isinstance(v, list) else v for k, v in advantage_metadata.items()}
    token_level_loss_metadata_agg = {}
    for k in token_level_loss_metadatas[0]:
        token_level_loss_metadata_agg[k] = sum([item[k] for item in token_level_loss_metadatas]) / len(token_level_loss_metadatas)
    metadata = {
        'num_prompts': num_prompts,
        'micro_batch_size': micro_batch_size,
        'macro_batch_size': macro_batch_size,
        'token_entropy': mean_token_entropy,
        'grad_norm': grad_norm,
        'ctx_len': input_ids.shape[-1],
        'min_response_len': tokenized['response_mask'].sum(dim=-1).amin(dim=-1),
        'max_response_len': tokenized['response_mask'].sum(dim=-1).amax(dim=-1)
    } | raw_rewards_metadata_agg | advantage_metadata_agg | token_level_loss_metadata_agg

    return batch_loss, metadata

"""
Usage:
uv run alignment/lrl.py --config_path configs/rl_config_olmo_base_gsm8k_r1zero_grpo.yaml --device cuda:0 --inference_device cuda:3 (on RTX 6000 Ada)
uv run alignment/lrl.py --config_path configs/rl_config_olmo_base_gsm8k_r1zero_grpo.yaml --trainer.gradient_accumulation_steps 32 (on H100)
taskset -c 0-5,8-31 uv run alignment/lrl.py --config_path configs/rl_config_olmo_base_gsm8k_r1zero_grpo.yaml --inference_backend lllm --device cuda --inference_device cuda --trainer.gradient_accumulation_steps 128 --trainer.rollout_batch_size 8 --trainer.max_seqlen_clipping 700 --run-name 4090 (on single 4090)
Learning rate ablation:
uv run alignment/lrl.py --config_path configs/rl_config_olmo_base_gsm8k_r1zero_grpo.yaml --device cuda:0 --inference_device cuda:3 --trainer.learning_rate 0.00002 --run-name lr_2e-5
uv run alignment/lrl.py --config_path configs/rl_config_olmo_base_gsm8k_r1zero_grpo.yaml --device cuda:0 --inference_device cuda:3 --trainer.learning_rate 0.000005 --run-name lr_5e-6
Prompt ablation:
uv run alignment/lrl.py --config_path configs/rl_config_olmo_base_gsm8k_r1zero_grpo.yaml --device cuda:0 --inference_device cuda:3 --prompt_template "alignment/prompts/question_only.prompt" --run-name prpt_qonly
uv run alignment/lrl.py --config_path configs/rl_config_olmo_base_gsm8k_r1zero_grpo.yaml --device cuda:0 --inference_device cuda:3 --prompt_template "alignment/prompts/r1_zero_three_shot_gsm8k.prompt" --run-name prpt_3shot
uv run alignment/lrl.py --config_path configs/sft_config_olmo_base_gsm8k.yaml --device cuda:0

I found best learning rate for online GRPO is 2e-5

sft baseline:
PYTORCH_ALLOC_CONF=expandable_segments:True uv run alignment/lrl.py --config_path configs/sft_config_olmo_base_gsm8k.yaml --device cuda:0
sft rl:
uv run alignment/lrl.py --config_path configs/rl_config_olmo_base_gsm8k_r1zero_grpo.yaml --device cuda:0 --inference_device cuda:3 --trainer.learning_rate 0.00002 --base_model_ckpt models/rl/olmo2_1B_gsm8k/base_sft_20260730_222909/hf_ckpts/step_0000799/ --save_path "models/rl/olmo2_1B_gsm8k/sft_rl_r1zero" --run-name lr_2e-5
"""

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
    assert config.trainer.batch_size % config.trainer.group_size == 0
    if config.base_model_format == 'lllm':
        if config.inference_backend == 'vllm':
            assert config.inference_vllm_dummy_model_path is not None, "if base_model_format == 'lllm', we need a dummy hf model path to launch vllm inference engine"
        assert config.tokenizer_path is not None, "if base_model_format == 'lllm', need to specify tokenizer_path"
        assert config.model_config is not None, "if base_model_format == 'lllm', need to specify model_config"
    if config.inference_backend == 'vllm':
        assert config.inference_device.startswith('cuda:'), 'VLLM inference device should be of the format cuda:X'
        assert config.device != config.inference_device, 'If using vllm, should be on different devices'
    if config.inference_backend == 'lllm':
        assert config.device == config.inference_device, 'If using lllm, should be on the same device'
    if config.inference_backend == 'sft':
        assert config.trainer.group_size == 1
        assert config.inference_sft_field_name is not None
        assert config.trainer.baseline == 'none'
        assert config.trainer.advantage_normalizer == 'none'
        assert config.trainer.loss_normalization == 'sequence'
    if config.trainer.importance_reweighting_method == 'gspo':
        assert config.trainer.loss_normalization == 'sequence'
    # if config.trainer.rollout_batch_size is not None:
    #     assert config.trainer.batch_size % config.trainer.rollout_batch_size == 0

    nowtime = datetime.now().strftime('_%Y%m%d_%H%M%S')
    original_save_path = config.save_path
    if config.save_path.endswith('/'):
        config.save_path = config.save_path[:-1]
    if config.run_name:
        config.save_path += '_' + config.run_name
    config.save_path += nowtime
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    config.git_hash = get_git_hash() or config.git_hash
    with open(Path(config.save_path) / 'configs.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(json.dumps(asdict(config), indent=2))
    print('Things saved to', config.save_path)
    save_path = Path(config.save_path) # for later convenience

    # construct model and optimizer
    print('Loading model and optimizer...')
    torch.cuda.set_device(torch.device(config.device))
    if config.base_model_format == 'lllm':
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
        print('Resume loading...')
        start_step = load_checkpoint(config.resume_path, model, optimizer)
        print('Resumed from', config.resume_path, 'at step', start_step)
    else:
        start_step = 0

    if config.trainer.max_seqlen_clipping is None:
        config.trainer.max_seqlen_clipping = model.max_seq_len
    if config.trainer.rollout_batch_size is None:
        config.trainer.rollout_batch_size = config.trainer.batch_size

    # data loading
    # now no shuffle to ensure reproducibility
    with open(config.prompt_template, 'r') as f:
        chat_template = f.read()
    train_datasets: list[dict] = []
    with open(config.train_data, 'r') as f:
        for line in f.readlines():
            if line.strip():
                train_datasets.append(json.loads(line))
    val_datasets = []
    # no need to load val data for now, unless it is for sft where we need to compute validation data loss
    if config.inference_backend == 'sft':
        with open(config.val_data, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    val_datasets.append(json.loads(line))

    # data moderator: (1) extract ground_truth answer str for grader to use; (2) if it is sft, moderate the response format to fit with grading criteria.
    # It is dataset specific. In the future need to move it to a better place.
    if config.task_name == 'gsm8k':
        for item in train_datasets:
            item['**final_answer'] = item['answer'][item['answer'].find('####') + 4:].strip()
        if config.inference_backend == 'sft':
            for item in train_datasets:
                item['answer'] = item['answer'].replace('\n#### ', '</think> <answer> ') + ' </answer>'
            for item in val_datasets:
                item['answer'] = item['answer'].replace('\n#### ', '</think> <answer> ') + ' </answer>'
    else:
        raise NotImplementedError

    # launch vllm server, then replace dummy parameter with the real one
    print('Setting up vllm server...')
    if config.inference_backend == 'vllm':
        if config.base_model_format == 'lllm':
            init_vllm_model_path = config.inference_vllm_dummy_model_path
        else:
            init_vllm_model_path = config.base_model_ckpt
        inference_device_no = int(config.inference_device[5:])
        rank_offset = 1
        vllm_base_url = f'http://{config.inference_vllm_ip}:{config.inference_vllm_port}'
        weight_format_converter = None
        if config.base_model_format in ['olmo_hf']:
            from basics.lmodeling_olmo import lolmo2_to_vllm_weights_converter
            weight_format_converter = lolmo2_to_vllm_weights_converter
        else:
            raise NotImplementedError

        vllm_sampling_params = {
            'temperature': config.trainer.rollout_temperature,
            'max_tokens': config.trainer.rollout_max_new_tokens,
            'n': 1,
            'seed': config.inference_vllm_seed,
            'stop': config.trainer.rollout_stop_words,
            'include_stop_str_in_output': True,
        }

        inference_serv_proc = vllm_utils.start_server(init_vllm_model_path, config.inference_vllm_ip, config.inference_vllm_port, config.dtype, inference_device_no, config.inference_vllm_seed, "auto", config.logging_level, config.inference_vllm_gpu_memory_utilization)
        vllm_utils.wait_for_server(vllm_base_url, inference_serv_proc, 300) # wait for 300s
        print('Inference server launched...\nNow sync up initial weights')

        weight_sync_group = vllm_utils.init_weight_sync(vllm_base_url, config.device, rank_offset)
        vllm_utils.sync_policy_weights(model, vllm_base_url, weight_sync_group, weight_format_converter)
    elif config.inference_backend == 'lllm':
        pass
    elif config.inference_backend == 'sft':
        pass
    else:
        raise NotImplementedError

    # task-specific formulator, grader, and valgrader fns
    question_formulator: Callable[[dict], str] = lambda x: get_question_formulator(config.task_name)(x, chat_template)
    eval_config_class, task_set_grader = get_testable_task_setgrader(config.task_name)
    if config.task_name == 'gsm8k':
        # need to first determine the prompt type
        guessed_prompt_type = config.prompt_template.split('/')[-1].split('.')[0] # not extensible, but now works
        assert guessed_prompt_type in ['r1_zero', 'question_only', 'r1_zero_three_shot_gsm8k']
        task_grader = get_task_grader(config.task_name, prompt_type=guessed_prompt_type)
        from alignment.benchmarks.lgsm8k_eval import GSM8KEvalConfig
        eval_config = {
            'backend': config.inference_backend,
            'dtype': config.dtype,
            'device': config.inference_device,
            'prompt_type': guessed_prompt_type,
            'max_new_tokens': config.trainer.rollout_max_new_tokens,
            'temperature': config.trainer.rollout_temperature,
            'n': config.trainer.group_size,
            'batch_size': config.trainer.rollout_batch_size,
            'data_file': config.val_data,
            'run_suffix': 'rl_eval_stepX',
            'save_dir': '', # will be overwritten
        }
        if config.inference_backend == 'vllm':
            eval_config |= {
                'model_dir': init_vllm_model_path,
                'vllm_server_no_host': True,
                'vllm_ip': config.inference_vllm_ip,
                'vllm_port': config.inference_vllm_port
            }
        elif config.inference_backend == 'lllm' or config.inference_backend == 'sft':
            # a side effect: if it is sft, since group size is 1, we only measure pass@1
            eval_config |= {
                'backend': 'lllm',
                'lllm_server_no_host': True,
                'lllm_max_seq_len': config.trainer.max_seqlen_clipping
            }
        eval_config = GSM8KEvalConfig(**eval_config)
    else:
        raise NotImplementedError

    n_samp_per_batch = int(config.trainer.batch_size // config.trainer.group_size)
    stats = {
        'stat_model_param': model.count_parameters()[0],
        'stat_model_non_embed_param': model.count_parameters()[1],
        'stat_dataset_len': len(train_datasets),
        'stat_epochs': n_samp_per_batch * config.trainer.tot_steps / len(train_datasets),
        'stat_n_samp_per_batch': n_samp_per_batch
    }
    print(stats)

    # wandb
    wandb.init(project=('LLLM/' + original_save_path).replace('/', '|'), name=config.save_path.replace('/', '|'), config=asdict(config) | stats, dir=os.path.join(config.save_path, 'wandb_logs'))

    stime = time.time()
    tot_sample_trained = 0

    # fuck antlr4
    soft_rlimit, hard_rlimit = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (min(1048576 * 1024, hard_rlimit), hard_rlimit))

    # main loop
    try:
        for now_step in tqdm(range(start_step, config.trainer.tot_steps), desc='training'):
            # TODO: support other LR
            now_lr = config.trainer.learning_rate

            # indexing the training data
            data_indexes = [idx % len(train_datasets) for idx in range(now_step * n_samp_per_batch, (now_step + 1) * n_samp_per_batch)] # wrap around
            train_data_batch = [train_datasets[idx] for idx in data_indexes]
            train_data_repeated = [x for y in [[item] * config.trainer.group_size for item in train_data_batch] for x in y] # [q1, q1, q1, q2, q2, q2, ...]
            prompts = [question_formulator(item) for item in train_data_repeated]
            ground_truths = []
            for item in train_data_repeated:
                ground_truths.append(item['**final_answer'])
            
            print('Inferencing...')
            if config.inference_backend == 'vllm':
                rollout_completions = vllm_utils.generate_completions(vllm_base_url, init_vllm_model_path, prompts, vllm_sampling_params, config.trainer.rollout_batch_size)
            elif config.inference_backend == 'lllm':
                # self batch
                rollout_completions = []
                for i in tqdm(range(0, len(prompts), config.trainer.rollout_batch_size)):
                    rollout_completions.extend(
                        generate(model, prompts[i: i + config.trainer.rollout_batch_size], tokenizer, config.trainer.rollout_max_new_tokens, config.trainer.rollout_temperature, extra_stop_tokens=config.trainer.rollout_stop_words, max_seq_len=config.trainer.max_seqlen_clipping, include_stop_str_in_output=True, verbose=False)
                    )
                torch.cuda.synchronize()
            elif config.inference_backend == 'sft':
                # extract
                rollout_completions = [LCompletion(prompt=prompt, 
                                                   text=full_item[config.inference_sft_field_name] + (tokenizer.eos_token if tokenizer.eos_token is not None else ''), token_ids=None, finish_reason='stop') 
                                        for prompt, full_item in zip(prompts, train_data_repeated)]
            else:
                raise NotImplementedError
            rollout_responses = [c.text if c.text else ' ' for c in rollout_completions]
            response_token_ids = [c.token_ids if c.token_ids else [0] for c in rollout_completions] if len(rollout_completions) > 0 and rollout_completions[0].token_ids else None
            # add thing to prevent empty response which results in div0 error.

            rollout_metadata = ({
                'length/mean': sum([len(c.token_ids) for c in rollout_completions]) / len(rollout_completions) } if rollout_completions and rollout_completions[0].token_ids else { }) | {
                    'length/truncate_ratio': sum([c.finish_reason != 'stop' for c in rollout_completions]) / len(rollout_completions)
                }

            print('Training...')
            # fully online RL
            loss, metadata = grpo_train_step(model, tokenizer, optimizer, config.trainer.gradient_accumulation_steps, config.trainer.gradient_clipping, task_grader, prompts, rollout_responses, ground_truths, config.trainer.group_size, config.trainer.baseline, config.trainer.advantage_eps, config.trainer.advantage_normalizer, "none", None, config.trainer.cliprange, config.trainer.clip_higher_range, config.trainer.loss_normalization, config.trainer.normalization_constant, response_token_ids=response_token_ids, max_seqlen_clipping=config.trainer.max_seqlen_clipping, is_sft=config.inference_backend == 'sft') # response_token_ids for debug

            print('Logging...')
            train_metadata = metadata | rollout_metadata | {'loss': loss.item(), 'time': time.time() - stime, 'lr': now_lr, 'epoch': (now_step + 1) * n_samp_per_batch / len(train_datasets)}
            train_metadata = {('train/' + k): v for k, v in train_metadata.items()}
            print('\n'.join([k + '=' + (f'{train_metadata[k].item()}' if isinstance(train_metadata[k], torch.Tensor) else f'{train_metadata[k]}') for k in ['train/loss', 'train/grad_norm', 'train/token_entropy', 'train/length/mean', 'train/reward/mean', 'train/min_response_len', 'train/max_response_len'] if k in train_metadata]))
            wandb.log(train_metadata, step=now_step)

            print('Weight sync...')
            if config.inference_backend == 'vllm':
                # no need to re-init
                vllm_utils.sync_policy_weights(model, vllm_base_url, weight_sync_group, weight_format_converter)
            elif config.inference_backend == 'lllm':
                torch.cuda.synchronize()
            elif config.inference_backend == 'sft':
                torch.cuda.synchronize()
            else:
                raise NotImplementedError

            if now_step % config.val_step == 0 or now_step == config.trainer.tot_steps - 1:
                print('Validation...')
                eval_config.run_suffix = f'rl_eval_step{now_step}'
                eval_config.save_dir = str(save_path / 'val' / f'step_{now_step:07}')
                overall_stats = None
                val_metadata = {}
                if config.inference_backend == 'vllm':
                    overall_stats, val_details = task_set_grader(eval_config, True, False, False, None, None)
                elif config.inference_backend == 'lllm' or (config.inference_backend == 'sft' and config.inference_sft_run_lllm_eval):
                    overall_stats, val_details = task_set_grader(eval_config, True, False, True, model, tokenizer)
                elif config.inference_backend == 'sft':
                    pass
                else:
                    raise NotImplementedError
                if overall_stats:
                    print(overall_stats)
                    if config.task_name == 'gsm8k':
                        for kk in ['pass1', 'passn']:
                            for kkk in ['reward', 'answer_reward', 'format_reward', 'stopped', 'ans_len']:
                                aggregated_num = sum([item[kk][kkk] for item in val_details]) / len(val_details)
                                val_metadata[f'val/{kk}/{kkk}'] = aggregated_num
                    else:
                        raise NotImplementedError
                    
                if config.inference_backend == 'sft':
                    new_overall_stats, new_val_details = compute_valloss(eval_config.batch_size, model, tokenizer, val_datasets, config.inference_sft_field_name, question_formulator)
                    val_metadata |= {f'val/{kk}': vv for kk, vv in new_val_details.items()}
                    print(new_val_details)
                
                wandb.log(val_metadata, step=now_step)

            if now_step % config.rollout_save_step == 0 or now_step == config.trainer.tot_steps - 1:
                print('Saving current rollout...')
                rollouts = []
                for prompt, ground_truth, completion in zip(prompts, ground_truths, rollout_completions):
                    rollouts.append({
                            'prompt': prompt,
                            'ground_truth': ground_truth,
                            'completion': completion.text,
                            'completion_reason': completion.finish_reason
                        } | ({} if completion.token_ids is None else {
                                'completion_len': len(completion.token_ids)
                            }))
                trace_save_path = save_path / 'train' / f'step_{now_step:07}'
                if not os.path.exists(trace_save_path):
                    os.makedirs(trace_save_path)
                with open(trace_save_path / 'traces.jsonl', 'w') as f:
                    for rollout in rollouts:
                        json.dump(rollout, f)
                        print('', file=f) # add \n
                with open(trace_save_path / 'metrics.json', 'w') as f:
                    json.dump({k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in train_metadata.items()}, f, indent=2)

            if now_step % config.ckpt_save_step == 0 or now_step == config.trainer.tot_steps - 1:
                print('Saving model checkpoints...')
                ckpt_save_path = save_path / 'ckpts' / f'step_{now_step:07}.pth'
                if not os.path.exists(save_path / 'ckpts'):
                    os.makedirs(save_path / 'ckpts')
                save_checkpoint(model, optimizer, now_step, ckpt_save_path)
                print(f'Checkpoint saved to {ckpt_save_path}')

                if now_step == config.trainer.tot_steps - 1:
                    print(json.dumps({k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in train_metadata.items()}, indent=2))

                    if config.base_model_format == 'olmo_hf':
                        from basics.lmodeling_olmo import lllm_ckpt_to_hf_ckpt
                        lllm_ckpt_to_hf_ckpt(str(ckpt_save_path), config.base_model_ckpt, str(save_path / 'hf_ckpts' / f'step_{now_step:07}'))
                        print(f'Huggingface format checkpoint saved to', str(save_path / 'hf_ckpts' / f'step_{now_step:07}'))

    finally:
        # sanitize
        resource.setrlimit(resource.RLIMIT_STACK, (soft_rlimit, hard_rlimit))
        if config.inference_backend == 'vllm':
            vllm_utils.stop_server(inference_serv_proc)

    print('Done!')
