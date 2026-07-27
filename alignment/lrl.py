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
from basics.ltrain_utils import dict_to_dataclass, load_checkpoint
from basics.lmodeling import LTransformerLM
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
    """If none, equal to batch_size, otherwise whould be a divisor of batch_size"""
    max_seqlen_clipping: int | None = None
    """If none, clip to model's context length"""

    # Reward normalization
    baseline: Literal["mean", "none"] = "mean"
    advantage_eps: float = 1e-6
    advantage_normalizer: Literal["std", "none", "mean"] = "std"
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none"
    cliprange: float | None = None
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence"
    normalization_constant: int | None = None

@dataclass
class RLConfig:
    trainer: RLTrainerConfig
    base_model_ckpt: str
    """Huggingface-style or LLLM-style base model dir, contain all information such as weights, tokenizer, and the architecture str"""
    base_model_format: Literal['olmo_hf', 'olmo_lllm']
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
    # """if base_model_format == 'olmo_lllm', need to specify tokenizer_path"""
    model_config: str | None = None
    # """if base_model_format == 'olmo_lllm', need to specify model_config"""
    inference_backend: Literal['vllm', 'lllm'] = 'vllm'
    inference_device: str='cuda:1'
    inference_vllm_server_no_host: bool = False # in this case, assume the host is already there
    inference_vllm_ip: str = '127.0.0.1'
    inference_vllm_port: int = 8080
    inference_vllm_seed: int | None = None #42
    inference_vllm_dummy_model_path: str | None = None
    inference_vllm_gpu_memory_utilization: float = 0.8
    # if base_model_format == 'olmo_lllm', we need a dummy hf model path to launch vllm inference engine
    run_name: str | None = ''
    """run_name is appended to both save_path and wandb"""
    val_step: int = 20
    rollout_save_step: int = 10
    ckpt_save_step: int = 50
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
    for i in tqdm(range(0, macro_batch_size, micro_batch_size)):
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
        'grad_norm': grad_norm,
        'ctx_len': input_ids.shape[-1]
    } | raw_rewards_metadata_agg | advantage_metadata_agg

    return batch_loss, metadata

"""
Usage:
uv run alignment/lrl.py --config_path configs/rl_config_olmo_base_gsm8k_r1zero.yaml --device cuda:0 --inference_device cuda:3
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
    if config.base_model_format == 'olmo_lllm':
        assert config.inference_vllm_dummy_model_path is not None, "if base_model_format == 'olmo_lllm', we need a dummy hf model path to launch vllm inference engine"
        assert config.tokenizer_path is not None, "if base_model_format == 'olmo_lllm', need to specify tokenizer_path"
        assert config.model_config is not None, "if base_model_format == 'olmo_lllm', need to specify model_config"
    assert config.inference_backend == 'vllm', 'backend engine from our own lllm is not supported yet until I know how to dynamically update engine weights'
    assert config.inference_device.startswith('cuda:'), 'VLLM inference device should be of the format cuda:X'
    if config.trainer.rollout_batch_size is not None:
        assert config.trainer.batch_size % config.trainer.rollout_batch_size == 0

    nowtime = datetime.now().strftime('_%Y%m%d_%H%M%S')
    original_save_path = config.save_path
    if config.save_path.endswith('/'):
        config.save_path = config.save_path[:-1]
    if config.run_name:
        config.save_path += '_' + config.run_name
    config.save_path += nowtime
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    with open(Path(config.save_path) / 'configs.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(json.dumps(asdict(config), indent=2))
    print('Things saved to', config.save_path)
    save_path = Path(config.save_path) # for later convenience

    # construct model and optimizer
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
    else:
        start_step = 0

    config.trainer.max_seqlen_clipping = model.max_seq_len
    if config.trainer.rollout_batch_size is None:
        config.trainer.rollout_batch_size = config.trainer.batch_size

    # data loading
    # now no shuffle to ensure reproducibility
    with open(config.prompt_template, 'r') as f:
        chat_template = f.read()
    train_datasets: list[dict] = []
    val_datasets: list[dict] = []
    with open(config.train_data, 'r') as f:
        for line in f.readlines():
            if line.strip():
                train_datasets.append(json.loads(line))
    # with open(config.val_data, 'r') as f:
    #     for line in f.readlines():
    #         if line.strip():
    #             val_datasets.append(json.loads(line))

    # launch vllm server, then replace dummy parameter with the real one
    print('Setting up vllm server...')
    if config.inference_backend == 'vllm':
        if config.base_model_format == 'olmo_lllm':
            init_vllm_model_path = config.inference_vllm_dummy_model_path
        else:
            init_vllm_model_path = config.base_model_ckpt
        inference_device_no = int(config.inference_device[5:])
        rank_offset = 1
        vllm_base_url = f'http://{config.inference_vllm_ip}:{config.inference_vllm_port}'
        weight_format_converter = None
        if config.base_model_format in ['olmo_lllm', 'olmo_hf']:
            weight_format_converter = lolmo2_to_vllm_weights_converter

        vllm_sampling_params = {
            'temperature': config.trainer.rollout_temperature,
            'max_tokens': config.trainer.rollout_max_new_tokens,
            'n': 1,
            'seed': config.inference_vllm_seed,
            'stop': config.trainer.rollout_stop_words,
            'include_stop_str_in_output': True,
        }

        inference_serv_proc = vllm_utils.start_server(init_vllm_model_path, config.inference_vllm_ip, config.inference_vllm_port, config.dtype, inference_device_no, config.inference_vllm_seed, "auto", 'INFO', config.inference_vllm_gpu_memory_utilization)
        vllm_utils.wait_for_server(vllm_base_url, inference_serv_proc, 300) # wait for 300s
        print('Inference server launched...\nNow sync up initial weights')

        weight_sync_group = vllm_utils.init_weight_sync(vllm_base_url, config.device, rank_offset)
        vllm_utils.sync_policy_weights(model, vllm_base_url, weight_sync_group, weight_format_converter)
    else:
        raise NotImplementedError

    # task-specific formulator, grader, and valgrader fns
    question_formulator: Callable[[dict, str], str] = get_question_formulator(config.task_name)
    eval_config_class, task_set_grader = get_testable_task_setgrader(config.task_name)
    if config.task_name == 'gsm8k':
        # need to first determine the prompt type
        guessed_prompt_type = config.prompt_template.split('/')[-1].split('.')[0] # not extensible, but now works
        assert guessed_prompt_type in ['r1_zero', 'question_only', 'r1_zero_three_shot_gsm8k']
        task_grader = get_task_grader(config.task_name, prompt_type=guessed_prompt_type)
        from alignment.benchmarks.lgsm8k_eval import GSM8KEvalConfig
        eval_config = GSM8KEvalConfig(
            backend=config.inference_backend,
            dtype=config.dtype,
            prompt_type=guessed_prompt_type,
            max_new_tokens=config.trainer.rollout_max_new_tokens,
            temperature=config.trainer.rollout_temperature,
            n=config.trainer.group_size,
            batch_size=config.trainer.rollout_batch_size,
            model_dir=init_vllm_model_path,
            data_file=config.val_data,
            run_suffix='rl_eval_stepX',
            save_dir='', # will be overwritten
            vllm_server_no_host=True,
            vllm_ip=config.inference_vllm_ip,
            vllm_port=config.inference_vllm_port
        )
    else:
        raise NotImplementedError

    n_samp_per_batch = int(config.trainer.batch_size // config.trainer.group_size)
    stats = {
        'stat_model_param': model.count_parameters()[0],
        'stat_model_non_embed_param': model.count_parameters()[1],
        'stat_dataset_len': len(train_datasets),
        'stet_caldataset_len': len(val_datasets),
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
            prompts = [question_formulator(item, chat_template) for item in train_data_repeated]
            ground_truths = []
            if config.task_name == 'gsm8k':
                for item in train_data_repeated:
                    final_answer = item['answer'][item['answer'].find('####') + 4:].strip()
                    ground_truths.append(final_answer)
            else:
                raise NotImplementedError

            print('Inferencing...')
            if config.inference_backend == 'vllm':
                rollout_completions = vllm_utils.generate_completions(vllm_base_url, init_vllm_model_path, prompts, vllm_sampling_params, config.trainer.rollout_batch_size)
            else:
                raise NotImplementedError
            rollout_responses = [c.text for c in rollout_completions]

            rollout_metadata = {
                'length/mean': sum([len(c.token_ids) for c in rollout_completions]) / len(rollout_completions),
                'length/truncate_ratio': sum([c.finish_reason != 'stop' for c in rollout_completions]) / len(rollout_completions)
            }

            print('Training...')
            # fully online RL
            loss, metadata = grpo_train_step(model, tokenizer, optimizer, config.trainer.gradient_accumulation_steps, config.trainer.gradient_clipping, task_grader, prompts, rollout_responses, ground_truths, config.trainer.group_size, config.trainer.baseline, config.trainer.advantage_eps, config.trainer.advantage_normalizer, "none", None, config.trainer.cliprange, config.trainer.loss_normalization, config.trainer.normalization_constant)

            print('Logging...')
            train_metadata = metadata | rollout_metadata | {'loss': loss.item(), 'time': time.time() - stime}
            train_metadata = {('train/' + k): v for k, v in train_metadata.items()}
            print('\n'.join([k + '=' + (f'{train_metadata[k].item()}' if isinstance(train_metadata[k], torch.Tensor) else f'{train_metadata[k]}') for k in ['train/loss', 'train/grad_norm', 'train/token_entropy', 'train/length/mean', 'train/reward/mean']]))
            wandb.log(train_metadata, step=now_step)

            print('Weight sync...')
            if config.inference_backend == 'vllm':
                # no need to re-init
                # weight_sync_group = vllm_utils.init_weight_sync(vllm_base_url, config.device, rank_offset)
                vllm_utils.sync_policy_weights(model, vllm_base_url, weight_sync_group, weight_format_converter)
            else:
                raise NotImplementedError

            if now_step % config.val_step == 0 or now_step == config.trainer.tot_steps - 1:
                print('Validation...')
                eval_config.run_suffix = f'rl_eval_step{now_step}'
                eval_config.save_dir = str(save_path / 'val' / f'step_{now_step:07}')
                overall_stats, val_details = task_set_grader(eval_config, True, False, False)
                print(overall_stats)
                val_metadata = {}
                if config.task_name == 'gsm8k':
                    for kk in ['pass1', 'passn']:
                        for kkk in ['reward', 'answer_reward', 'format_reward', 'stopped', 'ans_len']:
                            aggregated_num = sum([item[kk][kkk] for item in val_details]) / len(val_details)
                            val_metadata[f'val/{kk}/{kkk}'] = aggregated_num
                else:
                    raise NotImplementedError
                wandb.log(val_metadata, step=now_step)

            if now_step % config.rollout_save_step == 0 or now_step == config.trainer.tot_steps - 1:
                print('Saving current rollout...')
                rollouts = []
                for prompt, ground_truth, completion in zip(prompts, ground_truths, rollout_completions):
                    rollouts.append({
                        'prompt': prompt,
                        'ground_truth': ground_truth,
                        'completion': completion.text,
                        'completion_reason': completion.finish_reason,
                        'completion_len': len(completion.token_ids)
                    })
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

            if now_step == config.trainer.tot_steps - 1:
                print(json.dumps({k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in train_metadata.items()}, indent=2))

    finally:
        # sanitize
        resource.setrlimit(resource.RLIMIT_STACK, (soft_rlimit, hard_rlimit))
        if config.inference_backend == 'vllm':
            vllm_utils.stop_server(inference_serv_proc)

    print('Done!')
