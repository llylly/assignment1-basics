from typing import Callable, Literal
from dataclasses import dataclass
import torch
import transformers
from basics.ltokenizer import LTokenizer
from basics.ltrain import LGradientClipping
from basics.lmodeling import LTransformerLM
from alignment.lrl_utils import *


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
class RLConfig:
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
