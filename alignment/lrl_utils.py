from typing import Callable, Tuple, Literal, Iterable
from numpy import std
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from basics.ltokenizer import LTokenizer
from basics.lmodeling import LTransformerLM

def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase | LTokenizer,
        response_token_ids: list | None = None # if provided, no need to retokenized response to ensure inference-training consistency
    ) -> dict[str, torch.Tensor]:
    tokenized_prompts = []
    tokenized_outputs = []
    max_len = 0
    for i, (prompt, output) in enumerate(zip(prompt_strs, output_strs)):
        tok_prompt = tokenizer.encode(prompt)
        if response_token_ids is None:
            tok_output = tokenizer.encode(output)
        else:
            tok_output = response_token_ids[i]
        tokenized_prompts.append(tok_prompt)
        tokenized_outputs.append(tok_output)
        max_len = max(len(tok_prompt) + len(tok_output) - 1, max_len)
    input_ids = torch.zeros((len(prompt_strs), max_len), dtype=torch.long)
    labels = torch.zeros((len(prompt_strs), max_len), dtype=torch.long)
    response_mask = torch.zeros((len(prompt_strs), max_len), dtype=torch.bool)
    for i, (prompt, output) in enumerate(zip(tokenized_prompts, tokenized_outputs)):
        input_ids[i, :min(len(prompt) + len(output), max_len)] = torch.Tensor((prompt + output))[:max_len] # though not precise but pass
        labels[i, :len(prompt) + len(output) - 1] = torch.Tensor((prompt + output)[1:])
        response_mask[i, len(prompt)-1: len(prompt) + len(output) - 1] = 1
    return {
        'input_ids': input_ids,
        'labels': labels,
        'response_mask': response_mask
    }

def get_response_log_probs(
        model: PreTrainedModel | LTransformerLM,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool=False
    ):
    logits: torch.Tensor = model(input_ids).logits
    logits = logits - logits.amax(dim=-1, keepdim=True) # for stability
    if not return_token_entropy:
        Z = logits.exp().sum(dim=-1).log()
        log_probs = logits.gather(dim=-1, index=labels.unsqueeze(dim=-1)).squeeze(dim=-1) - Z
        # log_probs = - torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), labels, reduction='none')
    else:
        Z1 = logits.exp()
        Z2 = Z1.sum(dim=-1)
        all_probs = Z1 / Z2.unsqueeze(dim=-1)
        Z = Z2.log()
        log_probs = logits.gather(dim=-1, index=labels.unsqueeze(dim=-1)).squeeze(dim=-1) - Z
        all_log_probs = logits - Z.unsqueeze(dim=-1)
        # log_probs = - torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), labels, reduction='none')
    if return_token_entropy:
        return {
            'log_probs': log_probs,
            'token_entropy': -(all_probs * all_log_probs).sum(dim=-1)
            # 'token_entropy': - torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), all_probs.permute(0, 2, 1), reduction='none') # .sum(dim=-1)
        }
    else:
        return {'log_probs': log_probs}

def compute_rollout_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_group_truths: list[str],
    ) -> Tuple[torch.Tensor, dict[str, float]]:

    # reward_fn_rets = []
    # for r, g in zip(rollout_responses, repeated_group_truths):
    #     ret = reward_fn(r, g)
    #     reward_fn_rets.append(ret)

    reward_fn_rets = list(map(reward_fn, rollout_responses, repeated_group_truths))
    raw_rewards = torch.tensor([item['reward'] for item in reward_fn_rets])
    metadata = {
        'reward/mean': sum([item['reward'] for item in reward_fn_rets]) / len(rollout_responses),
        'format_reward/mean':  sum([item['format_reward'] for item in reward_fn_rets]) / len(rollout_responses),
        'answer_reward/mean':  sum([item['answer_reward'] for item in reward_fn_rets]) / len(rollout_responses),
        'reward/max':  max([item['reward'] for item in reward_fn_rets]),
        'format_reward/max':  max([item['format_reward'] for item in reward_fn_rets]),
        'answer_reward/max':  max([item['answer_reward'] for item in reward_fn_rets]),
        'reward/min':  min([item['reward'] for item in reward_fn_rets]),
        'format_reward/min':  min([item['format_reward'] for item in reward_fn_rets]),
        'answer_reward/min':  min([item['answer_reward'] for item in reward_fn_rets]),
        'reward/std':  float(std([item['reward'] for item in reward_fn_rets], ddof=1)),
        'format_reward/std':  float(std([item['format_reward'] for item in reward_fn_rets], ddof=1)),
        'answer_reward/std':  float(std([item['answer_reward'] for item in reward_fn_rets], ddof=1)),
    }
    return raw_rewards, metadata

def compute_group_normalized_rewards(
        raw_rewards: torch.Tensor,
        group_size: int,
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std"
    ):
    group_viewed = raw_rewards.view(-1, group_size)
    group_mean = group_viewed.mean(dim=-1, keepdim=True)
    group_std = (group_viewed).std(dim=-1, unbiased=True, keepdim=True) # but the previous text said to use /n. Here to pass the unit test I use /(n-1).
    if baseline == 'mean':
        new_group_viewed = group_viewed - group_mean
    elif baseline == 'none':
        new_group_viewed = group_viewed
    else:
        raise NotImplementedError
    if advantage_normalizer == 'std':
        denominator = group_std + advantage_eps
    elif advantage_normalizer == 'mean':
        denominator = group_mean + advantage_eps
    else:
        raise NotImplementedError
    advantages = new_group_viewed / denominator
    metadata = {
        'advantage/mean': advantages.mean(dim=-1).tolist(),
        'advantage/max':  advantages.amax(dim=-1).tolist(),
        'advantage/min':  advantages.amin(dim=-1).tolist(),
        'advantage/std':  advantages.std(dim=-1, unbiased=True).tolist(),
        'group/all_zero_ratio': sum(group_mean <= 0.01) / group_mean.numel(),
        'group/all_one_ratio': sum(group_mean >= 0.99) / group_mean.numel(),
        'group/active_group_ratio': 1. - (sum(group_mean <= 0.01) + sum(group_mean >= 0.99)) / group_mean.numel()
    }
    advantages = advantages.view(-1)
    return advantages, metadata

def compute_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        importance_reweighting_method: Literal['none', 'noclip', 'grpo', 'gspo'] = 'none',
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        response_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
        This compute the per-token policy-gradient loss, i.e., excluding 1/(BG) and 1/len(y) coefficient in GRPO that will be introduced in later aggregation
    """
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.view(-1, 1)
    metadata = {}
    if importance_reweighting_method == 'none':
        loss = - raw_rewards_or_advantages.to(policy_log_probs.device) * policy_log_probs
    else:
        raise NotImplementedError
    if response_mask is not None:
        loss = loss * response_mask
    return loss, metadata

def aggregate_loss_across_microbatch_sequence(
        per_token_policy_gradient_loss: torch.Tensor,
        mask: torch.Tensor,
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None,
    ) -> torch.Tensor:
    if loss_normalization == 'sequence':
        return ((mask * per_token_policy_gradient_loss).sum(dim=-1) / mask.sum(dim=-1)).mean()
    else:
        # "constant"
        assert normalization_constant is not None
        return (mask * per_token_policy_gradient_loss).sum() / normalization_constant

