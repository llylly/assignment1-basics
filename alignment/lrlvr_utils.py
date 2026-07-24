from typing import Callable, Tuple, Literal
from numpy import std
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from basics.ltokenizer import LTokenizer
from basics.lmodeling import LTransformerLM

def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase | LTokenizer
    ) -> dict[str, torch.Tensor]:
    tokenized_prompts = []
    tokenized_outputs = []
    max_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        print(prompt, output)
        tok_prompt = tokenizer.encode(prompt)
        tok_output = tokenizer.encode(output)
        tokenized_prompts.append(tok_prompt)
        tokenized_outputs.append(tok_output)
        print(tok_prompt + tok_output)
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
    else:
        Z1 = logits.exp()
        Z2 = Z1.sum(dim=-1)
        all_probs = Z1 / Z2.unsqueeze(dim=-1)
        Z = Z2.log()
        log_probs = logits.gather(dim=-1, index=labels.unsqueeze(dim=-1)).squeeze(dim=-1) - Z
        all_log_probs = logits - Z.unsqueeze(dim=-1)
    if return_token_entropy:
        return {
            'log_probs': log_probs,
            'token_entropy': -(all_probs * all_log_probs).sum(dim=-1)
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
        'mean/reward': sum([item['reward'] for item in reward_fn_rets]) / len(rollout_responses),
        'mean/format_reward':  sum([item['format_reward'] for item in reward_fn_rets]) / len(rollout_responses),
        'mean/answer_reward':  sum([item['answer_reward'] for item in reward_fn_rets]) / len(rollout_responses),
        'max/reward':  max([item['reward'] for item in reward_fn_rets]),
        'max/format_reward':  max([item['format_reward'] for item in reward_fn_rets]),
        'max/answer_reward':  max([item['answer_reward'] for item in reward_fn_rets]),
        'min/reward':  min([item['reward'] for item in reward_fn_rets]),
        'min/format_reward':  min([item['format_reward'] for item in reward_fn_rets]),
        'min/answer_reward':  min([item['answer_reward'] for item in reward_fn_rets]),
        'std/reward':  float(std([item['reward'] for item in reward_fn_rets], ddof=1)),
        'std/format_reward':  float(std([item['format_reward'] for item in reward_fn_rets], ddof=1)),
        'std/answer_reward':  float(std([item['answer_reward'] for item in reward_fn_rets], ddof=1)),
    }
    return raw_rewards, metadata

def compute_group_normalized_rewards(
        raw_rewards: torch.Tensor,
        group_size: int,
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std"
    ):
    print(raw_rewards)
    group_viewed = raw_rewards.view(-1, group_size)
    group_mean = group_viewed.mean(dim=-1, keepdim=True)
    group_std = (group_viewed).std(dim=-1, unbiased=True, keepdim=True) # but the previous text said to use /n. Here to pass the unit test I use /(n-1).
    print(group_size, group_viewed, group_mean, group_std)
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
    advantages = (new_group_viewed / denominator).view(-1)
    metadata = {
        'mean/advantage': advantages.mean().item(),
        'max/advantage':  advantages.max().item(),
        'min/advantage':  advantages.min().item(),
        'std/advantage':  advantages.std(unbiased=True).item(),
    }
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
        loss = - raw_rewards_or_advantages * policy_log_probs
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

