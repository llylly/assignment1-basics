"""
    Additional components to support OLMo models
"""
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Literal
import tyro
import json
import safetensors
import torch
from basics.ltokenizer import LTokenizer
from basics.lmodeling import *

class LOlmo2TransformerLM(LTransformerLM):

    def __init__(self, hidden_size: int, intermediate_size: int, max_position_embeddings: int, num_attention_heads: int, num_hidden_layers: int, num_key_value_heads: int, rms_norm_eps: float, rope_theta: int, tie_word_embeddings: bool, vocab_size: int, flash_attn: bool=False, device: torch.device | None = None, torch_dtype: torch.dtype | None = None) -> None:

        dummy = 0
        def customized_layer_constructor():
            return LTransformerBlock(
                hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, max_position_embeddings, 
                device, torch_dtype, 
                partial_post_norm=True,
                rms_norm_eps=rms_norm_eps, 
                attn_qk_norm=True,
                flash_attn=flash_attn,
                param_maps={'ln1': 'post_attention_layernorm', 'ln2': 'post_feedforward_layernorm', 'attn': 'self_attn', 'ffn': 'mlp'}, 
                ffn_param_maps={'w1': 'gate_proj', 'w3': 'up_proj', 'w2': 'down_proj'}, 
                attn_param_maps={'output_proj': 'o_proj'})

        super().__init__(hidden_size, num_attention_heads, dummy, max_position_embeddings, vocab_size, num_hidden_layers, rope_theta, rms_norm_eps, device, torch_dtype,
            customized_layer_constructor=customized_layer_constructor,
            interleave_rope=False,
            flash_attn=flash_attn,
            param_maps={
                'token_embeddings': 'embed_tokens',
                'ln_final': 'norm',
            })

        if tie_word_embeddings:
            getattr(self, self.param_maps['lm_head']).weight = getattr(self, self.param_maps['token_embeddings']).weight
    
def from_pretrained(model_dir: str, dtype=None, device='cuda', flash_attn=False) -> tuple[LOlmo2TransformerLM, LTokenizer]:
    supported_config_params = [
        'hidden_size',
        'intermediate_size',
        'max_position_embeddings',
        'num_attention_heads',
        'num_hidden_layers',
        'num_key_value_heads',
        'rms_norm_eps',
        'rope_theta',
        'tie_word_embeddings',
        'torch_dtype',
        'vocab_size'
    ]

    # load tokenizer
    tokenizer = LTokenizer.from_hf_tokenizers(model_dir)

    # construct model
    config_path = os.path.join(model_dir, 'config.json')
    
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    architecture = model_config['architectures'][0]
    assert architecture == 'Olmo2ForCausalLM'
    model_config = {k: v for k, v in model_config.items() if k in supported_config_params}
    if dtype is not None: # overwrite by args if exists
        model_config['torch_dtype'] = dtype
    torch_dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float}[model_config['torch_dtype']]
    model_config |= {'device': device, 'torch_dtype': torch_dtype, 'flash_attn': flash_attn}
    print(model_config)
    model = LOlmo2TransformerLM(**model_config)
    model.resource_count(1, 4096)

    # load weights
    tensor_index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    with open(tensor_index_path, 'r') as f:
        data = json.load(f)
        tensor_files = list(set(data['weight_map'].values()))
    weights = {}
    for tensor_file in tensor_files:
        with safetensors.safe_open(os.path.join(model_dir, tensor_file), framework='pt', device='cpu') as f:
            for k in f.keys():
                # remove starting `model.`
                if k.startswith('model.'):
                    weights[k[len('model.'):]] = f.get_tensor(k)
                else:
                    weights[k] = f.get_tensor(k)
    numel = 0
    for v in weights.values():
        numel += v.numel()

    for k in weights:
        weights[k] = weights[k].type(torch_dtype).to(device)
    model.load_state_dict(weights)
    del weights

    return model, tokenizer


# weight converters
def lolmo2_to_vllm_weights_converter(lweights: Iterable[tuple[str, torch.nn.Parameter | torch.Tensor]]) -> list[tuple[str, torch.nn.Parameter | torch.Tensor]]:
    # first, remove the side effect of torch.compile
    lweight_dict = dict([(k.replace('._orig_mod', ''), v) for k, v in lweights])
    assert 'embed_tokens.weight' in lweight_dict
    assert 'norm.weight' in lweight_dict
    assert 'lm_head.weight' in lweight_dict
    num_layers = 0
    while True:
        if any([not (f'layers.{num_layers}.{item}.weight' in lweight_dict) for item in [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.q_norm',
            'self_attn.k_norm',
            'self_attn.o_proj',
            'post_attention_layernorm',
            'mlp.gate_proj',
            'mlp.up_proj',
            'mlp.down_proj',
            'post_feedforward_layernorm',
        ]]):
            break
        num_layers += 1
    assert len(list(lweights)) == 3 + 11 * num_layers

    ans: list[tuple[str, torch.nn.Parameter | torch.Tensor]] = [
        ('model.embed_tokens.weight', lweight_dict['embed_tokens.weight']),
        ('model.norm.weight', lweight_dict['norm.weight']),
        ('lm_head.weight', lweight_dict['lm_head.weight'])
    ]
    for l in range(num_layers):
        ans.extend([
            (f'model.layers.{l}.self_attn.q_norm.weight', lweight_dict[f'layers.{l}.self_attn.q_norm.weight']),
            (f'model.layers.{l}.self_attn.k_norm.weight', lweight_dict[f'layers.{l}.self_attn.k_norm.weight']),
            (f'model.layers.{l}.self_attn.o_proj.weight', lweight_dict[f'layers.{l}.self_attn.o_proj.weight']),
            (f'model.layers.{l}.post_attention_layernorm.weight', lweight_dict[f'layers.{l}.post_attention_layernorm.weight']),
            (f'model.layers.{l}.mlp.down_proj.weight', lweight_dict[f'layers.{l}.mlp.down_proj.weight']),
            (f'model.layers.{l}.post_feedforward_layernorm.weight', lweight_dict[f'layers.{l}.post_feedforward_layernorm.weight']),

            (f'model.layers.{l}.self_attn.q_proj.weight', lweight_dict[f'layers.{l}.self_attn.q_proj.weight']),
            (f'model.layers.{l}.self_attn.k_proj.weight', lweight_dict[f'layers.{l}.self_attn.k_proj.weight']),
            (f'model.layers.{l}.self_attn.v_proj.weight', lweight_dict[f'layers.{l}.self_attn.v_proj.weight']),
            (f'model.layers.{l}.mlp.gate_proj.weight', lweight_dict[f'layers.{l}.mlp.gate_proj.weight']),
            (f'model.layers.{l}.mlp.up_proj.weight', lweight_dict[f'layers.{l}.mlp.up_proj.weight']),
            # VLLM secretly merge internally, so we don't need to care stacking by ourselves
            # (f'model.layers.{l}.self_attn.qkv_proj.weight', torch.cat([lweight_dict[f'layers.{l}.self_attn.{item}.weight'] for item in ['q_proj', 'k_proj', 'v_proj']], dim=0)),
            # (f'model.layers.{l}.mlp.gate_up_proj.weight', torch.cat([lweight_dict[f'layers.{l}.mlp.{item}.weight'] for item in ['gate_proj', 'up_proj']], dim=0))
        ])

    return ans


def to_hf_pretrained(model: LOlmo2TransformerLM, tokenizer: LTokenizer, output_dir: str):
    import safetensors
    import safetensors.torch
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer.save_hf_pretrained(output_dir)
    converted_weights = lolmo2_to_vllm_weights_converter(list(model.named_parameters()))
    converted_weights_dict = dict(converted_weights)
    safetensors.torch.save_file(converted_weights_dict, Path(output_dir) / 'model-00001-of-00001.safetensors')
    index_dict = {
        'metadata': {'total_size': model.count_parameters()[0] * 2}, # assume bf16
        'weight_map': {item: 'model-00001-of-00001.safetensors' for item, _ in converted_weights}
    }
    with open(Path(output_dir) / 'model.safetensors.index.json', 'w') as f:
        json.dump(index_dict, indent=2, fp=f)
    config_dict = {
        "architectures": [
            "Olmo2ForCausalLM"
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": converted_weights_dict['model.embed_tokens.weight'].shape[1],
        "initializer_range": 0.02,
        "intermediate_size": converted_weights_dict['model.layers.0.mlp.up_proj.weight'].shape[0],
        "max_position_embeddings": model.max_seq_len,
        "model_type": "olmo2",
        "num_attention_heads": getattr(getattr(model, model.param_maps['layers'])[0],getattr(model, model.param_maps['layers'])[0].param_maps['attn']).num_attention_heads,
        "num_hidden_layers": len(getattr(model, model.param_maps['layers'])),
        "num_key_value_heads": getattr(getattr(model, model.param_maps['layers'])[0],getattr(model, model.param_maps['layers'])[0].param_maps['attn']).num_key_value_heads,
        "rms_norm_eps": getattr(getattr(model, model.param_maps['layers'])[0],getattr(model, model.param_maps['layers'])[0].param_maps['ln1']).eps,
        "rope_scaling": None,
        "rope_theta": model.rope_cache.theta,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.50.3",
        "use_cache": True,
        "vocab_size": converted_weights_dict['model.embed_tokens.weight'].shape[0]
    }
    with open(Path(output_dir) / 'config.json', 'w') as f:
        json.dump(config_dict, indent=2, fp=f)

def lllm_ckpt_to_hf_ckpt(in_model_ckpt_pth: str, in_base_hf_model_path: str, out_dir: str):
    model, tokenizer = from_pretrained(in_base_hf_model_path, dtype='bfloat16', device='cpu') # by default write in bf16 to save space
    model.load_state_dict(torch.load(in_model_ckpt_pth, map_location='cpu')['model'])
    return to_hf_pretrained(model, tokenizer, out_dir)


@dataclass
class Usage:
    intent: Literal['inference', 'convert'] = 'inference'
    model_path: str = 'models/OLMo-2-0425-1B'
    base_model_path: str | None = None
    output_path: str | None = None
    device: str = 'cuda'

"""
Usage example:
uv run python basics/lmodeling_olmo.py
uv run python basics/lmodeling_olmo.py --model-path models/rl/olmo2_1B_gsm8k/base_rl_r1zero_20260727_141916/hf_ckpts/step_0000150
uv run python basics/lmodeling_olmo.py --intent convert --model-path models/rl/olmo2_1B_gsm8k/base_rl_r1zero_20260727_141916/ckpts/step_0000199.pth --base-model-path models/OLMo-2-0425-1B --output-path models/rl/olmo2_1B_gsm8k/base_rl_r1zero_20260727_141916/hf_ckpts/step_0000199
"""
if __name__ == '__main__':
    usage = tyro.cli(Usage)
    if usage.intent == 'inference':
        model, tokenizer = from_pretrained(usage.model_path, 'bfloat16', device=device, flash_attn=True)
        from basics.linference import generate
        while True:
            prompt = input('\n> ')
            if not prompt:
                break
            ret = generate(model, [prompt], tokenizer, max_new_tokens=1024, temperature=0.2)
            # print(ret)
    elif usage.intent == 'convert':
        assert usage.base_model_path
        assert usage.output_path
        lllm_ckpt_to_hf_ckpt(usage.model_path, usage.base_model_path, usage.output_path)
    else:
        raise NotImplementedError
