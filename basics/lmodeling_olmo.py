"""
    Additional components to support OLMo models
"""
import os
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

if __name__ == '__main__':
    model, tokenizer = from_pretrained('models/OLMo-2-0425-1B', 'bfloat16', flash_attn=True)
    from basics.linference import generate
    while True:
        prompt = input('\n> ')
        if not prompt:
            break
        ret = generate(model, [prompt], tokenizer, max_new_tokens=1024, temperature=0.2)
        # print(ret)
