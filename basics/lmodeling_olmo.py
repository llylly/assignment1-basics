"""
    Additional components to support OLMo models
"""
import os
import json
import safetensors
import torch
from basics.ltokenizer import LTokenizer
from lmodeling import *


class LOlmo2ROPE(torch.nn.Module):

    # need to be a singleton across layers
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        bases = torch.pow(torch.tensor(theta, device=device, dtype=torch.float32), -(torch.arange(1, d_k // 2 + 1, device=device, dtype=torch.float32) * 2 - 2) / d_k)
        angles = einsum(torch.arange(0, max_seq_len, device=device, dtype=torch.float32), bases, "seq_len, bases -> seq_len bases")
        sin_angles = torch.sin(angles) # [L, d_k / 2]
        cos_angles = torch.cos(angles) # [L, d_k / 2]
        self.register_buffer('sin_angles', sin_angles)
        self.register_buffer('cos_angles', cos_angles)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # same as LROPE, but apply this non-permute slicing (if False branch in LROPE)
        x_even = x.view(-1, self.d_k)[:,:self.d_k // 2].view(list(x.shape[:-1]) + [self.d_k // 2])
        x_odd = x.view(-1, self.d_k)[:,self.d_k // 2:].view(list(x.shape[:-1]) + [self.d_k // 2])

        cos_even_x = self.cos_angles[token_positions].to(x.dtype) * x_even
        cos_odd_x = self.cos_angles[token_positions].to(x.dtype) * x_odd
        nsin_odd_x = -self.sin_angles[token_positions].to(x.dtype) * x_odd
        sin_even_x = self.sin_angles[token_positions].to(x.dtype) * x_even

        ans_even = cos_even_x + nsin_odd_x
        ans_odd = cos_odd_x + sin_even_x
        ans = torch.stack([ans_even, ans_odd], dim=-1).contiguous().reshape(x.shape)
        return ans

class LOlmo2FFN(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None, custom_kernel: bool = True):
        super().__init__()
        self.custom_kernel = custom_kernel
        self.d_model, self.d_ff = d_model, d_ff # use passed in arg
        self.gate_proj = LLinear(self.d_model, self.d_ff, device, dtype)
        self.up_proj = LLinear(self.d_model, self.d_ff, device, dtype)
        self.down_proj = LLinear(self.d_ff, self.d_model, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.custom_kernel:
            t1 = self.gate_proj(x)
            t3 = self.up_proj(x)
            return self.down_proj(torch.sigmoid(t1) * t1 * t3)
        else:
            from systems.laccelerate import LSiLUFunc
            t1 = self.gate_proj(x)
            t3 = self.up_proj(x)
            return self.down_proj(LSiLUFunc.apply(t1) * t3)

class LOlmo2MHA(torch.nn.Module):
    # global singleton
    rope_cache: LOlmo2ROPE | None = None
    triu_cache: dict[int, torch.Tensor] = {}

    def __init__(self, d_model: int, num_attention_heads: int, num_key_value_heads: int, max_seq_len: int, rms_norm_eps: float, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.d_k = self.d_model // self.num_attention_heads
        self.max_seq_len = max_seq_len
        self.theta = theta # if theta is None, no ROPE will be applied

        self.q_proj = LLinear(d_model, d_model, device, dtype)
        self.k_proj = LLinear(d_model, self.num_key_value_heads * self.d_k, device, dtype)
        self.v_proj = LLinear(d_model, self.num_key_value_heads * self.d_k, device, dtype)
        self.o_proj = LLinear(d_model, d_model, device, dtype)

        self.q_norm = LRMSNorm(d_model, rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = LRMSNorm(self.num_key_value_heads * self.d_k, rms_norm_eps, device=device, dtype=dtype)

        if LOlmo2MHA.rope_cache is None and theta is not None:
            LOlmo2MHA.rope_cache = LOlmo2ROPE(theta, self.d_k, max_seq_len, device)
        if max_seq_len not in self.triu_cache:
            self.triu_cache[max_seq_len] = torch.triu(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device)).T
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, padded_tokens: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if kv_cache is not None:
            # using kv cache
            # another latent assumption: if using kv cache, we only compute the last token's output; in the output, other token places are zeros as placeholders...
            k_cache, v_cache = kv_cache # [B, L-1, D_M']
            assert tuple(k_cache.shape) == (x.shape[0], x.shape[1]-1, self.num_key_value_heads * self.d_k)
            assert tuple(v_cache.shape) == (x.shape[0], x.shape[1]-1, self.num_key_value_heads * self.d_k)
            new_k = self.k_norm(self.k_proj(x[:, -1:])) # [B, 1, D_M']
            new_v = self.v_proj(x[:, -1:]) # [B, 1, D_M']
            k = torch.concat([k_cache, new_k], dim=1) # [B, L, D_M']
            v = torch.concat([v_cache, new_v], dim=1) # [B, L, D_M']
            q = self.q_norm(self.q_proj(x[:, -1:])) # [B, 1, D_M]
        else:
            # not using kv cache
            k = self.k_norm(self.k_proj(x))
            v = self.v_proj(x)
            q = self.q_norm(self.q_proj(x))
        q = rearrange(q, '... seqlen (h d_k) -> ... seqlen h d_k', h=self.num_attention_heads, d_k=self.d_k)
        kk = rearrange(k, '... seqlen (h d_k) -> ... seqlen h d_k', h=self.num_key_value_heads, d_k=self.d_k)
        vv = rearrange(v, '... seqlen (h d_k) -> ... h 1 seqlen d_k', h=self.num_key_value_heads, d_k=self.d_k)
        if self.theta and LOlmo2MHA.rope_cache:
            if token_positions is not None:
                token_positions = token_positions.unsqueeze(-1) # add head dim
            else:
                token_positions = torch.arange(x.shape[1]).view(-1, 1) # default index for x
            if kv_cache is not None:
                # by latent assumption, q is of shape [B, 1, H_A, D_K]
                q = LOlmo2MHA.rope_cache(q, token_positions[:, -1:])
            else:
                q = LOlmo2MHA.rope_cache(q, token_positions)
            kk = LOlmo2MHA.rope_cache(kk, token_positions)
        q = rearrange(q, '... seqlen (h i) d_k -> ... h i seqlen d_k', i=self.num_attention_heads // self.num_key_value_heads, h=self.num_key_value_heads, d_k=self.d_k) # [B, H_K, R, 1, D_K] or [B, H_K, R, L, D_K]
        kk = rearrange(kk, '... seqlen h d_k -> ... h 1 seqlen d_k', h=self.num_key_value_heads, d_k=self.d_k) # [B, H_K, 1, L, D_K]
        if kv_cache is not None:
            # by latent assumption, q is of shape [B, H_K, R, 1, D_K] pointing to the last token place
            before_proj = LNaiveSDPA(q, kk, vv, self.triu_cache[self.max_seq_len][x.shape[1]-1: x.shape[1], :x.shape[1]], padded_tokens)
        else:
            before_proj = LNaiveSDPA(q, kk, vv, self.triu_cache[self.max_seq_len][:x.shape[1], :x.shape[1]], padded_tokens)
        before_proj = rearrange(before_proj, '... h i seqlen d_k -> ... seqlen (h i d_k)') # [B, L, D_M] or [B, 1, D_M]
        output = self.o_proj(before_proj)
        if kv_cache is not None:
            # add dummy output to keep the shape a constant, but actually only the last token is inferred and valuable
            dummy = torch.zeros((output.shape[0], x.shape[1]-1, output.shape[2]), dtype=output.dtype, device=output.device)
            output = torch.concat([dummy, output], dim=1)
        return output, (k, v)

class LOlmo2TransformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, num_attention_heads: int, num_key_value_heads: int, d_ff: int, max_seq_len: int, rms_norm_eps: float, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None, compile: bool = True, custom_kernel: bool = False) -> None:
        super().__init__()
        self.d_model = d_model

        assert not (compile and custom_kernel), "Cannot use both compile and custom_kernel!"

        self._compile = compile
        self._custom_kernel = custom_kernel

        if not custom_kernel:
            self.post_attention_layernorm = LRMSNorm(d_model, rms_norm_eps, device=device, dtype=dtype)
            self.post_feedforward_layernorm = LRMSNorm(d_model, rms_norm_eps, device=device, dtype=dtype)
        else:
            # use custom kernel LRMSNorm - slower than torch.compile :(
            from systems.laccelerate import LRMSNormFast
            self.post_attention_layernorm = LRMSNormFast(d_model, rms_norm_eps, device=device, dtype=dtype)
            self.post_feedforward_layernorm = LRMSNormFast(d_model, rms_norm_eps, device=device, dtype=dtype)

        self.self_attn = LOlmo2MHA(d_model, num_attention_heads, num_key_value_heads, max_seq_len, rms_norm_eps, rope_theta, device, dtype)
        self.mlp = LOlmo2FFN(d_model, d_ff, device, dtype)

        if compile:
            self.post_attention_layernorm = torch.compile(self.post_attention_layernorm)
            self.post_feedforward_layernorm = torch.compile(self.post_feedforward_layernorm)
            self.self_attn = torch.compile(self.self_attn)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        # overwrite to handle the name mapping problems with torch.compile
        to_add_orig_mod = []
        to_remove_orig_mod = []
        if self._compile:
            to_add_orig_mod.extend(['post_attention_layernorm', 'post_feedforward_layernorm', 'self_attn'])
        else:
            to_remove_orig_mod.extend(['post_attention_layernorm', 'post_feedforward_layernorm', 'self_attn'])
        new_state_dict = {}
        name_to_delete = []
        for k, v in state_dict.items():
            for pat in to_add_orig_mod:
                if pat in k and (not pat + '._orig_mod' in k):
                    new_state_dict[k.replace(pat, pat + '._orig_mod')] = v
                    name_to_delete.append(k)
                    break
            for pat in to_remove_orig_mod:
                if (pat + '._orig_mod') in k:
                    new_state_dict[k.replace(pat + '._orig_mod', pat)] = v
                    name_to_delete.append(k)
                    break
        state_dict |= new_state_dict
        for k in name_to_delete:
            del state_dict[k]
        return super()._load_from_state_dict(new_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, padded_tokens: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None)-> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # padded_tokens: torch.bool [B, T]
        attn_x, kv_cache = self.self_attn(x, token_positions, padded_tokens, kv_cache)
        t = x + self.post_attention_layernorm(attn_x)
        return t + self.post_feedforward_layernorm(self.mlp(t)), kv_cache

class LOlmo2TransformerLM(torch.nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, max_position_embeddings: int, num_attention_heads: int, num_hidden_layers: int, num_key_value_heads: int, rms_norm_eps: float, rope_theta: int, tie_word_embeddings: bool,vocab_size: int, device: torch.device | None = None, torch_dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.vocab_size = vocab_size

        self.embed_tokens = LEmbedding(vocab_size, hidden_size, device, torch_dtype)

        self.layers = nn.Sequential(*[LOlmo2TransformerBlock(hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, max_position_embeddings, rms_norm_eps, rope_theta, device, torch_dtype) for _ in range(num_hidden_layers)])

        self.norm = LRMSNorm(hidden_size, rms_norm_eps, device=device, dtype=torch_dtype)
        self.lm_head = LLinear(hidden_size, vocab_size, device=device, dtype=torch_dtype)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.device = device
        self.torch_dtype = torch_dtype
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed_tokens(x)
        kv_caches = []
        for layer in self.layers:
            x, kv = layer(x, token_positions)
            kv_caches.append(kv)
        x = self.norm(x)
        x = self.lm_head(x)
        return x # discard kv cache for now

    def batch_generate(self, x: torch.Tensor, kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None, pad_token_id: int | None = None):
        # changes to kv cache is in place
        padded_tokens = (x == pad_token_id)
        token_positions = (torch.cumsum(x != pad_token_id, dim=1) - 1).clamp(min=0)
        x = self.embed_tokens(x, clamp_pad=pad_token_id is not None)
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            # print('layer', i)
            if kv_cache is not None:
                x, new_layer_kvcache = layer(x, token_positions, padded_tokens, kv_cache[i])
            else:
                x, new_layer_kvcache = layer(x, token_positions, padded_tokens)
            new_kv_cache.append(new_layer_kvcache)
        x = self.norm(x)
        x = self.lm_head(x)
        return x, new_kv_cache
    
    def count_parameters(self) -> tuple[int, int]: # return tot params and tot non-embed params
        tot_params = 0
        tot_embed_params = 0
        for param in self.embed_tokens.parameters():
            tot_embed_params += param.numel()
        for param in self.lm_head.parameters():
            tot_embed_params += param.numel()
        for param in self.parameters():
            tot_params += param.numel()
        return tot_params, tot_params - tot_embed_params

    def resource_count(self, batch_size, seq_len):
        tot_params = 0
        tot_embed_params = 0
        each_block_params = 0
        tot_ffn_params = 0
        tot_attn_params = 0
        tot_ln_params = 0
        for param in self.embed_tokens.parameters():
            tot_embed_params += param.numel()
        for param in self.lm_head.parameters():
            tot_embed_params += param.numel()
        for param in self.norm.parameters():
            tot_ln_params += param.numel()
        for layer in self.layers:
            if each_block_params == 0:
                for param in layer.parameters():
                    each_block_params += param.numel()
            for param in layer.post_attention_layernorm.parameters():
                tot_ln_params += param.numel()
            for param in layer.post_feedforward_layernorm.parameters():
                tot_ln_params += param.numel()
            for param in layer.self_attn.q_norm.parameters():
                tot_ln_params += param.numel()
            for param in layer.self_attn.k_norm.parameters():
                tot_ln_params += param.numel()
            for param in layer.self_attn.q_proj.parameters():
                tot_attn_params += param.numel()
            for param in layer.self_attn.k_proj.parameters():
                tot_attn_params += param.numel()
            for param in layer.self_attn.v_proj.parameters():
                tot_attn_params += param.numel()
            for param in layer.self_attn.o_proj.parameters():
                tot_attn_params += param.numel()
            for param in layer.mlp.parameters():
                tot_ffn_params += param.numel()
            
        for param in self.parameters():
            tot_params += param.numel()
        assert tot_params == tot_embed_params + tot_ln_params + tot_attn_params + tot_ffn_params
        print('      Tot # param.:', f'{tot_params:20}', f' BF16 {(tot_params * 2e-9):5.2f} GB, BF32 {(tot_params * 4e-9):5.2f} GB')
        print('Tot # embed param.:', f'{tot_embed_params:20}', '{:.2f}%'.format(tot_embed_params / tot_params * 100.))
        print('   Tot # ln param.:', f'{tot_ln_params:20}', '{:.2f}%'.format(tot_ln_params / tot_params * 100.))
        print(' Tot # attn param.:', f'{tot_attn_params:20}', '{:.2f}%'.format(tot_attn_params / tot_params * 100.))
        print('  Tot # ffn param.:', f'{tot_ffn_params:20}', '{:.2f}%'.format(tot_ffn_params / tot_params * 100.))
        print(' Each blk # param.:', f'{each_block_params:20}', '{:.2f}%'.format(each_block_params / tot_params * 100.))

        activation_memory = 0
        # embed
        activation_memory += batch_size * seq_len * self.embed_tokens.weight.shape[1]
        # each layer
        for layer in self.layers:
            # ln1
            activation_memory += batch_size * seq_len * layer.d_model # pre ln1
            activation_memory += batch_size * seq_len * layer.d_model # after ln1 for attn usage
            # attn
            activation_memory += 4 * batch_size * seq_len * layer.d_model # q,k,v,output proj
            # need to cache attn scores before and after softmax because softmax backward needs that in naive (non-flash implementation)
            activation_memory += 3 * batch_size * layer.self_attn.num_attention_heads * seq_len * seq_len # LSoftmax is too naive, so it requires 3 [B,H,T,T] tensors cached for backward
            # ln2
            activation_memory += batch_size * seq_len * layer.d_model # pre ln2
            activation_memory += batch_size * seq_len * layer.d_model # after ln2
            # ffn
            activation_memory += 4 * batch_size * seq_len * layer.mlp.d_ff # t1, sigmoid(t1), sigmoid(t1)*t1, t3; d_ff is FFN layer width
            activation_memory += batch_size * seq_len * layer.mlp.d_ff # sigmoid(t1) * t1 * t3
        # ln_final
        activation_memory += batch_size * seq_len * layer.d_model
        # lm_head
        # activation_memory += batch_size * seq_len * self.lm_head.weight.shape[1]
        print(f'Activation memory estimation (bs={batch_size}, seqlen={seq_len}):', activation_memory, f' BF16 {(activation_memory * 2e-9):5.2f} GB, BF32 {(activation_memory * 4e-9):5.2f} GB')

        embed_flops = 0
        ln_flops = 0
        residual_flops = 0
        linear_flops = 0
        attn_flops = 0
        # token_embeddings
        embed_flops += batch_size * seq_len * layer.d_model
        # each layer
        for layer in self.layers:
            # ln1
            ln_flops += batch_size * seq_len * layer.d_model * 2 + batch_size * seq_len * 2 + batch_size * seq_len * layer.d_model * 2
            # ln2
            ln_flops += batch_size * seq_len * layer.d_model * 2 + batch_size * seq_len * 2 + batch_size * seq_len * layer.d_model * 2
            # residual
            residual_flops += batch_size * seq_len * layer.d_model * 2
            # attn
            linear_flops += batch_size * seq_len * layer.d_model * layer.d_model * 8
            attn_flops += batch_size * seq_len * seq_len * layer.d_model * 2
            # ffn
            linear_flops += batch_size * seq_len * layer.d_model * layer.mlp.d_ff * 6
        # lastln
        ln_flops += batch_size * seq_len * self.norm.d_model * 2 + batch_size * seq_len * 2 + batch_size * seq_len * self.norm.d_model * 2
        # last_embed
        embed_flops += batch_size * seq_len * self.norm.d_model * self.lm_head.weight.shape[1] * 2

        def format_flops(now_f, tot_f):
            s_nf = f'{now_f / 1e12:8.5f} TFlOPs '
            return s_nf + ' {:.2f}%'.format(now_f / tot_f * 100.)

        tot_flops = linear_flops + attn_flops + embed_flops + ln_flops + residual_flops
        compute_density = tot_flops / activation_memory
        print('Tot:     ', format_flops(tot_flops, tot_flops))
        print('linear:  ', format_flops(linear_flops, tot_flops))
        print('attn:    ', format_flops(attn_flops, tot_flops))
        print('embed:   ', format_flops(embed_flops, tot_flops))
        print('ln:      ', format_flops(ln_flops, tot_flops))
        print('residual:', format_flops(residual_flops, tot_flops))
        print(f'Training Density  (bs={batch_size}, seqlen={seq_len}):', compute_density)

def from_pretrained(model_dir, dtype=None, device='cuda'):
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
    tokenizer = LTokenizer.from_hf_tokenziers(model_dir)

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
    model_config |= {'device': device, 'torch_dtype': torch_dtype}
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
    model, tokenizer = from_pretrained('models/OLMo-2-0425-1B', 'bfloat16')
    from basics.linference import generate
    while True:
        prompt = input('\n> ')
        if not prompt:
            break
        ret = generate(model, [prompt], tokenizer, max_new_tokens=4096, temperature=0.0)
