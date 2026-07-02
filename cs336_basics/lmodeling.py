import math
from typing import Any
import torch
from torch import nn
from einops import rearrange, einsum
from contextlib import nullcontext
import torch.cuda.nvtx as nvtx

use_nvtx = True

range_ctx = nvtx.range if use_nvtx else lambda _: nullcontext()

INF_MIN = -1e+20

class LLinear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None=None):
        super().__init__()
        W_tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2. / (in_features+out_features))
        nn.init.trunc_normal_(W_tensor, mean=0., std=std, a=-3.*std, b=3.*std)
        self.weight = nn.Parameter(W_tensor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with range_ctx("Linear"):
            ret = x @ self.weight.T
        return ret

class LEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        W_embed = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(W_embed, mean=0., std=1., a=-3., b=3.)
        self.weight = nn.Parameter(W_embed)
    
    def forward(self, token_ids: torch.Tensor, clamp_pad: bool=False) -> torch.Tensor:
        # to prevent negative ids from pad, assign those ids a 0
        if clamp_pad:
            return self.weight[token_ids.clamp_min(0)]
        else:
            return self.weight[token_ids]

class LRMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float=1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        Wg = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(Wg)
        self.d_model = d_model
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x * x).sum(dim=-1) / self.d_model + self.eps)
        result = x / rms.unsqueeze(dim=-1) * self.weight
        return result.to(in_dtype)

class LFFN(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None, silu: bool = False, custom_kernel: bool = True):
        super().__init__()
        self.silu = silu
        self.custom_kernel = custom_kernel
        self.d_model, self.d_ff = d_model, d_ff # use passed in arg
        # if d_model % 24 == 0:
        #     self.d_ff = d_model * 8 // 3
        # else:
        #     self.d_ff = ((d_model * 8 // 3) // 64 + 1) * 64 # upper ceil to multiples of 64
        self.w1 = LLinear(self.d_model, self.d_ff, device, dtype)
        self.w2 = LLinear(self.d_ff, self.d_model, device, dtype)
        if not silu:
            self.w3 = LLinear(self.d_model, self.d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.custom_kernel:
            if not self.silu:
                t1 = self.w1(x)
                t3 = self.w3(x)
                return self.w2(torch.sigmoid(t1) * t1 * t3)
            else:
                t1 = self.w1(x)
                return self.w2(torch.sigmoid(t1) * t1)
        else:
            from cs336_systems.laccelerate import LSiLUFunc
            if not self.silu:
                t1 = self.w1(x)
                t3 = self.w3(x)
                return self.w2(LSiLUFunc.apply(t1) * t3)
            else:
                t1 = self.w1(x)
                return self.w2(LSiLUFunc.apply(t1))

class LROPE(torch.nn.Module):

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
        if True:
            x_even = x.view(-1, self.d_k)[:,::2].view(list(x.shape[:-1]) + [self.d_k // 2])
            x_odd = x.view(-1, self.d_k)[:,1::2].view(list(x.shape[:-1]) + [self.d_k // 2])
        else:
            x_even = x.view(-1, self.d_k)[:,:self.d_k // 2].view(list(x.shape[:-1]) + [self.d_k // 2]) # whether to permute indexes to achieve a faster implementation, appears to be not critical
            x_odd = x.view(-1, self.d_k)[:,self.d_k // 2:].view(list(x.shape[:-1]) + [self.d_k // 2])
        cos_even_x = self.cos_angles[token_positions].to(x.dtype) * x_even
        cos_odd_x = self.cos_angles[token_positions].to(x.dtype) * x_odd
        nsin_odd_x = -self.sin_angles[token_positions].to(x.dtype) * x_odd
        sin_even_x = self.sin_angles[token_positions].to(x.dtype) * x_even
        ans_even = cos_even_x + nsin_odd_x
        ans_odd = cos_odd_x + sin_even_x
        if True:
            ans = torch.stack([ans_even, ans_odd], dim=-1).contiguous().reshape(x.shape)
        else:
            ans = torch.concat([ans_even, ans_odd], dim=-1).contiguous().reshape(x.shape)
        return ans

def LSoftmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    rowmax = x.amax(dim=dim, keepdim=True)
    t = torch.exp(x - rowmax)
    return t / t.sum(dim=dim, keepdim=True)

@nvtx.range("SDPA")
def LNaiveSDPA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None, padded_tokens: torch.Tensor | None = None) -> torch.Tensor:
    with range_ctx("SDPA Attention Score"):
        QK = einsum(Q, K, '... queries d, ... keys d -> ... queries keys') / math.sqrt(Q.shape[-1])
    
    with range_ctx("SDPA Masking"):
        
        with range_ctx("SDPA Masking Causal"):
            if mask is not None:
                QK += (~mask) * INF_MIN
                # QK[~mask] = INF_MIN
        
        with range_ctx("SDPA Masking Context"):
            if padded_tokens is not None:
                QK = rearrange(QK, 'batch d_head queries keys -> queries d_head batch keys')
                QK += padded_tokens * INF_MIN
                # QK[padded_tokens] = INF_MIN
                QK = rearrange(QK, 'queries d_head batch keys -> batch d_head queries keys')
    
    with range_ctx("SDPA Softmax and V"):
        ret = einsum(LSoftmax(QK, dim=3), V, '... queries keys , ... keys d -> ... queries d')

    return ret

class LMHA(torch.nn.Module):
    # global singleton
    rope_cache: LROPE | None = None
    triu_cache: dict[int, torch.Tensor] = {}

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, nope: bool=False):
        super().__init__()
        self.nope = nope
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta # if theta is None, no ROPE will be applied

        self.q_proj = LLinear(d_model, d_model, device, dtype)
        self.k_proj = LLinear(d_model, d_model, device, dtype)
        self.v_proj = LLinear(d_model, d_model, device, dtype)
        self.output_proj = LLinear(d_model, d_model, device, dtype)
        if LMHA.rope_cache is None and theta is not None:
            LMHA.rope_cache = LROPE(theta, self.d_k, max_seq_len, device)
        if max_seq_len not in self.triu_cache:
            self.triu_cache[max_seq_len] = torch.triu(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device)).T
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, padded_tokens: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if kv_cache is not None:
            # using kv cache
            # another latent assumption: if using kv cache, we only compute the last token's output; in the output, other token places are zeros as placeholders...
            k_cache, v_cache = kv_cache # [B, L-1, D_M]
            assert tuple(k_cache.shape) == (x.shape[0], x.shape[1]-1, x.shape[2])
            assert tuple(v_cache.shape) == (x.shape[0], x.shape[1]-1, x.shape[2])
            new_k = self.k_proj(x[:, -1:]) # [B, 1, D_M]
            new_v = self.v_proj(x[:, -1:]) # [B, 1, D_M]
            k = torch.concat([k_cache, new_k], dim=1) # [B, L, D_M]
            v = torch.concat([v_cache, new_v], dim=1) # [B, L, D_M]
            q = self.q_proj(x[:, -1:]) # [B, 1, D_M]
        else:
            # not using kv cache
            k = self.k_proj(x)
            v = self.v_proj(x)
            q = self.q_proj(x)
        q = rearrange(q, '... seqlen (h d_k) -> ... seqlen h d_k', h=self.num_heads, d_k=self.d_k)
        kk = rearrange(k, '... seqlen (h d_k) -> ... seqlen h d_k', h=self.num_heads, d_k=self.d_k)
        vv = rearrange(v, '... seqlen (h d_k) -> ... h seqlen d_k', h=self.num_heads, d_k=self.d_k)
        if (not self.nope) and self.theta and LMHA.rope_cache:
            if token_positions is not None:
                token_positions = token_positions.unsqueeze(-1) # add head dim
            else:
                token_positions = torch.arange(x.shape[1]).view(-1, 1) # default index for x
            if kv_cache is not None:
                # by latent assumption, q is of shape [B, 1, D_M]
                q = LMHA.rope_cache(q, token_positions[:, -1:])
            else:
                q = LMHA.rope_cache(q, token_positions)
            kk = LMHA.rope_cache(kk, token_positions)
        q = rearrange(q, '... seqlen h d_k -> ... h seqlen d_k', h=self.num_heads, d_k=self.d_k) # [B, H, 1, D_K] or [B, H, L, D_K]
        kk = rearrange(kk, '... seqlen h d_k -> ... h seqlen d_k', h=self.num_heads, d_k=self.d_k) # [B, H, L, D_K]
        if kv_cache is not None:
            # by latent assumption, q is of shape [B, 1, D_M] pointing to the last token place
            before_proj = LNaiveSDPA(q, kk, vv, self.triu_cache[self.max_seq_len][x.shape[1]-1: x.shape[1], :x.shape[1]], padded_tokens)
        else:
            before_proj = LNaiveSDPA(q, kk, vv, self.triu_cache[self.max_seq_len][:x.shape[1], :x.shape[1]], padded_tokens)
        before_proj = rearrange(before_proj, '... h seqlen d_k -> ... seqlen (h d_k)') # [B, L, D_M] or [B, 1, D_M]
        output = self.output_proj(before_proj)
        if kv_cache is not None:
            # by latent assumption, need to add dummy output
            dummy = torch.zeros((output.shape[0], x.shape[1]-1, output.shape[2]), dtype=output.dtype, device=output.device)
            output = torch.concat([dummy, output], dim=1)
        return output, (k, v)

class LTransformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, no_rms_norm: bool = False, post_norm: bool = False, nope: bool = False, silu: bool = False, compile: bool = True, custom_kernel: bool = False) -> None:
        super().__init__()
        self.no_rms_norm = no_rms_norm
        self.post_norm = post_norm
        self.nope = nope
        self.d_model = d_model

        assert not (compile and custom_kernel), "Cannot use both compile and custom_kernel!"

        if no_rms_norm:
            self.ln1 = self.ln2 = None
        else:
            if not custom_kernel:
                self.ln1 = LRMSNorm(d_model, device=device, dtype=dtype)
                self.ln2 = LRMSNorm(d_model, device=device, dtype=dtype)
            else:
                # use custom kernel LRMSNorm - slower than torch.compile :(
                from cs336_systems.laccelerate import LRMSNormFast
                self.ln1 = LRMSNormFast(d_model, device=device, dtype=dtype)
                self.ln2 = LRMSNormFast(d_model, device=device, dtype=dtype)

        self.attn = LMHA(d_model, num_heads, max_seq_len, theta, device, dtype, nope)
        self.ffn = LFFN(d_model, d_ff, device, dtype, silu=silu)

        if self.ln1 and self.ln2 and compile:
            self.ln1 = torch.compile(self.ln1)
            self.ln2 = torch.compile(self.ln2)
        if compile:
            self.attn = torch.compile(self.attn)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, padded_tokens: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        # padded_tokens: torch.bool [B, T]
        if not self.post_norm:
            attn_x, kv_cache = self.attn(self.ln1(x) if not self.no_rms_norm else x, token_positions, padded_tokens, kv_cache)
            t = x + attn_x
            return t + self.ffn(self.ln2(t) if not self.no_rms_norm else t), kv_cache
        else:
            attn_x, kv_cache = self.attn(x, token_positions, padded_tokens, kv_cache)
            t = self.ln1(x + attn_x)
            return self.ln2(t + self.ffn(t)), kv_cache

class LTransformerLM(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int, vocab_size: int, num_layers: int, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, customizations: dict | None = None) -> None:
        super().__init__()
        self.customizations = customizations or {}
        no_rms_norm = self.customizations.get('no_rms_norm', False)
        post_norm = self.customizations.get('post_norm', False)
        nope = self.customizations.get('nope', False)
        silu = self.customizations.get('silu', False)
        if no_rms_norm or post_norm or nope or silu:
            print(f'Ablationed model architecture: no_rms_norm={no_rms_norm}, post_norm={post_norm}, nope={nope}, silu={silu}')
        assert not (no_rms_norm and post_norm), 'Cannot require both no_rms_norm and post_norm'

        self.token_embeddings = LEmbedding(vocab_size, d_model, device, dtype)
        self.layers: list[LTransformerBlock] = nn.Sequential(*[LTransformerBlock(d_model, num_heads, d_ff, context_length, theta, device, dtype, no_rms_norm=no_rms_norm, post_norm=post_norm, nope=nope, silu=silu) for _ in range(num_layers)])
        self.ln_final = LRMSNorm(d_model, device=device, dtype=dtype) if not no_rms_norm else None
        self.lm_head = LLinear(d_model, vocab_size, device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embeddings(x)
        kv_caches = []
        for layer in self.layers:
            x, kv = layer(x, token_positions)
            kv_caches.append(kv)
        x = self.ln_final(x) if self.ln_final else x
        x = self.lm_head(x)
        return x # discard kv cache for now

    def batch_generate(self, x: torch.Tensor, kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None, pad_token_id: int | None = None):
        # changes to kv cache is in place
        padded_tokens = (x == pad_token_id)
        token_positions = (torch.cumsum(x != pad_token_id, dim=1) - 1).clamp(min=0)
        x = self.token_embeddings(x, clamp_pad=pad_token_id is not None)
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            # print('layer', i)
            if kv_cache is not None:
                x, new_layer_kvcache = layer(x, token_positions, padded_tokens, kv_cache[i])
            else:
                x, new_layer_kvcache = layer(x, token_positions, padded_tokens)
            new_kv_cache.append(new_layer_kvcache)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x, new_kv_cache
    
    def count_parameters(self) -> tuple[int, int]: # return tot params and tot non-embed params
        tot_params = 0
        tot_embed_params = 0
        for param in self.token_embeddings.parameters():
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
        for param in self.token_embeddings.parameters():
            tot_embed_params += param.numel()
        for param in self.lm_head.parameters():
            tot_embed_params += param.numel()
        for param in self.ln_final.parameters():
            tot_ln_params += param.numel()
        for layer in self.layers:
            if each_block_params == 0:
                for param in layer.parameters():
                    each_block_params += param.numel()
            for param in layer.ln1.parameters():
                tot_ln_params += param.numel()
            for param in layer.ln2.parameters():
                tot_ln_params += param.numel()
            for param in layer.attn.parameters():
                tot_attn_params += param.numel()
            for param in layer.ffn.parameters():
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
        activation_memory += batch_size * seq_len * self.token_embeddings.weight.shape[1]
        # each layer
        for layer in self.layers:
            # ln1
            activation_memory += batch_size * seq_len * layer.d_model # pre ln1
            activation_memory += batch_size * seq_len * layer.d_model # after ln1 for attn usage
            # attn
            activation_memory += 4 * batch_size * seq_len * layer.d_model # q,k,v,output proj
            # need to cache attn scores before and after softmax because softmax backward needs that in naive (non-flash implementation)
            activation_memory += 3 * batch_size * layer.attn.num_heads * seq_len * seq_len # LSoftmax is too naive, so it requires 3 [B,H,T,T] tensors cached for backward
            # ln2
            activation_memory += batch_size * seq_len * layer.d_model # pre ln2
            activation_memory += batch_size * seq_len * layer.d_model # after ln2
            # ffn
            activation_memory += 4 * batch_size * seq_len * layer.ffn.d_ff # t1, sigmoid(t1), sigmoid(t1)*t1, t3; d_ff is FFN layer width
            activation_memory += batch_size * seq_len * layer.ffn.d_ff # sigmoid(t1) * t1 * t3
        # ln_final
        activation_memory += batch_size * seq_len * layer.d_model
        # lm_head
        # activation_memory += batch_size * seq_len * self.lm_head.weight.shape[1]
        print(f'Activation memory (bs={batch_size}, seqlen={seq_len}):', activation_memory, f' BF16 {(activation_memory * 2e-9):5.2f} GB, BF32 {(activation_memory * 4e-9):5.2f} GB')

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
            linear_flops += batch_size * seq_len * layer.d_model * layer.ffn.d_ff * 6
        # lastln
        ln_flops += batch_size * seq_len * self.ln_final.d_model * 2 + batch_size * seq_len * 2 + batch_size * seq_len * self.ln_final.d_model * 2
        # last_embed
        embed_flops += batch_size * seq_len * self.ln_final.d_model * self.lm_head.weight.shape[1] * 2



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

if __name__ == '__main__':
    import yaml

    with open('cs336_basics/configs/models/gpt2_xl.yaml', 'r') as f:
        gpt2_xl_model_config = yaml.safe_load(f)
    with open('cs336_basics/configs/models/gpt2_large.yaml', 'r') as f:
        gpt2_large_model_config = yaml.safe_load(f)
    with open('cs336_basics/configs/models/gpt2_medium.yaml', 'r') as f:
        gpt2_medium_model_config = yaml.safe_load(f)
    with open('cs336_basics/configs/models/gpt2_small.yaml', 'r') as f:
        gpt2_small_model_config = yaml.safe_load(f)
    with open('cs336_basics/configs/models/gpt2_tiny.yaml', 'r') as f:
        gpt2_tiny_model_config = yaml.safe_load(f)


    # config = gpt2_small_model_config
    # config = gpt2_xl_model_config
    config = gpt2_tiny_model_config
    config |= {
        'dtype': torch.bfloat16,
        'device': 'cuda'
    }

    lm = LTransformerLM(**config)
    lm.resource_count(batch_size=16, seq_len=512)
    # lm.resource_count(batch_size=1024, seq_len=1)

