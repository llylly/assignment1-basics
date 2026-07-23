import math
from typing import Any, Mapping, Callable
import torch
from torch import nn
from einops import rearrange, einsum
from contextlib import nullcontext
import torch.cuda.nvtx as nvtx

use_nvtx = True

range_ctx = nvtx.range if use_nvtx else lambda _: nullcontext()

INF_MIN = -1e+20

class LLinear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None=None, std_multiplier: float=1.0):
        super().__init__()
        W_tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(std_multiplier * 2. / (in_features+out_features))
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

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None, silu: bool = False, custom_kernel: bool = True, param_maps: dict | None = None):
        super().__init__()
        self.silu = silu
        self.custom_kernel = custom_kernel
        self.d_model, self.d_ff = d_model, d_ff # use passed in arg
        # if d_model % 24 == 0:
        #     self.d_ff = d_model * 8 // 3
        # else:
        #     self.d_ff = ((d_model * 8 // 3) // 64 + 1) * 64 # upper ceil to multiples of 64
        if param_maps is None: param_maps = {}
        param_maps['w1'] = param_maps.get('w1', 'w1')
        param_maps['w2'] = param_maps.get('w2', 'w2')
        param_maps['w3'] = param_maps.get('w3', 'w3')
        self.param_maps = param_maps

        setattr(self, param_maps['w1'], LLinear(self.d_model, self.d_ff, device, dtype))
        setattr(self, param_maps['w2'], LLinear(self.d_ff, self.d_model, device, dtype))
        if not silu:
            setattr(self, param_maps['w3'], LLinear(self.d_model, self.d_ff, device, dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.custom_kernel:
            if not self.silu:
                t1 = getattr(self, self.param_maps['w1'])(x)
                t3 = getattr(self, self.param_maps['w3'])(x)
                return getattr(self, self.param_maps['w2'])(torch.sigmoid(t1) * t1 * t3)
            else:
                t1 = getattr(self, self.param_maps['w1'])(x)
                return getattr(self, self.param_maps['w2'])(torch.sigmoid(t1) * t1)
        else:
            from systems.laccelerate import LSiLUFunc
            if not self.silu:
                t1 = getattr(self, self.param_maps['w1'])(x)
                t3 = getattr(self, self.param_maps['w3'])(x)
                return getattr(self, self.param_maps['w2'])(LSiLUFunc.apply(t1) * t3)
            else:
                t1 = getattr(self, self.param_maps['w1'])(x)
                return getattr(self, self.param_maps['w2'])(LSiLUFunc.apply(t1))

class LROPE(torch.nn.Module):

    # need to be a singleton across layers
    def __init__(self, theta: float, d_k: int, max_seq_len: int, interleave: bool = True, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.interleave = interleave
        bases = torch.pow(torch.tensor(theta, device=device, dtype=torch.float32), -(torch.arange(1, d_k // 2 + 1, device=device, dtype=torch.float32) * 2 - 2) / d_k)
        angles = einsum(torch.arange(0, max_seq_len, device=device, dtype=torch.float32), bases, "seq_len, bases -> seq_len bases")
        sin_angles = torch.sin(angles) # [L, d_k / 2]
        cos_angles = torch.cos(angles) # [L, d_k / 2]
        self.register_buffer('sin_angles', sin_angles, persistent=False)
        self.register_buffer('cos_angles', cos_angles, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: # x: [B, L, H, D_K], token_positions: [B, L, 1]
        if self.interleave:
            x_even = x.view(-1, x.shape[-2], x.shape[-1] // self.d_k, self.d_k)[..., ::2]
            x_odd = x.view(-1, x.shape[-2], x.shape[-1] // self.d_k, self.d_k)[..., 1::2]
        else:
            x_odd = x.view(-1, x.shape[-2], x.shape[-1] // self.d_k, self.d_k)[..., self.d_k // 2:]
            x_even = x.view(-1, x.shape[-2], x.shape[-1] // self.d_k, self.d_k)[..., :self.d_k // 2]

        cos_even_x = self.cos_angles[token_positions].to(x.dtype) * x_even
        cos_odd_x = self.cos_angles[token_positions].to(x.dtype) * x_odd
        nsin_odd_x = -self.sin_angles[token_positions].to(x.dtype) * x_odd
        sin_even_x = self.sin_angles[token_positions].to(x.dtype) * x_even

        ans_even = cos_even_x + nsin_odd_x
        ans_odd = cos_odd_x + sin_even_x
        if self.interleave:
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
    # print(Q.shape, K.shape, V.shape, mask.shape)
    with range_ctx("SDPA Attention Score"):
        QK = einsum(Q, K, '... queries d, ... keys d -> ... queries keys') / math.sqrt(Q.shape[-1])
    
    with range_ctx("SDPA Masking"):
        
        with range_ctx("SDPA Masking Causal"):
            if mask is not None:
                QK += (~mask) * INF_MIN
                # QK[~mask] = INF_MIN
        
        with range_ctx("SDPA Masking Context"):
            if padded_tokens is not None:
                QK = rearrange(QK, 'batch ... queries keys -> queries ... batch keys')
                QK += padded_tokens * INF_MIN
                # QK[padded_tokens] = INF_MIN
                QK = rearrange(QK, 'queries ... batch keys -> batch ... queries keys')
    
    with range_ctx("SDPA Softmax and V"):
        ret = einsum(LSoftmax(QK, dim=-1), V, '... queries keys , ... keys d -> ... queries d')

    return ret

class LMHA(torch.nn.Module):

    def __init__(self, d_model: int, num_attention_heads: int, num_key_value_heads: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None, qk_norm: bool = False, qk_norm_rms_eps: float | None=None, require_kv_cache: bool=True, flash_attn: bool=False, param_maps: dict | None=None):
        super().__init__()

        self._KVCACHE_BLOCKSIZE = 512

        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.d_k = self.d_model // self.num_attention_heads
        self.max_seq_len = max_seq_len
        self.qk_norm = qk_norm

        self.require_kv_cache = require_kv_cache
        self.flash_attn = flash_attn

        if param_maps is None: param_maps = {}
        self.param_maps = param_maps
        for k in ['q_proj', 'k_proj', 'v_proj', 'output_proj', 'q_norm', 'k_norm']:
            param_maps[k] = param_maps.get(k, k)

        setattr(self, param_maps['q_proj'], LLinear(d_model, d_model, device, dtype))
        setattr(self, param_maps['k_proj'], LLinear(d_model, self.num_key_value_heads * self.d_k, device, dtype))
        setattr(self, param_maps['v_proj'], LLinear(d_model, self.num_key_value_heads * self.d_k, device, dtype))
        setattr(self, param_maps['output_proj'], LLinear(d_model, d_model, device, dtype))

        if qk_norm:
            if qk_norm_rms_eps is not None:
                norm_eps = {'eps': qk_norm_rms_eps}
            else:
                norm_eps = {}
            setattr(self, param_maps['q_norm'], LRMSNorm(d_model, device=device, dtype=dtype, **norm_eps))
            setattr(self, param_maps['k_norm'], LRMSNorm(self.num_key_value_heads * self.d_k, device=device, dtype=dtype, **norm_eps))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, padded_tokens: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None, kv_cache_slice_to: int | None = None, rope_cache: LROPE | None = None, triu_cache: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert triu_cache is not None # casual mask must be there now

        if kv_cache is not None:
            # using kv cache
            k_cache, v_cache = kv_cache # [B, L-1, D_M']
            assert k_cache.shape[1] == x.shape[0] and k_cache.shape[0] >= kv_cache_slice_to and k_cache.shape[2] == self.num_key_value_heads * self.d_k
            assert v_cache.shape[1] == x.shape[0] and v_cache.shape[0] >= kv_cache_slice_to and v_cache.shape[2] == self.num_key_value_heads * self.d_k

        q = getattr(self, self.param_maps['q_proj'])(x) if not self.qk_norm else getattr(self, self.param_maps['q_norm'])(getattr(self, self.param_maps['q_proj'])(x)) # [B, 1, D_M]
        new_k = getattr(self, self.param_maps['k_proj'])(x) if not self.qk_norm else getattr(self, self.param_maps['k_norm'])(getattr(self, self.param_maps['k_proj'])(x)) # [B, 1, D_M']
        new_v = getattr(self, self.param_maps['v_proj'])(x) # [B, 1, D_M']

        if rope_cache is not None:
            if token_positions is not None:
                token_positions = token_positions.unsqueeze(-1) # add head dim
            else:
                token_positions = torch.arange(x.shape[1]).view(1, -1, 1) # default index for x
            q = rope_cache(q, token_positions)
            new_k = rope_cache(new_k, token_positions)
        else:
            print('Warning: no ROPE or positional rotary is applied.')
        
        if kv_cache is not None:
            # let's make kv cache [L, B, D] rather than [B, L, D]
            if (kv_cache_slice_to + new_k.shape[1] - 1) // self._KVCACHE_BLOCKSIZE > (k_cache.shape[0] - 1) // self._KVCACHE_BLOCKSIZE:
                # KV_cache needs expansion
                cells_to_add = (k_cache.shape[0] + new_k.shape[1] - 1) // self._KVCACHE_BLOCKSIZE * self._KVCACHE_BLOCKSIZE + self._KVCACHE_BLOCKSIZE - k_cache.shape[0]
                new_k_cache = torch.concat([k_cache, torch.zeros([cells_to_add, k_cache.shape[1], k_cache.shape[2]], dtype=k_cache.dtype, device=k_cache.device)], dim=0)
                new_v_cache = torch.concat([v_cache, torch.zeros([cells_to_add, k_cache.shape[1], k_cache.shape[2]], dtype=v_cache.dtype, device=v_cache.device)], dim=0)
                k_cache, v_cache = new_k_cache, new_v_cache
            k_cache[kv_cache_slice_to: kv_cache_slice_to + new_k.shape[1]] = new_k.transpose(0,1)
            v_cache[kv_cache_slice_to: kv_cache_slice_to + new_v.shape[1]] = new_v.transpose(0,1)
            k, v = k_cache[:kv_cache_slice_to + new_k.shape[1]], v_cache[:kv_cache_slice_to + new_v.shape[1]]
        else:
            cells_to_add = (new_k.shape[1] - 1) // self._KVCACHE_BLOCKSIZE * self._KVCACHE_BLOCKSIZE + self._KVCACHE_BLOCKSIZE
            k_cache = torch.zeros([cells_to_add, new_k.shape[0], new_k.shape[2]], dtype=new_k.dtype, device=new_k.device)
            v_cache = torch.zeros([cells_to_add, new_v.shape[0], new_v.shape[2]], dtype=new_v.dtype, device=new_v.device)
            # print(k_cache.shape, new_k.shape)
            k_cache[:new_k.shape[1]] = new_k.transpose(0,1)
            v_cache[:new_v.shape[1]] = new_v.transpose(0,1)
            k, v = k_cache[:new_k.shape[1]], v_cache[:new_v.shape[1]]

        q = rearrange(q, '... seqlen (h i d_k) -> ... h i seqlen d_k', i=self.num_attention_heads // self.num_key_value_heads, h=self.num_key_value_heads, d_k=self.d_k)
        kk = rearrange(k, 'seqlen ... (h d_k) -> ... h 1 seqlen d_k', h=self.num_key_value_heads, d_k=self.d_k)
        vv = rearrange(v, 'seqlen ... (h d_k) -> ... h 1 seqlen d_k', h=self.num_key_value_heads, d_k=self.d_k)

        if self.flash_attn:
            from systems.call_flash_attn import CallFlashAttn
            spda_fn = CallFlashAttn
        else:
            spda_fn = LNaiveSDPA

        if kv_cache is not None:
            # by latent assumption, q is of shape [B, 1, D_M] pointing to the last token place
            # actually triu_cache that get passed in is an all 1 masking matrix
            before_proj = spda_fn(q, kk, vv, None, padded_tokens)
        else:
            before_proj = spda_fn(q, kk, vv, triu_cache[:x.shape[1], :x.shape[1]], padded_tokens)
        before_proj = rearrange(before_proj, '... h i seqlen d_k -> ... seqlen (h i d_k)') # [B, L, D_M] or [B, 1, D_M]
        output = getattr(self, self.param_maps['output_proj'])(before_proj)
        return output, (k_cache, v_cache)

class LTransformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, num_attention_heads: int, num_key_value_heads: int, d_ff: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None, no_rms_norm: bool = False, post_norm: bool = False, partial_post_norm: bool = False, silu: bool = False, rms_norm_eps: float | None = None, compile: bool = True, custom_kernel: bool = False, flash_attn: bool = False, attn_qk_norm: bool=False, require_kv_cache: bool=True, param_maps: dict | None = None, attn_param_maps: dict | None = None, ffn_param_maps: dict | None = None) -> None:
        super().__init__()
        self.no_rms_norm = no_rms_norm
        self.post_norm = post_norm
        self.partial_post_norm = partial_post_norm
        self.silu = silu
        self.attn_qk_norm = attn_qk_norm
        self.d_model = d_model

        assert not (post_norm and partial_post_norm), "Cannot be both post_norm and partial_post_norm!"
        assert not (compile and custom_kernel), "Cannot use both compile and custom_kernel!"
        assert not (custom_kernel and flash_attn), "Cannot use both custom_kernel and flash_attn!"

        self._compile = compile
        self._custom_kernel = custom_kernel

        if param_maps is None: param_maps = {}
        for k in ['ln1', 'ln2', 'attn', 'ffn']:
            param_maps[k] = param_maps.get(k, k)
        self.param_maps, self.attn_param_maps, self.ffn_param_maps = param_maps, attn_param_maps, ffn_param_maps

        if rms_norm_eps is not None:
            norm_eps = {'eps': rms_norm_eps}
        else:
            norm_eps = {}

        if no_rms_norm:
            setattr(self, self.param_maps['ln1'], None)
            setattr(self, self.param_maps['ln2'], None)
        else:
            if not custom_kernel:
                setattr(self, self.param_maps['ln1'], LRMSNorm(d_model, device=device, dtype=dtype, **norm_eps))
                setattr(self, self.param_maps['ln2'], LRMSNorm(d_model, device=device, dtype=dtype, **norm_eps))
            else:
                # use custom kernel LRMSNorm - slower than torch.compile :(
                from systems.laccelerate import LRMSNormFast
                setattr(self, self.param_maps['ln1'], LRMSNormFast(d_model, device=device, dtype=dtype, **norm_eps))
                setattr(self, self.param_maps['ln2'], LRMSNormFast(d_model, device=device, dtype=dtype, **norm_eps))

        setattr(self, self.param_maps['attn'], LMHA(d_model, num_attention_heads, num_key_value_heads, max_seq_len, device, dtype, attn_qk_norm, require_kv_cache=require_kv_cache, flash_attn=flash_attn, qk_norm_rms_eps=rms_norm_eps, param_maps=attn_param_maps))
        setattr(self, self.param_maps['ffn'], LFFN(d_model, d_ff, device, dtype, silu=silu, param_maps=ffn_param_maps))

        if getattr(self, self.param_maps['ln1']) and getattr(self, self.param_maps['ln2']) and compile:
            setattr(self, self.param_maps['ln1'], torch.compile(getattr(self, self.param_maps['ln1'])))
            setattr(self, self.param_maps['ln2'], torch.compile(getattr(self, self.param_maps['ln2'])))
        if compile:
            setattr(self, self.param_maps['attn'], torch.compile(getattr(self, self.param_maps['attn'])))
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        # overwrite to handle the name mapping problems with torch.compile
        to_add_orig_mod = []
        to_remove_orig_mod = []
        if (getattr(self, self.param_maps['ln1']) is not None) and (getattr(self, self.param_maps['ln2']) is not None):
            if self._compile:
                to_add_orig_mod.extend([self.param_maps['ln1'], self.param_maps['ln2']])
            else:
                to_remove_orig_mod.extend([self.param_maps['ln1'], self.param_maps['ln2']])
        if self._compile:
            to_add_orig_mod.append(self.param_maps['attn'])
        else:
            to_remove_orig_mod.append(self.param_maps['attn'])
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
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, padded_tokens: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None, kv_cache_slice_to: int | None = None, rope_cache: LROPE | None = None, triu_cache: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # padded_tokens: torch.bool [B, T]
        if not self.post_norm and not self.partial_post_norm:
            # prenorm
            attn_x, kv_cache = getattr(self, self.param_maps['attn'])(getattr(self, self.param_maps['ln1'])(x) if not self.no_rms_norm else x, token_positions, padded_tokens, kv_cache, kv_cache_slice_to, rope_cache=rope_cache, triu_cache=triu_cache)
            t = x + attn_x
            return t + getattr(self, self.param_maps['ffn'])(getattr(self, self.param_maps['ln2'])(t) if not self.no_rms_norm else t), kv_cache
        elif self.partial_post_norm:
            # partial postnorm, adopted by Olmo2-1B
            attn_x, kv_cache = getattr(self, self.param_maps['attn'])(x, token_positions, padded_tokens, kv_cache, kv_cache_slice_to, rope_cache=rope_cache, triu_cache=triu_cache)
            t = x + (getattr(self, self.param_maps['ln1'])(attn_x) if not self.no_rms_norm else attn_x)
            mlp_x = getattr(self, self.param_maps['ffn'])(t)
            return t + (getattr(self, self.param_maps['ln2'])(mlp_x) if not self.no_rms_norm else mlp_x), kv_cache
        else:
            # full postnorm - suboptimal
            attn_x, kv_cache = getattr(self, self.param_maps['attn'])(x, token_positions, padded_tokens, kv_cache, kv_cache_slice_to, rope_cache=rope_cache, triu_cache=triu_cache)
            t = getattr(self, self.param_maps['ln1'])(x + attn_x)
            return getattr(self, self.param_maps['ln2'])(t + getattr(self, self.param_maps['ffn'])(t)), kv_cache

class LTransformerLM(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int, vocab_size: int, num_layers: int, theta: float | None = None, rms_norm_eps: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, customizations: dict | None = None, interleave_rope: bool = True, flash_attn: bool = False, require_kv_cache: bool=True, customized_layer_constructor: Callable | None = None, param_maps: dict | None = None) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = context_length

        self.customizations = customizations or {}
        no_rms_norm = self.customizations.get('no_rms_norm', False)
        post_norm = self.customizations.get('post_norm', False)
        partial_post_norm = self.customizations.get('partial_post_norm', False)
        nope = self.customizations.get('nope', False)
        silu = self.customizations.get('silu', False)
        qk_norm = self.customizations.get('qk_norm', False)
        if no_rms_norm or post_norm or nope or silu:
            print(f'Ablationed model architecture: no_rms_norm={no_rms_norm}, post_norm={post_norm}, nope={nope}, silu={silu}')
        assert not (no_rms_norm and post_norm), 'Cannot require both no_rms_norm and post_norm'
        assert not (partial_post_norm and post_norm), 'Cannot require both partial_post_norm and post_norm'

        if param_maps is None: param_maps = {}
        for k in ['token_embeddings', 'layers', 'ln_final', 'lm_head']:
            param_maps[k] = param_maps.get(k, k)
        self.param_maps = param_maps

        setattr(self, self.param_maps['token_embeddings'], LEmbedding(vocab_size, d_model, device, dtype))
        setattr(self, self.param_maps['layers'], nn.Sequential(*[LTransformerBlock(d_model, num_heads, num_heads, d_ff, context_length, device, dtype, rms_norm_eps=rms_norm_eps, no_rms_norm=no_rms_norm, post_norm=post_norm, silu=silu, partial_post_norm=partial_post_norm, attn_qk_norm=qk_norm, require_kv_cache=require_kv_cache, flash_attn=flash_attn) if customized_layer_constructor is None else customized_layer_constructor() for _ in range(num_layers)]))
        if rms_norm_eps is not None:
            norm_eps = {'eps': rms_norm_eps}
        else:
            norm_eps = {}
        setattr(self, self.param_maps['ln_final'], LRMSNorm(d_model, device=device, dtype=dtype, **norm_eps) if not no_rms_norm else None)
        setattr(self, self.param_maps['lm_head'], LLinear(d_model, vocab_size, device=device, dtype=dtype))

        self.device = device
        self.dtype = dtype

        self.rope_cache: LROPE | None = None
        if not nope and theta is not None:
            self.rope_cache = LROPE(theta, d_model // num_heads, self.max_seq_len, interleave_rope, device)
        self.register_buffer('triu_cache', torch.triu(torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.bool, device=device)).T, persistent=False) # casual mask
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, dump_act_norm: bool = False) -> torch.Tensor:
        x = getattr(self, self.param_maps['token_embeddings'])(x)
        kv_caches = []
        act_norm = {}
        layer_no = 0
        for layer in getattr(self, self.param_maps['layers']):
            new_x, kv = layer(x, token_positions, rope_cache=self.rope_cache, triu_cache=self.triu_cache) # TODO: Doesn't consider potential padded tokens which might need to put in
            if dump_act_norm:
                act_norm[layer_no] = {
                    'layer_act_norm': torch.linalg.vector_norm(new_x - x, ord=2, dim=-1).mean().item(),
                    'layer_res_norm': torch.linalg.vector_norm(new_x, ord=2, dim=-1).mean().item()
                }
            x = new_x
            kv_caches.append(kv)
            layer_no += 1
        x = getattr(self, self.param_maps['ln_final'])(x) if getattr(self, self.param_maps['ln_final']) else x
        x = getattr(self, self.param_maps['lm_head'])(x)
        if dump_act_norm:
            return x, act_norm # discard kv cache for now
        else:
            return x # discard kv cache for now
    
    def compute_layer_grad_norms(self) -> dict[int, float]:
        ret = {}
        layer_no = 0
        for layer in getattr(self, self.param_maps['layers']):
            now_norm = None
            for param in layer.parameters():
                if param.grad is not None:
                    if now_norm is None: 
                        now_norm = torch.sum(param.grad * param.grad) 
                    else: now_norm += torch.sum(param.grad * param.grad)
            now_norm = now_norm.sqrt().item()
            ret[layer_no] = now_norm
            layer_no += 1
        return ret

    def batch_generate(self, x: torch.Tensor, kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None, pad_token_id: int | None = None, token_positions: torch.Tensor | None = None, kv_cache_sliced_to: int | None = None, padded_tokens: torch.Tensor | None = None):
        # Note: padded_tokens should be of the shape [Batch, keys] at the globally scale, where keys = kv_cache's length dim + x length dim, i.e., it's a global mask
        # Note 2: changes to kv cache is in place
        if kv_cache is not None:
            assert token_positions is not None, "Need to provide token positions because x only contains the incremental indexes"
            assert kv_cache_sliced_to is not None, "Need to provide kv_cache_sliced_to because kv_cache could contain extra tailing space"
        if padded_tokens is None:
            local_mask = (x == pad_token_id)
            if kv_cache is None:
                padded_tokens = local_mask
            else:
                print('Warning: there is KV cache, but no padded token information - treating all KV cached tokens are unmasked ones.')
                prefill_mask = torch.zeros([x.shape[0], kv_cache_sliced_to], dtype=local_mask.dtype, device=local_mask.device)
                padded_tokens = torch.cat([prefill_mask, local_mask], dim=1)
        if token_positions is not None:
            assert tuple(x.shape) == tuple(token_positions.shape), f"x and token_positions should match their shape: {tuple(x.shape)} == {tuple(token_positions.shape)}"
            new_token_positions = token_positions
        else:
            new_token_positions = (torch.cumsum(x != pad_token_id, dim=1) - 1).clamp(min=0)
        x = getattr(self, self.param_maps['token_embeddings'])(x, clamp_pad=pad_token_id is not None)
        new_kv_cache = []
        for i, layer in enumerate(getattr(self, self.param_maps['layers'])):
            # print('layer', i)
            if kv_cache is not None:
                x, new_layer_kvcache = layer(x, new_token_positions, padded_tokens, kv_cache[i], kv_cache_sliced_to, rope_cache=self.rope_cache, triu_cache=self.triu_cache)
            else:
                x, new_layer_kvcache = layer(x, new_token_positions, padded_tokens, rope_cache=self.rope_cache, triu_cache=self.triu_cache)
            new_kv_cache.append(new_layer_kvcache)
        if not self.customizations.get('no_rms_norm', False):
            x = getattr(self, self.param_maps['ln_final'])(x)
        x = getattr(self, self.param_maps['lm_head'])(x)
        return x, new_kv_cache
    
    def count_parameters(self) -> tuple[int, int]: # return tot params and tot non-embed params
        tot_params = 0
        tot_embed_params = 0
        for param in getattr(self, self.param_maps['token_embeddings']).parameters():
            tot_embed_params += param.numel()
        for param in getattr(self, self.param_maps['lm_head']).parameters():
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
        for param in getattr(self, self.param_maps['token_embeddings']).parameters():
            tot_embed_params += param.numel()
        for param in getattr(self, self.param_maps['lm_head']).parameters():
            tot_embed_params += param.numel()
        for param in getattr(self, self.param_maps['ln_final']).parameters():
            tot_ln_params += param.numel()
        for layer in getattr(self, self.param_maps['layers']):
            if each_block_params == 0:
                for param in layer.parameters():
                    each_block_params += param.numel()
            for param in getattr(layer, layer.param_maps['ln1']).parameters():
                tot_ln_params += param.numel()
            for param in getattr(layer, layer.param_maps['ln2']).parameters():
                tot_ln_params += param.numel()
            for param in getattr(layer, layer.param_maps['attn']).parameters():
                tot_attn_params += param.numel()
            for param in getattr(layer, layer.param_maps['ffn']).parameters():
                tot_ffn_params += param.numel()
            attn_module = getattr(layer, layer.param_maps['attn'])
            if hasattr(attn_module, attn_module.param_maps['q_norm']):
                for param in getattr(attn_module, attn_module.param_maps['q_norm']).parameters():
                    tot_ln_params += param.numel()
                    tot_attn_params -= param.numel()
            if hasattr(attn_module, attn_module.param_maps['k_norm']):
                for param in getattr(attn_module, attn_module.param_maps['k_norm']).parameters():
                    tot_ln_params += param.numel()
                    tot_attn_params -= param.numel()
            
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
        activation_memory += batch_size * seq_len * getattr(self, self.param_maps['token_embeddings']).weight.shape[1]
        # each layer
        for layer in getattr(self, self.param_maps['layers']):
            # ln1
            activation_memory += batch_size * seq_len * layer.d_model # pre ln1
            activation_memory += batch_size * seq_len * layer.d_model # after ln1 for attn usage
            # attn
            activation_memory += 4 * batch_size * seq_len * layer.d_model # q,k,v,output proj
            # need to cache attn scores before and after softmax because softmax backward needs that in naive (non-flash implementation)
            activation_memory += 3 * batch_size * getattr(layer, layer.param_maps['attn']).num_attention_heads * seq_len * seq_len # LSoftmax is too naive, so it requires 3 [B,H,T,T] tensors cached for backward
            # ln2
            activation_memory += batch_size * seq_len * layer.d_model # pre ln2
            activation_memory += batch_size * seq_len * layer.d_model # after ln2
            # ffn
            activation_memory += 4 * batch_size * seq_len * getattr(layer, layer.param_maps['ffn']).d_ff # t1, sigmoid(t1), sigmoid(t1)*t1, t3; d_ff is FFN layer width
            activation_memory += batch_size * seq_len * getattr(layer, layer.param_maps['ffn']).d_ff # sigmoid(t1) * t1 * t3
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
        for layer in getattr(self, self.param_maps['layers']):
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
            linear_flops += batch_size * seq_len * layer.d_model * getattr(layer, layer.param_maps['ffn']).d_ff * 6
        # lastln
        ln_flops += batch_size * seq_len * getattr(self, self.param_maps['ln_final']).d_model * 2 + batch_size * seq_len * 2 + batch_size * seq_len * getattr(self, self.param_maps['ln_final']).d_model * 2
        # last_embed
        embed_flops += batch_size * seq_len * getattr(self, self.param_maps['ln_final']).d_model * getattr(self, self.param_maps['lm_head']).weight.shape[1] * 2

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

