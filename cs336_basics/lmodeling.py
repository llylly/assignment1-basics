import math
from typing import Any
import torch
from torch import nn
from einops import rearrange, einsum

class LLinear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None=None):
        super().__init__()
        W_tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2. / (in_features+out_features))
        nn.init.trunc_normal_(W_tensor, mean=0., std=std, a=-3.*std, b=3.*std)
        self.weight = nn.Parameter(W_tensor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class LEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        W_embed = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(W_embed, mean=0., std=1., a=-3., b=3.)
        self.weight = nn.Parameter(W_embed)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model, self.d_ff = d_model, d_ff # use passed in arg
        # if d_model % 24 == 0:
        #     self.d_ff = d_model * 8 // 3
        # else:
        #     self.d_ff = ((d_model * 8 // 3) // 64 + 1) * 64 # upper ceil to multiples of 64
        self.w1 = LLinear(self.d_model, self.d_ff, device, dtype)
        self.w2 = LLinear(self.d_ff, self.d_model, device, dtype)
        self.w3 = LLinear(self.d_model, self.d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1 = self.w1.forward(x)
        t3 = self.w3.forward(x)
        return self.w2.forward(torch.sigmoid(t1) * t1 * t3)

class LROPE(torch.nn.Module):

    # need to be a singleton across layers
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        bases = torch.pow(torch.tensor(theta, device=device, dtype=torch.float32), -(torch.arange(1, d_k // 2 + 1, device=device, dtype=torch.float32) * 2 - 2) / d_k)
        angles = einsum(torch.arange(0, max_seq_len, device=device, dtype=torch.float32), bases, "seq_len, bases -> seq_len bases")
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)
        self.register_buffer('sin_angles', sin_angles)
        self.register_buffer('cos_angles', cos_angles)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x.view(-1, self.d_k)[:,::2].view(list(x.shape[:-1]) + [self.d_k // 2])
        x_odd = x.view(-1, self.d_k)[:,1::2].view(list(x.shape[:-1]) + [self.d_k // 2])
        cos_even_x = self.cos_angles[token_positions].to(x.dtype) * x_even
        cos_odd_x = self.cos_angles[token_positions].to(x.dtype) * x_odd
        nsin_odd_x = -self.sin_angles[token_positions].to(x.dtype) * x_odd
        sin_even_x = self.sin_angles[token_positions].to(x.dtype) * x_even
        ans_even = cos_even_x + nsin_odd_x
        ans_odd = cos_odd_x + sin_even_x
        ans = torch.stack([ans_even, ans_odd], dim=-1).contiguous().reshape(x.shape)
        return ans

def LSoftmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    rowmax = x.amax(dim=dim, keepdim=True)
    t = torch.exp(x - rowmax)
    return t / t.sum(dim=dim, keepdim=True)

def LNaiveSDPA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    QK = einsum(Q, K, '... queries d, ... keys d -> queries keys ...') / math.sqrt(Q.shape[-1])
    mask = rearrange(mask, '... queries keys -> queries keys ...')
    if mask is not None:
        QK[~mask] = -torch.inf
    mask = rearrange(mask, 'queries keys ... -> ... queries keys')
    return einsum(LSoftmax(QK, dim=1), V, 'queries keys ..., ... keys d -> ... queries d')

class LMHA(torch.nn.Module):
    # global singleton
    rope_cache: LROPE | None = None
    triu_cache: dict[int, torch.Tensor] = {}

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
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
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q = rearrange(self.q_proj.forward(x), '... seqlen (h d_k) -> ... seqlen h d_k', h=self.num_heads, d_k=self.d_k)
        k = rearrange(self.k_proj.forward(x), '... seqlen (h d_k) -> ... seqlen h d_k', h=self.num_heads, d_k=self.d_k)
        v = rearrange(self.v_proj.forward(x), '... seqlen (h d_k) -> ... h seqlen d_k', h=self.num_heads, d_k=self.d_k)
        if self.theta and LMHA.rope_cache:
            if token_positions is not None:
                token_positions = token_positions.unsqueeze(-1) # add head dim
            else:
                token_positions = torch.arange(q.shape[-3]).view(-1, 1) # default index of x
            q = LMHA.rope_cache.forward(q, token_positions)
            k = LMHA.rope_cache.forward(k, token_positions)
        q = rearrange(q, '... seqlen h d_k -> ... h seqlen d_k', h=self.num_heads, d_k=self.d_k)
        k = rearrange(k, '... seqlen h d_k -> ... h seqlen d_k', h=self.num_heads, d_k=self.d_k)
        before_proj = rearrange(LNaiveSDPA(q, k, v, self.triu_cache[self.max_seq_len][:q.shape[-2], :q.shape[-2]]), '... h seqlen d_k -> ... seqlen (h d_k)')
        return self.output_proj.forward(before_proj)

class LTransformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.ln1 = LRMSNorm(d_model, device=device, dtype=dtype)
        self.attn = LMHA(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln2 = LRMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = LFFN(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        t = x + self.attn.forward(self.ln1.forward(x), token_positions)
        return t + self.ffn.forward(self.ln2.forward(t))

class LTransformerLM(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int, vocab_size: int, num_layers: int, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.token_embeddings = LEmbedding(vocab_size, d_model, device, dtype)
        self.layers = nn.Sequential(*[LTransformerBlock(d_model, num_heads, d_ff, context_length, theta, device, dtype) for _ in range(num_layers)])
        self.ln_final = LRMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = LLinear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embeddings.forward(x)
        for layer in self.layers:
            x = layer.forward(x, token_positions)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x  


if __name__ == '__main__':
    # llinear = LLinear(50, 100)
    # lembed = LEmbedding(100, 10)
    # print(lembed.forward(torch.tensor([[1, 2], [2, 3]])))
    # lnorm = LRMSNorm(100, device='cuda')
    # print(lnorm.forward(torch.randn((100,100,100), device='cuda')))
    # lffn = LFFN(96, 110)
    # print(lffn.forward(torch.randn(20, 10, 96)).shape)
    # rope = LROPE(10000, 100, 1000)
    # print(rope.forward(torch.randn((5,10,100),), torch.randint(0, 1000, (5,10))).shape)
    # print(LSoftmax(torch.randn(5,5,5), dim=0))
    mha = LMHA(96, 4, 100, None)
    print(mha.forward(torch.randn((5, 10, 96))))
    pass

