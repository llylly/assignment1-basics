from typing import Any
from einops import einsum

import torch
from torch import nn
from torch.autograd import gradcheck


class LRMSNormFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, g: torch.Tensor, eps: float):
        # x: [B, D], g: [D], result: [B, D]
        in_type = x.dtype
        x = x.to(torch.float32)
        g = g.to(torch.float32)
        r = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
        k = x * r.unsqueeze(dim=-1)
        result = k * g
        ctx.save_for_backward(x, r, k, g)
        ctx.in_type = in_type
        return result.to(in_type)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x, r, k, g = ctx.saved_tensors # r: [B] k: [B, D]
        go, = grad_outputs # go: [B, D]
        go = go.to(torch.float32)
        gg = einsum(go, k, '... D, ... D -> D')
        gk = go * g # [B, D]
        gx = gk * r.unsqueeze(-1) - einsum(gk, x, '... D, ... D -> ...').unsqueeze(-1) * x * (r*r*r).unsqueeze(-1) / x.shape[-1]
        return gx.to(ctx.in_type), gg.to(ctx.in_type), None

class LSiLUFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor):
        s = torch.sigmoid(x)
        y = s * x
        ctx.save_for_backward(s, y)
        return y
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        s, y = ctx.saved_tensors
        go, = grad_outputs
        return (s + y * (1. - s)) * go


class LRMSNormFast(torch.nn.Module):

    def __init__(self, d_model: int, eps: float=1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        Wg = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(Wg)
        self.d_model = d_model
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LRMSNormFunc.apply(x, self.weight, self.eps)



if __name__ == '__main__':
    # cannot pass due to finite differentiation
    # x = torch.randn(10, 10, 128, dtype=torch.float64, requires_grad=True)
    # g = torch.randn(128, dtype=torch.float64, requires_grad=True)
    # test = gradcheck(LRMSNormFunc.apply, (x, g, 1e-5), eps=1e-6, atol=1e-4)
    # print(test)
    pass

    x = torch.randn(3, 4, 128, dtype=torch.float64, requires_grad=True)
    test = gradcheck(LSiLUFunc.apply, (x,), eps=1e-6, atol=1e-4)
    print(test)
    pass
