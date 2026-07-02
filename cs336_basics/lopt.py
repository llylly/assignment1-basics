import math
from typing import Optional
from collections.abc import Callable, Iterable
import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int

def LCrossEntropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    col_max = inputs.amax(dim=-1, keepdim=True)
    second = torch.log(torch.exp(inputs - col_max).sum(dim=-1, keepdim=True))
    first = -torch.gather(inputs, -1, targets.unsqueeze(dim=-1)) + col_max
    return torch.mean(first + second)

class LSqaureRootSGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t+1
        return loss

class LSGD(torch.optim.Optimizer):

    def __init__(self, params, lr: float, momentum: float, weight_decay: float):
        if lr < 0 or (not 0. <= momentum <= 1.):
            raise ValueError('Invalid hyperparameters')
        defaults = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay, 'mementum_pow': 1.}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['m'] = torch.zeros_like(p)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, momentum, weight_decay = group['lr'], group['momentum'], group['weight_decay']
            group['momentum_pow'] *= momentum
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                state['m'] = momentum * state['m'] + (1. - momentum) * p.grad.data
                p.data = p.data - lr * weight_decay * p.data - lr * state['m'] / (1. - group['momentum_pow'])
        return loss

class LAdamW(torch.optim.Optimizer):

    def __init__(self, params, lr: float, betas: tuple[float, float], weight_decay: float, eps: float=1e-8):
        if lr < 0 or (not 0. <= betas[0] <= 1.) or (not 0. <= betas[1] <= 1.) or (not 0. <= weight_decay <= 1.):
            raise ValueError("Invalid hyperparameters")
        defaults = {'lr': lr, 'beta1': betas[0], 'beta2': betas[1], 'weight_decay': weight_decay, 'epsilon': eps,
                    'beta1pow': 1., 'beta2pow': 1.}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['m'] = torch.zeros_like(p)
                self.state[p]['v'] = torch.zeros_like(p)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, beta1, beta2, weight_decay, epsilon = group['lr'], group['beta1'], group['beta2'], group['weight_decay'], group['epsilon']
            group['beta1pow'] *= beta1
            group['beta2pow'] *= beta2
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                state['m'] = beta1 * state['m'] + (1. - beta1) * p.grad.data
                state['v'] = beta2 * state['v'] + (1. - beta2) * p.grad.data * p.grad.data
                p.data = p.data - lr * weight_decay * p.data - lr * math.sqrt(1. - group['beta2pow']) / (1. - group['beta1pow']) * state['m'] / (torch.sqrt(state['v']) + epsilon)
        return loss

def LCosineLR(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + (0.5 + 0.5 * math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def LGradientClipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> int:
    norm = None
    for param in parameters:
        if param.grad is not None:
            if norm is None: 
                norm = torch.sum(param.grad * param.grad) 
            else: norm += torch.sum(param.grad * param.grad)
    norm = norm.sqrt()
    if norm >= max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_l2_norm / (norm + 1e-6)
    return norm

if __name__ == '__main__':
    # minimal example
    weights = torch.nn.Parameter(5 * torch.randn((10, 10), device='cuda'))
    print('init:', (weights**2).mean())
    opt = LSqaureRootSGD([weights], lr=1000)
    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        if t % 10 == 0: print(loss.cpu().item())
        loss.backward()
        opt.step()