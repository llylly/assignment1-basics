import torch
import flash_attn

if __name__ == '__main__':
    output = flash_attn.flash_attn_qkvpacked_func(torch.randn(8, 128, 3, 8, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True), 0.0, causal=True)
    torch.sum(output).backward()
    print(output)