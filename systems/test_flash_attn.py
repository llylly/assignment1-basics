import torch
from basics.lmodeling import LNaiveSDPA
from call_flash_attn import CallFlashAttn

# q: [B,H,I,L,D] or [B,H,1,L,D]
# k & v: [B,H,1,L,D]

if __name__ == '__main__':

    q = torch.randn(4, 4, 3, 188, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(4, 4, 1, 188, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(4, 4, 1, 188, 64, device='cuda', dtype=torch.bfloat16)

    r_padded_tokens = torch.zeros([4, 188], dtype=torch.bool, device='cuda')
    r_padded_tokens[0, 150:] = True
    r_padded_tokens[1, 134:] = True
    r_padded_tokens[2, 134:] = True
    r_padded_tokens[3, 100:] = True

    mask = torch.triu(torch.ones((188, 188), dtype=torch.bool, device='cuda')).T


    my_output = LNaiveSDPA(q, k, v, padded_tokens=r_padded_tokens)
    output = CallFlashAttn(q, k, v, padded_tokens=r_padded_tokens)
    print(torch.max(torch.abs(output - my_output)))


    my_output = LNaiveSDPA(q, k, v, mask=mask, padded_tokens=r_padded_tokens)
    output = CallFlashAttn(q, k, v, mask=mask, padded_tokens=r_padded_tokens)
    my_output[0, :, :, 150:] = 0.
    my_output[1, :, :, 134:] = 0.
    my_output[2, :, :, 134:] = 0.
    my_output[3, :, :, 100:] = 0.
    print(torch.max(torch.abs(output - my_output)))

    l_padded_tokens = torch.zeros([4, 188], dtype=torch.bool, device='cuda')
    l_padded_tokens[0, :43] = True
    l_padded_tokens[1, :63] = True
    l_padded_tokens[2, :125] = True
    l_padded_tokens[3, :53] = True

    my_output = LNaiveSDPA(q, k, v, mask=mask, padded_tokens=l_padded_tokens)
    output = CallFlashAttn(q, k, v, mask=mask, padded_tokens=l_padded_tokens)
    my_output[0, :, :, :43] = 0.
    my_output[1, :, :, :63] = 0.
    my_output[2, :, :, :125] = 0.
    my_output[3, :, :, :53] = 0.
    print(torch.max(torch.abs(output - my_output)))


    my_output = LNaiveSDPA(q[:,:,:,-1:], k, v, padded_tokens=r_padded_tokens)
    output = CallFlashAttn(q[:,:,:,-1:], k, v, padded_tokens=r_padded_tokens)
    print(torch.max(torch.abs(output - my_output)))

    my_output = LNaiveSDPA(q[:,:,:,-1:], k, v, padded_tokens=l_padded_tokens)
    output = CallFlashAttn(q[:,:,:,-1:], k, v, padded_tokens=l_padded_tokens)
    print(torch.max(torch.abs(output - my_output)))

    my_output = LNaiveSDPA(q[:,:,:,-1:], k, v, padded_tokens=r_padded_tokens)
    output = CallFlashAttn(q[:,:,:,-1:], k, v, padded_tokens=r_padded_tokens)
    print(torch.max(torch.abs(output - my_output)))