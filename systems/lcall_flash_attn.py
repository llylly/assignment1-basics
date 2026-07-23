import torch
from einops import rearrange

def CallFlashAttn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None, padded_tokens: torch.Tensor | None = None) -> torch.Tensor:
    import flash_attn
    import flash_attn.bert_padding

    """
    flash_attn works only when:
    1. mask is None -> causal=False (inference decoding phase)
    2. mask is not None, then
        Require Q.len = K.len = V.len
        padded_tokens is None -> causal=True (pretraining)
        left_padding style padded_tokens -> casual=True (current batch inference)
        right_padding style padded_tokens -> casual=True (post-training)
    q: [B,H,I,L1,D] or [B,H,1,L1,D]
    k & v: [B,H,1,L2,D]
    mask: [L1, L2]
    padded_tokens: [B, L2]
    """
    if mask is None:
        # non-causal
        b, s, i, l = Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]
        q_new = rearrange(Q, 'b s i l d -> (b l) (s i) d')
        q_cu_seqlens = torch.arange(0, b*l + 1, l, device='cuda').type(torch.int32)
        if padded_tokens is not None:
            k_new, _, k_cu_seqlens, k_max_seqlen, _ = flash_attn.bert_padding.unpad_input(rearrange(K, 'b s i l d -> b l (s i) d'), ~padded_tokens)
            v_new, _, _, _, _ = flash_attn.bert_padding.unpad_input(rearrange(V, 'b s i l d -> b l (s i) d'), ~padded_tokens)
        else:
            k_max_seqlen = K.shape[3]
            k_cu_seqlens = torch.arange(0, b*k_max_seqlen + 1, k_max_seqlen, device='cuda').type(torch.int32)
            k_new = rearrange(K, 'b s i l d -> (b l) (s i) d')
            v_new = rearrange(V, 'b s i l d -> (b l) (s i) d')
        
        output = flash_attn.flash_attn_varlen_func(q_new, k_new, v_new, q_cu_seqlens, k_cu_seqlens, l, k_max_seqlen, causal=False)
        output = rearrange(output, '(b l) (s i) d -> b s i l d', b=b, l=l, s=s, i=i)
    else:
        # casual
        # TODO: assert that it is a casual mask.
        assert Q.shape[0] == K.shape[0] and K.shape[0] == V.shape[0]
        assert Q.shape[3] == K.shape[3] and K.shape[3] == V.shape[3]
        if padded_tokens is None:
            q_new = rearrange(Q, 'b s i l d -> b l (s i) d')
            k_new = rearrange(K, 'b s i l d -> b l (s i) d')
            v_new = rearrange(V, 'b s i l d -> b l (s i) d')
            output = flash_attn.flash_attn_func(q_new, k_new, v_new, causal=True)
            output = rearrange(output, 'b l (s i) d -> b s i l d', s=Q.shape[1], i=Q.shape[2])
        else:
            is_left_padding = (padded_tokens[:, :-1] & (~padded_tokens[:, 1:])).any(dim=-1).any(dim=0)
            is_right_padding = ((~padded_tokens[:, :-1]) & padded_tokens[:, 1:]).any(dim=-1).any(dim=0)
            assert not (is_left_padding and is_right_padding), "Flash attn doesn't support free padding. It has to be left padding or right padding"
            if not is_right_padding:
                # left padding branch
                # print('left padding branch')
                b, s, i, l = Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]
                q_new = rearrange(Q, 'b s i l d -> (b l) (s i) d')
                q_cu_seqlens = torch.arange(0, b*l + 1, l, device='cuda').type(torch.int32)
                k_new, _, k_cu_seqlens, k_max_seqlen, _ = flash_attn.bert_padding.unpad_input(rearrange(K, 'b s i l d -> b l (s i) d'), ~padded_tokens)
                v_new, _, _, _, _ = flash_attn.bert_padding.unpad_input(rearrange(V, 'b s i l d -> b l (s i) d'), ~padded_tokens)
                output = flash_attn.flash_attn_varlen_func(q_new, k_new, v_new, q_cu_seqlens, k_cu_seqlens, l, k_max_seqlen, causal=True)
                output = rearrange(output, '(b l) (s i) d -> b s i l d', b=b, l=l, s=s, i=i)
            else:
                # right padding branch
                # print('right padding branch')
                q_new = rearrange(Q, 'b s i l d -> b l (s i) d')
                k_new = rearrange(K, 'b s i l d -> b l (s i) d')
                v_new = rearrange(V, 'b s i l d -> b l (s i) d')
                output = flash_attn.flash_attn_func(q_new, k_new, v_new, causal=True)
                output[padded_tokens.bool()] = 0. # for efficiency concerns, we can even just discard this
                output = rearrange(output, 'b l (s i) d -> b s i l d', s=Q.shape[1], i=Q.shape[2])
    return output
