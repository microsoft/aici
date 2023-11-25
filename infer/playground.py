import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalFromBottomRightMask,
)
from typing import Sequence

class XorShiftRng:
    def __init__(self, seed=12345):
        self.state = seed

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x & 0xFFFFFFFF

    def urandom(self):
        return self.next() / 0xFFFFFFFF

    def srandom(self):
        return self.urandom() * 2.0 - 1.0

    def rand_tensor(self, shape, dtype=torch.bfloat16, device='cuda'):
        data = [self.srandom() for _ in range(torch.Size(shape).numel())]
        tensor = torch.tensor(data, device=device, dtype=torch.float32)
        tensor = tensor.reshape(shape).to(dtype=dtype)
        return tensor

def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlen: Sequence[int],
    kv_seqlen,
    _p: float,
    softmax_scale: float,
):
    out = xops.memory_efficient_attention_forward(
        q.unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        v.unsqueeze(0).transpose(1, 2),
        attn_bias=BlockDiagonalCausalFromBottomRightMask.from_seqlens(q_seqlen, kv_seqlen),
        p=_p,
        scale=softmax_scale,
    ).transpose(1, 2).squeeze(0)
    return out

def playground_1():
    torch.set_printoptions(sci_mode=False)
    xor = XorShiftRng()
    device = torch.device('cuda:0')

    slen = 5
    pref = 2
    head_dim = 8
    n_heads = 1
    query = xor.rand_tensor([n_heads, slen, head_dim], device=device)
    key = xor.rand_tensor([n_heads, slen, head_dim], device=device)
    value = xor.rand_tensor([n_heads, slen, head_dim], device=device)

    q1 = query[:, :pref, :]
    q2 = query[:, pref:, :]
    k1 = key[:, :pref, :]
    v1 = value[:, :pref, :]

    # out = flash_attn(query, key, value, [slen], [slen], 0.0, 1.0)
    # print(out)
    # o1 = flash_attn(q1, k1, v1, [pref], [pref], 0.0, 1.0)
    # print(o1)
    o2 = flash_attn(q2, key, value, [slen - pref], [slen], 0.0, 1.0)
    print(o2)

    # print(out.size())
    # print(o1.size())
    # print(o2.size())

    # print("pref")
    # check_all_close(out[:, :pref, :], o1)

    # print("suff")
    # check_all_close(out[:, pref:, :], o2)

def check_all_close(tensor1, tensor2, rtol=1e-05, atol=1e-08):
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

playground_1()