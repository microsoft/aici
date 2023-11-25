import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalFromBottomRightMask,
)

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

def move(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).cuda()


def test():
    xor = XorShiftRng()
    print(xor.rand_tensor((3, 3)))
    return

    slen = 12
    pref = 5
    head_dim = 128
    n_heads = 32
    q1 = move(torch.randn(1, n_heads, pref, head_dim))
    q2 = move(torch.randn(1, n_heads, slen-pref, head_dim))
    query = torch.cat([q1, q2], dim=-2)
    key = move(torch.randn(1, n_heads, slen, head_dim))
    value = move(torch.randn(1, n_heads, slen, head_dim))

    out = xops.memory_efficient_attention_forward(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_bias=BlockDiagonalCausalFromBottomRightMask.from_seqlens([slen], [slen]),
        p=0.0,
        scale=1.0,
    ).transpose(1, 2).squeeze(0)
    o1 = xops.memory_efficient_attention_forward(
        q1.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_bias=BlockDiagonalCausalFromBottomRightMask.from_seqlens([pref], [pref]),
        p=0.0,
        scale=1.0,
    ).transpose(1, 2).squeeze(0)
    o2 = xops.memory_efficient_attention_forward(
        q2.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_bias=BlockDiagonalCausalFromBottomRightMask.from_seqlens([slen-pref], [slen]),
        p=0.0,
        scale=1.0,
    ).transpose(1, 2).squeeze(0)
    print(o1.shape)
    print(o2.shape)
    assert torch.allclose(out[:, 0:pref, :], o1)
    assert torch.allclose(out[:, pref:, :], o2)


test()
