import safetensors
import sys
import torch

args = sys.argv[1:]


def check_all_close(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol=1e-05, atol=1e-10):
    # assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    if torch.equal(tensor1, tensor2):
        return
    d = tensor1.sub(tensor2).abs()
    print(d)
    print(d.max())
    assert False


def cmp(a: torch.Tensor, b: torch.Tensor):
    if b.shape[0] == 27:
        b = b[4:16, :, :]
    elif b.shape[1] == 27:
        b = b[:, 4:16, :]

    if a.shape == b.shape:
        print(a.shape)
        check_all_close(a, b)
    else:
        print(a.shape, b.shape)
        assert False


with safetensors.safe_open(args[0], framework="pt", device="cpu") as a:
    with safetensors.safe_open(args[1], framework="pt", device="cpu") as b:
        keys = a.keys()
        keys.sort()
        for key in keys:
            aa = a.get_tensor(key)
            bb = b.get_tensor(key)
            print(key)
            cmp(aa, bb)
            # break
