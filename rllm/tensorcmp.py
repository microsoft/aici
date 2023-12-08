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
    # if b.shape[0] == 27:
    #     b = b[4:16, :, :]
    # elif b.shape[1] == 27:
    #     b = b[:, 4:16, :]

    if a.shape == b.shape:
        print(a.shape)
        check_all_close(a, b)
    else:
        print("Size wrong", a.shape, b.shape)
        assert False

def load_all(fn: str) -> dict[str, torch.Tensor]:
    r = {}
    with safetensors.safe_open(fn, framework="pt", device="cuda") as a:
        keys = a.keys()
        keys.sort()
        for key in keys:
            kk = key.split("_")[1]
            if kk not in r:
                r[kk] = a.get_tensor(key)
    return r

def main():
    a = load_all(args[0])
    b = load_all(args[1])

    x1 = a["x1"]
    w1 = a["w1"].t().unsqueeze(0)

    m1 = a["m1"]
    print("X", x1.shape, w1.shape)
    m1p = x1.matmul(w1)
    #cmp(m1, m1p)

    bx1 = b["x1"]
    bw1 = b["w1"].t().unsqueeze(0)
    cmp(w1, bw1)
    cmp(bx1[:,4:16,:], x1)
    bm1 = bx1.matmul(bw1)
    cmp(bm1[:,4:16,:], m1p)



    
    # for key in a.keys():
    #     print(key)
    #     cmp(a[key], b[key])

main()