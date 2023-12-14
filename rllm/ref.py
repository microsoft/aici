import torch
import sys
from safetensors import safe_open
from safetensors.torch import save_file

# save_file(tensors, "model.safetensors")

N = 256

def main():
    inp = {}
    with safe_open("tmp/reference.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            inp[key] = f.get_tensor(key)
    tokens = []
    logits = []
    prob_mass = []
    for i in range(1000):
        l: torch.Tensor = inp.get("logits_%d" % i, None)
        if l is None:
            break
        print(".", end="")
        sys.stdout.flush()
        ll = l.tolist()
        with_id = list(zip(ll, range(len(ll))))
        with_id.sort()
        with_id.reverse()
        l = torch.tensor([p for (p, _) in with_id])
        toks = torch.tensor([idx for (_, idx) in with_id], dtype=torch.int32)
        prob = l.softmax(0)[0:N].sum().tolist()
        prob_mass.append(prob)
        logits.append(l[0:N])
        tokens.append(toks[0:N])
        # print(N, prob, toks[0:N], with_id[0:N])
        # break
    
    pm = torch.tensor(prob_mass, dtype=torch.float32)
    print("\n", N, pm.min(), pm.mean(), pm.max())

    out = {
        "prompt": inp["prompt"].to(torch.int32),
        "output": inp["output"].to(torch.int32),
        "prob_mass": torch.tensor(prob_mass, dtype=torch.float32),
        "tokens": torch.stack(tokens, 0).to(torch.int32),
        "logits": torch.stack(logits, 0).to(torch.half),
    }
    save_file(out, "tmp/compr.safetensors")


main()
