import torch

ts = []
k = 0
for n in range(10000):
    t = torch.zeros(32 * 1024, 1024, dtype=torch.half, device="mps")
    k += t.numel() * 2
    print(k / 1024 / 1024)
    ts.append(t)
