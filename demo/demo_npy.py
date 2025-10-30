import torch
import torch.nn as nn
import torch.nn.functional as F

import cmp.cubical as cubical

import numpy as np

FILE = "demo/nuclei_0000.npy"
with open(FILE, "rb") as f:
    image_np = np.load(f)
    image = torch.from_numpy(image_np)

image = image.float().to("cuda:0").unsqueeze(0).unsqueeze(0).clone().requires_grad_(True)
image = image.repeat(4, 2, 1, 1)
print(image.shape)


print("Full V")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
pairs, lengths = cubical.cubical_persistence_v_2d_full(image, 1000)
end.record()

torch.cuda.synchronize()
print(f"Took: {start.elapsed_time(end)} ms")

print(f"Pairs:\n{pairs} {pairs.shape}")
print(f"Lengths:\n{lengths} {lengths.shape}")