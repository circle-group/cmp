import torch
import torch.nn as nn
import torch.nn.functional as F

import cmp.cubical as cubical

image = torch.tensor([[
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
], [
    [1, 1, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 0, 1]
]], dtype=torch.float32, device="cuda:0", requires_grad=True).unsqueeze(1)
print(f"Image:\n{image} {image.shape}")

# print("T Construction")
# pairs, lengths = cubical.cubical_persistence_t_2d(image, 1000, False)
# print(f"Pairs:\n{pairs} {pairs.shape}")
# print(f"Lengths:\n{lengths} {lengths.shape}")

print("V Construction")
pairs, lengths = cubical.cubical_persistence_v_2d(image, 1000, 0, False, False)
print(f"Pairs:\n{pairs} {pairs.shape}")
print(f"Lengths:\n{lengths} {lengths.shape}")

print("Embedded V Construction - Alexander")
pairs, lengths = cubical.cubical_persistence_v_2d(image, 1000, 1, True, True)
print(f"Pairs:\n{pairs} {pairs.shape}")
print(f"Lengths:\n{lengths} {lengths.shape}")

print("Full V")
pairs, lengths = cubical.cubical_persistence_v_2d_full(image, 1000)
print(f"Pairs:\n{pairs} {pairs.shape}")
print(f"Lengths:\n{lengths} {lengths.shape}")