import torch
import torch.nn as nn
import torch.nn.functional as F

from .typing import *

__all__ = ["pad_stack_pairs"]

def pad_stack_pairs(pairs_l: List[Tensor], pad_value=0) -> Tensor:
    # dim = -2
    lens = [x.shape[-2] for x in pairs_l]
    target_len = max(lens)

    last_l = 2
    if len(pairs_l) > 0:
        for x, L in zip(pairs_l, lens):
            if L != 0:
                last_l = x.shape[-1]
                break

    pairs_l = [torch.cat((
        x,
        x.new_full((*x.shape[0:-2], target_len-L, x.shape[-1]), pad_value)
    ), dim=-2) if L != 0 else x.new_full((*x.shape[0:-2], target_len, last_l), pad_value) for x, L in zip(pairs_l, lens)]
    return torch.stack(pairs_l, dim=-3)
