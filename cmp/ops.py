import torch
from torch import Tensor

__all__ = ["enum_edges_t_2d", "joint_pairs_t_2d_d0"]


def enum_edges_t_2d(image: Tensor, threshold: float) -> Tensor:
    return torch.ops.cubical_ph.enum_edges_t_2d(image, threshold)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.impl_abstract("cubical_ph::enum_edges_t_2d")
def _(image: Tensor, threshold) -> Tensor:
    torch._check(len(image.shape) == 3)
    torch._check(threshold > 0)
    birth = torch.empty((image.shape[0], 2) + image.shape[1:], device=image.device, dtype=image.dtype)
    return birth


def joint_pairs_t_2d_d0(
    image: Tensor, values: Tensor, indices: Tensor, argmax_idx: Tensor, threshold: float, buffer_size=512
) -> tuple[Tensor, Tensor]:
    return torch.ops.cubical_ph.joint_pairs_t_2d_d0(image, values, indices, argmax_idx, threshold, buffer_size)


@torch.library.impl_abstract("cubical_ph::joint_pairs_t_2d_d0")
def _(
    image: Tensor, values: Tensor, indices: Tensor, argmax_idx: Tensor, threshold: float, buffer_size=512
) -> tuple[Tensor, Tensor]:
    M = 2
    torch._check(len(image.shape) == 3)
    torch._check(len(values.shape) == 2)
    torch._check(values.shape == indices.shape)

    torch._check(len(argmax_idx.shape) == 2)

    torch._check(image.shape[0] == indices.shape[0])
    torch._check(image.shape[1] * image.shape[2] * M == indices.shape[1])

    torch._check(indices.shape[0] == argmax_idx.shape[0])
    torch._check(argmax_idx.shape[0] == 2)

    torch._check(threshold > 0)
    return torch.empty((image.shape[0], buffer_size, 4), dtype=indices.dtype, device=indices.device), torch.empty(
        (image.shape[0]), dtype=indices.dtype, device=indices.device
    )


def enum_edges_v_2d(image: Tensor, threshold: float, alexander: bool = False) -> Tensor:
    return torch.ops.cubical_ph.enum_edges_v_2d(image, threshold, alexander)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.impl_abstract("cubical_ph::enum_edges_v_2d")
def _(image: Tensor, threshold: float, alexander: bool = False) -> Tensor:
    torch._check(len(image.shape) == 3)
    torch._check(threshold > 0)
    M = 4 if alexander else 2
    birth = torch.empty((image.shape[0], M) + image.shape[1:], device=image.device, dtype=image.dtype)
    return birth


def joint_pairs_v_2d(
    image: Tensor,
    values: Tensor,
    indices: Tensor,
    argmax_idx: Tensor,
    threshold: float,
    current_dim: int = 0,
    alexander: bool = False,
    buffer_size=512,
) -> tuple[Tensor, Tensor]:
    return torch.ops.cubical_ph.joint_pairs_v_2d(
        image, values, indices, argmax_idx, threshold, current_dim, alexander, buffer_size
    )


@torch.library.impl_abstract("cubical_ph::joint_pairs_v_2d")
def _(
    image: Tensor,
    values: Tensor,
    indices: Tensor,
    argmax_idx: Tensor,
    threshold: float,
    current_dim: int = 0,
    alexander: bool = False,
    buffer_size=512,
) -> tuple[Tensor, Tensor]:
    M = 4 if alexander else 2
    torch._check(len(image.shape) == 3)
    torch._check(len(values.shape) == 2)
    torch._check(values.shape == indices.shape)

    torch._check(len(argmax_idx.shape) == 2)

    torch._check(image.shape[0] == indices.shape[0])
    torch._check(image.shape[1] * image.shape[2] * M == indices.shape[1])

    torch._check(indices.shape[0] == argmax_idx.shape[0])
    torch._check(argmax_idx.shape[0] == 2)

    torch._check(threshold > 0)
    return torch.empty((image.shape[0], buffer_size, 4), dtype=indices.dtype, device=indices.device), torch.empty(
        (image.shape[0]), dtype=indices.dtype, device=indices.device
    )
