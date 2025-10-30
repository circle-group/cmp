import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops

from .util.typing import *
from .util.seq import pad_stack_pairs


def create_grid_tensor(image_data: Tensor, threshold: float = 1e10, embedded: bool = False) -> Tensor:
    dim = len(image_data.shape)
    assert 2 <= dim <= 4
    padding = (1, 1) * (dim - 1)
    if embedded:
        image_data = F.pad(-1 * image_data, padding, mode="constant", value=-threshold)
    return F.pad(image_data, padding, mode="constant", value=threshold)


def cubical_persistence_t_2d(
    image_data: Tensor, threshold: float = 1e10, embedded: bool = False, buffer_size: int = 512
) -> Tuple[Tensor, Tensor]:
    orig_shape = image_data.shape
    assert len(orig_shape) == 4
    image = image_data.view((orig_shape[0] * orig_shape[1], *orig_shape[2:])).contiguous()
    with torch.no_grad():
        argmax_idx = image.flatten(start_dim=1).argmax(dim=1)
        argmax_idx = torch.stack((argmax_idx / image.shape[-1], argmax_idx % image.shape[-1]), dim=1)
        if embedded:
            argmax_idx.add_(1)
        grid = create_grid_tensor(image, threshold=threshold, embedded=embedded)

        birth = ops.enum_edges_t_2d(grid, threshold)

        values, indices = birth.flatten(start_dim=1).sort(dim=-1, descending=True, stable=True)
        cofaces, lengths = ops.joint_pairs_t_2d_d0(
            grid, values, indices.int(), argmax_idx.int(), threshold=threshold, buffer_size=buffer_size
        )
        print(cofaces)
        if embedded:
            print(cofaces)
            cofaces.sub_(1).clamp_min_(-1)
            print(cofaces, lengths)

        birth_cy, birth_cx = cofaces[:, :, 0], cofaces[:, :, 1]
        death_cy, death_cx = cofaces[:, :, 2], cofaces[:, :, 3]

        batch_idx = torch.arange(image.shape[0]).view(-1, 1).expand_as(birth_cy)

    pairs_0d = torch.stack(
        (
            torch.where(birth_cy == -1, 0, image[batch_idx, birth_cy, birth_cx]),
            torch.where(death_cy == -1, 0, image[batch_idx, death_cy, death_cx]),
        ),
        dim=-1,
    )

    return pairs_0d.view((orig_shape[0], orig_shape[1], *pairs_0d.shape[1:])), lengths


def cubical_persistence_v_2d(
    image_data: Tensor,
    threshold: float = 1e10,
    current_dim=0,
    alexander: bool = False,
    embedded: bool = False,
    buffer_size: int = 512,
) -> Tuple[Tensor, Tensor]:
    orig_shape = image_data.shape
    assert len(orig_shape) == 4
    image = image_data.view((orig_shape[0] * orig_shape[1], *orig_shape[2:])).contiguous()
    with torch.no_grad():
        argmax_idx = image.flatten(start_dim=1).argmax(dim=1)
        argmax_idx = torch.stack((argmax_idx / image.shape[-1], argmax_idx % image.shape[-1]), dim=1)
        if embedded:
            argmax_idx.add_(1)

        grid = create_grid_tensor(image, threshold=threshold, embedded=embedded)

        birth = ops.enum_edges_v_2d(grid, threshold, alexander=alexander)

        values, indices = birth.flatten(start_dim=1).sort(dim=-1, descending=True, stable=True)
        cofaces, lengths = ops.joint_pairs_v_2d(
            grid,
            values,
            indices.int(),
            argmax_idx.int(),
            threshold=threshold,
            current_dim=current_dim,
            alexander=alexander,
            buffer_size=buffer_size,
        )
        if embedded:
            cofaces.sub_(1).clamp_min_(-1)

        birth_cy, birth_cx = cofaces[:, :, 0], cofaces[:, :, 1]
        death_cy, death_cx = cofaces[:, :, 2], cofaces[:, :, 3]

        batch_idx = torch.arange(image.shape[0]).view(-1, 1).expand_as(birth_cy)

    if cofaces.numel() == 0:
        return (
            torch.empty((orig_shape[0], orig_shape[1], *birth_cy.shape[1:], 0), device=image.device, dtype=image.dtype),
            lengths,
        )

    pairs_0d = torch.stack(
        (
            torch.where(birth_cy == -1, 0, image[batch_idx, birth_cy, birth_cx]),
            torch.where(death_cy == -1, 0, image[batch_idx, death_cy, death_cx]),
        ),
        dim=-1,
    )

    return pairs_0d.view((orig_shape[0], orig_shape[1], *pairs_0d.shape[1:])), lengths


def cubical_persistence_v_2d_full(
    image_data: Tensor, threshold: float = 1e10, buffer_size: int = 512
) -> Tuple[Tensor, Tensor]:
    orig_shape = image_data.shape
    assert len(orig_shape) == 4
    image = image_data.view((orig_shape[0] * orig_shape[1], *orig_shape[2:])).contiguous()
    with torch.no_grad():
        argmax_idx = image.flatten(start_dim=1).argmax(dim=1)
        argmax_idx = torch.stack((argmax_idx / image.shape[-1], argmax_idx % image.shape[-1]), dim=1)

        # 0d hom
        alexander = False
        embedded = False
        grid = create_grid_tensor(image, threshold=threshold, embedded=embedded)

        birth = ops.enum_edges_v_2d(grid, threshold, alexander=alexander)
        values, indices = birth.flatten(start_dim=1).sort(dim=-1, descending=True, stable=True)

        cofaces, lengths = ops.joint_pairs_v_2d(
            grid,
            values,
            indices.int(),
            argmax_idx.int(),
            threshold=threshold,
            current_dim=0,
            alexander=alexander,
            buffer_size=buffer_size,
        )

        birth_cy, birth_cx = cofaces[:, :, 0], cofaces[:, :, 1]
        death_cy, death_cx = cofaces[:, :, 2], cofaces[:, :, 3]

        batch_idx = torch.arange(image.shape[0]).view(-1, 1).expand_as(birth_cy)

    lengths_0d = lengths
    # if cofaces.numel() == 0:
    #     pairs_0d = torch.zeros(
    #         (orig_shape[0]*orig_shape[1], *birth_cy.shape[1:], 0), device=image.device, dtype=image.dtype)
    # else:
    pairs_0d = torch.stack(
        (
            torch.where(birth_cy == -1, 0, image[batch_idx, birth_cy, birth_cx]),
            torch.where(death_cy == -1, 0, image[batch_idx, death_cy, death_cx]),
        ),
        dim=-1,
    )

    with torch.no_grad():
        # 1d hom
        alexander = True
        embedded = True
        argmax_idx.add_(1)

        grid = create_grid_tensor(image, threshold=threshold, embedded=embedded)

        birth = ops.enum_edges_v_2d(grid, threshold, alexander=alexander)
        values, indices = birth.flatten(start_dim=1).sort(dim=-1, descending=True, stable=True)

        cofaces, lengths = ops.joint_pairs_v_2d(
            grid,
            values,
            indices.int(),
            argmax_idx.int(),
            threshold=threshold,
            current_dim=1,
            alexander=alexander,
            buffer_size=buffer_size,
        )
        if cofaces.numel() != 0:
            cofaces.sub_(1).clamp_min_(-1)

            birth_cy, birth_cx = cofaces[:, :, 0], cofaces[:, :, 1]
            death_cy, death_cx = cofaces[:, :, 2], cofaces[:, :, 3]

            batch_idx = torch.arange(image.shape[0]).view(-1, 1).expand_as(birth_cy)

    lengths_1d = lengths
    if cofaces.numel() == 0:
        pairs_1d = torch.zeros(
            (orig_shape[0] * orig_shape[1], *birth_cy.shape[1:], 2), device=image.device, dtype=image.dtype
        )
    else:
        pairs_1d = torch.stack(
            (
                torch.where(birth_cy == -1, 0, image[batch_idx, birth_cy, birth_cx]),
                torch.where(death_cy == -1, 0, image[batch_idx, death_cy, death_cx]),
            ),
            dim=-1,
        )

    pairs = pad_stack_pairs([pairs_0d, pairs_1d], pad_value=0)
    lengths = torch.stack((lengths_0d, lengths_1d), dim=-1)

    pairs = pairs.view((orig_shape[0], orig_shape[1], *pairs.shape[1:]))
    lengths = lengths.view((orig_shape[0], orig_shape[1], *lengths.shape[1:]))

    return pairs, lengths
