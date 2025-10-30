#pragma once

#include <torch/torch.h>

#include "common.h"
#include "cuda_common.cuh"

namespace cubical_ph::cuda {

CPH_HOST_DEVICE index_t uf_find_pc(AccessorRestrict<index_t, 1> uf_parent_b,
                                   index_t x) {
  index_t y = x;
  index_t z = uf_parent_b[x];
  while (z != y) {
    y = z;
    z = uf_parent_b[y];
  }
  // Path compression
  y = uf_parent_b[x];
  while (z != y) {
    uf_parent_b[x] = z;
    x = y;
    y = uf_parent_b[x];
  }
  return z;
}

template <typename scalar_t>
CPH_HOST_DEVICE void uf_link_rank(AccessorRestrict<index_t, 1> uf_parent_b,
                                  AccessorRestrict<scalar_t, 1> uf_birth_b,
                                  index_t x, index_t y) {
  if (x == y)
    return;

  if (uf_birth_b[x] >= uf_birth_b[y]) {
    uf_parent_b[x] = y;
  } else /* if(uf_birth_b[x] < uf_birth_b[y]) */ {
    uf_parent_b[y] = x;
  }
}

} // namespace cubical_ph::t_cons::cuda