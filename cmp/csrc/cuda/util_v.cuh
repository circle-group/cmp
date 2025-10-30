#pragma once

#include <torch/torch.h>

#include "common.h"
#include "cuda_common.cuh"

namespace cubical_ph::cuda::v_cons {

template <typename scalar_t>
CPH_HOST_DEVICE float get_birth_v_2d(const AccessorRestrict<scalar_t, 2> batch,
                                     index_t cy, index_t cx) {
  return batch[cy + 1][cx + 1];
}

template <typename scalar_t>
CPH_HOST_DEVICE float get_birth_v_2d_dim1(const AccessorRestrict<scalar_t, 2> batch,
                                        index_t cy, index_t cx,
                                        index_t m, float threshold) {
  switch (m) {
  case 0:
    return max(batch[cy + 1][cx + 1], batch[cy + 1][cx + 2]);
  case 1:
    return max(batch[cy + 1][cx + 1], batch[cy + 2][cx + 1]);
  case 2:
    return max(batch[cy + 1][cx + 1], batch[cy + 2][cx + 2]);
  case 3:
    return max(batch[cy + 1][cx + 1], batch[cy + 0][cx + 2]);
  default:
    break;
  }
  return threshold;
}

template <typename scalar_t>
CPH_HOST_DEVICE void get_parent_v_2d(const AccessorRestrict<scalar_t, 2> batch,
                                          index_t& cy, index_t& cx, scalar_t b) {
  if(b == batch[cy+1][cx+1]) {
    // pass
  } else if(b == batch[cy+1][cx+2]) {
    cx++;
  } else if(b == batch[cy+2][cx+2]) {
    cx++;
    cy++;
  } else if(b == batch[cy+2][cx+1]) {
    cy++;
  } else if(b == batch[cy+0][cx+2]) {
    cx++;
    cy--;
  } else {
    assert(false);
  }
}


} // namespace cubical_ph::t_cons::cuda
