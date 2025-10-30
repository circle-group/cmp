#pragma once

#include <torch/torch.h>

#include "common.h"
#include "cuda_common.cuh"

namespace cubical_ph::cuda::t_cons {
template <typename scalar_t>
CPH_HOST_DEVICE float get_birth_t_1d(const AccessorRestrict<scalar_t, 1> batch,
                                     index_t cx) {
  return min(batch[cx], batch[cx + 1]);
}

template <typename scalar_t>
CPH_HOST_DEVICE float get_birth_t_1d(const AccessorRestrict<scalar_t, 1> batch,
                                     index_t cx, index_t dim, float threshold) {
  switch (dim) {
  case 1:
    return batch[cx + 1];
  case 0:
    return min(batch[cx], batch[cx + 1]);
  default:
    break;
  }
  return threshold;
}

template <typename scalar_t>
CPH_HOST_DEVICE float get_birth_t_2d(const AccessorRestrict<scalar_t, 2> batch,
                                     index_t cy, index_t cx) {
  return min(min(batch[cy][cx], batch[cy + 1][cx]),
             min(batch[cy][cx + 1], batch[cy + 1][cx + 1]));
}

template <typename scalar_t>
CPH_HOST_DEVICE float get_birth_t_2d(const AccessorRestrict<scalar_t, 2> batch,
                                     index_t cy, index_t cx, index_t dim,
                                     index_t m, float threshold) {
  switch (dim) {
  case 2:
    return batch[cy + 1][cx + 1];
  case 1:
    switch (m) {
    case 0:
      return min(batch[cy][cx + 1], batch[cy + 1][cx + 1]);
    case 1:
      return min(batch[cy + 1][cx], batch[cy + 1][cx + 1]);
    default:
      break;
    }
    break;
  case 0:
    return min(min(batch[cy][cx], batch[cy + 1][cx]),
               min(batch[cy][cx + 1], batch[cy + 1][cx + 1]));
  default:
    break;
  }
  return threshold;
}

template <typename scalar_t>
CPH_HOST_DEVICE float get_birth_t_3d(const AccessorRestrict<scalar_t, 3> batch,
                                     index_t cz, index_t cy, index_t cx) {
  return min(
      min(min(batch[cz][cy][cx], batch[cz + 1][cy][cx]),
          min(batch[cz + 1][cy + 1][cx], batch[cz][cy + 1][cx])),
      min(min(batch[cz][cy][cx + 1], batch[cz + 1][cy][cx + 1]),
          min(batch[cz + 1][cy + 1][cx + 1], batch[cz][cy + 1][cx + 1])));
}

template <typename scalar_t>
CPH_HOST_DEVICE float get_birth_t_3d(const AccessorRestrict<scalar_t, 3> batch,
                                     index_t cz, index_t cy, index_t cx,
                                     index_t dim, index_t m, float threshold) {
  switch (dim) {
  case 3:
    return batch[cz + 1][cy + 1][cx + 1];
  case 2:
    switch (m) {
    case 0:
      return min(batch[cz + 1][cy + 1][cx], batch[cz + 1][cy + 1][cx + 1]);
    case 1:
      return min(batch[cz + 1][cy][cx + 1], batch[cz + 1][cy + 1][cx + 1]);
    case 2:
      return min(batch[cz][cy + 1][cx + 1], batch[cz + 1][cy + 1][cx + 1]);
    default:
      break;
    }
  case 1:
    switch (m) {
    case 0:
      return min(min(batch[cz + 1][cy + 1][cx + 1], batch[cz + 1][cy + 1][cx]),
                 min(batch[cz + 1][cy][cx + 1], batch[cz + 1][cy][cx]));
    case 1:
      return min(min(batch[cz + 1][cy + 1][cx + 1], batch[cz][cy + 1][cx + 1]),
                 min(batch[cz + 1][cy + 1][cx], batch[cz][cy + 1][cx]));
    case 2:
      return min(min(batch[cz + 1][cy + 1][cx + 1], batch[cz][cy + 1][cx + 1]),
                 min(batch[cz + 1][cy][cx + 1], batch[cz][cy][cx + 1]));
    default:
      break;
    }
    break;
  case 0:
    return min(
        min(min(batch[cz][cy][cx], batch[cz + 1][cy][cx]),
            min(batch[cz + 1][cy + 1][cx], batch[cz][cy + 1][cx])),
        min(min(batch[cz][cy][cx + 1], batch[cz + 1][cy][cx + 1]),
            min(batch[cz + 1][cy + 1][cx + 1], batch[cz][cy + 1][cx + 1])));
  default:
    break;
  }
  return threshold;
}

template <typename scalar_t>
CPH_HOST_DEVICE void get_parent_t_2d(const AccessorRestrict<scalar_t, 2> batch,
                                          index_t& cy, index_t& cx, scalar_t b) {
  if(b == batch[cy+1][cx+1]) {
    // pass
  } else if(b == batch[cy+1][cx]) {
    cx--;
  } else if(b == batch[cy][cx+1]) {
    cy--;
  } else if(b == batch[cy][cx]) {
    cx--;
    cy--;
  } else {
    assert(false);
  }
}


} // namespace cubical_ph::t_cons::cuda
