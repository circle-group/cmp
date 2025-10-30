#pragma once

#define CPH_HOST_DEVICE __host__ __device__ __forceinline__
#define CPH_CONSTEXPR constexpr

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

namespace cubical_ph::cuda {
using index_t_2 = int2;
}
