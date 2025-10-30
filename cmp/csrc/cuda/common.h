#pragma once

#include <torch/torch.h>

namespace cubical_ph {

namespace cuda {
using index_t = int32_t;

template <typename T, size_t N>
using PackedAccessorRestrict =
    torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, index_t>;

template <typename T, size_t N>
using AccessorRestrict =
    torch::TensorAccessor<T, N, torch::RestrictPtrTraits, index_t>;
} // namespace cuda

using index_t = cuda::index_t;
constexpr auto kIndex = at::kInt;
} // namespace cubical_ph