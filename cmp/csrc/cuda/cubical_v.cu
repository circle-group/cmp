#include <vector>

#include <cutlass/fast_math.h>
#include <torch/torch.h>

#include "union_find.cuh"
#include "util_v.cuh"

#define CT_CUDA_DEBUG 0

using namespace torch::indexing;

namespace cubical_ph {

namespace cuda {

namespace v_cons {

template <typename scalar_t>
__global__ void
enum_edges_v_2d_kernel(const PackedAccessorRestrict<scalar_t, 3> image,
                       PackedAccessorRestrict<scalar_t, 4> birth_value,
                       float threshold) {
  index_t x = blockIdx.x * blockDim.x + threadIdx.x;
  index_t y = blockIdx.y * blockDim.y + threadIdx.y;
  index_t b = blockIdx.z * blockDim.z + threadIdx.z;

  auto B = image.size(0);
  auto H = image.size(1);
  auto W = image.size(2);
  if (b >= B || y >= H || x >= W)
    return;

  auto M = birth_value.size(1);

  auto batch = image[b];
  auto birth_b = birth_value[b];

  index_t HW = H * W;
  index_t idx = y * W + x;

  if (y >= H - 2 || x >= W - 2) {
    for (index_t m = 0; m < M; m++) {
      birth_b[m][y][x] = threshold;
    }
    return;
  }

  for (index_t m = 0; m < M; m++) {
    float birth = get_birth_v_2d_dim1(batch, y, x, m, threshold);
    birth_b[m][y][x] = birth;
  }
}

template <typename scalar_t>
__global__ void
union_find_v_2d_init_kernel(const PackedAccessorRestrict<scalar_t, 3> image,
                            PackedAccessorRestrict<index_t, 2> uf_parent,
                            PackedAccessorRestrict<scalar_t, 2> uf_birth) {
  index_t x = blockIdx.x * blockDim.x + threadIdx.x;
  index_t y = blockIdx.y * blockDim.y + threadIdx.y;
  index_t b = blockIdx.z * blockDim.z + threadIdx.z;

  auto B = image.size(0);
  auto H = image.size(1);
  auto W = image.size(2);
  if (b >= B || y >= (H - 1) || x >= (W - 1))
    return;

  index_t idx = y * W + x;
  float birth = get_birth_v_2d(image[b], y, x);

  uf_parent[b][idx] = idx;
  uf_birth[b][idx] = birth;
}

template <typename scalar_t>
__global__ void
joint_pairs_v_2d_kernel(const PackedAccessorRestrict<scalar_t, 2> values,
                        PackedAccessorRestrict<index_t, 2> indices,
                        PackedAccessorRestrict<index_t, 2> uf_parent,
                        PackedAccessorRestrict<scalar_t, 2> uf_birth,
                        PackedAccessorRestrict<index_t, 3> output_idx,
                        PackedAccessorRestrict<scalar_t, 3> output_val,
                        PackedAccessorRestrict<index_t, 2> aux_indices,
                        index_t *__restrict__ max_end_idx, float threshold,
                        int64_t current_dim, cutlass::FastDivmod width_divmod,
                        cutlass::FastDivmod m_divmod, bool first_iter) {
  index_t b = blockIdx.x * blockDim.x + threadIdx.x;

  auto B = indices.size(0);
  // constexpr index_t M = 2;

  if (b >= B)
    return;

  auto S = indices.size(1);

  auto aux_b = aux_indices[b];
  if (first_iter) {
    aux_b[0] = S - 1;
    aux_b[1] = -1;
    aux_b[2] = 0;
  }

  auto start_idx = aux_b[0];
  if (start_idx == -2)
    return; // Already finished this element

  auto W = width_divmod.divisor;

  auto UF_S = uf_parent.size(1);

  auto values_b = values[b];
  auto indices_b = indices[b];
  auto uf_parent_b = uf_parent[b];
  auto uf_birth_b = uf_birth[b];

  auto out_idx_b = output_idx[b];
  auto out_val_b = output_val[b];

  auto BUF_S = output_idx.size(1);
  index_t insertion_loc = 0;

  scalar_t min_birth = aux_b[1] == -1 ? threshold : uf_birth_b[aux_b[1]];

  // start_idx should be S-1 in the first iteration
  for (index_t i = start_idx; i >= 0; i--) {
    auto death = values_b[i];
    if (death >= threshold) {
      break; // Sorted, stop when we hit the threshold
    }

    auto idx = indices_b[i];
    index_t m, uind, cx, cy;
    // uind = idx % (H*W), m = (idx-uind) / (H*W);
    m_divmod(m, uind, idx);
    // cx = uind % W, cy = (uind-cx) / W
    width_divmod(cy, cx, uind);

#if CT_CUDA_DEBUG
    printf("%d %d %d (%d): %f\n", cy, cx, m, idx, death);
#endif

    index_t u = uf_find_pc(uf_parent_b, uind);

    index_t vind = -1;
    switch (m) {
    case 0:
      vind = uind + 1;
      break;
    case 1:
      vind = uind + W;
      break;
    case 2:
      vind = uind + 1 + W;
      break;
    case 3:
      vind = uind + 1 - W;
      break;
    default:
      assert(false); // Shouldn't get here
    }

    index_t v = uf_find_pc(uf_parent_b, vind);

#if CT_CUDA_DEBUG
    printf("\t uind vind: %d %d\n", uind, vind);
    printf("\t u v: %d %d\n", u, v);
#endif

    if (u == v) {
      continue;
    }

    index_t birth_ind;
    scalar_t cur_birth;

    if (uf_birth_b[u] >= uf_birth_b[v]) {
      // Younger component u is killed
      // current_dim == 0
      cur_birth = uf_birth_b[u];
      birth_ind = u;
      if (uf_birth_b[v] < min_birth) {
        min_birth = uf_birth_b[v];
        aux_b[1] = v;
      }
    } else {
      // Younger component v is killed
      // current_dim == 0
      cur_birth = uf_birth_b[v];
      birth_ind = v;
      if (uf_birth_b[u] < min_birth) {
        min_birth = uf_birth_b[u];
        aux_b[1] = u;
      }
    }
    uf_link_rank(uf_parent_b, uf_birth_b, u, v);

    // Column clearing
    // e->index = NONE; in the original
    indices_b[i] = -1;
    // Do the clearing above since this if may return

    if (cur_birth != death) {
      index_t death_ind;
      if (uf_birth_b[uind] > uf_birth_b[vind]) {
        death_ind = uind;
      } else {
        death_ind = vind;
      }

      if (current_dim != 0) {
        // swap birth/death indices
        cutlass::swap(birth_ind, death_ind);
      }

#if CT_CUDA_DEBUG
      index_t bcx, bcy, dcx, dcy;
      width_divmod(bcy, bcx, birth_ind);
      width_divmod(dcy, dcx, death_ind);

      printf("Found dim=0 feature: Birth: (%d, %d), %f, Death: (%d, %d), %f\n",
             bcy, bcx, cur_birth, dcy, dcx, death);
#endif

      out_idx_b[insertion_loc][0] = birth_ind;
      out_idx_b[insertion_loc][1] = death_ind;

      out_val_b[insertion_loc][0] = cur_birth;
      out_val_b[insertion_loc][1] = death;

      aux_b[2]++;

      if ((++insertion_loc) == BUF_S) {
        // Done all the calculation for this kernel call, defer to next call
        aux_b[0] = i - 1; // Next starting loc
        atomicMax(max_end_idx, i - 1);
        return;
      }
    }
  }

  if (start_idx != -2) {    // Only last element left
    if (current_dim == 0) { // add dim 0 feature
      out_idx_b[insertion_loc][0] = aux_b[1];
      out_idx_b[insertion_loc][1] = -1;

      out_val_b[insertion_loc][0] = min_birth;
      out_val_b[insertion_loc][1] = threshold;

      aux_b[2]++;

#if CT_CUDA_DEBUG
      index_t bcx, bcy;
      width_divmod(bcy, bcx, aux_b[1]);

      printf("dim=0 base point feature: %d %d, Birth: %f\n", bcy, bcx,
             min_birth);
#endif
    }
    // finish the iteration for this image
    aux_b[0] = -2;
    atomicMax(max_end_idx, -1);
  }
}

template <typename scalar_t>
__global__ void joint_pairs_v_2d_out_parents_kernel(
    const PackedAccessorRestrict<scalar_t, 3> image,
    const PackedAccessorRestrict<index_t, 1> lengths,
    const PackedAccessorRestrict<index_t, 3> output_idx,
    const PackedAccessorRestrict<scalar_t, 3> output_val,
    const PackedAccessorRestrict<index_t, 2> argmax_idx,
    PackedAccessorRestrict<index_t, 3> cofaces,
    cutlass::FastDivmod width_divmod) {
  index_t s = blockIdx.x * blockDim.x + threadIdx.x;
  index_t b = blockIdx.y * blockDim.y + threadIdx.y;

  auto B = image.size(0);
  auto H = image.size(1);
  auto W = image.size(2);

  if (b >= B)
    return;

  auto l = lengths[b];
  if (s >= l)
    return;

  auto birth_idx = output_idx[b][s][0];
  auto death_idx = output_idx[b][s][1];

  auto birth_val = output_val[b][s][0];
  auto death_val = output_val[b][s][1];

  auto image_b = image[b];
  auto cofaces_bs = cofaces[b][s];

  index_t cy, cx;
  width_divmod(cy, cx, birth_idx);
  // No get_parent in v cons
  cofaces_bs[0] = cy;
  cofaces_bs[1] = cx;

  if (death_idx != -1) {
    width_divmod(cy, cx, death_idx);
    // No get_parent in v cons
    cofaces_bs[2] = cy;
    cofaces_bs[3] = cx;
  } else {
    cofaces_bs[2] = argmax_idx[b][0];
    cofaces_bs[3] = argmax_idx[b][1];
  }
}

} // namespace v_cons

} // namespace cuda

torch::Tensor enum_edges_v_2d(const torch::Tensor &image, double threshold,
                              bool alexander = false) {
  TORCH_CHECK(image.sizes().size() == 3);
  TORCH_CHECK(threshold > 0);
  TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);

  at::Tensor image_contig = image.contiguous();

  auto batch = image_contig.size(0);
  auto height = image_contig.size(1);
  auto width = image_contig.size(2);
  int64_t M = 2;
  if (alexander) {
    M = 4;
  }

  std::vector out_sz{batch, M, height, width};

  at::Tensor birth_value = torch::empty({out_sz}, image_contig.options());

  const dim3 threads{16, 16, 4};
  const dim3 blocks((width + threads.x - 1) / threads.x,
                    (height + threads.y - 1) / threads.y,
                    (batch + threads.z - 1) / threads.z);

  AT_DISPATCH_FLOATING_TYPES(
      image.scalar_type(), "enum_edges_v_2d", ([&] {
        cuda::v_cons::enum_edges_v_2d_kernel<scalar_t><<<blocks, threads>>>(
            image_contig
                .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            birth_value
                .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            threshold);
      }));

  return birth_value;
}

std::vector<torch::Tensor>
joint_pairs_v_2d(const torch::Tensor &image, const torch::Tensor &values,
                 torch::Tensor &indices, const torch::Tensor &argmax_idx,
                 double threshold, int64_t current_dim = 0,
                 bool alexander = false, int64_t block_size = 512) {
  int64_t M = 2;
  if (alexander) {
    M = 4;
  }

  TORCH_CHECK(image.sizes().size() == 3);

  TORCH_CHECK(values.sizes().size() == 2);
  TORCH_CHECK(values.sizes() == indices.sizes());

  TORCH_CHECK(image.size(0) == indices.size(0));
  TORCH_CHECK(image.size(1) * image.size(2) * M == indices.size(1));

  TORCH_CHECK(argmax_idx.sizes().size() == 2);
  TORCH_CHECK(indices.size(0) == argmax_idx.size(0));
  TORCH_CHECK(argmax_idx.size(1) == 2);

  TORCH_CHECK(threshold > 0);

  TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(indices.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(argmax_idx.device().type() == at::DeviceType::CUDA);

  TORCH_CHECK(image.dtype() == values.dtype());
  TORCH_CHECK(indices.dtype() == kIndex);
  TORCH_CHECK(argmax_idx.dtype() == kIndex);

  auto batch = image.size(0);
  auto height = image.size(1);
  auto width = image.size(2);

  auto S = indices.size(1);
  auto UF_S = height * width;

  cutlass::FastDivmod width_divmod(width);
  cutlass::FastDivmod m_divmod(width * height);

  at::Tensor out_idx, out_val;
  at::Tensor lengths;
  {
    std::vector uf_reps{batch, UF_S};

    at::Tensor uf_parent = torch::full({uf_reps}, -1, indices.options());
    at::Tensor uf_birth = torch::full({uf_reps}, -1, values.options());

    const dim3 threads_3d{16, 16, 4};
    const dim3 blocks_3d((width + threads_3d.x - 1) / threads_3d.x,
                         (height + threads_3d.y - 1) / threads_3d.y,
                         (batch + threads_3d.z - 1) / threads_3d.z);

    AT_DISPATCH_FLOATING_TYPES(
        image.scalar_type(), "union_find_v_2d_init", ([&] {
          cuda::v_cons::union_find_v_2d_init_kernel<
              scalar_t><<<blocks_3d, threads_3d>>>(
              image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
              uf_parent
                  .packed_accessor32<index_t, 2, torch::RestrictPtrTraits>(),
              uf_birth
                  .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        }));

    at::Tensor out = torch::empty_like(image, image.options());

    std::vector out_sz{batch, block_size, static_cast<int64_t>(2)};

    {
      std::vector<at::Tensor> out_idx_vec{};
      std::vector<at::Tensor> out_val_vec{};

      std::vector aux_idx_sz{batch, static_cast<int64_t>(3)};
      at::Tensor aux_idx = torch::empty({aux_idx_sz}, indices.options());
      int max_end_idx;

      const dim3 threads{256};
      const dim3 blocks((batch + threads.x - 1) / threads.x);

      bool first_iter = true;

      do {
        at::Tensor max_end_idx_t = torch::scalar_tensor(-1, indices.options());

        at::Tensor out_idx_buf = torch::full({out_sz}, -1, indices.options());
        at::Tensor out_val_buf = torch::full({out_sz}, -1, values.options());

        AT_DISPATCH_FLOATING_TYPES(
            image.scalar_type(), "joint_pairs_v_2d", ([&] {
              cuda::v_cons::joint_pairs_v_2d_kernel<scalar_t>
                  <<<blocks, threads>>>(
                      values.packed_accessor32<scalar_t, 2,
                                               torch::RestrictPtrTraits>(),
                      indices.packed_accessor32<index_t, 2,
                                                torch::RestrictPtrTraits>(),
                      uf_parent.packed_accessor32<index_t, 2,
                                                  torch::RestrictPtrTraits>(),
                      uf_birth.packed_accessor32<scalar_t, 2,
                                                 torch::RestrictPtrTraits>(),
                      out_idx_buf.packed_accessor32<index_t, 3,
                                                    torch::RestrictPtrTraits>(),
                      out_val_buf.packed_accessor32<scalar_t, 3,
                                                    torch::RestrictPtrTraits>(),
                      aux_idx.packed_accessor32<index_t, 2,
                                                torch::RestrictPtrTraits>(),
                      max_end_idx_t.data_ptr<index_t>(), threshold,
                      current_dim,
                      width_divmod, m_divmod, first_iter);
            }));

        first_iter = false;
        max_end_idx = max_end_idx_t.item().to<index_t>();
        out_idx_vec.emplace_back(out_idx_buf);
        out_val_vec.emplace_back(out_val_buf);
      } while (max_end_idx >= 0);

      out_idx = torch::cat({out_idx_vec}, 1);
      out_val = torch::cat({out_val_vec}, 1);
      lengths = aux_idx.index({Slice(), 2}).clone();
    }
  }

  auto max_len = static_cast<int64_t>(torch::max(lengths).item().to<index_t>());
  std::vector cofaces_sz{batch, max_len, static_cast<int64_t>(4)};

  if(max_len == 0) {
    at::Tensor cofaces = torch::zeros({cofaces_sz}, indices.options());

    return {cofaces, lengths};
  }
  at::Tensor cofaces = torch::full({cofaces_sz}, -1, indices.options());

  const dim3 threads{32, 32};
  const dim3 blocks((max_len + threads.x - 1) / threads.x,
                    (batch + threads.y - 1) / threads.y);
  AT_DISPATCH_FLOATING_TYPES(
      image.scalar_type(), "joint_pairs_v_2d_out_parents", ([&] {
        cuda::v_cons::joint_pairs_v_2d_out_parents_kernel<
            scalar_t><<<blocks, threads>>>(
            image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            lengths.packed_accessor32<index_t, 1, torch::RestrictPtrTraits>(),
            out_idx.packed_accessor32<index_t, 3, torch::RestrictPtrTraits>(),
            out_val.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            argmax_idx
                .packed_accessor32<index_t, 2, torch::RestrictPtrTraits>(),
            cofaces.packed_accessor32<index_t, 3, torch::RestrictPtrTraits>(),
            width_divmod);
      }));

  return {cofaces, lengths};
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(cubical_ph, CUDA, m) {
  m.impl("enum_edges_v_2d", &enum_edges_v_2d);
  m.impl("joint_pairs_v_2d", &joint_pairs_v_2d);
}

} // namespace cubical_ph
