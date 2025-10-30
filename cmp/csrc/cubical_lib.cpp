#include <torch/extension.h>

namespace cubical_ph {

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(cubical_ph, m) {
  m.def("enum_edges_t_2d(Tensor image, float threshold) -> Tensor");
  m.def("joint_pairs_t_2d_d0(Tensor image, Tensor values, Tensor(a!) indices, "
        "Tensor argmax_idx, float threshold, int block_size) -> Tensor[]");

  m.def("enum_edges_v_2d(Tensor image, float threshold, bool alexander) -> "
        "Tensor");
  m.def("joint_pairs_v_2d(Tensor image, Tensor values, Tensor(a!) indices, "
        "Tensor argmax_idx, float threshold, int current_dim, bool alexander, "
        "int block_size) -> Tensor[]");
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(cubical_ph, CPU, m) {
  // m.impl("enum_edges_t_2d", &t_cons::enum_edges_t_2d);
}

} // namespace cubical_ph
