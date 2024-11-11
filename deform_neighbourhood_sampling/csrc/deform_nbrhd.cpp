#include "deform_nbrhd.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/types.h>

namespace deform_neighbourhood_sampling {

at::Tensor deform_neighbourhood(const at::Tensor &input,
                                const at::Tensor &offset, int64_t nbrhd_h,
                                int64_t nbrhd_w, int64_t stride_h,
                                int64_t stride_w, int64_t pad_h, int64_t pad_w,
                                int64_t dilation_h, int64_t dilation_w,
                                int64_t offset_groups) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "deform_neighbourhood_sampling::deform_neighbourhood", "")
          .typed<decltype(deform_neighbourhood)>();
  return op.call(input, offset, nbrhd_h, nbrhd_w, stride_h, stride_w, pad_h,
                 pad_w, dilation_h, dilation_w, offset_groups);
}

at::Tensor deform_neighbourhood_symint(
    const at::Tensor &input, const at::Tensor &offset, c10::SymInt nbrhd_h,
    c10::SymInt nbrhd_w, c10::SymInt stride_h, c10::SymInt stride_w,
    c10::SymInt pad_h, c10::SymInt pad_w, c10::SymInt dilation_h,
    c10::SymInt dilation_w, c10::SymInt offset_groups) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "deform_neighbourhood_sampling::deform_neighbourhood", "")
          .typed<decltype(deform_neighbourhood_symint)>();
  return op.call(input, offset, nbrhd_h, nbrhd_w, stride_h, stride_w, pad_h,
                 pad_w, dilation_h, dilation_w, offset_groups);
}

std::tuple<at::Tensor, at::Tensor> deform_neighbourhood_backward(
    const at::Tensor &grad_columns, const at::Tensor &input,
    const at::Tensor &offset, int64_t nbrhd_h, int64_t nbrhd_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t offset_groups) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "deform_neighbourhood_sampling::deform_neighbourhood_backward",
              "")
          .typed<decltype(deform_neighbourhood_backward)>();
  return op.call(grad_columns, input, offset, nbrhd_h, nbrhd_w, stride_h,
                 stride_w, pad_h, pad_w, dilation_h, dilation_w, offset_groups);
}

std::tuple<at::Tensor, at::Tensor> deform_neighbourhood_backward_symint(
    const at::Tensor &grad_columns, const at::Tensor &input,
    const at::Tensor &offset, c10::SymInt nbrhd_h, c10::SymInt nbrhd_w,
    c10::SymInt stride_h, c10::SymInt stride_w, c10::SymInt pad_h,
    c10::SymInt pad_w, c10::SymInt dilation_h, c10::SymInt dilation_w,
    c10::SymInt offset_groups) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "deform_neighbourhood_sampling::deform_neighbourhood_backward",
              "")
          .typed<decltype(deform_neighbourhood_backward_symint)>();
  return op.call(grad_columns, input, offset, nbrhd_h, nbrhd_w, stride_h,
                 stride_w, pad_h, pad_w, dilation_h, dilation_w, offset_groups);
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(deform_neighbourhood_sampling, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "deform_neighbourhood_sampling::deform_neighbourhood(Tensor input, "
      "Tensor offset, SymInt nbrhd_h, SymInt nbrhd_w, SymInt stride_h, SymInt "
      "stride_w, "
      "SymInt pad_h, SymInt pad_w, SymInt dilation_h, SymInt dilation_w, "
      "SymInt "
      "offset_groups) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "deform_neighbourhood_sampling::deform_neighbourhood_backward(Tensor "
      "grad_columns, Tensor input, Tensor offset, SymInt nbrhd_h, SymInt "
      "nbrhd_w, "
      "SymInt stride_h, SymInt stride_w, SymInt pad_h, SymInt pad_w, SymInt "
      "dilation_h, SymInt "
      "dilation_w, SymInt offset_groups) -> (Tensor, Tensor)"));
}

} // namespace deform_neighbourhood_sampling
