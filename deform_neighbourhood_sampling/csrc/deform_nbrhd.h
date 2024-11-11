#pragma once

#include <ATen/ATen.h>

namespace deform_neighbourhood_sampling {

at::Tensor deform_neighbourhood(const at::Tensor &input,
                                const at::Tensor &offset, int64_t nbrhd_h,
                                int64_t nbrhd_w, int64_t stride_h,
                                int64_t stride_w, int64_t pad_h, int64_t pad_w,
                                int64_t dilation_h, int64_t dilation_w,
                                int64_t offset_groups);

at::Tensor deform_neighbourhood_symint(
    const at::Tensor &input, const at::Tensor &offset, c10::SymInt nbrhd_h,
    c10::SymInt nbrhd_w, c10::SymInt stride_h, c10::SymInt stride_w,
    c10::SymInt pad_h, c10::SymInt pad_w, c10::SymInt dilation_h,
    c10::SymInt dilation_w, c10::SymInt offset_groups);

std::tuple<at::Tensor, at::Tensor> deform_neighbourhood_backward(
    const at::Tensor &grad_columns, const at::Tensor &input,
    const at::Tensor &offset, int64_t nbrhd_h, int64_t nbrhd_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t offset_groups);

std::tuple<at::Tensor, at::Tensor> deform_neighbourhood_backward_symint(
    const at::Tensor &grad_columns, const at::Tensor &input,
    const at::Tensor &offset, c10::SymInt nbrhd_h, c10::SymInt nbrhd_w,
    c10::SymInt stride_h, c10::SymInt stride_w, c10::SymInt pad_h,
    c10::SymInt pad_w, c10::SymInt dilation_h, c10::SymInt dilation_w,
    c10::SymInt offset_groups);

} // namespace deform_neighbourhood_sampling
