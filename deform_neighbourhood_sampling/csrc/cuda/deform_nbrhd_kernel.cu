/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp

// modified from
// https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace deform_neighbourhood_sampling {

namespace {

#define CUDA_1D_KERNEL_LOOP_T(i, n, index_t)                                   \
  for (index_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n);           \
       i += (blockDim.x * gridDim.x))

const int kMaxParallelImgs = 32;

inline unsigned int GET_THREADS() {
#ifdef WITH_HIP
  return 256;
#endif
  return 512;
}

inline unsigned int GET_BLOCKS(const unsigned int THREADS, const int64_t N) {
  int64_t kMaxGridNum = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  return (unsigned int)std::min(kMaxGridNum, (N + THREADS - 1) / THREADS);
}

template <typename scalar_t, typename index_t>
__device__ scalar_t bilinear_interpolate(const scalar_t *in, index_t height,
                                         index_t width, scalar_t h,
                                         scalar_t w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  index_t h_low = floor(h);
  index_t w_low = floor(w);
  index_t h_high = h_low + 1;
  index_t w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = in[h_low * width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = in[h_low * width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = in[h_high * width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t, typename index_t>
__global__ void deformable_im2col_kernel(
    index_t n, const scalar_t *input_ptr, const scalar_t *offset_ptr,
    const scalar_t *mask_ptr, index_t height, index_t width, index_t weight_h,
    index_t weight_w, index_t pad_h, index_t pad_w, index_t stride_h,
    index_t stride_w, index_t dilation_h, index_t dilation_w, index_t batch_sz,
    index_t n_in_channels, index_t offset_groups, index_t out_h, index_t out_w,
    bool use_mask, scalar_t *columns_ptr) {
  CUDA_1D_KERNEL_LOOP_T(index, n, index_t) {
    const index_t out_x = index % out_w;
    const index_t out_y = (index / out_w) % out_h;
    const index_t out_b = (index / (out_w * out_h)) % batch_sz;
    const index_t in_c = index / (out_w * out_h * batch_sz);
    const index_t out_c = in_c * weight_h * weight_w;

    index_t c_per_offset_grp = n_in_channels / offset_groups;
    const index_t grp_idx = in_c / c_per_offset_grp;

    columns_ptr += (out_c * (batch_sz * out_h * out_w) +
                    out_b * (out_h * out_w) + out_y * out_w + out_x);

    input_ptr +=
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    offset_ptr += (out_b * offset_groups + grp_idx) * 2 * weight_h * weight_w *
                  out_h * out_w;

    if (use_mask) {
      mask_ptr += (out_b * offset_groups + grp_idx) * weight_h * weight_w *
                  out_h * out_w;
    }

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const index_t mask_idx = i * weight_w + j;
        const index_t offset_idx = 2 * mask_idx;

        scalar_t mask_value = 1;
        if (use_mask) {
          mask_value =
              mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
        }

        const scalar_t offset_h =
            offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t offset_w =
            offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w +
                       out_x];
        const scalar_t y =
            (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
        const scalar_t x =
            (out_x * stride_w - pad_w) + j * dilation_w + offset_w;
        *columns_ptr =
            mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

void deformable_im2col(const at::Tensor &input, const at::Tensor &data_offset,
                       const at::Tensor &data_mask, int n_in_channels,
                       int height, int width, int weight_h, int weight_w,
                       int pad_h, int pad_w, int stride_h, int stride_w,
                       int dilation_h, int dilation_w, int out_h, int out_w,
                       int parallel_imgs, int deformable_group, bool use_mask,
                       at::Tensor data_col) {
  at::cuda::CUDAGuard device_guard(input.get_device());

  const int64_t num_kernels =
      (int64_t)n_in_channels * out_h * out_w * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  // Checks if we should use 64bits indexing
  // https://github.com/pytorch/vision/issues/4269
  bool use_64bits_indexing = false;
  // Checks if num_kernels or columns numel larger than 2 ** 31
  use_64bits_indexing |= num_kernels > (1 << 31);
  use_64bits_indexing |= ((int64_t)n_in_channels * weight_h * weight_w *
                              parallel_imgs * out_h * out_w >
                          (1 << 31));

  if (use_64bits_indexing) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "deformable_im2col", ([&] {
          deformable_im2col_kernel<scalar_t, int64_t><<<blocks, threads>>>(
              num_kernels, input.data_ptr<scalar_t>(),
              data_offset.data_ptr<scalar_t>(), data_mask.data_ptr<scalar_t>(),
              height, width, weight_h, weight_w, pad_h, pad_w, stride_h,
              stride_w, dilation_h, dilation_w, parallel_imgs, n_in_channels,
              deformable_group, out_h, out_w, use_mask,
              data_col.data_ptr<scalar_t>());
        }));

  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "deformable_im2col", ([&] {
          deformable_im2col_kernel<scalar_t, int><<<blocks, threads>>>(
              num_kernels, input.data_ptr<scalar_t>(),
              data_offset.data_ptr<scalar_t>(), data_mask.data_ptr<scalar_t>(),
              height, width, weight_h, weight_w, pad_h, pad_w, stride_h,
              stride_w, dilation_h, dilation_w, parallel_imgs, n_in_channels,
              deformable_group, out_h, out_w, use_mask,
              data_col.data_ptr<scalar_t>());
        }));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int get_greatest_divisor_below_bound(int n, int bound) {
  for (int k = bound; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

template <typename scalar_t, typename index_t>
__global__ void deformable_col2im_kernel(
    index_t n, const scalar_t *col, const scalar_t *offset_ptr,
    const scalar_t *mask_ptr, index_t channels, index_t height, index_t width,
    index_t kernel_h, index_t kernel_w, index_t pad_h, index_t pad_w,
    index_t stride_h, index_t stride_w, index_t dilation_h, index_t dilation_w,
    index_t batch_sz, index_t offset_groups, index_t out_h, index_t out_w,
    bool use_mask, scalar_t *grad_im) {
  const index_t grad_im_numel = width * height * channels * batch_sz;

  CUDA_1D_KERNEL_LOOP_T(index, n, int64_t) {
    const index_t out_x = index % out_w;
    const index_t out_y = (index / out_w) % out_h;
    const index_t b = (index / (out_w * out_h)) % batch_sz;
    const index_t j = (index / (out_w * out_h * batch_sz)) % kernel_w;
    const index_t i =
        (index / (out_w * out_h * batch_sz * kernel_w)) % kernel_h;
    const index_t c = index / (out_w * out_h * batch_sz * kernel_w * kernel_h);

    index_t c_per_offset_grp = channels / offset_groups;
    const index_t offset_grp = c / c_per_offset_grp;

    offset_ptr += (b * offset_groups + offset_grp) * 2 * kernel_h * kernel_w *
                  out_h * out_w;

    if (use_mask) {
      mask_ptr += (b * offset_groups + offset_grp) * kernel_h * kernel_w *
                  out_h * out_w;
    }

    const index_t mask_idx = i * kernel_w + j;
    const index_t offset_idx = 2 * mask_idx;

    const index_t offset_h_ptr = ((offset_idx)*out_h + out_y) * out_w + out_x;
    const index_t offset_w_ptr =
        ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

    const scalar_t offset_h = offset_ptr[offset_h_ptr];
    const scalar_t offset_w = offset_ptr[offset_w_ptr];

    scalar_t mask_value = 1;
    if (use_mask) {
      mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
    }

    const scalar_t y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
    const scalar_t x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

    for (index_t dy = -1; dy <= 1; dy++) {
      for (index_t dx = -1; dx <= 1; dx++) {
        index_t yp = (index_t)y + dy;
        index_t xp = (index_t)x + dx;
        if (0 <= yp && yp < height && 0 <= xp && xp < width &&
            std::abs(y - yp) < 1 && std::abs(x - xp) < 1) {
          index_t grad_pos = ((b * channels + c) * height + yp) * width + xp;
          scalar_t weight = (1 - std::abs(y - yp)) * (1 - std::abs(x - xp));
          at::native::fastAtomicAdd(grad_im, grad_pos, grad_im_numel,
                                    mask_value * weight * col[index], true);
        }
      }
    }
  }
}

template <typename scalar_t, typename index_t>
__device__ scalar_t get_coordinate_weight(const scalar_t *im_data,
                                          index_t height, index_t width,
                                          scalar_t y, scalar_t x,
                                          bool is_y_direction) {
  index_t y_l = floor(y);
  index_t x_l = floor(x);
  index_t y_h = y_l + 1;
  index_t x_h = x_l + 1;

  bool valid_y_l = 0 <= y_l && y_l < height;
  bool valid_y_h = 0 <= y_h && y_h < height;
  bool valid_x_l = 0 <= x_l && x_l < width;
  bool valid_x_h = 0 <= x_h && x_h < width;

  scalar_t zero = 0;
  scalar_t v_yx = (valid_y_l && valid_x_l) ? im_data[y_l * width + x_l] : zero;
  scalar_t v_yX = (valid_y_l && valid_x_h) ? im_data[y_l * width + x_h] : zero;
  scalar_t v_Yx = (valid_y_h && valid_x_l) ? im_data[y_h * width + x_l] : zero;
  scalar_t v_YX = (valid_y_h && valid_x_h) ? im_data[y_h * width + x_h] : zero;

  if (is_y_direction) {
    scalar_t dx = x - x_l;
    return dx * (v_YX - v_yX) + (1 - dx) * (v_Yx - v_yx);
  } else {
    scalar_t dy = y - y_l;
    return dy * (v_YX - v_Yx) + (1 - dy) * (v_yX - v_yx);
  }
}

template <typename scalar_t, typename index_t>
__global__ void deformable_col2im_coord_kernel(
    index_t n, const scalar_t *col_ptr, const scalar_t *im_ptr,
    const scalar_t *offset_ptr, const scalar_t *mask_ptr, index_t channels,
    index_t height, index_t width, index_t weight_h, index_t weight_w,
    index_t pad_h, index_t pad_w, index_t stride_h, index_t stride_w,
    index_t dilation_h, index_t dilation_w, index_t batch_sz,
    index_t offset_channels, index_t offset_groups, index_t out_h,
    index_t out_w, const bool use_mask, scalar_t *grad_offset,
    scalar_t *grad_mask) {
  CUDA_1D_KERNEL_LOOP_T(index, n, int64_t) {
    scalar_t grad_offset_val = 0;
    scalar_t grad_mask_val = 0;

    index_t w = index % out_w;
    index_t h = (index / out_w) % out_h;
    index_t w_w = (index / (out_w * out_h * 2)) % weight_w;
    index_t w_h = (index / (out_w * out_h * 2 * weight_w)) % weight_h;
    index_t c = (index / (out_w * out_h)) % offset_channels;
    index_t b = index / (out_w * out_h * offset_channels);

    const index_t offset_grp = c / (2 * weight_h * weight_w);
    const index_t col_step = weight_h * weight_w;

    index_t c_per_offset_grp = channels / offset_groups;

    col_ptr += offset_grp * c_per_offset_grp * weight_h * weight_w * batch_sz *
               out_w * out_h;
    im_ptr +=
        (b * offset_groups + offset_grp) * c_per_offset_grp * height * width;
    offset_ptr += (b * offset_groups + offset_grp) * 2 * weight_h * weight_w *
                  out_h * out_w;

    if (use_mask) {
      mask_ptr += (b * offset_groups + offset_grp) * weight_h * weight_w *
                  out_h * out_w;
    }

    const index_t offset_c = c - offset_grp * 2 * weight_h * weight_w;
    const bool is_y_direction = offset_c % 2 == 0;

    const index_t c_bound = c_per_offset_grp * weight_h * weight_w;
    for (index_t col_c = (offset_c / 2); col_c < c_bound; col_c += col_step) {
      const index_t col_pos =
          (((col_c * batch_sz + b) * out_h) + h) * out_w + w;

      index_t out_x = col_pos % out_w;
      index_t out_y = (col_pos / out_w) % out_h;
      index_t j = (col_pos / (out_w * out_h * batch_sz)) % weight_w;
      index_t i = (col_pos / (out_w * out_h * batch_sz * weight_w)) % weight_h;

      const index_t mask_idx = i * weight_w + j;

      const index_t offset_h_ptr =
          (((2 * mask_idx) * out_h + out_y) * out_w + out_x);
      const index_t offset_w_ptr =
          (((2 * mask_idx + 1) * out_h + out_y) * out_w + out_x);
      const scalar_t offset_h = offset_ptr[offset_h_ptr];
      const scalar_t offset_w = offset_ptr[offset_w_ptr];

      scalar_t mask_value = 1;
      if (use_mask) {
        mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
      }

      scalar_t y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
      scalar_t x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

      const scalar_t weight =
          get_coordinate_weight(im_ptr, height, width, y, x, is_y_direction);
      grad_offset_val += mask_value * weight * col_ptr[col_pos];

      if (use_mask && is_y_direction) {
        grad_mask_val += col_ptr[col_pos] *
                         bilinear_interpolate(im_ptr, height, width, y, x);
      }

      im_ptr += height * width;
    }

    grad_offset[index] = grad_offset_val;

    if (use_mask && is_y_direction) {
      const index_t idx =
          ((((b * offset_groups + offset_grp) * weight_h + w_h) * weight_w +
            w_w) *
               out_h +
           h) *
              out_w +
          w;
      grad_mask[idx] = grad_mask_val;
    }
  }
}

void compute_grad_offset(const at::Tensor &columns, const at::Tensor &input,
                         const at::Tensor &offset, int channels, int height,
                         int width, int nbrhd_h, int nbrhd_w, int pad_h,
                         int pad_w, int stride_h, int stride_w, int dilation_h,
                         int dilation_w, int parallel_imgs, int offset_groups,
                         at::Tensor grad_offset) {
  at::cuda::CUDAGuard device_guard(columns.get_device());

  const int out_h =
      (height + 2 * pad_h - (dilation_h * (nbrhd_h - 1) + 1)) / stride_h + 1;
  const int out_w =
      (width + 2 * pad_w - (dilation_w * (nbrhd_w - 1) + 1)) / stride_w + 1;
  const int64_t num_kernels = (int64_t)out_h * out_w * 2 * nbrhd_h * nbrhd_w *
                              offset_groups * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  bool use_64bits_indexing = false;
  use_64bits_indexing |= num_kernels > (1 << 31);
  use_64bits_indexing |=
      ((int64_t)channels * nbrhd_h * nbrhd_w * parallel_imgs * out_h * out_w >
       (1 << 31));

  if (use_64bits_indexing) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        columns.scalar_type(), "compute_grad_offset", ([&] {
          deformable_col2im_coord_kernel<scalar_t, int64_t>
              <<<blocks, threads>>>(
                  num_kernels, columns.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(), offset.data_ptr<scalar_t>(),
                  static_cast<scalar_t *>(nullptr), /* mask ptr */
                  channels, height, width, nbrhd_h, nbrhd_w, pad_h, pad_w,
                  stride_h, stride_w, dilation_h, dilation_w, parallel_imgs,
                  2 * nbrhd_h * nbrhd_w * offset_groups, offset_groups, out_h,
                  out_w, false, /* use_mask */
                  grad_offset.data_ptr<scalar_t>(),
                  static_cast<scalar_t *>(nullptr) /* grad_mask */);
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        columns.scalar_type(), "compute_grad_offset", ([&] {
          deformable_col2im_coord_kernel<scalar_t, int><<<blocks, threads>>>(
              num_kernels, columns.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), offset.data_ptr<scalar_t>(),
              static_cast<scalar_t *>(nullptr), /* mask ptr */
              channels, height, width, nbrhd_h, nbrhd_w, pad_h, pad_w, stride_h,
              stride_w, dilation_h, dilation_w, parallel_imgs,
              2 * nbrhd_h * nbrhd_w * offset_groups, offset_groups, out_h,
              out_w, false, /* use_mask */
              grad_offset.data_ptr<scalar_t>(),
              static_cast<scalar_t *>(nullptr) /* grad_mask */);
        }));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_grad_input(const at::Tensor &columns, const at::Tensor &offset,
                        int channels, int height, int width, int nbrhd_h,
                        int nbrhd_w, int pad_h, int pad_w, int stride_h,
                        int stride_w, int dilation_h, int dilation_w,
                        int parallel_imgs, int offset_groups,
                        at::Tensor grad_im) {
  at::cuda::CUDAGuard device_guard(columns.get_device());

  const int out_h =
      (height + 2 * pad_h - (dilation_h * (nbrhd_h - 1) + 1)) / stride_h + 1;
  const int out_w =
      (width + 2 * pad_w - (dilation_w * (nbrhd_w - 1) + 1)) / stride_w + 1;

  const int64_t num_kernels =
      (int64_t)channels * nbrhd_h * nbrhd_w * out_h * out_w * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  bool use_64bits_indexing = false;
  use_64bits_indexing |= num_kernels > (1 << 31);

  at::globalContext().alertNotDeterministic("compute_grad_input");

  if (use_64bits_indexing) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        columns.scalar_type(), "compute_grad_input", ([&] {
          deformable_col2im_kernel<scalar_t, int64_t><<<blocks, threads>>>(
              num_kernels, columns.data_ptr<scalar_t>(),
              offset.data_ptr<scalar_t>(),
              static_cast<scalar_t *>(nullptr), /* mask_ptr */
              channels, height, width, nbrhd_h, nbrhd_w, pad_h, pad_w, stride_h,
              stride_w, dilation_h, dilation_w, parallel_imgs, offset_groups,
              out_h, out_w, false, /* use_mask */
              grad_im.data_ptr<scalar_t>());
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        columns.scalar_type(), "compute_grad_input", ([&] {
          deformable_col2im_kernel<scalar_t, int><<<blocks, threads>>>(
              num_kernels, columns.data_ptr<scalar_t>(),
              offset.data_ptr<scalar_t>(),
              static_cast<scalar_t *>(nullptr), /* mask_ptr */
              channels, height, width, nbrhd_h, nbrhd_w, pad_h, pad_w, stride_h,
              stride_w, dilation_h, dilation_w, parallel_imgs, offset_groups,
              out_h, out_w, false, /* use_mask */
              grad_im.data_ptr<scalar_t>());
        }));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor deform_nbrhd_forward_kernel(const at::Tensor &input,
                                       const at::Tensor &offset,
                                       int64_t nbrhd_h, int64_t nbrhd_w,
                                       int64_t stride_h, int64_t stride_w,
                                       int64_t pad_h, int64_t pad_w,
                                       int64_t dilation_h, int64_t dilation_w,
                                       int64_t offset_groups) {

  at::Tensor input_c = input.contiguous();
  at::Tensor offset_c = offset.contiguous();

  TORCH_CHECK(input_c.ndimension() == 4);
  TORCH_CHECK(offset_c.ndimension() == 4);
  TORCH_CHECK(input_c.is_cuda(), "input must be a CUDA tensor");

  at::DeviceGuard guard(input_c.device());

  const int batch_sz = input_c.size(0);
  const int in_channels = input_c.size(1);
  const int in_h = input_c.size(2);
  const int in_w = input_c.size(3);
  const int n_offset_vals = offset_c.size(1);

  const int n_parallel_images =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  const int ker_h = dilation_h * (nbrhd_h - 1) + 1;
  const int ker_w = dilation_w * (nbrhd_w - 1) + 1;
  const int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  const int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  TORCH_CHECK(nbrhd_h > 0 && nbrhd_w > 0, "nbrhd_h: ", nbrhd_h,
              " nbrhd_w: ", nbrhd_w);
  TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride_h: ", stride_h,
              " stride_w: ", stride_w);
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad_h: ", pad_h, " pad_w: ", pad_w);
  TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation_h: ", dilation_h,
              " dilation_w: ", dilation_w);
  TORCH_CHECK((n_offset_vals == offset_groups * 2 * nbrhd_h * nbrhd_w),
              "offset.shape[1] is not valid: got: ", n_offset_vals,
              " expected: ", offset_groups * 2 * nbrhd_h * nbrhd_w);
  TORCH_CHECK(in_channels % offset_groups == 0);

  TORCH_CHECK(offset_c.size(0) == batch_sz, "invalid batch size of offset");
  TORCH_CHECK((offset_c.size(2) == out_h && offset_c.size(3) == out_w),
              "offset output dims: (", offset_c.size(2), ", ", offset_c.size(3),
              ") - ", "computed output dims: (", out_h, ", ", out_w, ")");
  TORCH_CHECK(out_h > 0 && out_w > 0,
              "Calculated output size too small - out_h: ", out_h,
              " out_w: ", out_w);

  auto columns =
      at::zeros({batch_sz / n_parallel_images, in_channels * nbrhd_h * nbrhd_w,
                 n_parallel_images * out_h * out_w},
                input_c.options());

  auto mask =
      at::empty({2 * nbrhd_h * nbrhd_w, out_h, out_w}, input_c.options());

  for (int b = 0; b < batch_sz / n_parallel_images; b++) {
    deformable_im2col(input_c[b], offset_c[b], mask, in_channels, in_h, in_w,
                      nbrhd_h, nbrhd_w, pad_h, pad_w, stride_h, stride_w,
                      dilation_h, dilation_w, out_h, out_w, n_parallel_images,
                      offset_groups, false, /* use_mask */
                      columns[b]);
  }

  columns =
      columns
          .reshape({batch_sz / n_parallel_images,
                    in_channels * nbrhd_h * nbrhd_w, n_parallel_images, out_h,
                    out_w})
          .permute({2, 0, 1, 3, 4})
          .reshape({batch_sz, in_channels * nbrhd_h * nbrhd_w, out_h, out_w})
          .contiguous();

  return columns;
}

std::tuple<at::Tensor, at::Tensor> deform_nbrhd_backward_kernel(
    const at::Tensor &grad_columns, const at::Tensor &input,
    const at::Tensor &offset, int64_t nbrhd_h, int64_t nbrhd_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t offset_groups) {

  at::Tensor input_c = input.contiguous();
  at::Tensor offset_c = offset.contiguous();

  at::DeviceGuard guard(input_c.device());

  const int batch_sz = input_c.size(0);
  const int in_channels = input_c.size(1);
  const int in_h = input_c.size(2);
  const int in_w = input_c.size(3);
  const int n_offset_vals = offset_c.size(1);

  const int n_parallel_images =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  const int ker_h = dilation_h * (nbrhd_h - 1) + 1;
  const int ker_w = dilation_w * (nbrhd_w - 1) + 1;
  const int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  const int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;
  const int64_t num_kernels = (int64_t)out_h * out_w * 2 * nbrhd_h * nbrhd_w *
                              offset_groups * n_parallel_images;

  auto grad_input = at::zeros_like(input_c);
  auto grad_offset = at::zeros_like(offset_c);

  if (batch_sz == 0) {
    return std::make_tuple(std::move(grad_input), std::move(grad_offset));
  }

  auto grad_columns_c =
      grad_columns
          .reshape({n_parallel_images, batch_sz / n_parallel_images,
                    in_channels * nbrhd_h * nbrhd_w, out_h * out_w})
          .permute({1, 2, 0, 3})
          .reshape({batch_sz / n_parallel_images,
                    in_channels * nbrhd_h * nbrhd_w,
                    n_parallel_images * out_h * out_w})
          .contiguous();

  for (int b = 0; b < batch_sz / n_parallel_images; b++) {
    compute_grad_offset(grad_columns_c[b], input_c[b], offset_c[b], in_channels,
                        in_h, in_w, nbrhd_h, nbrhd_w, pad_h, pad_w, stride_h,
                        stride_w, dilation_h, dilation_w, n_parallel_images,
                        offset_groups, grad_offset[b]);
    compute_grad_input(grad_columns_c[b], offset_c[b], in_channels, in_h, in_w,
                       nbrhd_h, nbrhd_w, pad_h, pad_w, stride_h, stride_w,
                       dilation_h, dilation_w, n_parallel_images, offset_groups,
                       grad_input[b]);
  }

  return std::make_tuple(std::move(grad_input), std::move(grad_offset));
}

} // namespace

TORCH_LIBRARY_IMPL(deform_neighbourhood_sampling, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME(
             "deform_neighbourhood_sampling::deform_neighbourhood"),
         TORCH_FN(deform_nbrhd_forward_kernel));
  m.impl(TORCH_SELECTIVE_NAME(
             "deform_neighbourhood_sampling::deform_neighbourhood_backward"),
         TORCH_FN(deform_nbrhd_backward_kernel));
}

} // namespace deform_neighbourhood_sampling
