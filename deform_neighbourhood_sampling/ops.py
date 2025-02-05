import torch
from torch import Tensor

__all__ = ["deform_neighbourhood"]


def deform_neighbourhood(
    input: Tensor,
    offset: Tensor,
    neighbourhood_size: tuple[int, int] | int,
    stride: tuple[int, int] | int,
    padding: tuple[int, int] | int,
    dilation: tuple[int, int] | int,
    offset_groups: int,
) -> Tensor:
    if isinstance(neighbourhood_size, int):
        neighbourhood_size = (neighbourhood_size, neighbourhood_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return torch.ops.deform_neighbourhood_sampling.deform_neighbourhood.default(
        input,
        offset,
        neighbourhood_size[0],
        neighbourhood_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        offset_groups,
    )


@torch.library.register_fake("deform_neighbourhood_sampling::deform_neighbourhood")
def _(
    input,
    offset,
    nbrhd_h,
    nbrhd_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    offset_groups,
):
    torch._check(input.ndimension() == 4)
    torch._check(offset.ndimension() == 4)

    torch._check(nbrhd_h > 0 and nbrhd_w > 0)
    torch._check(stride_h > 0 and stride_w > 0)
    torch._check(pad_h >= 0 and pad_w >= 0)
    torch._check(dilation_h > 0 and dilation_w > 0)
    torch._check(offset.size(1) == offset_groups * 2 * nbrhd_h * nbrhd_w)
    torch._check(input.size(1) % offset_groups == 0)
    torch._check(offset.size(0) == input.size(0))

    ker_h = dilation_h * (nbrhd_h - 1) + 1
    ker_w = dilation_w * (nbrhd_w - 1) + 1
    out_h = ((input.size(2) + 2 * pad_h - ker_h) // stride_h) + 1
    out_w = ((input.size(3) + 2 * pad_w - ker_w) // stride_w) + 1
    torch._check(offset.size(2) == out_h)
    torch._check(offset.size(3) == out_w)
    torch._check(out_h > 0 and out_w > 0)

    torch._check(input.dtype == offset.dtype)
    torch._check(input.device == offset.device)
    return torch.empty(
        (input.size(0), input.size(1) * nbrhd_h * nbrhd_w, out_h, out_w),
        dtype=input.dtype,
        device=input.device,
    )


def _backward(ctx, grad):
    input, offset = ctx.saved_tensors
    grad_input, grad_offset = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        grad_input, grad_offset = (
            torch.ops.deform_neighbourhood_sampling.deform_neighbourhood_backward.default(
                grad,
                input,
                offset,
                ctx.nbrhd_h,
                ctx.nbrhd_w,
                ctx.stride_h,
                ctx.stride_w,
                ctx.pad_h,
                ctx.pad_w,
                ctx.dilation_h,
                ctx.dilation_w,
                ctx.offset_groups,
            )
        )
    return grad_input, grad_offset, None, None, None, None, None, None, None, None, None


def _setup_context(ctx, inputs, output):
    (
        input,
        offset,
        nbrhd_h,
        nbrhd_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        offset_groups,
    ) = inputs
    ctx.nbrhd_h = nbrhd_h
    ctx.nbrhd_w = nbrhd_w
    ctx.stride_h = stride_h
    ctx.stride_w = stride_w
    ctx.pad_h = pad_h
    ctx.pad_w = pad_w
    ctx.dilation_h = dilation_h
    ctx.dilation_w = dilation_w
    ctx.offset_groups = offset_groups
    saved_input, saved_offset = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        saved_input = input
        saved_offset = offset
    ctx.save_for_backward(saved_input, saved_offset)


torch.library.register_autograd(
    "deform_neighbourhood_sampling::deform_neighbourhood",
    _backward,
    setup_context=_setup_context,
)


@torch.library.register_fake(
    "deform_neighbourhood_sampling::deform_neighbourhood_backward"
)
def _(
    grad,
    input,
    offset,
    nbrhd_h,
    nbrhd_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    offset_groups,
):
    grad_input = torch.empty_like(input)
    grad_offset = torch.empty_like(offset)
    return grad_input, grad_offset
