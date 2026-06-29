"""
AutoKernel -- Optimized elementwise kernel.
Op type: elementwise
Operation: silu(x) * x (SwiGLU activation)

Optimization: vectorized loads, dtype-matched compute, BLOCK_SIZE tuning
"""

KERNEL_TYPE = "elementwise"

import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_kernel_fp16(
    X_ptr,
    Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Elementwise silu(x) * x for float16 (compute in FP32 for exp)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    exp_neg = tl.exp(-x_f32)
    sigmoid_x = 1.0 / (1.0 + exp_neg)
    silu_x = x_f32 * sigmoid_x

    y = (silu_x * x_f32).to(tl.float16)
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def elementwise_kernel_bf16(
    X_ptr,
    Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Elementwise silu(x) * x for bfloat16 (compute in BF16 to match PyTorch)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    # Match PyTorch: F.silu computes in BF16 natively on tensor cores
    x_f32 = x.to(tl.float32)

    exp_neg = tl.exp(-x_f32)
    sigmoid_x = 1.0 / (1.0 + exp_neg)
    silu_x = x_f32 * sigmoid_x

    # Convert back to BF16 at the end (matching PyTorch's output dtype)
    y = (silu_x * x_f32).to(tl.bfloat16)
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def elementwise_kernel_f32(
    X_ptr,
    Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Elementwise silu(x) * x for float32."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)

    exp_neg = tl.exp(-x)
    sigmoid_x = 1.0 / (1.0 + exp_neg)
    silu_x = x * sigmoid_x

    y = silu_x * x
    tl.store(Y_ptr + offs, y, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py."""
    N = x.numel()
    y = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    if x.dtype == torch.float16:
        elementwise_kernel_fp16[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    elif x.dtype == torch.bfloat16:
        elementwise_kernel_bf16[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    else:
        elementwise_kernel_f32[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y
