"""
AutoKernel -- RMS Normalization kernel.

RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight
Row-parallel: one program instance per row.
"""

KERNEL_TYPE = "rmsnorm"

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr,
    W_ptr,
    OUT_ptr,
    stride_xm,
    stride_om,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel RMS normalization with multi-block reduction."""
    row = tl.program_id(0)

    # Pass 1: Compute sum of squares across all blocks
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    rrms = tl.rsqrt(sum_sq / N + eps)

    # Pass 2: Normalize and write output
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = (x * rrms) * w
        tl.store(OUT_ptr + row * stride_om + offs, out, mask=mask)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024

    rmsnorm_kernel[(M,)](
        x,
        weight,
        out,
        x.stride(0),
        out.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
