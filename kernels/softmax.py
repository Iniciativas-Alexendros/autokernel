"""
AutoKernel -- Softmax kernel.

Current kernel: Online Softmax (row-parallel, chunked)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Each program instance handles one row of the input tensor.
Uses online softmax: max and sum of exp(x - max) computed in a single
forward pass over column chunks, then a second pass writes the output.
This avoids two full loads of the data when n_cols >> BLOCK_SIZE.
"""

KERNEL_TYPE = "softmax"

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel online softmax. One program per row, chunked over columns."""
    row_idx = tl.program_id(0)

    row_start_input = input_ptr + row_idx * stride_input_row
    row_start_output = output_ptr + row_idx * stride_output_row

    # ── Pass 1: online accumulation of max and sum(exp(x - max)) ────────
    running_max = tl.full((), value=float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((), dtype=tl.float32)

    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        chunk = tl.load(row_start_input + col_offsets, mask=mask, other=float("-inf")).to(
            tl.float32
        )

        chunk_max = tl.max(tl.where(mask, chunk, float("-inf")), axis=0)

        # Rescale previous accumulator when a larger value is found
        new_max = tl.maximum(running_max, chunk_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(chunk - new_max) * mask, axis=0
        )
        running_max = new_max

    # ── Pass 2: write normalised output ──────────────────────────────────
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        chunk = tl.load(row_start_input + col_offsets, mask=mask, other=float("-inf")).to(
            tl.float32
        )

        result = tl.exp(chunk - running_max) / running_sum
        tl.store(row_start_output + col_offsets, result, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.softmax_ref signature."""
    assert x.is_cuda

    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = 1024

    grid = (n_rows,)
    softmax_kernel[grid](
        x,
        output,
        n_cols,
        x.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )

    return output.view(orig_shape)
