"""
AutoKernel -- Fused SwiGLU MLP kernel.

Current kernel: Fused Gate + Up projection + SiLU + elementwise multiply + Down projection
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Fuses the following operations:
  gate = x @ w_gate.T
  up   = x @ w_up.T
  hidden = silu(gate) * up
  out  = hidden @ w_down.T

The first three operations (gate proj, up proj, activation, multiply) are fused
into a single kernel. The down projection is a separate Triton matmul kernel.
"""

KERNEL_TYPE = "fused_mlp"

import torch
import triton
import triton.language as tl


@triton.jit
def fused_gate_up_kernel(
    X_ptr,
    W_gate_ptr,
    W_up_ptr,
    Out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wgk,
    stride_wgn,
    stride_wuk,
    stride_wun,
    stride_om,
    stride_on,
    USE_SILU: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel: computes activation(X @ W_gate^T) * (X @ W_up^T).
    W_gate and W_up are [intermediate_size, hidden_size] (transposed access).
    X is [M, K], output is [M, N] where N = intermediate_size, K = hidden_size.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    wg_ptrs = W_gate_ptr + offs_k[:, None] * stride_wgk + offs_n[None, :] * stride_wgn
    wu_ptrs = W_up_ptr + offs_k[:, None] * stride_wuk + offs_n[None, :] * stride_wun

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offs = k_start + offs_k
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        wg_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        wg = tl.load(wg_ptrs, mask=wg_mask, other=0.0)

        wu_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        wu = tl.load(wu_ptrs, mask=wu_mask, other=0.0)

        acc_gate += tl.dot(x, wg)
        acc_up += tl.dot(x, wu)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        wg_ptrs += BLOCK_SIZE_K * stride_wgk
        wu_ptrs += BLOCK_SIZE_K * stride_wuk

    if USE_SILU:
        gate_activated = acc_gate * tl.sigmoid(acc_gate)
    else:
        gate_activated = (
            0.5
            * acc_gate
            * (
                1.0
                + tl.math.tanh(
                    0.7978845608 * (acc_gate + 0.044715 * acc_gate * acc_gate * acc_gate)
                )
            )
        )

    result = gate_activated * acc_up

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Tiled matmul with L2 cache reuse via CTA grouping."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    c = acc.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def kernel_fn(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """
    Entry point called by bench.py. Must match reference.fused_mlp_ref signature.

    SwiGLU MLP:
      hidden = activation(x @ w_gate.T) * (x @ w_up.T)
      out = hidden @ w_down.T

    Args:
        x: [batch, hidden_size] or [batch, seq_len, hidden_size]
        w_gate: [intermediate_size, hidden_size]
        w_up: [intermediate_size, hidden_size]
        w_down: [hidden_size, intermediate_size]
        activation: "silu" or "gelu"
    """
    assert x.is_cuda

    orig_shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    M, K = x.shape
    N, K2 = w_gate.shape
    assert K == K2, f"Hidden dim mismatch: x has {K}, w_gate has {K2}"
    assert w_up.shape == (N, K), "w_up shape mismatch"

    hidden = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    fused_gate_up_kernel[grid](
        x,
        w_gate,
        w_up,
        hidden,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_gate.stride(1),
        w_gate.stride(0),
        w_up.stride(1),
        w_up.stride(0),
        hidden.stride(0),
        hidden.stride(1),
        USE_SILU=(activation == "silu"),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Down projection: hidden [M, N] @ w_down.T [N, hidden_size] -> [M, hidden_size]
    out_size = w_down.shape[0]
    out = torch.empty((M, out_size), device=x.device, dtype=x.dtype)

    grid_down = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(out_size, BLOCK_SIZE_N),)

    # w_down is [out_size, N], accessed transposed: hidden [M, N] @ w_down^T [N, out_size]
    matmul_kernel[grid_down](
        hidden,
        w_down,
        out,
        M,
        out_size,
        N,
        hidden.stride(0),
        hidden.stride(1),
        w_down.stride(1),
        w_down.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    if len(orig_shape) > 2:
        out = out.view(*orig_shape[:-1], out.shape[-1])

    return out
