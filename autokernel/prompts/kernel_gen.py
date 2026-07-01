"""Prompt for kernel implementation."""

KERNEL_GEN_PROMPT = """Implement a Triton kernel based on this spec and tests.

Spec:
{spec}

Tests (must pass):
{tests}

Kernel type: {kernel_type}

Requirements:
1. Output raw Python code only — NO markdown, NO code blocks, NO explanations
2. File must define KERNEL_TYPE = "{kernel_type}" at the top
3. File must define kernel_fn(A, B) -> C as the entry point (or appropriate signature for this kernel type)
4. Use @triton.jit decorator on the kernel function
5. Follow this exact structure (example for matmul):

```python
KERNEL_TYPE = "matmul"

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # ... kernel implementation ...
    pass

def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32)
    return C
```

6. Use tl.load() and tl.store() for memory access (not direct indexing)
7. Use triton.cdiv() for grid calculations
8. Define constants like ceil_div BEFORE they are used
"""
