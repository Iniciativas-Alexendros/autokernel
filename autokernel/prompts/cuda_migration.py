"""Prompt for Triton → CUDA C++ migration."""

CUDA_MIGRATION_PROMPT = """Migrate this Triton kernel to CUDA C++ for Blackwell SM 12.0.

Kernel type: {kernel_type}
Target arch: {target_arch}

Triton kernel:
```python
{triton_kernel}
```

Spec:
{spec}

Requirements:
1. Complete, compilable CUDA C++ code with pybind11 bindings
2. Use CUTLASS for matmul kernels if applicable
3. Use wmma/mma.sync for tensor cores on SM 12.0
4. Include proper shared memory management
5. Handle FP16/BF16/FP32 dtypes
6. Include a Python wrapper class that matches PyTorch's interface

Output complete CUDA code with headers and Python bindings.
"""
