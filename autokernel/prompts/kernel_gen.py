"""Prompt for kernel implementation."""

KERNEL_GEN_PROMPT = """Implement a Triton kernel based on this spec and tests.

Spec:
{spec}

Tests (must pass):
{tests}

Kernel type: {kernel_type}

Requirements:
1. Use @triton.jit decorator
2. Include autotuning if beneficial
3. Handle multiple dtypes
4. Follow existing kernel conventions in the codebase
5. Include docstring with memory/compute analysis

Output only the kernel code, no explanation.
"""
