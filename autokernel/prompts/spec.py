"""Prompt for kernel optimization spec generation."""

SPEC_PROMPT = """You are generating a Triton kernel optimization spec.

Kernel type: {kernel_type}
Profile data:
{profile_data}

RAG context (Triton docs + similar kernels):
{context}

Generate a complete optimization spec with:
1. **Goal**: What the kernel does, target throughput, peak % goal
2. **Current bottlenecks**: From profile data (memory bandwidth, compute, latency)
3. **Optimization strategies**: Ranked list with expected impact
4. **Config parameters**: BLOCK_SIZE_K, GROUP_SIZE_M, stages, etc.
5. **Input shapes**: Test cases from profile data
6. **Acceptance criteria**: min speedup, correctness threshold

Output as JSON with keys: goal, bottlenecks, strategies, config, test_shapes, acceptance
"""
