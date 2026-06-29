"""Prompt for Nemotron kernel review."""

REVIEW_PROMPT = """Review this Triton GPU kernel for correctness, performance, and safety.

Kernel type: {kernel_type}

```python
{kernel_code}
```

Optimization spec:
{spec}

Benchmark results:
{bench_results}

NCU profiling log:
{ncu_log}

Evaluate:
1. **Correctness**: Does the kernel compute correct results? Check boundary conditions, dtype handling.
2. **Performance**: Is it using tensor cores? Memory coalescing? Occupancy?
3. **Memory safety**: Race conditions, out-of-bounds access, shared memory bank conflicts?
4. **Code quality**: Clean, maintainable, follows Triton best practices?
5. **Improvements**: Specific suggestions with expected speedup impact.

Format your response as:
## Verdict: APPROVED / REJECTED

## Critical Issues (if any)
- ...

## Warnings
- ...

## Suggestions
- ...
"""
