"""Prompt for NCU log analysis."""

NCU_ANALYSIS_PROMPT = """Analyze this NCU profiling log and suggest optimizations.

Kernel type: {kernel_type}

NCU log:
{ncu_log}

Identify:
1. **Bottleneck**: memory-bound vs compute-bound, with evidence
2. **Occupancy**: current vs theoretical max
3. **Memory**: cache hit rate, bank conflicts, uncoalesced accesses
4. **Compute**: tensor core utilization, warp divergence
5. **Config suggestions**: specific BLOCK_SIZE, stages, GROUP_SIZE changes

Output as JSON with keys: bottleneck, occupancy, memory_issues, compute_issues, config_changes
"""
