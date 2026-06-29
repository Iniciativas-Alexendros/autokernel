"""Prompt for TDD test generation."""

TEST_GEN_PROMPT = """Generate pytest tests for a Triton kernel.

Spec:
{spec}

Kernel type: {kernel_type}

Generate complete pytest tests that:
1. Test correctness against PyTorch reference for each input shape
2. Test edge cases (empty, single element, power-of-2 sizes)
3. Test dtypes (fp16, bf16, f32 where applicable)
4. Use pytest.mark.parametrize for shapes
5. Assert torch.allclose with appropriate tolerances

Output only the test code, no explanation.
"""
