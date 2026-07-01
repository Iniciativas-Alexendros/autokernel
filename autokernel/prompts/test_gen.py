"""Prompt for TDD test generation."""

TEST_GEN_PROMPT = """Generate pytest tests for a Triton kernel.

Spec:
{spec}

Kernel type: {kernel_type}

Generate complete pytest tests that:
1. Import the kernel_fn from kernel.py
2. Test correctness against PyTorch reference for each input shape
3. Test edge cases (empty, single element, power-of-2 sizes)
4. Test dtypes (fp16, bf16, f32 where applicable)
5. Use pytest.mark.parametrize for shapes
6. Assert torch.allclose with appropriate tolerances
7. Run with: pytest test_kernel.py -v

Output only raw Python test code — NO markdown, NO code blocks, NO explanations.
"""
