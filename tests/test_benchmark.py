import json
import os
import pytest

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "workspace")


class TestBenchResult:
    """Tests del resultado de benchmark."""

    RESULT_PATH = os.path.join(WORKSPACE, "bench_result.json")

    def test_bench_result_exists(self):
        """bench_result.json existe después de benchmark."""
        if not os.path.exists(self.RESULT_PATH):
            pytest.skip("Benchmark no ejecutado aún")
        with open(self.RESULT_PATH) as f:
            result = json.load(f)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_bench_result_has_speedup(self):
        """Resultado tiene campo speedup_vs_pytorch."""
        if not os.path.exists(self.RESULT_PATH):
            pytest.skip("Benchmark no ejecutado aún")
        with open(self.RESULT_PATH) as f:
            result = json.load(f)
        assert "speedup_vs_pytorch" in result or "speedup" in result

    def test_bench_result_has_correctness(self):
        """Resultado tiene campo correctness."""
        if not os.path.exists(self.RESULT_PATH):
            pytest.skip("Benchmark no ejecutado aún")
        with open(self.RESULT_PATH) as f:
            result = json.load(f)
        assert "correctness" in result


class TestKernelPerformance:
    """Tests de rendimiento de kernels optimizados."""

    KERNEL_PATH = os.path.join(WORKSPACE, "kernel_matmul_1.py")

    def test_kernel_importable(self):
        """Kernel se puede importar sin errores de compilación."""
        if not os.path.exists(self.KERNEL_PATH):
            pytest.skip("Kernel no extraído aún")
        import ast

        with open(self.KERNEL_PATH) as f:
            source = f.read()
        # Verify it's valid Python (syntax check)
        ast.parse(source)

    def test_kernel_has_kernel_type(self):
        """Kernel define KERNEL_TYPE."""
        if not os.path.exists(self.KERNEL_PATH):
            pytest.skip("Kernel no extraído aún")
        with open(self.KERNEL_PATH) as f:
            content = f.read()
        assert "KERNEL_TYPE" in content, "kernel_file sin KERNEL_TYPE"

    def test_kernel_has_kernel_fn(self):
        """Kernel define kernel_fn."""
        if not os.path.exists(self.KERNEL_PATH):
            pytest.skip("Kernel no extraído aún")
        with open(self.KERNEL_PATH) as f:
            content = f.read()
        assert "kernel_fn" in content, "kernel_file sin kernel_fn"
