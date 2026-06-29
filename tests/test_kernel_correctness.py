import ast
import os
import pytest

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "workspace")


class TestKernelCorrectness:
    """Tests de corrección de kernels extraídos."""

    KERNELS = [
        "kernel_matmul_1.py",
        "kernel_elementwise_2.py",
        "kernel_elementwise_3.py",
        "kernel_flash_attention_4.py",
        "kernel_elementwise_5.py",
    ]

    def _load_kernel(self, name: str) -> str:
        path = os.path.join(WORKSPACE, name)
        if not os.path.exists(path):
            pytest.skip(f"{name} no existe")
        with open(path) as f:
            return f.read()

    @pytest.mark.parametrize("kernel_name", KERNELS)
    def test_kernel_valid_python(self, kernel_name: str):
        """Kernel es Python válido (sin errores de sintaxis)."""
        source = self._load_kernel(kernel_name)
        ast.parse(source)

    @pytest.mark.parametrize("kernel_name", KERNELS)
    def test_kernel_has_triton_import(self, kernel_name: str):
        """Kernel importa triton."""
        source = self._load_kernel(kernel_name)
        assert "import triton" in source, f"{kernel_name} sin import triton"

    @pytest.mark.parametrize("kernel_name", KERNELS)
    def test_kernel_has_kernel_type(self, kernel_name: str):
        """Kernel define KERNEL_TYPE."""
        source = self._load_kernel(kernel_name)
        assert "KERNEL_TYPE" in source, f"{kernel_name} sin KERNEL_TYPE"
