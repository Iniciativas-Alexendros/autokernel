import json
import os
import pytest

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "workspace")
REPORT_PATH = os.path.join(WORKSPACE, "profile_report.json")
PLAN_PATH = os.path.join(WORKSPACE, "optimization_plan.json")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


class TestProfileReport:
    """Tests de carga y validación del profile report."""

    def test_report_exists(self):
        """profile_report.json existe."""
        assert os.path.exists(REPORT_PATH), f"No existe: {REPORT_PATH}"

    def test_report_has_kernels(self):
        """Reporte tiene lista de kernels."""
        report = _load_json(REPORT_PATH)
        assert report, "Reporte vacío"
        assert "kernels" in report
        assert len(report["kernels"]) > 0

    def test_kernel_has_required_fields(self):
        """Cada kernel tiene campos requeridos."""
        report = _load_json(REPORT_PATH)
        required = ["kernel_name", "op_type", "total_time_us", "fraction"]
        for kernel in report["kernels"]:
            for field in required:
                assert field in kernel, (
                    f"Kernel {kernel.get('kernel_name', '?')} sin campo: {field}"
                )

    def test_kernels_sorted_by_time(self):
        """Kernels ordenados por tiempo descendente."""
        report = _load_json(REPORT_PATH)
        times = [k["total_time_us"] for k in report["kernels"]]
        assert times == sorted(times, reverse=True)


class TestOptimizationPlan:
    """Tests del plan de optimización generado por extract.py."""

    def test_plan_exists(self):
        """optimization_plan.json existe después de extract."""
        assert os.path.exists(PLAN_PATH), "Ejecuta: uv run extract.py --top 5"

    def test_plan_has_targets(self):
        """Plan tiene targets de optimización."""
        plan = _load_json(PLAN_PATH)
        assert plan, "Plan vacío"
        has_key = (
            "targets" in plan or "kernels" in plan or "kernels_to_optimize" in plan
        )
        assert has_key, f"Plan sin targets. Keys: {list(plan.keys())}"

    def test_plan_kernel_types_valid(self):
        """Tipos de kernel son válidos."""
        plan = _load_json(PLAN_PATH)
        valid_types = {
            "matmul",
            "flash_attention",
            "elementwise",
            "layernorm",
            "softmax",
            "reduce",
        }
        targets = plan.get("targets", plan.get("kernels", []))
        for t in targets:
            op = t.get("op_type", t.get("type", ""))
            assert op in valid_types, f"Tipo inválido: {op}"


class TestKernelFiles:
    """Tests de archivos de kernel generados."""

    def test_kernel_matmul_exists(self):
        """kernel_matmul_1.py existe."""
        path = os.path.join(WORKSPACE, "kernel_matmul_1.py")
        assert os.path.exists(path), f"No existe: {path}"

    def test_kernel_file_has_kernel_type(self):
        """Archivo de kernel define KERNEL_TYPE."""
        path = os.path.join(WORKSPACE, "kernel_matmul_1.py")
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            assert "KERNEL_TYPE" in content, "kernel_file sin KERNEL_TYPE"
