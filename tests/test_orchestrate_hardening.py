import importlib.util
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_orchestrate_module():
    """Load orchestrate.py as a module without executing __main__."""
    path = os.path.join(REPO_ROOT, "orchestrate.py")
    spec = importlib.util.spec_from_file_location("orchestrate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orchestrate"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestBenchOutputParser:
    """Tests del parseador de salida de bench.py."""

    def test_parse_valid_output(self):
        orchestrate = _load_orchestrate_module()
        _parse_bench_output = orchestrate._parse_bench_output

        text = """
=== FINAL ===
correctness: PASS
throughput_tflops: 32.456
speedup_vs_pytorch: 1.218x
latency_us: 123.45
pct_peak_compute: 96.5%
pct_peak_bandwidth: 45.2%
peak_vram_mb: 2048.5
"""
        result = _parse_bench_output(text)
        assert result["correctness"] == "PASS"
        assert result["throughput_tflops"] == pytest.approx(32.456)
        assert result["speedup_vs_pytorch"] == pytest.approx(1.218)
        assert result["latency_us"] == pytest.approx(123.45)
        assert result["pct_peak_compute"] == pytest.approx(96.5)
        assert result["pct_peak_bandwidth"] == pytest.approx(45.2)
        assert result["peak_vram_mb"] == pytest.approx(2048.5)

    def test_parse_fail_output(self):
        orchestrate = _load_orchestrate_module()
        _parse_bench_output = orchestrate._parse_bench_output

        text = """
correctness: FAIL
throughput_tflops: 0.000
"""
        result = _parse_bench_output(text)
        assert result["correctness"] == "FAIL"
        assert result["throughput_tflops"] == 0.0


class TestNightlyPipelineScript:
    """Tests del script de pipeline nocturno."""

    def test_nightly_pipeline_syntax(self):
        """El script nocturno debe tener sintaxis válida de zsh."""
        result = subprocess.run(
            ["zsh", "-n", "scripts/nightly_pipeline.sh"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"

    def test_nightly_pipeline_no_compglob(self):
        """No debe quedar el typo 'compglob' no existente."""
        script_path = os.path.join(REPO_ROOT, "scripts", "nightly_pipeline.sh")
        with open(script_path) as f:
            content = f.read()
        assert "compglob" not in content, "comando inexistente compglob aún presente"
        assert 'find "$MODEL_WS" -maxdepth 1' in content, "debe usar find para globbing portable"
