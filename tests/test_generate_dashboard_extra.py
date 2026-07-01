import json
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_dashboard import _load_results, _load_state, _load_verify


class TestGenerateDashboardExtra:
    """Tests adicionales para generate_dashboard."""

    def test_load_state_missing(self, tmp_path):
        state = _load_state(tmp_path)
        assert state == {"kernels": []}

    def test_load_results_parses_tsv(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "2026-07-01_results.tsv").write_text(
            "experiment\ttag\tkernel_type\tthroughput_tflops\n1\ta\tmatmul\t10.0\n"
        )
        results = _load_results(tmp_path)
        assert "2026-07-01_results" in results
        assert results["2026-07-01_results"][0]["kernel_type"] == "matmul"

    def test_load_verify_parses_json(self, tmp_path):
        (tmp_path / "verification_matmul.json").write_text(
            json.dumps({"correctness": True, "speedup": 1.2})
        )
        verify = _load_verify(tmp_path)
        assert verify == {"correctness": True, "speedup": 1.2}

    def test_load_profile_parses_json(self, tmp_path):
        from scripts.generate_dashboard import _load_profile

        (tmp_path / "profile").mkdir()
        (tmp_path / "profile" / "profile_report.json").write_text(
            json.dumps({"kernels": [{"op_type": "matmul", "time_pct": 80.0}]})
        )
        profile = _load_profile(tmp_path)
        assert profile["kernels"][0]["op_type"] == "matmul"
