import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def _load_self_audit_module():
    path = os.path.join(REPO_ROOT, "scripts", "self_audit.py")
    spec = importlib.util.spec_from_file_location("self_audit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["self_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestSelfAudit:
    """Tests del auto-auditor."""

    def test_generate_report_runs(self, tmp_path):
        mod = _load_self_audit_module()
        # Override REPO_ROOT to a temp dir so it doesn't depend on real workspace
        mod.REPO_ROOT = tmp_path
        report = mod.generate_report()
        assert "AutoKernel — Self-Audit & Evolution Report" in report
        assert "Runtime Metrics" in report
        assert "Checklist for /criticar" in report

    def test_avg_phase_duration(self, tmp_path):
        mod = _load_self_audit_module()
        metrics = {"phase_durations_sec": {"profile": [10, 20, 30], "optimize": [5]}}
        avg = mod.avg_phase_duration(metrics)
        assert avg["profile"] == 20.0
        assert avg["optimize"] == 5.0

    def test_count_verified_kernels(self, tmp_path):
        mod = _load_self_audit_module()
        mod.REPO_ROOT = tmp_path
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "verification_20260701.json").write_text('{"correctness": true}')
        (ws / "verification_20260702.json").write_text('{"correctness": false}')
        verified, failed = mod.count_verified_kernels()
        assert verified == 1
        assert failed == 1
