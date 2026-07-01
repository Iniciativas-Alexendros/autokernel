import json
import os
import subprocess
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestOrchestrateCLI:
    """Tests de la interfaz CLI de orchestrate.py."""

    def test_orchestrate_help(self):
        """orchestrate.py muestra ayuda con --help."""
        result = subprocess.run(
            ["uv", "run", "orchestrate.py", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )
        assert result.returncode == 0

    def test_orchestrate_status(self):
        """orchestrate.py status no crashea."""
        result = subprocess.run(
            ["uv", "run", "orchestrate.py", "status"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )
        # Should not crash (returncode 0 or show status output)
        assert result.returncode == 0 or "PENDING" in result.stdout or "DONE" in result.stdout

    def test_orchestrate_next(self):
        """orchestrate.py next retorna respuesta de decision."""
        result = subprocess.run(
            ["uv", "run", "orchestrate.py", "next"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )
        output = result.stdout + result.stderr
        valid_responses = ["NEXT:", "CONTINUE:", "DONE", "REVISIT:", "DECISION:"]
        assert any(r in output for r in valid_responses), f"Respuesta inesperada: {output[:200]}"

    def test_orchestrate_plan(self):
        """orchestrate.py plan muestra el plan de optimizacion."""
        result = subprocess.run(
            ["uv", "run", "orchestrate.py", "plan"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0


class TestOrchestrateState:
    """Tests de persistencia de estado de orchestrate.py."""

    STATE_PATH = os.path.join(REPO_ROOT, "workspace", "orchestration_state.json")

    def test_state_file_has_required_fields(self):
        """orchestration_state.json tiene campos requeridos."""
        if not os.path.exists(self.STATE_PATH):
            pytest.skip("Estado no inicializado aun")
        with open(self.STATE_PATH) as f:
            state = json.load(f)
        assert "kernels" in state or "current_kernel" in state
