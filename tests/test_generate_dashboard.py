import json
import sys


sys.path.insert(0, "/home/alexendros/repositorios/org-iniciativas-alexendros/autokernel")

from scripts.generate_dashboard import generate_dashboard


class TestGenerateDashboard:
    """Tests para generate_dashboard."""

    def test_generates_html_with_meta(self, tmp_path):
        state = {
            "kernels": [
                {
                    "kernel_type": "matmul",
                    "status": "completed",
                    "best_speedup": 1.2,
                    "experiments_run": 3,
                    "time_spent_minutes": 10,
                    "correctness": True,
                }
            ]
        }
        state_path = tmp_path / "orchestration_state.json"
        state_path.write_text(json.dumps(state))
        config_path = tmp_path / "pipeline.yaml"
        config_path.write_text("pipeline:\n  target_models: []\n")
        output = tmp_path / "index.html"
        generate_dashboard(tmp_path, config_path, output)
        html = output.read_text()
        assert '<meta name="description"' in html
        assert "<title>AutoKernel" in html
        assert "completed" in html

    def test_generates_robots_meta(self, tmp_path):
        state = {"kernels": []}
        state_path = tmp_path / "orchestration_state.json"
        state_path.write_text(json.dumps(state))
        config_path = tmp_path / "pipeline.yaml"
        config_path.write_text("pipeline:\n  target_models: []\n")
        output = tmp_path / "index.html"
        generate_dashboard(tmp_path, config_path, output)
        html = output.read_text()
        assert "Content-Security-Policy" in html
        assert "X-Content-Type-Options" in html
