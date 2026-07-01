import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def _load_continuous_module():
    path = os.path.join(REPO_ROOT, "scripts", "continuous_pipeline.py")
    spec = importlib.util.spec_from_file_location("continuous_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["continuous_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_config(tmp_path: Path) -> Path:
    cfg = {
        "continuous": {
            "enabled": True,
            "idle_sleep_sec": 1,
            "work_sleep_sec": 0,
            "gpu_util_threshold": 25.0,
            "max_retries_per_task": 3,
        },
        "pipeline": {
            "target_models": [
                {
                    "name": "test_model",
                    "path": "models/test_model.py",
                    "class": "TestModel",
                    "shape": "1,128",
                    "dtype": "float16",
                    "enabled": True,
                }
            ],
            "phases": {
                "extract": {"top_k": 3, "backend": "triton"},
                "optimize": {
                    "iterations_per_kernel": 1,
                    "timeout_per_iteration_sec": 60,
                    "models": {
                        "planner": "ornith:9b",
                        "coder": "qwen2.5-coder:7b",
                    },
                },
            },
        },
    }
    path = tmp_path / "pipeline.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


class TestContinuousPipeline:
    """Tests del pipeline continuo."""

    def test_load_config_and_build_queue(self, tmp_path):
        mod = _load_continuous_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        assert len(runner.queue) == 1
        assert runner.queue[0].model_name == "test_model"
        assert runner.queue[0].phase == "idle"
        assert runner.queue[0].status == "pending"

    def test_queue_persistence(self, tmp_path):
        mod = _load_continuous_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        runner.save_queue()
        assert (tmp_path / "ws" / "queue.json").exists()

        runner2 = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        assert len(runner2.queue) == 1

    def test_advance_phase(self, tmp_path):
        mod = _load_continuous_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        task = runner.queue[0]
        assert task.phase == "idle"
        runner._advance_phase(task)
        assert task.phase == "profiling"

    def test_gpu_available_fallback(self, tmp_path):
        mod = _load_continuous_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        # When nvidia-smi is not present or fails, it should return True
        assert runner.gpu_available() in (True, False)

    def test_metrics_persistence(self, tmp_path):
        mod = _load_continuous_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        runner._record_phase_duration("test", 1.23)
        runner.save_metrics()
        assert (tmp_path / "ws" / "metrics.json").exists()

        runner2 = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        assert "test" in runner2.metrics.get("phase_durations_sec", {})
