import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def _load_module():
    path = os.path.join(REPO_ROOT, "scripts", "continuous_pipeline.py")
    spec = importlib.util.spec_from_file_location("continuous_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["continuous_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_config(tmp_path: Path):
    cfg = {
        "continuous": {
            "enabled": True,
            "idle_sleep_sec": 1,
            "work_sleep_sec": 0,
            "gpu_util_threshold": 25.0,
            "max_retries_per_task": 2,
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
            ]
        },
    }
    path = tmp_path / "pipeline.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


class TestContinuousPipelinePhases:
    """Tests para fases y comandos del pipeline continuo."""

    def test_kernel_types_from_plan_default(self, tmp_path):
        mod = _load_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        types = runner._kernel_types_from_plan(tmp_path / "ws" / "model")
        assert "matmul" in types
        assert "flash_attention" in types

    def test_kernel_types_from_plan_file(self, tmp_path):
        mod = _load_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        plan_path = tmp_path / "optimization_plan.json"
        plan_path.write_text(json.dumps({"kernels_to_optimize": [{"op_type": "layernorm"}]}))
        types = runner._kernel_types_from_plan(tmp_path)
        assert types == ["layernorm"]

    def test_run_command_success(self, tmp_path):
        mod = _load_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        with mock.patch("scripts.continuous_pipeline.subprocess.run") as fake_run:
            fake_run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
            assert runner._run_command(["echo", "hi"], "test") is True
        assert "test" in runner.metrics["phase_durations_sec"]

    def test_run_command_failure(self, tmp_path):
        mod = _load_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        with mock.patch("scripts.continuous_pipeline.subprocess.run") as fake_run:
            fake_run.return_value = mock.Mock(returncode=1, stdout="", stderr="err")
            assert runner._run_command(["false"], "test") is False

    def test_run_phase_profiling(self, tmp_path):
        mod = _load_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        task = runner.queue[0]
        task.phase = "profiling"
        with mock.patch.object(runner, "_run_command", return_value=True):
            assert runner._run_phase(task, "profiling") is True
