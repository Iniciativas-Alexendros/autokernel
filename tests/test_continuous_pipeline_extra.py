import importlib.util
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


class TestContinuousPipelineExtra:
    """Tests adicionales para continuous_pipeline."""

    def test_gpu_available_when_nvidia_smi_missing(self, tmp_path):
        mod = _load_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        with mock.patch("shutil.which", return_value=None):
            assert runner.gpu_available() is True

    def test_gpu_available_below_threshold(self, tmp_path):
        mod = _load_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        with (
            mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            mock.patch("pathlib.Path.exists", return_value=True),
            mock.patch("scripts.continuous_pipeline.subprocess.run") as fake_run,
        ):
            fake_run.return_value = mock.Mock(returncode=0, stdout="10\n")
            assert runner.gpu_available() is True

    def test_gpu_available_above_threshold(self, tmp_path):
        mod = _load_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        with (
            mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            mock.patch("pathlib.Path.exists", return_value=True),
            mock.patch("scripts.continuous_pipeline.subprocess.run") as fake_run,
        ):
            fake_run.return_value = mock.Mock(returncode=0, stdout="80\n")
            assert runner.gpu_available() is False

    def test_record_phase_duration(self, tmp_path):
        mod = _load_module()
        cfg_path = _make_config(tmp_path)
        runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
        runner._record_phase_duration("profile", 1.5)
        assert runner.metrics["phase_durations_sec"]["profile"] == [1.5]
