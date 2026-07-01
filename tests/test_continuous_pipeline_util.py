import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def _load_module():
    path = os.path.join(REPO_ROOT, "scripts", "continuous_pipeline.py")
    spec = importlib.util.spec_from_file_location("continuous_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["continuous_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_task_advance_phase(tmp_path):
    import yaml

    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text("pipeline:\n  target_models: []\n")
    mod = _load_module()
    runner = mod.ContinuousPipeline(cfg_path, workspace=tmp_path / "ws")
    task = mod.Task("model", "path", "Class", "1,128", "float16")
    task.phase = "profiling"
    runner._advance_phase(task)
    assert task.phase == "extracting"
    runner._advance_phase(task)
    assert task.phase == "optimizing"
