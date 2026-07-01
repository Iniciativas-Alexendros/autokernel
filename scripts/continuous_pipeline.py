#!/usr/bin/env python3
"""AutoKernel Continuous Pipeline -- 24x7 autonomous kernel evolution.

Usage:
    uv run python scripts/continuous_pipeline.py --config config/pipeline.yaml

The runner maintains a persistent queue of (model, kernel) optimization tasks
and executes them in a state machine loop. It is meant to run under systemd as
a long-lived service.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


class ShutdownRequest(Exception):
    """Raised when the runner receives a shutdown signal."""


@dataclass
class Task:
    """A single optimization task in the continuous queue."""

    model_name: str
    model_path: str
    model_class: str
    input_shape: str
    dtype: str
    kernel_type: str | None = None
    phase: str = "idle"
    status: str = "pending"
    attempts: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_class": self.model_class,
            "input_shape": self.input_shape,
            "dtype": self.dtype,
            "kernel_type": self.kernel_type,
            "phase": self.phase,
            "status": self.status,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            model_name=data.get("model_name", ""),
            model_path=data.get("model_path", ""),
            model_class=data.get("model_class", ""),
            input_shape=data.get("input_shape", ""),
            dtype=data.get("dtype", "float16"),
            kernel_type=data.get("kernel_type"),
            phase=data.get("phase", "idle"),
            status=data.get("status", "pending"),
            attempts=data.get("attempts", 0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )


class ContinuousPipeline:
    """State-machine driven continuous optimization pipeline."""

    PHASES = ["idle", "profiling", "extracting", "optimizing", "verifying", "reporting"]

    def __init__(self, config_path: Path, workspace: Path | None = None):
        self.config_path = config_path
        self.workspace = (workspace or REPO_ROOT / "workspace" / "continuous").resolve()
        self.queue_path = self.workspace / "queue.json"
        self.metrics_path = self.workspace / "metrics.json"
        self.state_path = self.workspace / "state.json"
        self.config: dict[str, Any] = {}
        self.queue: list[Task] = []
        self.metrics: dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "tasks_completed": 0,
            "tasks_failed": 0,
            "phase_durations_sec": {},
            "errors": [],
        }
        self._shutdown = False
        self._setup_signals()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.load_config()
        self.load_queue()
        self.load_metrics()

    def _setup_signals(self) -> None:
        def handler(signum, frame):
            self._shutdown = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def load_config(self) -> None:
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f) or {}

    def load_queue(self) -> None:
        if self.queue_path.exists():
            with open(self.queue_path) as f:
                raw = json.load(f)
            self.queue = [Task.from_dict(item) for item in raw]
        else:
            self.queue = self._build_initial_queue()
            self.save_queue()

    def _build_initial_queue(self) -> list[Task]:
        """Build queue from enabled models in the config."""
        queue: list[Task] = []
        models = self.config.get("pipeline", {}).get("target_models", [])
        for m in models:
            if not m.get("enabled", False):
                continue
            queue.append(
                Task(
                    model_name=m["name"],
                    model_path=m.get("path", f"models/{m['name']}.py"),
                    model_class=m.get("class", ""),
                    input_shape=m.get("shape", ""),
                    dtype=m.get("dtype", "float16"),
                )
            )
        return queue

    def save_queue(self) -> None:
        tmp = self.queue_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump([t.to_dict() for t in self.queue], f, indent=2)
        tmp.replace(self.queue_path)

    def load_metrics(self) -> None:
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                self.metrics = json.load(f)

    def save_metrics(self) -> None:
        tmp = self.metrics_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.metrics, f, indent=2)
        tmp.replace(self.metrics_path)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def gpu_available(self) -> bool:
        """Return True if the GPU is not being used by another process.

        If nvidia-smi is unavailable, assumes the GPU is available.
        """
        nvidia_smi = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
        if not Path(nvidia_smi).exists():
            return True
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return True
            util = float(result.stdout.strip().splitlines()[0].strip())
            threshold = self.config.get("continuous", {}).get("gpu_util_threshold", 25.0)
            return util < threshold
        except Exception as exc:
            logger.warning("gpu availability check failed: %s", exc)
            return True

    def _model_workspace(self, task: Task) -> Path:
        return self.workspace / "models" / task.model_name

    def _run_phase(self, task: Task, phase: str) -> bool:
        """Execute a single phase for a task. Returns True on success."""
        model_ws = self._model_workspace(task)
        model_ws.mkdir(parents=True, exist_ok=True)

        if phase == "profiling":
            cmd = [
                sys.executable,
                str(REPO_ROOT / "profile.py"),
                "--model",
                task.model_path,
                "--class-name",
                task.model_class,
                "--input-shape",
                task.input_shape,
                "--dtype",
                task.dtype,
                "--output",
                str(model_ws / "profile"),
            ]
            return self._run_command(cmd, "profile")

        if phase == "extracting":
            cmd = [
                sys.executable,
                str(REPO_ROOT / "extract.py"),
                "--top",
                str(
                    self.config.get("pipeline", {})
                    .get("phases", {})
                    .get("extract", {})
                    .get("top_k", 5)
                ),
                "--backend",
                self.config.get("pipeline", {})
                .get("phases", {})
                .get("extract", {})
                .get("backend", "triton"),
                "--report",
                str(model_ws / "profile" / "profile_report.json"),
            ]
            return self._run_command(cmd, "extract")

        if phase == "optimizing":
            opt_cfg = self.config.get("pipeline", {}).get("phases", {}).get("optimize", {})
            kernel_types = (
                [task.kernel_type] if task.kernel_type else self._kernel_types_from_plan(model_ws)
            )
            for kt in kernel_types:
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "orchestrate.py"),
                    "--workspace",
                    str(model_ws),
                    "auto",
                    "--kernel",
                    kt,
                    "--llm-planner",
                    opt_cfg.get("models", {}).get("planner", "ornith:9b"),
                    "--llm-coder",
                    opt_cfg.get("models", {}).get("coder", "qwen2.5-coder:7b"),
                    "--iterations",
                    str(opt_cfg.get("iterations_per_kernel", 5)),
                    "--timeout",
                    str(opt_cfg.get("timeout_per_iteration_sec", 1800)),
                ]
                if not self._run_command(cmd, f"optimize-{kt}"):
                    return False
            return True

        if phase == "verifying":
            cmd = [
                sys.executable,
                str(REPO_ROOT / "verify.py"),
                "--model",
                task.model_path,
                "--class-name",
                task.model_class,
                "--input-shape",
                task.input_shape,
                "--dtype",
                task.dtype,
                "--workspace",
                str(model_ws),
            ]
            return self._run_command(cmd, "verify")

        if phase == "reporting":
            cmd = [
                sys.executable,
                str(REPO_ROOT / "orchestrate.py"),
                "--workspace",
                str(model_ws),
                "report-extended",
            ]
            return self._run_command(cmd, "report")

        return True

    def _kernel_types_from_plan(self, model_ws: Path) -> list[str]:
        """Read optimization plan and return top kernel types."""
        plan_path = model_ws / "optimization_plan.json"
        if not plan_path.exists():
            return ["matmul", "flash_attention", "softmax", "rmsnorm", "elementwise"]
        with open(plan_path) as f:
            plan = json.load(f)
        kernels = plan.get("kernels_to_optimize", plan.get("kernels", []))
        seen: set[str] = set()
        out: list[str] = []
        for k in kernels[:5]:
            op = k.get("op_type", "")
            if op and op not in seen:
                seen.add(op)
                out.append(op)
        return out or ["matmul"]

    def _run_command(self, cmd: list[str], label: str) -> bool:
        """Run a command and record duration."""
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd, cwd=REPO_ROOT, timeout=3600, capture_output=True, text=True
            )
            elapsed = time.monotonic() - t0
            self._record_phase_duration(label, elapsed)
            if result.returncode != 0:
                self._record_error(f"{label} failed: {result.stderr[:500]}")
                return False
            return True
        except Exception as exc:
            elapsed = time.monotonic() - t0
            self._record_phase_duration(label, elapsed)
            self._record_error(f"{label} exception: {exc}")
            return False

    def _record_phase_duration(self, label: str, elapsed: float) -> None:
        durations = self.metrics.setdefault("phase_durations_sec", {})
        durations.setdefault(label, []).append(round(elapsed, 2))

    def _record_error(self, message: str) -> None:
        self.metrics.setdefault("errors", []).append({"timestamp": self._now(), "message": message})
        self.metrics["errors"] = self.metrics["errors"][-100:]  # keep last 100

    def _advance_phase(self, task: Task) -> None:
        """Move task to the next phase in the state machine."""
        idx = self.PHASES.index(task.phase)
        if idx < len(self.PHASES) - 1:
            task.phase = self.PHASES[idx + 1]
        else:
            task.status = "done"

    def tick(self) -> bool:
        """Run one iteration of the continuous loop. Returns True if work was done."""
        if self._shutdown:
            raise ShutdownRequest()

        # Find next pending task
        task = None
        for t in self.queue:
            if t.status == "pending":
                task = t
                break
        if task is None:
            self._rebuild_queue_if_done()
            return False

        # Wait for GPU if needed
        while not self.gpu_available() and not self._shutdown:
            time.sleep(10)
        if self._shutdown:
            raise ShutdownRequest()

        task.attempts += 1
        task.updated_at = self._now()
        self.save_queue()

        success = self._run_phase(task, task.phase)
        if success:
            self._advance_phase(task)
            if task.status == "done":
                self.metrics["tasks_completed"] += 1
        else:
            if task.attempts >= 3:
                task.status = "failed"
                self.metrics["tasks_failed"] += 1
            # else stay in same phase for retry
        task.updated_at = self._now()
        self.save_queue()
        self.save_metrics()
        return True

    def _rebuild_queue_if_done(self) -> None:
        """If all tasks are done, queue the next cycle."""
        if all(t.status in ("done", "failed") for t in self.queue):
            for t in self.queue:
                t.status = "pending"
                t.phase = "idle"
                t.attempts = 0
                t.updated_at = self._now()
            self.save_queue()

    def run(self) -> None:
        """Main continuous loop."""
        cfg = self.config.get("continuous", {})
        idle_sleep = cfg.get("idle_sleep_sec", 60)
        work_sleep = cfg.get("work_sleep_sec", 5)
        while not self._shutdown:
            try:
                worked = self.tick()
            except ShutdownRequest:
                break
            sleep = work_sleep if worked else idle_sleep
            time.sleep(sleep)
        self.save_queue()
        self.save_metrics()


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoKernel Continuous Pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "pipeline.yaml",
        help="Path to pipeline config (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Override continuous workspace directory",
    )
    args = parser.parse_args()

    runner = ContinuousPipeline(args.config, workspace=args.workspace)
    runner.run()


if __name__ == "__main__":
    main()
