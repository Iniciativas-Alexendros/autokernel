"""GitHub publisher for AutoKernel -- creates PRs with validated kernel optimizations."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PublishResult:
    """Outcome of a publish attempt."""

    success: bool
    branch: str
    pr_url: str = ""
    merged: bool = False
    message: str = ""


class GitHubPublisher:
    """Publish optimized kernels to GitHub via `gh` CLI."""

    def __init__(self, repo_dir: Path | None = None, default_branch: str = "main"):
        self.repo_dir = (repo_dir or REPO_ROOT).resolve()
        self.default_branch = default_branch
        self._ensure_gh()

    def _ensure_gh(self) -> None:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True,
            cwd=self.repo_dir,
        )
        if result.returncode != 0:
            raise RuntimeError("gh CLI no disponible")

    def _run(self, cmd: list[str], **kwargs: Any) -> tuple[int, str, str]:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.repo_dir,
            **kwargs,
        )
        return result.returncode, result.stdout, result.stderr

    def _git(self, args: list[str]) -> tuple[int, str, str]:
        return self._run(["git"] + args)

    def _current_branch(self) -> str:
        _, out, _ = self._git(["branch", "--show-current"])
        return out.strip()

    def _create_branch(self, branch: str) -> bool:
        rc, _, err = self._git(["checkout", "-b", branch])
        if rc != 0:
            # Maybe branch exists; try to checkout
            rc2, _, err2 = self._git(["checkout", branch])
            if rc2 != 0:
                raise RuntimeError(f"No se pudo crear/ checkout rama {branch}: {err}\n{err2}")
        return True

    def _commit_changes(self, message: str, paths: list[str]) -> bool:
        self._git(["add"] + paths)
        rc, _, err = self._git(["commit", "-m", message])
        if rc != 0 and "nothing to commit" not in err:
            return False
        return True

    def _push(self, branch: str) -> bool:
        rc, _, err = self._git(["push", "-u", "origin", branch])
        if rc != 0:
            raise RuntimeError(f"push failed: {err}")
        return True

    def _create_pr(
        self,
        branch: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> str:
        cmd = [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--base",
            self.default_branch,
            "--head",
            branch,
        ]
        if labels:
            cmd.extend(["--label", ",".join(labels)])
        rc, out, err = self._run(cmd)
        if rc != 0:
            raise RuntimeError(f"gh pr create failed: {err}")
        # Extract URL from gh output (last line is usually the URL)
        url = out.strip().splitlines()[-1]
        return url

    def _enable_auto_merge(self, branch: str) -> bool:
        rc, _, err = self._run(["gh", "pr", "merge", branch, "--auto", "--squash"])
        if rc != 0:
            # Auto-merge not always available; leave PR open
            return False
        return True

    def publish(
        self,
        kernel_path: Path,
        model_name: str,
        kernel_type: str,
        verification: dict[str, Any],
        report_text: str = "",
    ) -> PublishResult:
        """Publish a validated kernel as a GitHub PR.

        Args:
            kernel_path: Path to the optimized kernel file.
            model_name: Name of the target model (e.g. llama_7b).
            kernel_type: Kernel type (e.g. matmul).
            verification: Dict with at least 'correctness' (bool) and 'speedup' (float).
            report_text: Markdown report to include in the PR body.
        """
        correctness = verification.get("correctness", False)
        speedup = verification.get("speedup", 0.0)
        if not correctness or speedup <= 1.0:
            return PublishResult(
                success=False,
                branch="",
                message="publicación bloqueada: correctness o speedup no cumplen umbral",
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        branch = f"autokernel/{kernel_type}-{timestamp}"
        original_branch = self._current_branch()

        try:
            self._git(["checkout", self.default_branch])
            self._git(["pull", "--rebase", "origin", self.default_branch])
            self._create_branch(branch)

            # Destination for optimized kernels
            dest_dir = self.repo_dir / "kernels" / "optimized" / model_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"{kernel_type}_optimized.py"

            # Copy kernel and report
            import shutil
            shutil.copy2(kernel_path, dest_file)
            report_path = dest_dir / f"{kernel_type}_report.md"
            with open(report_path, "w") as f:
                f.write(report_text or f"AutoKernel optimization report for {kernel_type}")

            commit_msg = f"autokernel: optimize {kernel_type} for {model_name} (+{speedup:.2f}x)"
            self._commit_changes(commit_msg, [str(dest_file), str(report_path)])
            self._push(branch)

            title = f"autokernel: optimize {kernel_type} for {model_name} (+{speedup:.2f}x)"
            body_lines = [
                f"## AutoKernel Optimization Report",
                "",
                f"- **Model**: {model_name}",
                f"- **Kernel**: {kernel_type}",
                f"- **Correctness**: {'PASS' if correctness else 'FAIL'}",
                f"- **Speedup**: {speedup:.3f}x",
                "",
                "### Report",
                report_text or "_No detailed report provided._",
            ]
            pr_url = self._create_pr(
                branch, title, "\n".join(body_lines), labels=["autokernel", "automated"]
            )

            merged = self._enable_auto_merge(branch)

            self._git(["checkout", original_branch])
            return PublishResult(
                success=True,
                branch=branch,
                pr_url=pr_url,
                merged=merged,
                message="PR creado" + (" y auto-merge habilitado" if merged else ", pendiente de merge"),
            )
        except Exception as exc:
            # Try to return to original branch
            try:
                self._git(["checkout", original_branch])
            except Exception:
                pass
            return PublishResult(
                success=False,
                branch=branch,
                message=f"error durante publicación: {exc}",
            )
