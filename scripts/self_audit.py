#!/usr/bin/env python3
"""AutoKernel Self-Audit -- generate a living health report from runtime metrics.

Usage:
    uv run python scripts/self_audit.py --output docs/EVOLUTION.md

Reads:
    - workspace/continuous/metrics.json
    - workspace/*/verification_*.json
    - workspace/results/*_results.tsv
    - Current git stats

Outputs:
    - Markdown report with recommendations and a checklist for /criticar
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path = REPO_ROOT) -> tuple[int, str, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.returncode, result.stdout, result.stderr


def load_metrics() -> dict[str, Any]:
    path = REPO_ROOT / "workspace" / "continuous" / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def count_verified_kernels() -> tuple[int, int]:
    """Return (verified, failed) counts from verification JSON files."""
    verified = 0
    failed = 0
    for f in (REPO_ROOT / "workspace").rglob("verification_*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            if data.get("correctness"):
                verified += 1
            else:
                failed += 1
        except Exception as exc:
            logger.warning("could not read verification file %s: %s", f, exc)
    return verified, failed


def count_results_tsv() -> int:
    return len(list((REPO_ROOT / "workspace" / "results").glob("*_results.tsv")))


def git_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {}
    rc, out, _ = _run(["git", "rev-parse", "--short", "HEAD"])
    stats["commit"] = out.strip() if rc == 0 else "unknown"
    rc, out, _ = _run(["git", "status", "--short"])
    stats["uncommitted_files"] = len([line for line in out.splitlines() if line.strip()])
    rc, out, _ = _run(["git", "log", "--oneline", "-5"])
    stats["recent_commits"] = out.strip().splitlines() if rc == 0 else []
    return stats


def avg_phase_duration(metrics: dict[str, Any]) -> dict[str, float]:
    durations = metrics.get("phase_durations_sec", {})
    out: dict[str, float] = {}
    for label, values in durations.items():
        if values:
            out[label] = round(sum(values) / len(values), 2)
    return out


def generate_report() -> str:
    metrics = load_metrics()
    verified, failed = count_verified_kernels()
    result_tsv_count = count_results_tsv()
    stats = git_stats()
    avg_durations = avg_phase_duration(metrics)

    lines = [
        "# AutoKernel — Self-Audit & Evolution Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}Z",
        f"**Commit:** `{stats['commit']}`",
        "",
        "## Runtime Metrics",
        "",
        f"- Tasks completed: {metrics.get('tasks_completed', 0)}",
        f"- Tasks failed: {metrics.get('tasks_failed', 0)}",
        f"- Kernels verified (PASS): {verified}",
        f"- Kernels failed verification: {failed}",
        f"- Result TSV files: {result_tsv_count}",
        "",
        "## Average Phase Durations",
        "",
    ]
    if avg_durations:
        for label, avg in avg_durations.items():
            lines.append(f"- {label}: {avg:.2f}s")
    else:
        lines.append("- No phase data yet.")

    lines.extend(
        [
            "",
            "## Recent Commits",
            "",
        ]
    )
    for commit in stats["recent_commits"]:
        lines.append(f"- `{commit}`")

    lines.extend(
        [
            "",
            f"- Uncommitted files: {stats['uncommitted_files']}",
            "",
            "## Checklist for /criticar",
            "",
            "- [ ] Pipeline continuo sin intervención > 24 h.",
            "- [ ] Todos los tests unitarios/integración pasan.",
            "- [ ] Ningún secret hardcodeado en el repo.",
            "- [ ] Dependencias revisadas y sin vulnerabilidades críticas.",
            "- [ ] Documentación (README, ARCHITECTURE, PLAYBOOK) actualizada.",
            "- [ ] Servicios systemd validados con `systemd-analyze verify`.",
            "",
            "## Recommendations",
            "",
        ]
    )

    if metrics.get("tasks_failed", 0) > metrics.get("tasks_completed", 0):
        lines.append("- High failure rate: review error log and reduce model complexity.")
    if not avg_durations:
        lines.append("- No runtime data yet: run the continuous pipeline to populate metrics.")
    if stats["uncommitted_files"] > 0:
        lines.append("- Uncommitted files present: review and commit or clean up.")
    if verified == 0 and (metrics.get("tasks_completed", 0) > 0):
        lines.append("- Completed tasks but no verified kernels: check verification thresholds.")

    if len(lines) == 0 or lines[-1].startswith("## Recommendations"):
        lines.append("- No critical issues detected.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoKernel Self-Audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "EVOLUTION.md",
        help="Output markdown file",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report()
    tmp = args.output.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(report)
    tmp.replace(args.output)
    print(f"Self-audit report written to {args.output}")


if __name__ == "__main__":
    main()
