#!/usr/bin/env python3
"""Generate AutoKernel HTML dashboard from orchestration state + results."""

import json
import sys
from pathlib import Path


def _load_state(workspace: Path) -> dict:
    state_path = workspace / "orchestration_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"kernels": []}


def _load_results(workspace: Path) -> dict[str, list[dict]]:
    results_dir = workspace / "results"
    out = {}
    if not results_dir.exists():
        return out
    for tsv in results_dir.glob("*.tsv"):
        lines = tsv.read_text().strip().split("\n")
        if len(lines) < 2:
            continue
        headers = lines[0].split("\t")
        rows = []
        for line in lines[1:]:
            vals = line.split("\t")
            rows.append(dict(zip(headers, vals)))
        out[tsv.stem] = rows
    return out


def _load_verify(workspace: Path) -> dict | None:
    for vf in sorted(workspace.glob("verification_*.json")):
        return json.loads(vf.read_text())
    return None


def _load_profile(workspace: Path) -> dict | None:
    for pf in [
        workspace / "profile" / "profile_report.json",
        workspace / "profile_report.json",
    ]:
        if pf.exists():
            return json.loads(pf.read_text())
    return None


def _kernel_rows(kernels: list[dict]) -> str:
    rows = ""
    for k in kernels:
        op = k.get("op_type", "?")
        status = k.get("status", "pending")
        baseline = k.get("baseline_tflops") or "-"
        best = k.get("best_tflops") or "-"
        speedup = k.get("speedup") or "-"
        if isinstance(speedup, float):
            speedup = f"{speedup:.3f}x"
        exps = k.get("experiments_run", 0)
        kept = k.get("experiments_kept", 0)
        reverts = k.get("consecutive_reverts", 0)
        minutes = k.get("time_spent_minutes", 0)

        badge = "bg-success" if status == "completed" else "bg-secondary"
        rows += f"""
        <tr>
          <td><span class="badge {badge}">{status}</span></td>
          <td><code>{op}</code></td>
          <td>{baseline}</td>
          <td>{best}</td>
          <td>{speedup}</td>
          <td>{exps}</td>
          <td>{kept}</td>
          <td>{reverts}</td>
          <td>{minutes}m</td>
        </tr>"""
    return rows


def _experiment_table(results: dict[str, list[dict]]) -> str:
    sections = ""
    for kernel_name, rows in results.items():
        if not rows:
            continue
        headers = rows[0].keys()
        th = "".join(f"<th>{h}</th>" for h in headers)
        trs = ""
        for r in rows:
            tds = ""
            for v in r.values():
                val = str(v) if v else "-"
                tds += f"<td><small>{val}</small></td>"
            trs += f"<tr>{tds}</tr>"
        sections += f"""
      <h6 class="mt-3">{kernel_name}</h6>
      <div class="table-responsive">
        <table class="table table-sm table-striped">
          <thead><tr>{th}</tr></thead>
          <tbody>{trs}</tbody>
        </table>
      </div>"""
    return sections


def generate_dashboard(workspace: Path, config_path: Path, output: Path) -> None:
    state = _load_state(workspace)
    results = _load_results(workspace)
    verify = _load_verify(workspace)
    profile = _load_profile(workspace)

    kernels = state.get("kernels", [])
    completed = [k for k in kernels if k.get("status") == "completed"]
    pending = [k for k in kernels if k.get("status") != "completed"]
    avg_speedup = 0.0
    if completed:
        speedups = [k["speedup"] for k in completed if k.get("speedup")]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

    target_models = []
    try:
        import yaml

        cfg = yaml.safe_load(config_path.read_text())
        for m in cfg.get("pipeline", {}).get("target_models", []):
            if m.get("enabled", True):
                target_models.append(m)
    except Exception:
        pass

    models_html = ""
    for m in target_models:
        models_html += f"""
        <tr>
          <td><code>{m["name"]}</code></td>
          <td>{m["class"]}</td>
          <td>{m["shape"]}</td>
          <td>{m["dtype"]}</td>
        </tr>"""

    verify_html = ""
    if verify:
        for k, v in verify.items():
            verify_html += f"<tr><td>{k}</td><td>{v}</td></tr>"

    profile_html = ""
    if profile:
        for entry in profile.get("top_kernels", profile.get("kernels", []))[:10]:
            name = entry.get("name", entry.get("op_type", "?"))
            pct = entry.get("pct_total", entry.get("percentage", 0))
            profile_html += f"<tr><td><code>{name}</code></td><td>{pct:.1f}%</td></tr>"

    plotly_data = {
        "x": [k.get("op_type", "?") for k in kernels],
        "y": [k.get("speedup") or 0 for k in kernels],
        "type": "bar",
        "marker": {
            "color": [
                "#198754" if k.get("status") == "completed" else "#6c757d"
                for k in kernels
            ]
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoKernel Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
  <style>
    body {{ background: #0d1117; color: #c9d1d9; }}
    .accordion-button {{ background: #161b22; color: #c9d1d9; }}
    .accordion-button:not(.collapsed) {{ background: #1f2937; color: #58a6ff; }}
    .accordion-body {{ background: #161b22; }}
    .card {{ background: #161b22; border-color: #30363d; }}
    .table {{ color: #c9d1d9; }}
    code {{ color: #79c0ff; }}
    .badge {{ font-weight: 400; }}
    .stat-card {{ text-align: center; padding: 1.5rem; }}
    .stat-card h2 {{ font-size: 2.5rem; margin: 0; }}
    .stat-card p {{ margin: 0; color: #8b949e; }}
  </style>
</head>
<body>
  <div class="container-fluid py-4">
    <h1 class="mb-4">AutoKernel Dashboard</h1>
    <p class="text-muted">Last updated: {_now()}</p>

    <div class="row mb-4">
      <div class="col-md-3">
        <div class="card stat-card">
          <h2 class="text-success">{len(completed)}</h2>
          <p>Completed</p>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card stat-card">
          <h2 class="text-warning">{len(pending)}</h2>
          <p>Pending</p>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card stat-card">
          <h2 class="text-info">{avg_speedup:.2f}x</h2>
          <p>Avg Speedup</p>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card stat-card">
          <h2 class="text-light">{len(target_models)}</h2>
          <p>Target Models</p>
        </div>
      </div>
    </div>

    <div class="accordion" id="dashboard">
      <div class="accordion-item">
        <h2 class="accordion-header"><button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#s-models">Target Models</button></h2>
        <div id="s-models" class="accordion-collapse collapse show" data-bs-parent="#dashboard">
          <div class="accordion-body">
            <table class="table table-sm">
              <thead><tr><th>Name</th><th>Class</th><th>Shape</th><th>Dtype</th></tr></thead>
              <tbody>{models_html}</tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="accordion-item">
        <h2 class="accordion-header"><button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#s-kernels">Kernel Optimization</button></h2>
        <div id="s-kernels" class="accordion-collapse collapse" data-bs-parent="#dashboard">
          <div class="accordion-body">
            <div id="speedup-chart"></div>
            <table class="table table-sm mt-3">
              <thead><tr><th>Status</th><th>Type</th><th>Baseline</th><th>Best</th><th>Speedup</th><th>Exps</th><th>Kept</th><th>Reverts</th><th>Time</th></tr></thead>
              <tbody>{_kernel_rows(kernels)}</tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="accordion-item">
        <h2 class="accordion-header"><button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#s-experiments">Experiments</button></h2>
        <div id="s-experiments" class="accordion-collapse collapse" data-bs-parent="#dashboard">
          <div class="accordion-body">{_experiment_table(results)}</div>
        </div>
      </div>

      <div class="accordion-item">
        <h2 class="accordion-header"><button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#s-profile">Profile Top Kernels</button></h2>
        <div id="s-profile" class="accordion-collapse collapse" data-bs-parent="#dashboard">
          <div class="accordion-body">
            <table class="table table-sm">
              <thead><tr><th>Kernel</th><th>% Time</th></tr></thead>
              <tbody>{profile_html}</tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="accordion-item">
        <h2 class="accordion-header"><button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#s-verify">Verification</button></h2>
        <div id="s-verify" class="accordion-collapse collapse" data-bs-parent="#dashboard">
          <div class="accordion-body">
            {"<table class='table table-sm'><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>" + verify_html + "</tbody></table>" if verify_html else "<p class='text-muted'>No verification data yet.</p>"}
          </div>
        </div>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    Plotly.newPlot('speedup-chart', [{json.dumps(plotly_data)}], {{
      title: 'Speedup by Kernel',
      paper_bgcolor: '#161b22',
      plot_bgcolor: '#0d1117',
      font: {{ color: '#c9d1d9' }},
      xaxis: {{ title: 'Kernel' }},
      yaxis: {{ title: 'Speedup (x)', rangemode: 'tozero' }}
    }}, {{responsive: true}});
  </script>
</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Dashboard written to {output}")


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate AutoKernel HTML dashboard")
    p.add_argument("--workspace", type=str, default="workspace")
    p.add_argument("--config", type=str, default="config/pipeline.yaml")
    p.add_argument("--output", type=str, default="docs/index.html")
    args = p.parse_args()

    workspace = Path(args.workspace)
    config = Path(args.config)
    output = Path(args.output)

    if not workspace.exists():
        print(f"ERROR: workspace not found: {workspace}", file=sys.stderr)
        sys.exit(1)

    generate_dashboard(workspace, config, output)


if __name__ == "__main__":
    main()
