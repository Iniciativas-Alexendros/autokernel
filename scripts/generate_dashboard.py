#!/usr/bin/env python3
"""Generate AutoKernel HTML dashboard — professional, clean, clear design."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


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
        rows = [dict(zip(headers, line.split("\t"))) for line in lines[1:]]
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


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")


def _status_badge(status: str) -> str:
    colors = {
        "completed": "#10b981",
        "optimizing": "#f59e0b",
        "pending": "#6b7280",
        "failed": "#ef4444",
    }
    color = colors.get(status, "#6b7280")
    return f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:500;color:#fff;background:{color}">{status}</span>'


def _speedup_cell(speedup) -> str:
    if speedup is None or speedup == "-":
        return '<span style="color:#6b7280">—</span>'
    val = float(speedup)
    if val >= 1.5:
        color = "#10b981"
    elif val >= 1.1:
        color = "#f59e0b"
    elif val >= 1.0:
        color = "#3b82f6"
    else:
        color = "#ef4444"
    return f'<span style="color:{color};font-weight:600">{val:.3f}x</span>'


def _kernel_rows(kernels: list[dict]) -> str:
    rows = ""
    for k in kernels:
        op = k.get("op_type", "?")
        status = k.get("status", "pending")
        baseline = k.get("baseline_tflops") or "—"
        best = k.get("best_tflops") or "—"
        speedup = k.get("speedup")
        exps = k.get("experiments_run", 0)
        kept = k.get("experiments_kept", 0)
        minutes = k.get("time_spent_minutes", 0)

        rows += f"""
        <tr>
          <td>{_status_badge(status)}</td>
          <td><code style="color:#6366f1">{op}</code></td>
          <td style="text-align:right">{baseline}</td>
          <td style="text-align:right">{best}</td>
          <td style="text-align:right">{_speedup_cell(speedup)}</td>
          <td style="text-align:center">{exps}</td>
          <td style="text-align:center">{kept}</td>
          <td style="text-align:center">{minutes}m</td>
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
            tds = "".join(f"<td>{str(v) if v else '—'}</td>" for v in r.values())
            trs += f"<tr>{tds}</tr>"
        sections += f"""
        <h6 style="margin-top:1.5rem;color:#374151;font-weight:600">{kernel_name}</h6>
        <div style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
            <thead><tr style="border-bottom:2px solid #e5e7eb">{th}</tr></thead>
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

    total_exps = sum(k.get("experiments_run", 0) for k in kernels)
    total_time = sum(k.get("time_spent_minutes", 0) for k in kernels)

    target_models = []
    try:
        import yaml

        cfg = yaml.safe_load(config_path.read_text())
        for m in cfg.get("pipeline", {}).get("target_models", []):
            if m.get("enabled", True):
                target_models.append(m)
    except Exception as exc:
        logger.warning("could not load target models from %s: %s", config_path, exc)

    models_rows = ""
    for m in target_models:
        models_rows += f"""
        <tr>
          <td><code style="color:#6366f1">{m["name"]}</code></td>
          <td>{m["class"]}</td>
          <td><code>{m["shape"]}</code></td>
          <td>{m["dtype"]}</td>
        </tr>"""

    verify_rows = ""
    if verify:
        for k, v in verify.items():
            status_icon = (
                "✅"
                if v in ("PASS", True, "pass")
                else "❌"
                if v in ("FAIL", False, "fail")
                else "—"
            )
            verify_rows += f"<tr><td>{k}</td><td>{status_icon} {v}</td></tr>"

    profile_rows = ""
    if profile:
        for entry in profile.get("top_kernels", profile.get("kernels", []))[:10]:
            name = entry.get("name", entry.get("op_type", "?"))
            pct = entry.get("pct_total", entry.get("percentage", 0))
            bar_width = min(pct, 100)
            profile_rows += f"""
            <tr>
              <td><code style="color:#6366f1">{name}</code></td>
              <td style="width:60%">
                <div style="background:#e5e7eb;border-radius:4px;height:8px;width:100%">
                  <div style="background:#6366f1;border-radius:4px;height:8px;width:{bar_width}%"></div>
                </div>
              </td>
              <td style="text-align:right;font-weight:500">{pct:.1f}%</td>
            </tr>"""

    plotly_kernels = [k.get("op_type", "?") for k in kernels]
    plotly_speedups = [k.get("speedup") or 0 for k in kernels]
    plotly_colors = ["#10b981" if k.get("status") == "completed" else "#d1d5db" for k in kernels]

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoKernel — Dashboard</title>
  <meta name="description" content="Dashboard del pipeline de optimización autónoma de kernels GPU para RTX 5060">
  <link rel="canonical" href="https://iniciativas-alexendros.github.io/autokernel/">
  <meta property="og:title" content="AutoKernel — Dashboard">
  <meta property="og:description" content="Pipeline de optimización autónoma de kernels GPU para RTX 5060">
  <meta property="og:url" content="https://iniciativas-alexendros.github.io/autokernel/">
  <meta property="og:type" content="website">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="AutoKernel — Dashboard">
  <meta name="twitter:description" content="Pipeline de optimización autónoma de kernels GPU para RTX 5060">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' https://cdn.plot.ly 'unsafe-inline'; style-src 'self' https://fonts.googleapis.com https://cdn.jsdelivr.net 'unsafe-inline'; font-src https://fonts.gstatic.com; img-src 'self' data:;">
  <meta http-equiv="X-Content-Type-Options" content="nosniff">
  <meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">
  <link rel="preconnect" href="https://fonts.googleapis.com" crossorigin="anonymous">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" crossorigin="anonymous">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" crossorigin="anonymous">
  <script src="https://cdn.plot.ly/plotly-2.35.0.min.js" defer crossorigin="anonymous"></script>
  <script>
    if (window.top !== window.self) {{ window.top.location = window.self.location; }}
  </script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Inter', -apple-system, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; }}
    .header {{ background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: #fff; padding: 2rem 0; }}
    .header h1 {{ font-size: 1.75rem; font-weight: 700; margin: 0; }}
    .header p {{ color: #94a3b8; margin: 0.25rem 0 0; font-size: 0.9rem; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: -1.5rem; padding: 0 1rem; position: relative; z-index: 1; }}
    .stat-card {{ background: #fff; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
    .stat-card .value {{ font-size: 2rem; font-weight: 700; line-height: 1; }}
    .stat-card .label {{ font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .section {{ background: #fff; border-radius: 12px; padding: 1.5rem; margin: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    .section-title {{ font-size: 1rem; font-weight: 600; color: #1e293b; margin: 0 0 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; gap: 0.5rem; }}
    .section-title .icon {{ font-size: 1.1rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    th {{ text-align: left; font-weight: 600; color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; padding: 0.5rem 0.75rem; border-bottom: 2px solid #e2e8f0; }}
    td {{ padding: 0.625rem 0.75rem; border-bottom: 1px solid #f1f5f9; }}
    tr:hover {{ background: #f8fafc; }}
    code {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }}
    .footer {{ text-align: center; padding: 2rem; color: #94a3b8; font-size: 0.8rem; }}
    @media (max-width: 768px) {{
      .stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .section {{ margin: 0.5rem; padding: 1rem; }}
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="container">
      <h1>⚡ AutoKernel</h1>
      <p>Pipeline nocturno de optimización de kernels GPU — RTX 5060</p>
      <p style="color:#64748b;font-size:0.8rem">Última actualización: {_now()}</p>
    </div>
  </div>

  <div class="container">
    <div class="stat-grid">
      <div class="stat-card">
        <div class="value" style="color:#10b981">{len(completed)}</div>
        <div class="label">Completados</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color:#f59e0b">{len(pending)}</div>
        <div class="label">Pendientes</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color:#6366f1">{avg_speedup:.2f}x</div>
        <div class="label">Speedup Medio</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color:#1e293b">{total_exps}</div>
        <div class="label">Experimentos</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color:#1e293b">{total_time}m</div>
        <div class="label">Tiempo Total</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color:#1e293b">{len(target_models)}</div>
        <div class="label">Modelos</div>
      </div>
    </div>

    <div class="section">
      <div class="section-title"><span class="icon">📊</span> Speedup por Kernel</div>
      <div id="speedup-chart" style="height:300px"></div>
    </div>

    <div class="section">
      <div class="section-title"><span class="icon">🎯</span> Modelos Objetivo</div>
      <table>
        <thead><tr><th>Modelo</th><th>Clase</th><th>Shape</th><th>Dtype</th></tr></thead>
        <tbody>{models_rows if models_rows else '<tr><td colspan="4" style="color:#94a3b8;text-align:center">Sin modelos configurados</td></tr>'}</tbody>
      </table>
    </div>

    <div class="section">
      <div class="section-title"><span class="icon">🔧</span> Optimización de Kernels</div>
      <table>
        <thead><tr><th>Estado</th><th>Tipo</th><th>Baseline</th><th>Mejor</th><th>Speedup</th><th>Exps</th><th>Conservados</th><th>Tiempo</th></tr></thead>
        <tbody>{_kernel_rows(kernels) if kernels else '<tr><td colspan="8" style="color:#94a3b8;text-align:center">Sin datos de optimización</td></tr>'}</tbody>
      </table>
    </div>

    <div class="section">
      <div class="section-title"><span class="icon">🧪</span> Perfil — Top Kernels</div>
      <table>
        <thead><tr><th>Kernel</th><th>Distribución</th><th style="text-align:right">% Tiempo</th></tr></thead>
        <tbody>{profile_rows if profile_rows else '<tr><td colspan="3" style="color:#94a3b8;text-align:center">Sin datos de perfilado</td></tr>'}</tbody>
      </table>
    </div>

    <div class="section">
      <div class="section-title"><span class="icon">✅</span> Verificación End-to-End</div>
      <table>
        <thead><tr><th>Check</th><th>Resultado</th></tr></thead>
        <tbody>{verify_rows if verify_rows else '<tr><td colspan="2" style="color:#94a3b8;text-align:center">Sin datos de verificación</td></tr>'}</tbody>
      </table>
    </div>

    <div class="section">
      <div class="section-title"><span class="icon">📋</span> Experimentos</div>
      {_experiment_table(results) if results else '<p style="color:#94a3b8;text-align:center;margin:0">Sin experimentos registrados.</p>'}
    </div>
  </div>

  <div class="footer">
    AutoKernel — Pipeline nocturno de optimización de kernels GPU<br>
    <a href="https://github.com/Iniciativas-Alexendros/autokernel" style="color:#6366f1;text-decoration:none">GitHub</a> ·
    <a href="https://iniciativas-alexendros.github.io/autokernel/" style="color:#6366f1;text-decoration:none">Dashboard</a>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    const kernels = {json.dumps(plotly_kernels)};
    const speedups = {json.dumps(plotly_speedups)};
    const colors = {json.dumps(plotly_colors)};

    if (kernels.length > 0) {{
      Plotly.newPlot('speedup-chart', [{{
        x: kernels,
        y: speedups,
        type: 'bar',
        marker: {{ color: colors, cornerradius: 6 }},
        text: speedups.map(s => s > 0 ? s.toFixed(2) + 'x' : ''),
        textposition: 'outside',
        textfont: {{ size: 12, color: '#64748b' }}
      }}], {{
        margin: {{ t: 10, b: 40, l: 50, r: 20 }},
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ family: 'Inter, sans-serif', color: '#64748b' }},
        xaxis: {{ gridcolor: '#f1f5f9', tickfont: {{ size: 11 }} }},
        yaxis: {{ title: 'Speedup (x)', gridcolor: '#f1f5f9', rangemode: 'tozero', tickfont: {{ size: 11 }} }},
        bargap: 0.3
      }}, {{ responsive: true, displayModeBar: false }});
    }} else {{
      document.getElementById('speedup-chart').innerHTML = '<p style="color:#94a3b8;text-align:center;padding:2rem">Sin datos de kernels</p>';
    }}
  </script>

</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Dashboard written to {output}")


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
