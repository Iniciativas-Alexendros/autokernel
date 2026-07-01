# Contract: AutoKernel Ecosystem Adaptation

## Componentes y responsabilidades

| Componente | Responsabilidad |
| ---------- | --------------- |
| `Orchestrator` | Máquina de estados del pipeline; decide qué kernel/modelo procesar, cuándo moverse, cuándo dar por terminado. |
| `PipelineRunner` | Ejecuta las fases en orden: profile → extract → optimize → verify → export → report. |
| `LLM Router` | Enruta peticiones a Ollama, Nemotron o modelos `opencode` según tarea y disponibilidad. |
| `Benchmark Harness` | Mide throughput, latencia, correctness, VRAM y clasifica bottleneck. |
| `Kernel` | Archivo `kernel.py` exportable con `KERNEL_TYPE`, `BACKEND`, `kernel_fn()`. |
| `Verifier` | Compara salida end-to-end del modelo original vs modelo con kernels optimizados. |
| `GitHub Publisher` | Crea ramas, commits, PRs y merges vía MCP `github` o `gh`. |
| `Dashboard` | Genera HTML/JSON del estado actual en `docs/`. |
| `Agent Bridge` | Expone AutoKernel a `.devin/AGENTS.md` y skills (`criticar`, `spec-driven`). |

## Contratos de interfaz

### 1. Orchestrator state (`workspace/orchestration_state.json`)

```json
{
  "current_kernel_idx": 0,
  "current_kernel_file": "workspace/llama_7b/kernel_matmul_1.py",
  "started_at": "2026-07-01T02:00:00Z",
  "phase": "optimizing",
  "kernels": [
    {
      "rank": 1,
      "file": "workspace/llama_7b/kernel_matmul_1.py",
      "op_type": "matmul",
      "pct_total": 62.3,
      "status": "optimizing",
      "baseline_tflops": 28.5,
      "best_tflops": 32.1,
      "speedup": 1.126,
      "pct_peak": 96.5,
      "experiments_run": 12,
      "experiments_kept": 4,
      "consecutive_reverts": 0,
      "time_spent_minutes": 45
    }
  ]
}
```

Invariantes:
- `status` ∈ {pending, optimizing, done, skipped, failed}.
- `speedup` se calcula como `best_tflops / baseline_tflops` cuando ambos existen.
- `current_kernel_idx` apunta siempre a un kernel con estado `optimizing` o al primer `pending` disponible.

### 2. Optimization plan (`workspace/{model}/optimization_plan.json`)

```json
{
  "model": "llama_7b",
  "input_shape": "1,512",
  "dtype": "float16",
  "kernels_to_optimize": [
    {
      "rank": 1,
      "op_type": "matmul",
      "pct_total": 62.3,
      "file": "workspace/llama_7b/kernel_matmul_1.py"
    }
  ]
}
```

### 3. Results TSV (`workspace/results/{kernel}_results.tsv`)

Columnas obligatorias, separadas por tabuladores:

```
experiment\ttag\tkernel_type\tthroughput_tflops\tlatency_us\tpct_peak\tspeedup_vs_pytorch\tcorrectness\tpeak_vram_mb\tdescription
```

Valores de `correctness` ∈ {PASS, FAIL, TIMEOUT, CRASH}.

### 4. Kernel contract (`kernel.py`)

Cada kernel debe exportar:

```python
KERNEL_TYPE = "matmul"          # str: matmul, flash_attention, softmax, ...
BACKEND = "triton"              # str: triton | cuda

def kernel_fn(*args, **kwargs):
    """Firma compatible con reference.py para el mismo KERNEL_TYPE."""
    ...
```

### 5. Pipeline config (`config/pipeline.yaml`)

El fichero YAML sigue siendo la fuente de verdad de:
- Modelos objetivo (`target_models`).
- Modelos LLM (`models.planner`, `models.coder`, `models.reviewer`).
- Límites operativos (`max_duration_hours`, `max_parallel_kernels`, `timeout_per_iteration_sec`).
- Umbrales de aceptación (`min_speedup`, `max_regression`, `correctness`).

### 6. Verifier output (`workspace/{model}/verification_YYYYMMDD.json`)

```json
{
  "model": "llama_7b",
  "input_shape": "1,512",
  "dtype": "float16",
  "reference_latency_ms": 142.5,
  "optimized_latency_ms": 98.3,
  "end_to_end_speedup": 1.45,
  "correctness": true,
  "max_abs_diff": 0.0012,
  "per_kernel": {
    "matmul": {"correctness": true, "speedup": 1.22}
  }
}
```

Invariantes:
- `correctness == true` es requisito para cualquier merge/PR.
- `end_to_end_speedup > 1.0` es requisito para merge automático.

### 7. GitHub PR contract

- Rama: `autokernel/<kernel>-<timestamp>`.
- Título: `autokernel: optimize <kernel> for <model> (+<speedup>x)`.
- Body: reporte markdown con benchmark, correctness, diff resumido y link al dashboard.
- Labels: `autokernel`, `kernel-optimization`, `automated`.
- Merge condition: CI verde + verificación e2e + revisión por Nemotron/opencode (si está habilitada).

### 8. Agent Bridge contract

- `.devin/AGENTS.md` o `AGENTS.md` raíz incluye sección `AutoKernel` con:
  - Paths de trabajo.
  - Comandos rápidos (`uv run orchestrate.py`, `bash scripts/nightly_pipeline.sh`, etc.).
  - Convenciones de commit/rama.
  - Política de secretos y hardware.
- Skills: `criticar` puede audit AutoKernel; `spec-driven` puede generar specs de nuevos kernels.

## Dependencias entre fases

| Fase | Entrada | Salida | Bloquea si falla |
| ---- | ------- | ------ | ---------------- |
| 0. Saneamiento | Repos duplicados | Repo único limpio | Todo |
| 1. Hardening | Pipeline actual | Pipeline estable | 2, 3, 4 |
| 2. Ecosistema | AGENTS.md, MCPs | Agentes operativos | 3, 4 |
| 3. Continuo | systemd, cola | Servicio persistente | 4 |
| 4. Auto-PR | Verifier, GitHub | PRs auto-generados | 5 |
| 5. Meta | Métricas, auditorías | Mejora auto-sostenida | — |
