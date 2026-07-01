# Test Plan: AutoKernel Ecosystem Adaptation

## Estrategia

Tests first para cada fase. Los tests de integración y end-to-end se ejecutan en CI y localmente. Los tests que requieren GPU se marcan con `@pytest.mark.slow` y se ejecutan solo en el pipeline o bajo demanda.

## Tests por fase

### Fase 0 — Saneamiento y unificación

| ID | Test | Tipo | Verificación |
| -- | ---- | ---- | ------------ |
| T0.1 | `projects/autokernel` no existe | acceptance | `not os.path.exists("/home/alexendros/projects/autokernel")` |
| T0.2 | Repo fuente es clon limpio de GitHub | unit | `git remote -v` muestra `Iniciativas-Alexendros/autokernel` |
| T0.3 | `git status` no tiene archivos sin trackear relevantes | unit | `git status --short` vacío o solo artifacts esperados |
| T0.4 | `AGENTS.md` contiene sección AutoKernel | unit | `grep -i "autokernel" AGENTS.md` |

### Fase 1 — Hardening y correcciones críticas

| ID | Test | Tipo | Verificación |
| -- | ---- | ---- | ------------ |
| T1.1 | `scripts/nightly_pipeline.sh` sin errores de sintaxis | unit | `bash -n scripts/nightly_pipeline.sh` |
| T1.2 | Pipeline puede parsear `config/pipeline.yaml` | unit | `uv run python -c "import yaml; yaml.safe_load(open('config/pipeline.yaml'))"` |
| T1.3 | `orchestrate.py` subcomandos `status`, `next`, `record` funcionan | integration | `uv run python orchestrate.py --workspace workspace status` sin crash |
| T1.4 | `orchestrate.py` puede inicializar estado desde plan | integration | `uv run python orchestrate.py --workspace workspace plan` |
| T1.5 | `systemd` units parsean correctamente | unit | `systemd-analyze verify systemd/autokernel-nightly.service` |
| T1.6 | Tests existentes pasan | unit | `uv run pytest -m "not slow"` |

### Fase 2 — Integración con ecosistema de agentes

| ID | Test | Tipo | Verificación |
| -- | ---- | ---- | ------------ |
| T2.1 | `.devin/AGENTS.md` o `AGENTS.md` contiene contexto AutoKernel | unit | `grep -A 20 "AutoKernel" AGENTS.md` |
| T2.2 | LLM Router puede instanciarse con modelos opencode | unit | `uv run python -c "from autokernel.llm_assistant import LLMAssistant; LLMAssistant(planner_model='opencode/mimo-v2.5-free')"` |
| T2.3 | MCP `github` responde (si token disponible) | integration | `mcp_call_tool github / user` o equivalente |
| T2.4 | `codebase-memory` puede indexar kernels | integration | indexación de `kernels/` sin error |
| T2.5 | Prompts de agentes respetan contratos de kernel | unit | validación de `KERNEL_TYPE` y `BACKEND` |

### Fase 3 — Pipeline continuo 24×7

| ID | Test | Tipo | Verificación |
| -- | ---- | ---- | ------------ |
| T3.1 | Máquina de estados transita correctamente | unit | simular `idle → profiling → extracting → optimizing → verifying → reporting` |
| T3.2 | Scheduler respeta timeout de VRAM | integration | mock de uso de GPU > umbral → pipeline pausa |
| T3.3 | Cache de resultados persiste entre ejecuciones | integration | segundo run reutiliza baseline si no hay cambios |
| T3.4 | Recovery ante fallo reinicia desde último estado | integration | corrupt `orchestration_state.json` → reinit from plan |
| T3.5 | Servicio systemd puede iniciar/detener sin errores | e2e | `systemctl --user start autokernel-continuous` (o system) |

### Fase 4 — Auto-PR y auto-merge

| ID | Test | Tipo | Verificación |
| -- | ---- | ---- | ------------ |
| T4.1 | `GitHub Publisher` crea rama con nombre correcto | integration | `git branch --list 'autokernel/*'` |
| T4.2 | PR generado contiene reporte de benchmark | integration | `gh pr view <branch> --json title,body` |
| T4.3 | Merge bloqueado si `correctness` es false | integration | verificar que `auto-merge` no se activa |
| T4.4 | Dashboard se actualiza tras cambios en `docs/` | e2e | `git diff --name-only docs/` y HTTP 200 en GitHub Pages |
| T4.5 | Notificación webhook se envía en estado final | integration | mock de servidor ntfy o captura de curl |

### Fase 5 — Meta-evolución

| ID | Test | Tipo | Verificación |
| -- | ---- | ---- | ------------ |
| T5.1 | Métricas del pipeline se registran | integration | `workspace/metrics.json` con duración, modelos, kernels, reverts |
| T5.2 | Auditoría `/criticar` produce checklist | e2e | skill `criticar` ejecuta sobre repo y genera output |
| T5.3 | Actualización automática de docs no rompe build | e2e | `uv run python scripts/generate_dashboard.py` y `git status` limpio |

## Gates de calidad por fase

| Fase | Cobertura mínima | Lint errors | CI verde | GPU opcional |
| ---- | ---------------- | ----------- | -------- | -------------- |
| 0    | —                | 0           | Sí       | No             |
| 1    | 60%              | 0           | Sí       | No             |
| 2    | 70%              | 0           | Sí       | No             |
| 3    | 75%              | 0           | Sí       | Sí (slow)      |
| 4    | 80%              | 0           | Sí       | Sí (slow)      |
| 5    | 80%              | 0           | Sí       | No             |

## Comandos de ejecución

```bash
# Tests unitarios/integración (sin GPU)
uv run pytest -m "not slow"

# Tests lentos (requieren GPU)
uv run pytest -m "slow"

# Todos los tests
uv run pytest

# Lint
uv run ruff check .

# Typecheck (si se añade mypy)
uv run mypy autokernel/ kernels/ models/
```

## Notas

- Los tests de integración que dependan de `ollama` o `nemotron` deben poder mockearse con fixtures.
- Los tests de GitHub deben usar un repo de prueba o modo dry-run si no hay token.
- Los tests de systemd se ejecutan con `systemctl --user` si el usuario no tiene permisos de system.
