# AutoKernel — Reglas para agentes

## Contexto

- **Repo**: `/home/alexendros/repositorios/org-iniciativas-alexendros/autokernel`
- **GitHub**: `https://github.com/Iniciativas-Alexendros/autokernel`
- **Propósito**: Evolución autónoma de kernels GPU para el stack propio de Alexendros.
- **Hardware**: NVIDIA RTX 5060 Laptop (SM 12.0, 8 GB VRAM), CUDA 13.1, GCC 15.2.0.

## Comandos operativos

| Comando | Acción |
| ------- | ------ |
| `uv run orchestrate.py --workspace workspace status` | Estado del pipeline |
| `uv run orchestrate.py --workspace workspace next` | Siguiente decisión del orquestador |
| `uv run orchestrate.py --workspace workspace plan` | Plan de optimización con Amdahl |
| `uv run orchestrate.py --workspace workspace auto --kernel <tipo>` | Optimizar un kernel con LLM |
| `uv run orchestrate.py --workspace workspace migrate-cuda --kernel <tipo>` | Migrar Triton a CUDA C++ |
| `uv run python profile.py --model <path> --class-name <Name> --input-shape <shape>` | Profilear modelo |
| `uv run python verify.py --model <path> --class-name <Name> --input-shape <shape>` | Verificación end-to-end |
| `bash scripts/nightly_pipeline.sh` | Pipeline nocturno manual |
| `uv run python scripts/generate_dashboard.py --workspace workspace --config config/pipeline.yaml --output docs/index.html` | Generar dashboard |
| `uv run pytest -m "not slow" -q --cov=autokernel --cov=scripts --cov-fail-under=70` | Tests con cobertura mínima 70% |
| `uv run pytest` | Todos los tests (incluye GPU) |
| `uv run ruff check . && uv run ruff format --check .` | Lint y formato |
| `uv run bandit -r autokernel scripts -ll && gitleaks detect --source .` | Seguridad básica |

## Convenciones

- Nuevas features: `specs/<feature>/` con `spec.md`, `contract.md`, `scenarios.md`, `test-plan.md`.
- Ramas de kernel: `autokernel/<kernel>-<timestamp>`.
- Commits automáticos: `autokernel: <acción> <kernel> for <model> (+<speedup>x)`.
- No modificar `bench.py`, `reference.py`, `prepare.py`, `verify.py` salvo bug crítico con spec y test.
- No introducir dependencias sin justificar en `pyproject.toml`.
- No comments añadidos salvo petición explícita.

## Modelos LLM

| Rol | Modelo por defecto | Fallback |
| --- | ------------------ | -------- |
| Planner | `ornith:9b` (Ollama) | `opencode/mimo-v2.5-free` |
| Coder | `qwen2.5-coder:7b` (Ollama) | `opencode/deepseek-v4-flash-free` |
| Reviewer | `nvidia/nemotron-3-ultra` (NVIDIA API) | `opencode/nemotron-3-ultra-free` |
| Embeddings | `bge-m3` (Ollama) | — |

El fallback a `opencode/*` requiere `OPENROUTER_API_KEY` o token configurado en env; nunca hardcodear.

## MCPs recomendados

- `github`: crear issues/PRs, listar ramas, mergear cambios validados.
- `codebase-memory`: indexar kernels, resultados y specs para RAG.
- `context7`: consultar documentación Triton, CUDA, PyTorch.
- `proton-pass-community-mcp` / `pass-cli`: gestión de secretos.

## Skills recomendadas

- `/spec <feature>` para nuevas capacidades o kernels.
- `/criticar autokernel/` para auditorías periódicas del pipeline.
- `/deps` para revisar dependencias obsoletas o inseguras.
- `/fmt` para formatear código Python antes de commit.

## Seguridad

- Secretos (Nemotron, OpenRouter, GitHub): solo vía env o `pass-cli`. Nunca en JSON, YAML ni chat.
- `.env` files: denegar lectura/escritura salvo `.env.example`.
- `git push` y `rm`: requiere confirmación del operador; en scripts autónomos, usar PR + auto-merge con protecciones.

## Métricas de éxito

- correctness ≥ 99%.
- speedup end-to-end > 1.0x.
- Pipeline continuo sin intervención humana > 24 h.
- PRs automáticos validados por CI y reviewer.
- Cobertura de tests ≥ 70% y sin issues de seguridad en bandit/gitleaks.

## Propuestas de magnificación (evaluadas)

| Prioridad | Propuesta | Impacto | Esfuerzo | Próximo paso |
| --- | --- | --- | --- | --- |
| 1 | Venderizar Bootstrap/Plotly/Inter y añadir SRI | Seguridad, offline | Medio | Descargar assets y generar hashes sha384 en CI. |
| 2 | Hardening de URLs y añadir HSTS preload | Seguridad | Bajo | Audit redirects DNS y enviar dominio a hstspreload.org. |
| 3 | Test suite de GPU reales (pytest slow) | Fiabilidad | Alto | Job CI auto-hospedado con runner que tenga RTX 5060. |
| 4 | Alertas/métricas Prometheus + Grafana | Observabilidad | Medio | Exponer /metrics en continuous_pipeline y scrapear. |
| 5 | Multi-GPU scheduling y cola prioritaria | Escalabilidad | Alto | Refactor ResourceSemaphore con semáforos por GPU. |
| 6 | Auto-rollback de kernels con regresión | Robustez | Medio | Comparar verification_*.json y restaurar baseline. |
