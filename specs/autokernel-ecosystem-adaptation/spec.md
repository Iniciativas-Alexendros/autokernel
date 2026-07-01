# Spec: AutoKernel Ecosystem Adaptation

## Objetivo

Transformar el repositorio `autokernel` en una herramienta de mejora y evolución interna del software-hardware propio, integrada en el ecosistema de agentes y automatización de Alexendros. El pipeline actual (nocturno, 6 h, manual) evolucionará hacia un modo continuo, autónomo, versionado y documentado que optimiza kernels GPU, valida resultados y propone/integra cambios sin intervención humana.

## Contexto

- **Fuente de verdad**: `/home/alexendros/repositorios/org-iniciativas-alexendros/autokernel` (GitHub: `Iniciativas-Alexendros/autokernel`).
- **Clon obsoleto**: `/home/alexendros/projects/autokernel` se eliminará para evitar bifurcación de estado.
- **Hardware**: NVIDIA RTX 5060 Laptop GPU (SM 12.0, 8 GB VRAM), CUDA 13.1, GCC 15.2.0.
- **Stack actual**: Python 3.10+, `uv`, Triton, PyTorch, Ollama, Nemotron API, systemd, GitHub Pages.
- **Ecosistema de agentes**: `.devin/AGENTS.md`, modelos `opencode/*`, skills (`criticar`, `spec-driven`, `web-audit`), MCPs (`github`, `codebase-memory`, `context7`, `playwright`).

## Requerimientos funcionales

### RF1: Unificación del repositorio
- Eliminar el clon duplicado `/home/alexendros/projects/autokernel`.
- Sincronizar `origin/main` como única fuente de verdad.
- Documentar en `AGENTS.md` las rutas, comandos y convenciones de AutoKernel.

### RF2: Hardening del pipeline actual
- Corregir bugs en `scripts/nightly_pipeline.sh` (ej. `compglob`, defaults de `yq`, paths).
- Completar/auditar los subcomandos de `orchestrate.py` (`auto`, `migrate-cuda`, `report-extended`).
- Reforzar `systemd` units con recovery, variables de entorno y timeouts adecuados.
- Añadir tests de integración que validen el pipeline sin necesidad de GPU.

### RF3: Integración con el ecosistema de agentes
- Extender `.devin/AGENTS.md` (o crear `.devin/autokernel/AGENTS.md`) con contexto específico de AutoKernel.
- Configurar modelos `opencode` como alternativas locales a Ollama (`ornith:9b`, `qwen2.5-coder:7b`).
- Integrar MCPs: `github` para issues/PRs, `codebase-memory` para indexar kernels y resultados, `context7` para documentación Triton/CUDA.
- Definir prompts y skills estándar para operar AutoKernel desde agentes.

### RF4: Pipeline continuo 24×7
- Rediseñar `systemd` como servicio persistente con máquina de estados clara.
- Implementar cola de modelos y kernels priorizada.
- Añadir resiliencia: reintentos, backoff, rollback automático, límites de VRAM/CPU.
- Cache persistente de resultados de benchmark y perfiles.
- Scheduler que evite OOM y respete uso interactivo de la máquina.

### RF5: Auto-PR, auto-merge y gobierno
- Cada kernel optimizado y validado genera una rama `autokernel/<kernel>-<timestamp>`.
- Crear PR automáticamente vía MCP `github` con reporte de benchmark.
- Auto-merge solo si pasa CI, verificación end-to-end y no hay regresión.
- Políticas de protección: correctness ≥ umbral, speedup ≥ mínimo, latencia máxima.
- Publicar dashboard actualizado en GitHub Pages y notificar por `ntfy`.

### RF6: Meta-evolución y documentación viva
- Aplicar auditorías periódicas (`/criticar`) sobre el propio código de AutoKernel.
- Usar métricas del pipeline para auto-ajustar parámetros (timeouts, modelos LLM, prioridades).
- Actualizar automáticamente `README.md`, `docs/ARCHITECTURE.md` y `docs/PLAYBOOK.md` con resultados recientes.

## Criterios de aceptación

| ID | Criterio | Verificación |
| -- | -------- | ------------ |
| CA1 | No existe `/home/alexendros/projects/autokernel` | `ls` |
| CA2 | `git status` limpio en repo fuente | `git status` |
| CA3 | `AGENTS.md` documenta rutas y comandos de AutoKernel | `grep` |
| CA4 | `nightly_pipeline.sh` ejecuta sin errores de sintaxis/CLI | `bash -n` + ejecución manual |
| CA5 | Tests de integración pasan en CI | `uv run pytest` |
| CA6 | Pipeline puede ejecutarse con modelos `opencode` | Ejecución de prueba |
| CA7 | Servicio systemd continuo activo 24 h sin intervención | `systemctl status` + logs |
| CA8 | Se genera al menos un PR automático con kernel validado | `gh pr list` |
| CA9 | Dashboard refleja el último estado | HTTP 200 + contenido actualizado |
| CA10 | Documentación se actualiza automáticamente tras cada fase | diff en `docs/` |

## Restricciones

- No hardcodear secretos ni API keys; usar `pass-cli` / Proton Pass MCP.
- No modificar el sentido del pipeline actual hasta que la Fase 1 esté validada.
- No introducir dependencias nuevas sin justificación y revisión de seguridad.
- Mantener compatibilidad con RTX 5060 (SM 12.0) y CUDA 13.1.
- Respetar el code style existente (sin comentarios añadidos salvo petición).

## Fuera de alcance

- Cambiar el hardware objetivo.
- Soporte multi-GPU o cluster.
- Reescribir kernels desde cero fuera del flujo de optimización existente.
- Migrar el repo a privado o cambiar de organización.
