# Spec: Limpieza del Repo AutoKernel

## Objetivo

Convertir el repositorio `autokernel` en la configuración local de Alexendros para el pipeline nocturno de optimización de kernels GPU, eliminando todo contenido no esencial heredado del upstream RightNowAI.

## Contexto

- **Hardware**: RTX 5060 Laptop GPU (SM 12.0, 8 GB VRAM), CUDA 13.1, GCC 15.2.0
- **Pipeline**: nocturno 2:00–8:00 AM, systemd timer, Ollama + Nemotron
- **Dashboard**: GitHub Pages desde `/docs` en main
- **Webhook**: ntfy.sh/autokernel-alexendros
- **Repo**: público en `Iniciativas-Alexendros/autokernel`

## Requerimientos funcionales

### RF1: Eliminar contenido upstream no esencial

- Eliminar `examples/` (ejemplos HF de RightNowAI)
- Eliminar `kernelbench/` (benchmark suite upstream)
- Eliminar `CHANGELOG.md` (changelog upstream)
- Eliminar `SUMMARY.txt` (resumen temporal)
- Eliminar `LICENSE` (heredado, no necesario para uso local)
- Eliminar `progress.png` (imagen temporal)

### RF2: Reorganizar documentación

- Renombrar `PROPOSAL.md` → `docs/ARCHITECTURE.md`
- Mover `program.md` → `docs/PLAYBOOK.md`
- Reescribir `README.md` en español con contexto propio
- Mantener `docs/index.html` (dashboard generado)

### RF3: Renombrar carpetas para claridad

- Renombrar `cuda-lab/` → `cuda-samples/` (muestras CUDA manuales de referencia)

### RF4: Mantener utilidades

- Mantener `export_hf.py` (exportación a HuggingFace)

### RF5: Adaptar tests

- Revisar `tests/` para que reflejen solo la configuración local
- Eliminar tests que dependan de archivos eliminados

### RF6: Mantener intacto

- `config/pipeline.yaml`
- `scripts/nightly_pipeline.sh`
- `scripts/generate_dashboard.py`
- `systemd/autokernel-nightly.service`
- `systemd/autokernel-nightly.timer`
- `models/` (phi3_mini, llama_7b, bert_base)
- `kernels/`, `kernels/cuda/`
- `autokernel/` (semáforo, nemotron, prompts, llm_assistant, rag_index)
- `orchestrate.py`, `verify.py`, `extract.py`, `prepare.py`, `kernel.py`, `profile.py`, `bench.py`, `reference.py`
- `specs/cuda-optimization-pipeline/`
- `.github/workflows/`
- `.gitignore`, `pyproject.toml`, `uv.lock`, `.python-version`

## Criterios de aceptación

| ID   | Criterio                                                                                         | Verificación                      |
| ---- | ------------------------------------------------------------------------------------------------ | --------------------------------- |
| CA1  | No existen `examples/`, `kernelbench/`, `CHANGELOG.md`, `SUMMARY.txt`, `LICENSE`, `progress.png` | `ls` no los encuentra             |
| CA2  | `PROPOSAL.md` no existe, `docs/ARCHITECTURE.md` sí                                               | `ls docs/ARCHITECTURE.md`         |
| CA3  | `program.md` no existe, `docs/PLAYBOOK.md` sí                                                    | `ls docs/PLAYBOOK.md`             |
| CA4  | `README.md` existe y contiene "RTX 5060" y "pipeline nocturno"                                   | `grep`                            |
| CA5  | `cuda-lab/` no existe, `cuda-samples/` sí                                                        | `ls cuda-samples/`                |
| CA6  | `tests/` existe y pasa sintaxis                                                                  | `zsh -n` o `python -m py_compile` |
| CA7  | `export_hf.py` existe                                                                            | `ls export_hf.py`                 |
| CA8  | `git status` limpio tras commit                                                                  | Sin archivos sin trackear         |
| CA9  | Push a origin/main exitoso                                                                       | `git push` sin errores            |
| CA10 | Dashboard sigue accesible en GitHub Pages                                                        | HTTP 200                          |

## Restricciones

- No modificar el contenido de `config/pipeline.yaml`
- No modificar `orchestrate.py`, `verify.py`, `extract.py`, `prepare.py`
- No modificar `scripts/nightly_pipeline.sh`
- No modificar `systemd/` units
- No modificar `models/` (phi3_mini, llama_7b, bert_base)
- No modificar `kernels/` ni `kernels/cuda/`
- No modificar `autokernel/`
- No romper el pipeline nocturno existente
- README y docs deben estar en español

## Fuera de alcance

- Cambiar visibilidad del repo (se mantiene público)
- Modificar el contenido de los kernels optimizados
- Cambiar la configuración de modelos LLM
- Modificar el webhook o el schedule del timer
