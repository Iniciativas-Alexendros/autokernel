# AutoKernel — RTX 5060

[![Release](https://img.shields.io/github/v/release/Iniciativas-Alexendros/autokernel?style=flat-square&color=blue)](https://github.com/Iniciativas-Alexendros/autokernel/releases)
[![CI](https://github.com/Iniciativas-Alexendros/autokernel/actions/workflows/ci.yml/badge.svg)](https://github.com/Iniciativas-Alexendros/autokernel/actions/workflows/ci.yml)
[![Pipeline](https://img.shields.io/badge/pipeline-nocturno-2%3A00--8%3A00-ff6b35?style=flat-square)](https://iniciativas-alexendros.github.io/autokernel/)
[![Dashboard](https://img.shields.io/badge/dashboard-GitHub%20Pages-2ea44f?style=flat-square)](https://iniciativas-alexendros.github.io/autokernel/)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Triton](https://img.shields.io/badge/triton-latest-ff6b35?style=flat-square)](https://triton-lang.org/)
[![CUDA](https://img.shields.io/badge/cuda-13.1-76b900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

Pipeline de optimización autónoma de kernels GPU para el stack propio de Alexendros. Perfilado, extracción de cuellos de botella, generación de kernels Triton/CUDA y verificación end-to-end orquestados por LLM local y Nemotron como reviewer.

## Hardware

| Componente | Especificación                                  |
| ---------- | ----------------------------------------------- |
| GPU        | NVIDIA RTX 5060 Laptop GPU (SM 12.0, 8 GB VRAM) |
| CUDA       | 13.1                                            |
| GCC        | 15.2.0                                          |
| Triton     | latest                                          |

## Modelos configurados

| Modelo             | Params | VRAM    | Archivo               |
| ------------------ | ------ | ------- | --------------------- |
| LLaMA 7B (compact) | 125 M  | 0,30 GB | `models/llama_7b.py`  |
| Phi-3 Mini         | 1,1 B  | 2,30 GB | `models/phi3_mini.py` |
| BERT Base          | 109 M  | 0,28 GB | `models/bert_base.py` |

## Pipeline

| Parámetro    | Valor                                                                 |
| ------------ | --------------------------------------------------------------------- |
| Horario      | 2:00–8:00 AM (6 h)                                                    |
| Timer        | systemd (`autokernel-nightly.timer`)                                  |
| Planner LLM  | `ornith:9b` (Ollama) / `opencode/mimo-v2.5-free` (fallback)             |
| Coder LLM    | `qwen2.5-coder:7b` (Ollama) / `opencode/deepseek-v4-flash-free` (fallback) |
| Reviewer LLM | `nvidia/nemotron-3-ultra` (NVIDIA API) / `opencode/nemotron-3-ultra-free` (fallback) |
| Webhook      | ntfy.sh/autokernel-alexendros                                         |
| Dashboard    | [GitHub Pages](https://iniciativas-alexendros.github.io/autokernel/)    |
| CI           | lint + tests (≥70% cobertura) + bandit/gitleaks en cada push a `main` |

## Kernels optimizados

| Kernel          | Speedup | Status  |
| --------------- | ------- | ------- |
| matmul          | 1,218×  | PASS |
| flash_attention | 1,418×  | PASS |
| elementwise     | 1,74×   | PASS |
| softmax         | 1,79×   | PASS |
| rmsnorm         | 3,84×   | PASS |
| fused_mlp       | 1,10×   | PASS |

## Comandos rápidos

```bash
# Estado del pipeline
uv run orchestrate.py --workspace workspace status

# Siguiente decisión del orquestador
uv run orchestrate.py --workspace workspace next

# Plan de optimización con Amdahl
uv run orchestrate.py --workspace workspace plan

# Optimizar un kernel con LLM
uv run orchestrate.py --workspace workspace auto --kernel matmul

# Migrar Triton a CUDA C++
uv run orchestrate.py --workspace workspace migrate-cuda --kernel flash_attention

# Verificar modelo end-to-end
uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512

# Generar dashboard HTML
uv run python scripts/generate_dashboard.py --workspace workspace --config config/pipeline.yaml --output docs/index.html

# Ejecutar pipeline manual
bash scripts/nightly_pipeline.sh

# Tests con cobertura
uv run pytest -m "not slow" -q --cov=autokernel --cov=scripts --cov-fail-under=70

# Lint + seguridad
uv run ruff check . && uv run ruff format --check .
uv run bandit -r autokernel scripts -ll
gitleaks detect --source .
```

## Estructura del proyecto

```
├── config/pipeline.yaml          # Configuración del pipeline
├── scripts/
│   ├── nightly_pipeline.sh       # Script principal nocturno
│   ├── generate_dashboard.py     # Generador de dashboard HTML
│   ├── continuous_pipeline.py    # Pipeline continuo
│   └── self_audit.py             # Auditoría de salud del repo
├── models/                       # Modelos PyTorch
├── kernels/                      # Kernels Triton optimizados
│   └── cuda/                     # Kernels CUDA optimizados
├── autokernel/                   # Integración LLM
│   ├── llm_assistant.py          # Asistente LLM (Ollama + Nemotron)
│   ├── nemotron_client.py        # Cliente API Nemotron
│   ├── semaphore.py              # Semáforo de recursos VRAM
│   ├── rag_index.py              # Índice RAG para contexto
│   └── prompts/                  # Prompts para generación
├── orchestrate.py                # Orquestador principal
├── verify.py                     # Verificación end-to-end
├── extract.py                    # Extracción de kernels
├── profile.py                    # Perfilado de modelos
├── bench.py                      # Benchmark de kernels
├── tests/                        # Suite de tests (≥70% cobertura)
├── systemd/                      # Servicios systemd
├── docs/                         # Dashboard + documentación
└── specs/                        # Especificaciones de features
```

## Calidad y seguridad

- **Lint**: `ruff` en cada push.
- **Tests**: `pytest` con cobertura mínima del 70%.
- **Seguridad**: `bandit` y `gitleaks` en cada push.
- **Servicios**: `systemd-analyze verify` para `systemd/*.service` y `*.timer`.
- **Secretos**: solo vía variables de entorno o `pass-cli`; nunca hardcodeados.

## Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Ollama con modelos: `ornith:9b`, `qwen2.5-coder:7b`, `bge-m3`
- NVIDIA GPU con soporte CUDA
- `pass-cli` (para secretos Proton Pass)
- `zsh` (para `scripts/nightly_pipeline.sh`)

## Licencia

[MIT](LICENSE) — Basado en [AutoKernel](https://github.com/RightNowAI/auto-kernel) de RightNowAI.
