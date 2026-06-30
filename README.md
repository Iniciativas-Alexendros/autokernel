# AutoKernel — RTX 5060

[![Release](https://img.shields.io/github/v/release/Iniciativas-Alexendros/autokernel?style=flat-square&color=blue)](https://github.com/Iniciativas-Alexendros/autokernel/releases)
[![Pipeline](https://img.shields.io/badge/pipeline-nocturno-2%3A00--8%3A00-ff6b35?style=flat-square)](https://iniciativas-alexendros.github.io/autokernel/)
[![Dashboard](https://img.shields.io/badge/dashboard-GitHub%20Pages-2ea44f?style=flat-square)](https://iniciativas-alexendros.github.io/autokernel/)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Triton](https://img.shields.io/badge/triton-latest-ff6b35?style=flat-square)](https://triton-lang.org/)
[![CUDA](https://img.shields.io/badge/cuda-13.1-76b900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

Pipeline nocturno de optimización de kernels GPU con Ollama + Nemotron en RTX 5060.

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

## Pipeline nocturno

| Parámetro    | Valor                                                                |
| ------------ | -------------------------------------------------------------------- |
| Horario      | 2:00–8:00 AM (6 h)                                                   |
| Timer        | systemd (`autokernel-nightly.timer`)                                 |
| Planner LLM  | ornith:9b                                                            |
| Coder LLM    | qwen2.5-coder:7b                                                     |
| Reviewer LLM | nemotron-3-ultra (API)                                               |
| Webhook      | ntfy.sh/autokernel-alexendros                                        |
| Dashboard    | [GitHub Pages](https://iniciativas-alexendros.github.io/autokernel/) |

## Kerneles optimizados

| Kernel          | Speedup | Status  |
| --------------- | ------- | ------- |
| matmul          | 1,218×  | ✅ PASS |
| flash_attention | 1,418×  | ✅ PASS |
| elementwise     | 1,74×   | ✅ PASS |
| softmax         | 1,79×   | ✅ PASS |
| rmsnorm         | 3,84×   | ✅ PASS |
| fused_mlp       | 1,10×   | ✅ PASS |

## Comandos rápidos

```bash
# Ver estado de optimización
uv run python orchestrate.py --workspace workspace status

# Verificar modelo end-to-end
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512

# Generar dashboard HTML
uv run python scripts/generate_dashboard.py --workspace workspace --config config/pipeline.yaml --output docs/index.html

# Ejecutar pipeline manual
bash scripts/nightly_pipeline.sh
```

## Estructura del proyecto

```
├── config/pipeline.yaml          # Configuración del pipeline
├── scripts/
│   ├── nightly_pipeline.sh       # Script principal nocturno
│   └── generate_dashboard.py     # Generador de dashboard HTML
├── models/                       # Modelos PyTorch
│   ├── llama_7b.py
│   ├── phi3_mini.py
│   └── bert_base.py
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
├── specs/                        # Especificaciones
├── tests/                        # Suite de tests
├── systemd/                      # Servicios systemd
├── docs/                         # Dashboard + documentación
└── cuda-samples/                 # Muestras CUDA de referencia
```

## Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Ollama con modelos: ornith:9b, qwen2.5-coder:7b, phi3:mini
- NVIDIA GPU con soporte CUDA
- pass-cli (para secretos Proton Pass)

## Licencia

[MIT](LICENSE) — Basado en [AutoKernel](https://github.com/RightNowAI/auto-kernel) de RightNowAI.
