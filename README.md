# AutoKernel — RTX 5060

Pipeline nocturno de optimización de kernels GPU con Ollama + Nemotron.

**Repo**: [Iniciativas-Alexendros/autokernel](https://github.com/Iniciativas-Alexendros/autokernel)

## Hardware

- NVIDIA RTX 5060 Laptop GPU (SM 12.0, 8 GB VRAM)
- CUDA 13.1, GCC 15.2.0, Triton latest

## Modelos configurados

| Modelo             | Params | VRAM    | Archivo               |
| ------------------ | ------ | ------- | --------------------- |
| LLaMA 7B (compact) | 125 M  | 0,30 GB | `models/llama_7b.py`  |
| Phi-3 Mini         | 1,1 B  | 2,30 GB | `models/phi3_mini.py` |
| BERT Base          | 109 M  | 0,28 GB | `models/bert_base.py` |

## Pipeline nocturno

- **Horario**: 2:00–8:00 AM (6 h), systemd timer
- **Modelos LLM**: ornith:9b (planner), qwen2.5-coder:7b (coder), nemotron-3-ultra (reviewer)
- **Webhook**: ntfy.sh/autokernel-alexendros
- **Dashboard**: https://iniciativas-alexendros.github.io/autokernel/

## Comandos rápidos

```bash
# Ver estado de optimización
uv run python orchestrate.py --workspace workspace status

# Verificar modelo end-to-end
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512

# Generar dashboard HTML
uv run python scripts/generate_dashboard.py --workspace workspace --config config/pipeline.yaml --output docs/index.html

# Dry-run del pipeline completo
bash scripts/nightly_pipeline.sh --dry-run
```

## Estructura del proyecto

```
config/pipeline.yaml          # Configuración del pipeline
scripts/nightly_pipeline.sh   # Script principal nocturno
scripts/generate_dashboard.py # Generador de dashboard HTML
models/                       # Modelos PyTorch (llama, phi3, bert)
kernels/                      # Kernels Triton optimizados
kernels/cuda/                 # Kernels CUDA optimizados
autokernel/                   # Integración LLM (semáforo, nemotron, prompts)
specs/                        # Especificación y contrato del pipeline
tests/                        # Suite de tests
systemd/                      # Servicios y timers systemd
docs/                         # Dashboard HTML + documentación técnica
cuda-samples/                 # Muestras CUDA manuales de referencia
```

## Kerneles optimizados (resultados)

| Kernel          | Speedup | Status  |
| --------------- | ------- | ------- |
| matmul          | 1,218×  | ✅ PASS |
| flash_attention | 1,418×  | ✅ PASS |
| elementwise     | 1,74×   | ✅ PASS |
| softmax         | 1,79×   | ✅ PASS |
| rmsnorm         | 3,84×   | ✅ PASS |
| fused_mlp       | 1,10×   | ✅ PASS |

## Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Ollama con modelos: ornith:9b, qwen2.5-coder:7b, phi3:mini
- NVIDIA GPU con soporte CUDA
