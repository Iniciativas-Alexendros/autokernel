# Contract: Estructura del Repo después de la limpieza

## Estructura de archivos esperada

```
autokernel/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── pages.yml
├── .gitignore
├── .python-version
├── README.md                          # ESPAÑOL, contexto propio
├── config/
│   └── pipeline.yaml
├── cuda-samples/                      # RENOMBRADO de cuda-lab/
│   ├── fix_math/
│   │   └── math.h
│   ├── hello.cu
│   ├── hello_final.cu
│   └── main.cpp
├── docs/
│   ├── ARCHITECTURE.md                # RENOMBRADO de PROPOSAL.md
│   ├── PLAYBOOK.md                    # MOVIDO de program.md
│   └── index.html
├── export_hf.py                       # MANTENIDO
├── models/
│   ├── __init__.py
│   ├── bert_base.py
│   ├── custom.py
│   ├── gpt2.py
│   ├── llama_7b.py
│   └── phi3_mini.py
├── kernels/
│   ├── __init__.py
│   ├── cross_entropy.py
│   ├── elementwise.py
│   ├── flash_attention.py
│   ├── fused_mlp.py
│   ├── layernorm.py
│   ├── matmul.py
│   ├── reduce.py
│   ├── rmsnorm.py
│   ├── rotary_embedding.py
│   ├── softmax.py
│   └── cuda/
│       ├── __init__.py
│       ├── _compile.py
│       ├── cross_entropy.py
│       ├── flash_attention.py
│       ├── fused_mlp.py
│       ├── layernorm.py
│       ├── matmul.py
│       ├── reduce.py
│       ├── rmsnorm.py
│       ├── rotary_embedding.py
│       └── softmax.py
├── autokernel/
│   ├── __init__.py
│   ├── llm_assistant.py
│   ├── nemotron_client.py
│   ├── rag_index.py
│   ├── semaphore.py
│   └── prompts/
│       ├── __init__.py
│       ├── cuda_migration.py
│       ├── kernel_gen.py
│       ├── ncu.py
│       ├── review.py
│       ├── spec.py
│       └── test_gen.py
├── orchestrate.py
├── extract.py
├── prepare.py
├── profile.py
├── bench.py
├── kernel.py
├── reference.py
├── verify.py
├── analysis.py
├── pyproject.toml
├── uv.lock
├── scripts/
│   ├── nightly_pipeline.sh
│   └── generate_dashboard.py
├── specs/
│   ├── cuda-optimization-pipeline/
│   │   ├── contract.md
│   │   ├── scenarios.md
│   │   ├── spec.md
│   │   └── test-plan.md
│   └── repo-cleanup/                 # NUEVO (esta spec)
│       ├── contract.md
│       ├── scenarios.md
│       ├── spec.md
│       └── test-plan.md
├── systemd/
│   ├── autokernel-nightly.service
│   └── autokernel-nightly.timer
└── tests/
    ├── test_benchmark.py
    ├── test_kernel_correctness.py
    ├── test_kernel_extraction.py
    ├── test_ollama_integration.py
    └── test_orchestrate.py
```

## Archivos ELIMINADOS

| Archivo        | Razón                     |
| -------------- | ------------------------- |
| `examples/`    | Ejemplos HF de RightNowAI |
| `kernelbench/` | Benchmark suite upstream  |
| `CHANGELOG.md` | Changelog upstream        |
| `SUMMARY.txt`  | Resumen temporal          |
| `LICENSE`      | Heredado, no necesario    |
| `progress.png` | Imagen temporal           |

## Archivos RENOMBRADOS

| Original      | Nuevo                  | Razón             |
| ------------- | ---------------------- | ----------------- |
| `PROPOSAL.md` | `docs/ARCHITECTURE.md` | Integrar en docs/ |
| `program.md`  | `docs/PLAYBOOK.md`     | Integrar en docs/ |
| `cuda-lab/`   | `cuda-samples/`        | Claridad          |

## README.md en español — Contrato de contenido

El README debe contener como mínimo:

1. **Título**: `AutoKernel — RTX 5060`
2. **Descripción**: Pipeline nocturno de optimización de kernels GPU con Ollama + Nemotron
3. **Hardware**: RTX 5060 Laptop GPU (SM 12.0, 8 GB VRAM), CUDA 13.1, GCC 15.2.0
4. **Modelos configurados**: Tabla con LLaMA 7B, Phi-3 Mini, BERT Base
5. **Pipeline nocturno**: Horario, modelos LLM, webhook, dashboard URL
6. **Comandos rápidos**: status, verify, dashboard, dry-run
7. **Estructura del proyecto**: Árbol de directorios
8. **Kerneles optimizados**: Tabla con speedup y status
9. **Requisitos**: Python 3.10+, uv, Ollama, NVIDIA GPU

## Tests adaptados — Contrato

Los tests en `tests/` deben:

- No importar archivos eliminados (`examples/`, `kernelbench/`)
- No referenciar `CHANGELOG.md`, `SUMMARY.txt`, `LICENSE`
- Funcionar con la estructura de archivos resultante
- Mantener cobertura mínima del 80% en módulos críticos
