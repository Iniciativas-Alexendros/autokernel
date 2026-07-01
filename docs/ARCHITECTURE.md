# Arquitectura — AutoKernel RTX 5060 (Blackwell SM 12.0)

**Fecha:** 2026-07-01
**Hardware:** NVIDIA RTX 5060 Laptop GPU (SM 12.0, 26 SMs, 8 GB VRAM)
**Stack:** Triton latest + CUDA Toolkit 13.1 + GCC 15.2.0

---

## 1. Estado actual

| Kernel          | Rendimiento              | Status     | Nota                              |
| --------------- | ------------------------ | ---------- | --------------------------------- |
| Matmul          | 32 TFLOPS (96.5% cuBLAS) | PASS       | Edge cases FP16 bajo control      |
| Flash Attention | 31.9 TFLOPS (153% peak)  | PASS       | Optimizaciones Blackwell aplicadas |
| Elementwise     | 1.74× speedup            | PASS       | Kernel funcional                 |
| Softmax         | 1.79× speedup            | PASS       | Kernel funcional                 |
| RMSNorm         | 3.84× speedup            | PASS       | Kernel funcional                 |
| FusedMLP        | 1.10× speedup            | PASS       | Kernel funcional                 |

La suite de tests en CI alcanza ≥70% de cobertura y los jobs de lint, tests y seguridad (`ruff`, `pytest`, `bandit`, `gitleaks`) son exitosos en cada push a `main`.

---

## 2. Componentes del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                         Orquestador                              │
│                   orchestrate.py / continuous_pipeline.py         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ profile.py  │  │ extract.py  │  │  bench.py   │  │verify.py│ │
│  │  (perfil)   │  │  (plan)     │  │ (benchmark) │  │ (e2e)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────┬────┘ │
│         └─────────────────┴─────────────────┴─────────────┘      │
│                           │                                      │
│                    ┌────────▼────────┐                            │
│                    │  workspace/     │                            │
│                    │  state, kernels │                            │
│                    └────────┬────────┘                            │
│                             │                                     │
│  ┌──────────────────────────┼──────────────────────────┐         │
│  │                          │                          │         │
│  ▼                          ▼                          ▼         │
│  autokernel/llm_assistant.py  autokernel/rag_index.py     scripts/ │
│  (planner + coder + reviewer) (contexto RAG)            generate_dashboard.py│
│       ┌─────────────┐                                      │      │
│       │  Ollama     │  ┌──────────────┐                   │      │
│       │  Nemotron   │  │   faiss-cpu  │                   │      │
│       └─────────────┘  └──────────────┘                   │      │
│                                                             │      │
└─────────────────────────────────────────────────────────────┼──────┘
                                                                │
                                                                ▼
                                                    docs/index.html (GitHub Pages)
```

### 2.1 Orquestador

- `orchestrate.py`: CLI para estado, plan, siguiente decisión y acciones puntuales.
- `continuous_pipeline.py`: bucle continuo con semáforos de VRAM y modelos configurados.
- `autokernel/semaphore.py`: control de recursos GPU y evita colisiones entre tareas.

### 2.2 Integración LLM

- `autokernel/llm_assistant.py`: enruta a Ollama para planner/coder y a Nemotron para revisiones.
- `autokernel/nemotron_client.py`: lee la API key de `NEMOTRON_API_KEY` o Proton Pass.
- `autokernel/rag_index.py`: índice RAG sobre kernels, specs y docs para contexto de generación.

### 2.3 Seguridad y calidad

- CI con `ruff`, `pytest`, `bandit`, `gitleaks`, `systemd-analyze verify`.
- Cobertura mínima 70%.
- Secretos solo por entorno o `pass-cli`.

---

## 3. Mejoras de rendimiento aplicadas

### 3.1 Matmul — Alineación de memoria

```python
offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
```

### 3.2 Flash Attention — Optimizaciones Blackwell

- `exp2` en lugar de `exp` para precisión nativa.
- `warp_specialize=True` cuando la versión de Triton lo permite.
- Detección de SM ≥ 10 en lugar de igualdad exacta.

### 3.3 Autotuning

Configuración de autotuning para matmul y flash attention:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK,
                       'GROUP_SIZE_M': 8}, num_stages=s, num_warps=w)
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ],
    key=["M", "N", "K"],
)
```

---

## 4. Pipeline de trabajo

1. **Perfilado**: `profile.py` genera `workspace/profile_report.json`.
2. **Extracción**: `extract.py` crea kernels candidatos y `orchestration_state.json`.
3. **Optimización**: `orchestrate.py auto` o `continuous_pipeline.py` itera sobre kernels.
4. **Verificación**: `bench.py` para un kernel; `verify.py` para end-to-end.
5. **Publicación**: `github_publisher.py` abre PR automático con kernel validado.
6. **Dashboard**: `generate_dashboard.py` actualiza `docs/index.html` para GitHub Pages.

---

## 5. Métricas de éxito

| Objetivo        | Métrica                | Target           |
| --------------- | ---------------------- | ---------------- |
| Matmul          | TFLOPS en 2048³ FP16   | ≥33 (99% cuBLAS) |
| Flash Attention | TFLOPS en 2,32,1024,64 | ≥35 (103% peak)  |
| Softmax         | Throughput             | ≥50 GB/s         |
| RMSNorm         | Latencia               | ≤0.5 ms          |
| End-to-end      | Speedup vs PyTorch     | ≥2x              |
| Correctness     | Todos los edge cases   | PASS             |
| CI              | Tests + lint + seguridad | Verde           |
| Cobertura       | `pytest --cov`         | ≥70%             |

---

## 6. Propuestas de magnificación

| Prioridad | Propuesta | Impacto | Esfuerzo | Próximo paso |
| --- | --- | --- | --- | --- |
| 1 | Venderizar Bootstrap/Plotly/Inter + SRI | Seguridad, offline | Medio | Descargar assets y generar hashes sha384 en CI. |
| 2 | Hardening de URLs y HSTS preload | Seguridad | Bajo | Audit redirects DNS y enviar dominio a hstspreload.org. |
| 3 | Test suite con GPU reales (pytest slow) | Fiabilidad | Alto | Runner auto-hospedado con RTX 5060. |
| 4 | Métricas Prometheus + Grafana | Observabilidad | Medio | Exponer `/metrics` en `continuous_pipeline`. |
| 5 | Multi-GPU scheduling y cola prioritaria | Escalabilidad | Alto | Refactor `ResourceSemaphore` por GPU. |
| 6 | Auto-rollback de kernels con regresión | Robustez | Medio | Comparar `verification_*.json` y restaurar baseline. |

---

## 7. Referencias

- Triton Tutorial 02: Fused Softmax — `triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html`
- Triton Tutorial 06: Fused Attention — `triton-lang.org/main/getting-started/tutorials/06-fused-attention.html`
- Triton Tutorial 09: Persistent Matmul — `triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html`
- Triton Gluon: Persistence — `triton-lang.org/main/getting-started/tutorials/gluon/persistence.html`
- Triton Gluon: TCGen05 — `triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html`
