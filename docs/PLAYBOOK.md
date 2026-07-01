# PLAYBOOK — AutoKernel

Guía de trabajo para el agente autónomo de optimización de kernels GPU en el stack propio de Alexendros.

---

## Contexto

- **Repo**: `Iniciativas-Alexendros/autokernel`
- **Hardware**: NVIDIA RTX 5060 Laptop (SM 12.0, 8 GB VRAM), CUDA 13.1, GCC 15.2.0
- **Stack**: Triton latest, PyTorch, CUDA C++ backend opcional
- **Modelos**: LLaMA 7B compact, Phi-3 Mini, BERT Base
- **CI**: lint (`ruff`), tests (`pytest` ≥70% cobertura), seguridad (`bandit`, `gitleaks`)
- **Dashboard**: GitHub Pages desde `docs/index.html`

---

## Fases del pipeline

| Fase | Propósito | Comando clave |
| ---- | --------- | ------------- |
| A. Perfilado | Detectar cuellos de botella del modelo | `uv run profile.py --model <path> --class-name <Name> --input-shape <shape>` |
| B. Extracción | Generar kernels candidatos y plan | `uv run extract.py --top 5` |
| C. Optimización | Mejorar cada kernel automáticamente | `uv run orchestrate.py --workspace workspace auto --kernel <tipo>` |
| D. Verificación | Validar correctness y speedup | `uv run bench.py` y `uv run verify.py` |
| E. Publicación | Abrir PR con el kernel validado | `autokernel/github_publisher.py` |
| F. Dashboard | Actualizar GitHub Pages | `uv run python scripts/generate_dashboard.py --workspace workspace --config config/pipeline.yaml --output docs/index.html` |

---

## Comandos operativos

```bash
# Estado del pipeline
uv run orchestrate.py --workspace workspace status

# Siguiente decisión
uv run orchestrate.py --workspace workspace next

# Plan con Amdahl
uv run orchestrate.py --workspace workspace plan

# Optimizar un kernel
uv run orchestrate.py --workspace workspace auto --kernel matmul

# Migrar Triton a CUDA C++
uv run orchestrate.py --workspace workspace migrate-cuda --kernel flash_attention

# Pipeline continuo
uv run python scripts/continuous_pipeline.py --config config/pipeline.yaml

# Pipeline nocturno manual
bash scripts/nightly_pipeline.sh

# Verificación end-to-end
uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512

# Tests y calidad
uv run pytest -m "not slow" -q --cov=autokernel --cov=scripts --cov-fail-under=70
uv run ruff check . && uv run ruff format --check .
uv run bandit -r autokernel scripts -ll
gitleaks detect --source .
```

---

## Fase A: Perfilado y plan

1. Recibe el modelo (local, HuggingFace o `transformers`).
2. Ejecuta `profile.py` para obtener `workspace/profile_report.json`.
3. Revisa el reporte: top ops por tiempo, compute-bound vs memory-bound.
4. Ejecuta `extract.py --top N` para generar kernels y `orchestration_state.json`.
5. Presenta el plan con estimaciones de Amdahl.
6. El operador confirma el plan.
7. Crea rama `autokernel/<modelo>-<timestamp>`.

---

## Fase B: Optimización de un kernel

1. Consulta el orquestador: `uv run orchestrate.py next`.
2. Copia el kernel objetivo a `kernel.py`:
   ```bash
   cp workspace/kernel_<tipo>_<rank>.py kernel.py
   ```
3. Lanza baseline: `uv run bench.py > run.log 2>&1`.
4. Registra baseline: `uv run orchestrate.py record kernel_<tipo>_<rank>.py <tflops> keep "baseline"`.
5. Bucle de experimentos:
   - Hipótesis de una sola optimización.
   - Edita `kernel.py`.
   - Commitea: `git add kernel.py && git commit -m "exp N: <hipótesis>"`.
   - Ejecuta: `uv run bench.py > run.log 2>&1`.
   - Decide:
     - `correctness = FAIL` → `git reset --hard HEAD~1`.
     - `correctness = PASS` y mejora ≥1% → keep.
     - `correctness = PASS` y igual/peor → revert.
   - Registra: `uv run orchestrate.py record kernel_<tipo>_<rank>.py <tflops> keep|revert "<desc>"`.
   - Añade fila a `results.tsv` (tab-separated, no commitear).
   - Consulta `uv run orchestrate.py next` para continuar o cambiar de kernel.
6. Guarda el kernel optimizado:
   ```bash
   cp kernel.py workspace/kernel_<tipo>_<rank>_optimized.py
   ```

---

## Fase C: Verificación end-to-end

```bash
uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512
```

| Resultado | Acción |
| --------- | ------ |
| PASS, speedup > 1.0 | Generar reporte final y publicar PR. |
| PASS, speedup ≤ 1.0 | Revisar estrategia de reemplazo; posible overhead. |
| FAIL | `uv run verify.py --diagnose` para aislar el kernel culpable. |

---

## Fase D: Publicación

- `github_publisher.py` crea rama, sube el kernel y abre PR.
- Requisitos para publicar: `correctness = True` y `speedup > 1.0`.
- Auto-merge si CI pasa y el reviewer (Nemotron) aprueba.

---

## Fase E: Dashboard y auditoría

```bash
# Generar dashboard
uv run python scripts/generate_dashboard.py --workspace workspace --config config/pipeline.yaml --output docs/index.html

# Auditoría de salud
uv run python scripts/self_audit.py --output docs/EVOLUTION.md
```

El dashboard se despliega automáticamente en GitHub Pages cuando cambia `docs/**`.

---

## Playbook de optimización

### Tier 1: Tamaño de bloques

- Barrer `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` en potencias de 2.
- Probar tiles rectangulares (128×64, 128×256).
- Ajustar `num_warps` y `num_stages` como knobs secundarios.

### Tier 2: Acceso a memoria

- Coalescing: cargas consecutivas en el warp.
- Prefetching con `tl.prefetch` o `num_stages`.
- Swizzling de L2 para reutilización entre bloques.
- Padding de 1 elemento para evitar bank conflicts.

### Tier 3: Optimización de compute

- `tl.dot(a, b, allow_tf32=True)` con acumulador fp32.
- Fusionar elementwise en epílogo.
- Minimizar operaciones en el bucle interno.

### Tier 4: Técnicas avanzadas

- Split-K para matrices pequeñas.
- Persistent kernels: un bloque por SM, bucle sobre tiles.
- `@triton.autotune` con espacio de configuraciones.
- Warp specialization.

### Tier 5: Optimizaciones por arquitectura

- **Blackwell SM 12.0**: TMA, TCGen05, FP8, `exp2`, `warp_specialize`.
- **Hopper SM 90**: WGMMA, TMA, cluster shared memory.
- **Ampere SM 80**: `cp.async`, TF32, sparsidad 2:4.

### Tier 6: Trucos por kernel

- **Matmul**: swizzle, epílogo fusionado, split-K.
- **Softmax**: online softmax de dos pasos, multi-row.
- **LayerNorm/RMSNorm**: Welford, fusión de pesos/bias, multi-row.
- **Flash Attention**: online softmax, causal masking, block-sparse.
- **RoPE**: fusionar con Q/K, precomputar frecuencias.

---

## Anti-patrones

- Bloques muy grandes (512+): register spill.
- `num_stages` > 5: overflow de shared memory.
- `tl.debug_barrier` innecesario.
- Unrolling manual cuando Triton ya lo hace.
- `atomic_add` prematuro.
- Cargas desalineadas.
- Control flow complejo en bucle interno.

---

## CUDA C++ backend

Cuando se usa `--backend cuda`, el agente edita `CUDA_SRC` en `kernel.py`. El compilador es `torch.utils.cpp_extension.load_inline()`. El contrato sigue siendo `KERNEL_TYPE`, `BACKEND = "cuda"` y `kernel_fn()` con la misma firma.

### Tier 1 CUDA: configuración de bloques

- Barrer `blockDim.x` en 128, 256, 512.
- Usar `__launch_bounds__` para controlar registros.

---

## Reglas de oro

1. Nunca modificar `bench.py`, `reference.py`, `prepare.py`, `verify.py` salvo bug crítico con spec y test.
2. Un solo cambio por experimento.
3. Siempre commitear antes de `bench.py`.
4. Nunca añadir dependencias sin justificar en `pyproject.toml`.
5. No comments añadidos salvo petición explícita.
6. Secretos solo vía env o `pass-cli`.
7. Cada feature nueva debe tener `specs/<feature>/` con `spec.md`, `contract.md`, `scenarios.md`, `test-plan.md`.
