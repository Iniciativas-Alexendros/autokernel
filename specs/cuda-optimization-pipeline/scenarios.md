# Scenarios: Pipeline de Optimización CUDA

## Fase 1: RightNow + Ollama

### Happy Path

1. **S1.1**: Ollama está corriendo → RightNow detecta y usa el modelo
   - Pre: `ollama serve` activo
   - When: Usuario escribe código CUDA en RightNow
   - Then: Autocompletado aparece con sugerencias relevantes

2. **S1.2**: RightNow inicia sin Ollama → fallback graceful
   - Pre: Ollama no está corriendo
   - When: RightNow abre archivo .cu
   - Then: Editor funciona sin AI, muestra warning

### Edge Cases

3. **S1.3**: Ollama se cae durante uso
   - When: Ollama se detiene mientras RightNow está abierto
   - Then: RightNow detecta y muestra "AI offline", no crashea

4. **S1.4**: Modelo no descargado
   - When: `qwen2.5-coder:7b` no existe en Ollama
   - Then: RightNow sugiere descargar o usar modelo alternativo

---

## Fase 2: Optimización de Kernels

### Happy Path

2. **S2.1**: Kernel matmul optimizado supera baseline
   - Pre: Profile report con matmul como #1
   - When: extract.py genera kernel optimizado
   - Then: bench.py muestra speedup ≥1.5x

3. **S2.2**: Kernel flash_attention optimizado
   - Pre: Profile report con flash_fwd_kernel
   - When: Se extrae y optimiza
   - Then: Correctness PASS, speedup ≥1.2x

### Edge Cases

5. **S2.3**: Kernel no soportado por Triton
   - When: extract.py encuentra kernel sin starter template
   - Then: Se omite con warning, se reporta como "needs custom implementation"

6. **S2.4**: GPU sin soporte CUPTI
   - When: profiling falla con CUPTI_ERROR_INVALID_DEVICE
   - Then: Pipeline usa nsys como alternativa automáticamente

---

## Fase 3: Pipeline Automation

### Happy Path

3. **S3.1**: Ejecución completa con un comando
   - Pre: Modelo .py con clase PyTorch
   - When: `uv run pipeline.py --model models/llama_7b.py`
   - Then: Genera profile → extract → optimize → benchmark → report

4. **S3.2**: Re-ejecución incremental
   - Pre: Pipeline ejecutado previamente
   - When: Se re-ejecuta con mismo modelo
   - Then: Reusa profiling si modelo no cambió

### Edge Cases

7. **S3.3**: Modelo muy grande para VRAM
   - When: Modelo necesita más VRAM disponible
   - Then: Pipeline reduce batch size automáticamente o falla con mensaje claro

8. **S3.4**: Timeout en benchmark
   - When: Benchmark toma >5 minutos
   - Then: Se interrumpe, se reporta parcial, se sugiere reducir iteraciones

---

## Fase 4: Validación

### Happy Path

4. **S4.1**: Reporte HTML generado
   - Pre: Benchmark completado
   - When: Se genera reporte
   - Then: HTML con gráficas de speedup por kernel y modelo

### Edge Cases

9. **S4.2**: Sin datos de baseline
   - When: No hay PyTorch baseline para comparar
   - Then: Reporte muestra solo TFLOPS absolutos, marca "no comparison"

---

## Matrix de Cobertura

| Escenario | Fase | Tipo       | Prioridad |
| --------- | ---- | ---------- | --------- |
| S1.1      | 1    | Happy Path | Alta      |
| S1.2      | 1    | Happy Path | Alta      |
| S1.3      | 1    | Edge Case  | Media     |
| S1.4      | 1    | Edge Case  | Baja      |
| S2.1      | 2    | Happy Path | Alta      |
| S2.2      | 2    | Happy Path | Alta      |
| S2.3      | 2    | Edge Case  | Media     |
| S2.4      | 2    | Edge Case  | Alta      |
| S3.1      | 3    | Happy Path | Alta      |
| S3.2      | 3    | Happy Path | Media     |
| S3.3      | 3    | Edge Case  | Media     |
| S3.4      | 3    | Edge Case  | Baja      |
| S4.1      | 4    | Happy Path | Alta      |
| S4.2      | 4    | Edge Case  | Baja      |
