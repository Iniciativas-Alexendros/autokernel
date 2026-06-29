# Contract: Pipeline de Optimización CUDA

## Interfaces por Fase

### Fase 1: RightNow + Ollama

**Ollama API Contract**

```
POST http://localhost:11434/api/generate
{
  "model": "qwen2.5-coder:7b",
  "prompt": "<cuda_code_context>",
  "stream": false
}

Response:
{
  "response": "<suggested_code>",
  "done": true
}
```

**RightNow Config** (`.rightnowrules`)

```json
{
  "ai_provider": "ollama",
  "ollama_url": "http://localhost:11434",
  "model": "qwen2.5-coder:7b",
  "languages": ["cuda", "cpp"]
}
```

### Fase 2: Kernel Optimization

**Input Contract** (`profile_report.json`)

```json
{
  "model": "string",
  "gpu": "string",
  "total_time_ms": "number",
  "kernels": [
    {
      "kernel_name": "string",
      "op_type": "matmul|flash_attention|elementwise|reduction",
      "total_time_us": "number",
      "fraction": "number",
      "autokernel_supported": "boolean",
      "rank": "number"
    }
  ]
}
```

**Output Contract** (`kernel_optimized.py`)

```python
# Kernel optimizado con Triton
KERNEL_TYPE = "matmul"
OPTIMIZED = True
SPEEDUP_TARGET = 1.5  # vs PyTorch

def optimized_kernel(...):
    # Implementación Triton optimizada
    pass
```

**Benchmark Contract** (`bench_result.json`)

```json
{
  "kernel_type": "string",
  "correctness": "pass|fail",
  "speedup_vs_pytorch": "number",
  "throughput_tflops": "number",
  "pct_peak_compute": "number",
  "sizes_tested": ["tiny", "small", "medium", "large"]
}
```

### Fase 3: Pipeline Automation

**CLI Contract**

```bash
# Ejecutar pipeline completo
uv run pipeline.py --model models/llama_7b.py --class-name LlamaModel

# Solo profiling
uv run pipeline.py --stage profile --model ...

# Solo optimización (requiere profile previo)
uv run pipeline.py --stage optimize --report workspace/profile_report.json

# Solo benchmark
uv run pipeline.py --stage benchmark --kernel workspace/kernel_optimized.py
```

**Pipeline State** (`pipeline_state.json`)

```json
{
  "run_id": "uuid",
  "stages": {
    "profile": { "status": "done", "output": "workspace/profile_report.json" },
    "extract": {
      "status": "done",
      "output": "workspace/optimization_plan.json"
    },
    "optimize": { "status": "pending", "output": null },
    "benchmark": { "status": "pending", "output": null }
  },
  "started_at": "ISO8601",
  "completed_at": null
}
```

### Fase 4: Validation Report

**Report Contract** (`report.json`)

```json
{
  "summary": {
    "models_tested": "number",
    "kernels_optimized": "number",
    "avg_speedup": "number",
    "total_gpu_time_saved_ms": "number"
  },
  "results": [
    {
      "model": "string",
      "kernel": "string",
      "baseline_tflops": "number",
      "optimized_tflops": "number",
      "speedup": "number",
      "correctness": "boolean"
    }
  ],
  "gpu_info": {
    "name": "string",
    "driver": "string",
    "cuda_version": "string"
  }
}
```

## Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ profile.py  │ →  │ extract.py   │ →  │ optimize.py  │ →  │ bench.py     │
│             │    │              │    │ (nuestro)    │    │              │
│ nsys trace  │    │ top kernels  │    │ Triton kernels│   │ vs PyTorch   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       ↓                  ↓                   ↓                    ↓
  profile_report.json  plan.json          kernel.py          bench_result.json
```

## Error Handling

| Error                   | Fase | Recovery                        |
| ----------------------- | ---- | ------------------------------- |
| Ollama not running      | 1    | `ollama serve &`                |
| CUPTI unavailable       | 2    | Usar nsys alternativo           |
| Kernel compilation fail | 3    | Revertir a PyTorch fallback     |
| Benchmark timeout       | 4    | Reducir iteraciones, reintentar |
