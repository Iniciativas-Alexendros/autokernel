# Test Plan: Pipeline de Optimizacion CUDA

## Estrategia

Tests escritos ANTES de la implementacion. Cada fase tiene tests independientes que validan el contrato.

## Fase 1: RightNow + Ollama

### Unit Tests (tests/test_ollama_integration.py)

```python
class TestOllamaHealth:
    test_ollama_responds()           # Ollama responde en puerto 11434
    test_ornith_model_available()    # ornith:9b esta disponible
    test_qwen_model_available()      # qwen2.5-coder:7b sigue disponible

class TestOllamaGenerate:
    test_generate_cuda_context()     # Ollama genera completado para CUDA
    test_generate_response_time()    # Respuesta en <15 segundos

class TestRightNowConfig:
    test_rightnow_config_exists()    # .rightnowrules existe
    test_rightnow_config_has_ollama()# Config apunta a Ollama
    test_rightnow_config_uses_ornith()# Config usa ornith
```

## Fase 2: Optimizacion de Kernels

### Unit Tests (tests/test_kernel_extraction.py)

```python
class TestProfileReport:
    test_report_exists()             # profile_report.json existe
    test_report_has_kernels()        # Reporte tiene lista de kernels
    test_kernel_has_required_fields()# Cada kernel tiene campos requeridos
    test_kernels_sorted_by_time()    # Kernels ordenados por tiempo

class TestOptimizationPlan:
    test_plan_exists()               # optimization_plan.json existe
    test_plan_has_targets()          # Plan tiene targets
    test_plan_kernel_types_valid()   # Tipos de kernel validos

class TestKernelFiles:
    test_kernel_matmul_exists()      # kernel_matmul_1.py existe
    test_kernel_file_has_kernel_type()# KERNEL_TYPE definido
```

### Correctness Tests (tests/test_kernel_correctness.py)

```python
class TestKernelCorrectness:
    test_kernel_valid_python()       # Kernel es Python valido
    test_kernel_has_triton_import()  # Kernel importa triton
    test_kernel_has_kernel_type()    # KERNEL_TYPE definido
    # Parametrizado para los 5 kernels extraidos
```

### Performance Tests (tests/test_benchmark.py)

```python
class TestBenchResult:
    test_bench_result_exists()       # bench_result.json existe
    test_bench_result_has_speedup()  # Tiene campo speedup
    test_bench_result_has_correctness()# Tiene campo correctness

class TestKernelPerformance:
    test_kernel_importable()         # Kernel es Python valido
    test_kernel_has_kernel_type()    # KERNEL_TYPE definido
    test_kernel_has_kernel_fn()      # kernel_fn definido
```

## Fase 3: Automatizacion

### Unit Tests (tests/test_orchestrate.py)

```python
class TestOrchestrateCLI:
    test_orchestrate_help()          # --help funciona
    test_orchestrate_status()        # status no crashea
    test_orchestrate_next()          # next retorna respuesta valida
    test_orchestrate_plan()          # plan muestra optimizacion

class TestOrchestrateState:
    test_state_file_has_required_fields()# orchestration_state.json valido
```

## Orden de Ejecucion

```
1. tests/test_ollama_integration.py    (Fase 1)
2. tests/test_kernel_extraction.py     (Fase 2)
3. tests/test_kernel_correctness.py    (Fase 2)
4. tests/test_benchmark.py             (Fase 2)
5. tests/test_orchestrate.py           (Fase 3)
```

## Cobertura Objetivo

| Modulo             | Tests | Cobertura |
| ------------------ | ----- | --------- |
| ollama_integration | 9     | 90%       |
| kernel_extraction  | 8     | 85%       |
| kernel_correctness | 3     | 80%       |
| benchmark          | 6     | 85%       |
| orchestrate        | 5     | 80%       |

## Gates de Calidad

- [x] Todos los tests escritos antes de implementacion
- [x] Red state observado (tests fallan sin implementacion)
- [x] Green state alcanzado (todos pasan tras implementacion)
- [ ] Cobertura >=80%
- [ ] 0 lint errors
- [ ] Complejidad ciclomatica <=10 por funcion
