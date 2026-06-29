# Spec: Pipeline de Optimización CUDA en Cascada

## Objetivo

Crear un pipeline end-to-end de optimización de kernels CUDA que funcione en cascada sin bloqueantes, donde cada fase alimenta la siguiente de forma independiente.

## Arquitectura en Cascada

```
Fase 1: Fundación AI        → Fase 2: Optimización      → Fase 3: Automatización → Fase 4: Validación
(RightNow + Ollama)          (AutoKernel kernels)        (Pipeline completo)      (Benchmark + Reporte)
```

**Principio**: Cada fase puede ejecutarse independientemente si la anterior esta completa. No hay bloqueantes cruzados.

## Fases

### Fase 1: Integración RightNow + Ollama

**Objetivo**: RightNow Editor usa Ollama local para code completion en CUDA.

- RightNow detecta Ollama en `localhost:11434`
- Modelo `qwen2.5-coder:7b` configurado como backend
- Code completion funciona sin conexión a internet
- **Criterio de aceptación**: Autocompletado CUDA funcional en RightNow

### Fase 2: Optimización de Kernels

**Objetivo**: Optimizar los 5 kernels bottleneck extraídos del profiling.

- Kernel #1: matmul (66.4% del tiempo GPU)
- Kernel #2: elementwise (27.2% combinado)
- Kernel #3: flash_attention (5.3%)
- Cada kernel se optimiza independientemente
- **Criterio de aceptación**: Speedup ≥1.5x vs PyTorch baseline

### Fase 3: Pipeline Automatizado

**Objetivo**: Script que ejecuta el ciclo completo automáticamente.

- Input: archivo `.py` con modelo PyTorch
- Output: kernels optimizados + reporte JSON
- Pasos: profile → extract → optimize → benchmark → report
- **Criterio de aceptación**: Ejecución con un solo comando

### Fase 4: Validación y Benchmark

**Objetivo**: Medir y reportar mejora real del pipeline.

- Benchmark completo con múltiples modelos
- Comparación antes/después de optimización
- Reporte HTML con gráficas
- **Criterio de aceptación**: Reporte generado con datos reales

## Dependencias entre Fases

| Fase | Depende de                | Bloquea |
| ---- | ------------------------- | ------- |
| 1    | Setup previo (completado) | Ninguna |
| 2    | Ninguna (independiente)   | 3, 4    |
| 3    | 2                         | 4       |
| 4    | 2, 3                      | Ninguna |

**Nota**: La Fase 1 (RightNow + Ollama) es opcional y no bloquea ninguna otra fase. La Fase 2 (optimizacion de kernels) es independiente y puede ejecutarse directamente.

## Restricciones

- GPU: RTX 5060 Laptop (driver 595.71.05)
- CUDA: Toolkit 13.1 (parcheado para GCC 15)
- CUPTI: No disponible (usar nsys como alternativa)
- RAM: 29GB (suficiente para LLaMA-7B en fp16)
