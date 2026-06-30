# Propuesta de Optimización — AutoKernel RTX 5060 (Blackwell SM 12.0)

**Fecha:** 2026-06-29
**Hardware:** NVIDIA RTX 5060 Laptop GPU (SM 12.0, 26 SMs, 8GB VRAM)
**Stack:** Triton latest + CUDA Toolkit 13.1 + GCC 15.2.0

---

## 1. Estado Actual

| Kernel          | Rendimiento              | Status          | Problema                              |
| --------------- | ------------------------ | --------------- | ------------------------------------- |
| Matmul          | 32 TFLOPS (96.5% cuBLAS) | Funcional       | Edge cases FP16 (limitación hardware) |
| Flash Attention | 31.9 TFLOPS (153% peak)  | Funcional       | Sin optimizar进一步                   |
| Elementwise     | N/A                      | Stub            | No soportado por bench.py             |
| Softmax         | N/A                      | No implementado | —                                     |
| LayerNorm       | N/A                      | No implementado | —                                     |
| RMSNorm         | N/A                      | No implementado | —                                     |
| FusedMLP        | N/A                      | No implementado | —                                     |

---

## 2. Correcciones Críticas (Inmediatas)

### 2.1 Matmul — Tolerancias de Edge Cases

**Problema:** Error 0.0625-0.125 en K no alineado a 32 (Blackwell tensor core rounding).
**Solución documentada:** Triton tutorial 09-persistent-matmul usa `tl.max_contiguous` y `tl.multiple_of` para alinear accesos a memoria. Aplicar:

```python
offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
```

### 2.2 Flash Attention — Detección de Blackwell

**Problema:** Nuestro `is_blackwell()` podría no detectar SM 12.0 (documentación dice SM 10).
**Solución:** Usar `torch.cuda.get_device_capability()[0] >= 10` en lugar de `== 10`.

---

## 3. Mejoras de Rendimiento (Corto Plazo)

### 3.1 Matmul — Block Sizes Óptimos

**Fuente:** Triton tutorial 09-persistent-matmul y gluon/persistence

| Parámetro    | Actual | Propuesto | Justificación                      |
| ------------ | ------ | --------- | ---------------------------------- |
| BLOCK_SIZE_M | 128    | 128       | Óptimo para Blackwell              |
| BLOCK_SIZE_N | 128    | **256**   | Máximo instruction throughput      |
| BLOCK_SIZE_K | 32     | **64**    | Mejor utilización de shared memory |
| num_warps    | 4      | **4 u 8** | Autotuning                         |
| num_stages   | 3      | **2-4**   | Pipelining según shared memory     |

**Configuración Autotuning sugerida:**

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

### 3.2 Flash Attention — Optimizaciones Blackwell

**Fuente:** Triton tutorial 06-fused-attention

| Mejora                   | Descripción                                  | Impacto Esperado |
| ------------------------ | -------------------------------------------- | ---------------- |
| `exp2` en lugar de `exp` | Más preciso en Blackwell, instrucción nativa | +5-10%           |
| `warp_specialize=True`   | Usa tensor memory accelerator                | +15-20%          |
| TensorDescriptor (TMA)   | Acceso a memoria acelerado por hardware      | +20-30%          |
| FP8 output               | Soportado en Blackwell                       | +2x throughput   |
| `maxnreg` tuning         | Control de registros por warp                | +10%             |

**Kernel mejorado (simplificado):**

```python
@triton.jit
def flash_attention_kernel_optimized(...):
    # Usar exp2 en lugar de exp
    p = tl.math.exp2(qk * sm_scale - m_ij[:, None])

    # Warp specialize para Blackwell
    acc = tl.dot(p, v, acc, warp_specialize=True)
```

### 3.3 Flash Attention — Block Sizes Dinámicos

**Propuesto:**

| head_dim | BLOCK_M | BLOCK_N | Warps |
| -------- | ------- | ------- | ----- |
| ≤32      | 128     | 128     | 4     |
| 64       | 64      | 64      | 4     |
| 128      | 32      | 32      | 4     |
| 256      | 16      | 16      | 4     |

---

## 4. Nuevos Kernels (Mediano Plazo)

### 4.1 Softmax Fused

**Fuente:** Triton tutorial 02-fused-softmax

```python
# Configuración óptima para RTX 5060
BLOCK_SIZE = triton.next_power_of_2(n_cols)
num_warps = 8
num_stages = 2  # 101376 bytes shared mem disponible
```

**Tamaño:** ~200 líneas
**Impacto:** Elimina 1 kernel separado en inference

### 4.2 RMSNorm Fused

**Implementación:** Reducción paralela + normalización en un solo kernel

```python
@triton.jit
def rmsnorm_kernel(X_ptr, W_ptr, O_ptr, N, eps, BLOCK_SIZE: tl.constexpr):
    # 1. Cargar bloque
    # 2. Calcular x^2 + mean
    # 3. rsqrt + multiply + scale
    # 4. Almacenar
```

**Tamaño:** ~150 líneas
**Impacto:** 2x speedup vs PyTorch F.rms_norm

### 4.3 Fused MLP (SwiGLU)

**Operación:** `down(silu(x @ gate.T) * (x @ up.T))`

```python
@triton.jit
def fused_mlp_kernel(...):
    # Gate projection
    gate = tl.dot(x, gate_ptr)
    # Up projection
    up = tl.dot(x, up_ptr)
    # SiLU activation + elementwise multiply
    gate = gate * tl.sigmoid(gate)
    out = gate * up
    # Down projection
    result = tl.dot(out, down_ptr)
```

**Tamaño:** ~300 líneas
**Impacto:** 3x speedup vs separado

### 4.4 Rotary Embedding Fused

**Operación:** `x * cos + rotate_half(x) * sin`

---

## 5. Arquitectura del Sistema (Largo Plazo)

### 5.1 Auto-Tuning Framework

```python
class KernelAutotuner:
    def __init__(self, kernel_fn, config_space):
        self.kernel = kernel_fn
        self.configs = config_space
        self.results = {}

    def tune(self, input_shapes, dtypes):
        """Prueba todas las configuraciones y guarda las óptimas"""
        for shape in input_shapes:
            for dtype in dtypes:
                best = self._benchmark_configs(shape, dtype)
                self.results[(shape, dtype)] = best

    def get_config(self, shape, dtype):
        """Retorna la configuración óptima para un shape/dtype"""
        return self.results.get((shape, dtype), self.default_config)
```

### 5.2 Kernel Dispatch por Tamaño

```python
class AdaptiveMatmul:
    def __init__(self):
        self.kernels = {
            'tiny': self._tiny_kernel,    # M,N,K < 128
            'small': self._small_kernel,  # 128-512
            'medium': self._medium_kernel, # 512-2048
            'large': self._large_kernel,  # >2048
        }

    def __call__(self, A, B):
        M, K = A.shape
        size_class = self._classify(M, K)
        return self.kernels[size_class](A, B)
```

### 5.3 Shared Memory Pool

```python
# Optimización de shared memory para RTX 5060 (101376 bytes)
SHARED_MEM_POOL = {
    'matmul_128x256x64': 128 * 64 * 2 + 256 * 64 * 2,  # 49152 bytes
    'flash_attn_64x64': 64 * 64 * 4 * 2,  # 32768 bytes
    'softmax_1024': 1024 * 4,  # 4096 bytes
}
```

---

## 6. Métricas de Éxito

| Objetivo        | Métrica                | Target           |
| --------------- | ---------------------- | ---------------- |
| Matmul          | TFLOPS en 2048³ FP16   | ≥33 (99% cuBLAS) |
| Flash Attention | TFLOPS en 2,32,1024,64 | ≥35 (103% peak)  |
| Softmax         | Throughput             | ≥50 GB/s         |
| RMSNorm         | Latencia               | ≤0.5 ms          |
| End-to-end      | Speedup vs PyTorch     | ≥2x              |
| Correctness     | Todos los edge cases   | PASS             |

---

## 7. Priorización

### Fase 1 (Esta semana)

1. ✅ Aplicar `tl.max_contiguous` y `tl.multiple_of` al matmul
2. ✅ Implementar autotuning para matmul (BLOCK_N=256, BK=64)
3. ✅ Optimizar flash attention con `exp2` y `warp_specialize`

### Fase 2 (Próxima semana)

4. Implementar softmax fused
5. Implementar RMSNorm fused
6. Implementar Fused MLP (SwiGLU)

### Fase 3 (Siguiente)

7. Auto-tuning framework
8. Kernel dispatch por tamaño
9. FP8 support para Blackwell

---

## 8. Riesgos y Mitigaciones

| Riesgo                  | Probabilidad | Mitigación                               |
| ----------------------- | ------------ | ---------------------------------------- |
| Shared memory overflow  | Media        | Dinamic block sizing                     |
| Precision loss con exp2 | Baja         | Testing con torch.allclose               |
| Autotuning timeout      | Media        | Limitar a 50 configs por kernel          |
| VRAM OOM en LLM 7B      | Alta         | Model parallelism o batch size reduction |

---

## 9. Referencias

- Triton Tutorial 02: Fused Softmax — `triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html`
- Triton Tutorial 06: Fused Attention — `triton-lang.org/main/getting-started/tutorials/06-fused-attention.html`
- Triton Tutorial 09: Persistent Matmul — `triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html`
- Triton Gluon: Persistence — `triton-lang.org/main/getting-started/tutorials/gluon/persistence.html`
- Triton Gluon: TCGen05 — `triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html`
