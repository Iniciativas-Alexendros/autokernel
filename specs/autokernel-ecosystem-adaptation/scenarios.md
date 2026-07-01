# Scenarios: AutoKernel Ecosystem Adaptation

## Happy paths

### H1: Pipeline nocturno completo exitoso
1. El timer systemd arranca el servicio a las 2:00.
2. `PipelineRunner` itera sobre los modelos habilitados en `config/pipeline.yaml`.
3. Para cada modelo: profile → extract → optimize → verify → export → report.
4. Cada kernel optimizado supera umbrales de speedup y correctness.
5. Se genera dashboard, se commitea a `docs/` y se notifica por `ntfy`.
6. El servicio termina antes del timeout de 6 h.

### H2: Optimización de un único kernel bajo demanda
1. Operador ejecuta: `uv run orchestrate.py --workspace workspace/llama_7b auto --kernel matmul`.
2. El orquestador carga el plan, establece baseline y entra en loop de experimentos.
3. Cada experimento se mantiene o revierte según reglas de `bench.py`.
4. Al alcanzar criterio de move-on, el kernel se guarda como `_optimized.py`.
5. Se ejecuta verificación end-to-end y se genera PR automático.

### H3: Auto-PR y auto-merge de kernel validado
1. Un kernel optimizado pasa verificación con `correctness=true` y `speedup > 1.05`.
2. `GitHub Publisher` crea rama y commit con el kernel optimizado y el reporte.
3. Abre PR con labels `autokernel`, `automated`.
4. CI ejecuta tests y verificación; Nemotron/opencode revisa el diff.
5. Si todo pasa, se mergea a `main` y se publica dashboard.

### H4: Auditoría y mejora del propio pipeline
1. Skill `/criticar autokernel/` revisa código, docs y métricas.
2. Se genera spec de refactorización con propuestas concretas.
3. Se implementa el cambio mínimo con tests first.
4. Se mide impacto en duración y tasa de éxito del pipeline.
5. Se actualiza `docs/PLAYBOOK.md` con la lección aprendida.

## Edge cases

### E1: Ollama no responde
- El LLM Router intenta el modelo local configurado.
- Si falla tras 3 reintentos, cambia a modelo `opencode` remoto (si está habilitado) o pausa la fase LLM.
- El pipeline continúa con kernels ya generados o entra en estado `degraded`.

### E2: OOM durante profiling de modelo grande
- El `ResourceSemaphore` detecta VRAM > umbral antes de lanzar perfil.
- Reduce batch size o secuencia automáticamente según `config/pipeline.yaml`.
- Si no es recuperable, marca el modelo como `skipped` y notifica.

### E3: Ningún kernel alcanza speedup mínimo
- El orquestador aplica criterio de move-on por tiempo o reverts consecutivos.
- El kernel se marca como `done` sin mejoras.
- El PR no se genera; solo se registra en reporte.

### E4: GPU en uso por otro proceso
- El scheduler detecta actividad de GPU (por ejemplo, via `nvidia-smi`).
- Postpone benchmarks hasta que la GPU esté libre o por debajo de umbral de uso.
- Si el usuario está activo, el pipeline pasa a modo `idle`.

### E5: Fallo de verificación end-to-end
- Se activa modo `diagnose` de `verify.py` para aislar el kernel culpable.
- Se revierte el cambio problemático y se reintentan optimizaciones más conservadoras.
- No se genera PR hasta que `correctness` sea `true`.

## Failure modes

### F1: Estado de orquestación corrupto
- `load_state()` detecta JSON inválido.
- Se reinicia el estado desde `optimization_plan.json` con una copia de seguridad del estado anterior.
- Se registra el incidente en `workspace/incidents.log`.

### F2: Push a GitHub falla
- Se reintentan hasta 3 veces con backoff.
- Si persiste, el cambio se deja en rama local y se alerta por `ntfy`.
- El operador puede resolver manualmente sin perder resultados.

### F3: Secretos no disponibles
- Si `pass-cli` o Proton Pass MCP no responden, el pipeline falla de forma segura.
- No se escribe nunca una API key en disco ni en logs.
- Se notifica al operador para que verifique el vault.

### F4: CI bloquea merge automático
- El PR queda abierto para revisión humana.
- El dashboard marca el kernel como `pending-review`.
- El operador puede mergear manualmente tras revisar.

## Escenarios de agente externo

### A1: Agente invoca `/spec` para nuevo kernel
1. Lee `AGENTS.md` para conocer rutas y comandos.
2. Genera spec en `specs/<kernel>/`.
3. Escribe tests first, implementa, verifica red/green.
4. Registra resultados y actualiza dashboard.

### A2: Agente invoca `/criticar` sobre AutoKernel
1. El skill audita código, docs y configuración.
2. Genera checklist de saneamiento con severidad.
3. El operador (o agente) prioriza y ejecuta fases.
4. Se actualiza la spec con lecciones aprendidas.
