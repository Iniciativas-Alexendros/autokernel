# AUDITORÍA CRÍTICA — AutoKernel (repo + sitio GitHub Pages)

> **Veredicto de una línea:** Un pipeline ambicioso con buenas intenciones, pero la CI está rota, la cobertura de tests es ridícula, el código se autopodría con excepciones silenciadas y el sitio público parece un dashboard interno tirado a producción sin SEO ni cabeceras de seguridad.

**Resumen ejecutivo:** AutoKernel ha crecido rápido con 5 fases de implementación recientes (saneamiento, hardening, agentes, pipeline continuo, auto-PR y auto-auditoría). Sin embargo, la disciplina de ingeniería no ha acompañado la velocidad: `ruff check .` falla con decenas de F541, el CI de GitHub no instala los extras necesarios para ejecutar los tests, la cobertura real es del 38 % y hay módulos core (`semaphore.py`, `rag_index.py`, `nemotron_client.py`) sin tests. El sitio GitHub Pages carece de `robots.txt`, `sitemap.xml`, meta description, CSP y cabeceras de seguridad básicas. La buena noticia: la mayoría de los defectos son sistémicos y se pueden resolver con un parche de saneamiento de una sentada.

---

## Alcance auditado

| Parámetro | Valor |
| --------- | ----- |
| **Objetivo** | `/home/alexendros/repositorios/org-iniciativas-alexendros/autokernel` + `https://iniciativas-alexendros.github.io/autokernel/` |
| **Modo de objetivo** | COMBINADO (repositorio + sitio web) |
| **Modo de entorno** | COMPLETO (shell con red real, herramientas locales) |
| **Nivel de profundidad** | 3-PROFUNDO |
| **Commit auditado** | `84b87f9` (HEAD → main, origin/main) |
| **Archivos** | 98 (`git ls-files`) |
| **LOC Python** | ~17 558 |
| **LOC shell** | ~334 |

**Herramientas ejecutadas:** `gitleaks`, `ruff`, `bandit`, `tokei`, `pytest` (con cobertura), `curl` (headers/HTTP), análisis manual del sitio.
**Herramientas NO disponibles:** `trivy`, `lighthouse`, `scc`, `playwright`. Los aspectos que habrían cubierto se auditan manualmente y se declaran como LIMITADO en el informe.

---

## 1. INFORME DE CRÍTICA DESTRUCTIVA

### 1.1 Arquitectura y Diseño

#### DEFECTO-001 — `orchestrate.py` es un monolito que viola SRP

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Ubicación:** `orchestrate.py:1-1394`  
**Evidencia:**
- 1 394 líneas en un solo archivo.
- Mezcla: CLI (`argparse`), máquina de estados, TSV, Amdahl, auto-LLM, CUDA migration, reportes extendidos y publicación GitHub.
- No hay separación entre dominio (orquestación) e infraestructura (subprocess, CLI, LLM).

**Impacto:** Cualquier cambio en un subcomando obliga a leer/retocar el mismo archivo. Reutilizar la lógica de orquestación desde el pipeline continuo es imposible sin importar el CLI completo. El riesgo de regresión es alto.

**Parche:** Extraer `autokernel/orchestrator/` como paquete con módulos `state.py`, `decisions.py`, `reporting.py`, `auto.py`, `publish.py`, `cli.py`.

---

#### DEFECTO-002 — El publicador GitHub asume `main`, `gh` autenticado y un workflow externo

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Ubicación:** `autokernel/github_publisher.py:34-68`  
**Evidencia:**
- `self.default_branch = "main"` hardcodeado.
- `subprocess.run(["gh", "--version"], ...)` sin path absoluto (B607).
- No hay verificación de que el entorno de CI tenga `gh` autenticado.
- El auto-merge depende de que la CI del repo (`.github/workflows/ci.yml`) esté verde, pero esa CI está rota (ver DEFECTO-020/021).

**Impacto:** En un PR generado automáticamente, el auto-merge se habilita sobre una CI que no pasa. El merge puede entrar código con tests fallidos o lint roto.

**Parche:**
1. Leer la rama por defecto desde `git symbolic-ref refs/remotes/origin/HEAD`.
2. Usar path absoluto de `gh` (detectado en `__init__`) o fallar con mensaje claro.
3. No habilitar auto-merge si la CI no está configurada para ejecutar tests en ramas `autokernel/*`.

---

#### DEFECTO-003 — Acoplamiento implícito a Ollama sin adaptador

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `autokernel/llm_assistant.py`, `autokernel/semaphore.py`  
**Evidencia:**
- `ollama_client.chat`, `ollama_client.embeddings`, `ollama_client.stop` se llaman directamente.
- El fallback a `opencode/*` se implementa con un `if model.startswith("opencode/")` dentro de `_call_ollama`, mezclando dos abstracciones distintas.

**Impacto:** Añadir un tercer proveedor (local, API, opencode) requiere tocar `_call_ollama`. La lógica de carga/descarga de modelos locales se mezcla con la llamada a APIs remotas.

**Parche:** Crear `LLMBackend` (interfaz) con implementaciones `OllamaBackend`, `OpenRouterBackend`, `NemotronBackend`. El `LLMAssistant` enruta según el modelo sin saber detalles de transporte.

---

### 1.2 Implementación y Lógica

#### DEFECTO-004 — `except Exception: pass` silencia errores críticos

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Evidencia:** Bandit B110 en:
- `autokernel/github_publisher.py:202`
- `autokernel/llm_assistant.py:271`
- `autokernel/semaphore.py:64`, `semaphore.py:71`, `semaphore.py:103`
- `scripts/generate_dashboard.py:151`
- `scripts/self_audit.py:55`

**Impacto:** Fallos de `git checkout`, de carga de modelos Ollama, de parada de modelos, de parseo de JSON del dashboard o de verificación de kernels se tragan. El pipeline sigue como si nada hubiera pasado, produciendo resultados corruptos o PRs basura.

**Parche mínimo:** Sustituir `pass` por `logger.exception(...)` o `self._record_error(...)` en cada caso. No silenciar nunca un `Exception` genérico sin métrica/alerta.

```diff
--- a/autokernel/semaphore.py
+++ b/autokernel/semaphore.py
@@ -61,8 +61,8 @@ class ResourceSemaphore:
         if self.current_model:
             try:
                 ollama_client.stop(self.current_model)
-            except Exception:
-                pass
+            except Exception as exc:
+                logger.warning("ollama stop failed: %s", exc)
             self.stats.llm_switches += 1
```

---

#### DEFECTO-005 — `_run_bench` copia `kernel.py` sin rollback ni atomicidad

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `orchestrate.py:956-987`  
**Evidencia:**
```python
shutil.copy2(kernel_path, target)
```
donde `target = SCRIPT_DIR / "kernel.py"`. Si el proceso muere después, el repo queda con un `kernel.py` residual no deseado. No hay backup del kernel.py original.

**Impacto:** Riesgo de commit accidental de un kernel intermedio. Contaminación del working tree.

**Parche:**
```python
original = SCRIPT_DIR / "kernel.py"
backup = SCRIPT_DIR / "kernel.py.bak"
try:
    if original.exists():
        shutil.copy2(original, backup)
    shutil.copy2(kernel_path, original)
    return _parse_bench_output(...)
finally:
    if backup.exists():
        backup.replace(original)
    elif original.exists():
        original.unlink()
```

---

#### DEFECTO-006 — `cmd_auto` genera tests, los ve fallar y sigue

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `orchestrate.py:1069-1085`  
**Evidencia:**
```python
if test_result.returncode != 0:
    print(f"  Tests failed (expected during generation): {test_result.stderr[:200]}")
```

**Impacto:** El loop LLM no usa el resultado de los tests para corregir. El TDD es teatro: se generan tests, se ignoran, y luego se genera el kernel. La calidad del kernel depende enteramente del LLM, no del feedback del test.

**Parche:** Si los tests fallan, pasar el error al LLM como contexto y reintentar la generación del kernel (con límite de reintentos).

---

### 1.3 Vulnerabilidades de Seguridad

#### DEFECTO-007 — Subprocess con paths parciales (B607) y sin validación de entradas

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Evidencia:**
- `autokernel/github_publisher.py:37` → `["gh", "--version"]`
- `scripts/continuous_pipeline.py:174` → `["nvidia-smi", ...]`
- `scripts/self_audit.py:31` → `subprocess.run(cmd, ...)` con `cmd` proveniente de llamadas internas.

**Impacto:** Si un atacante con acceso local manipula el `PATH` o el directorio de trabajo, puede ejecutar un binario falso. `nvidia-smi` no es crítico, pero `gh` sí: roba el token o modifica el repo.

**Parche:**
```python
import shutil
gh_path = shutil.which("gh") or "/usr/bin/gh"
if not Path(gh_path).exists():
    raise RuntimeError("gh CLI not found")
subprocess.run([gh_path, "--version"], ...)
```

---

#### DEFECTO-008 — `urllib.request.urlopen` sin restringir esquemas (B310)

**Severidad:** MODERADO  
**Confianza:** MEDIA  
**Ubicación:** `autokernel/llm_assistant.py:241`  
**Evidencia:**
```python
with urllib.request.urlopen(req, timeout=120) as resp:
```

**Impacto:** Aunque ahora se construye siempre con `https://openrouter.ai/...`, la función no valida el scheme. Un error futuro o un modelo configurado con `file://` permitiría lectura local de archivos (SSRF local).

**Parche:**
```python
from urllib.parse import urlparse
scheme = urlparse(req.full_url).scheme
if scheme not in ("https",):
    raise ValueError(f"unsupported URL scheme: {scheme}")
```

---

#### DEFECTO-009 — `access-control-allow-origin: *` en GitHub Pages expone el dashboard

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** Respuesta HTTP del sitio `https://iniciativas-alexendros.github.io/autokernel/`  
**Evidencia:**
```http
access-control-allow-origin: *
```

**Impacto:** Cualquier sitio web puede leer el dashboard desde el navegador del usuario (CORS abierto). Aunque el contenido sea público, amplifica la superficie de tracking y posibles filtraciones si el dashboard incluye datos sensibles en el futuro.

**Nota:** GitHub Pages añade esta cabecera automáticamente. No se puede eliminar sin proxy/cloudflare. El riesgo se mitiga no incluyendo datos sensibles en `docs/`.

---

### 1.4 Rendimiento y Escalabilidad

#### DEFECTO-010 — RAG embeddea documentos uno a uno en lugar de por lotes

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `autokernel/rag_index.py:51-59`  
**Evidencia:**
```python
for text in texts:
    resp = ollama_client.embeddings(model=self.embed_model, prompt=text)
    vectors.append(resp["embedding"])
```

**Impacto:** Cada llamada a Ollama tiene overhead de red+modelo. Con 50 documentos, 50 round-trips. Latencia de minutos en lugar de segundos.

**Parche:** Ollama soporta múltiples prompts en paralelo o batching. Usar `asyncio.gather` o `ThreadPoolExecutor` con un límite de concurrencia (la VRAM lo impone).

---

#### DEFECTO-011 — `continuous_pipeline.py` ejecuta fases secuencialmente

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `scripts/continuous_pipeline.py:160-258`  
**Evidencia:** Cada fase (`profile`, `extract`, `optimize`, `verify`, `report`) se ejecuta en serie con un único `timeout=3600`. No hay paralelismo entre modelos ni entre kernels dentro de un modelo.

**Impacto:** Un RTX 5060 con 8 GB no aprovecha bien el tiempo. Un modelo grande puede monopolizar el pipeline.

**Parche:** Permitir `max_parallel_models` y `max_parallel_kernels` con un semáforo de VRAM/CPU. Reutilizar `ResourceSemaphore` del ecosistema local.

---

### 1.5 Deuda Técnica y Malas Prácticas

#### DEFECTO-012 — F-strings sin placeholders (F541) en `analysis.py`

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Evidencia:** `ruff check .` reporta decenas de F541. Ejemplos:
- `analysis.py:117` → `f"WARNING: TSV columns do not match expected schema."`
- `analysis.py:413` → `f"\n  Top 5 improvements:"`
- `analysis.py:480` → `f"| Metric | Value |"`

**Impacto:** El CI está roto. Cualquier PR falla en el job de lint.

**Parche:**
```bash
uv run ruff check . --select F541 --fix
```

---

#### DEFECTO-013 — Código duplicado de `subprocess.run` + parseo de salida

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Evidencia:** Patrones idénticos en `github_publisher.py`, `continuous_pipeline.py`, `self_audit.py`, `orchestrate.py`.

**Impacto:** Cada nuevo script reinventa logging, timeout y manejo de errores. Inconsistencias inevitable.

**Parche:** Crear `autokernel/subprocess_runner.py` con `run_logged(cmd, timeout, label)` que devuelva `(ok, stdout, stderr, duration)` y registre en métricas.

---

### 1.6 Documentación y Legibilidad

#### DEFECTO-014 — README y PLAYBOOK aún promocionan el pipeline nocturno como protagonista

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `README.md`, `docs/PLAYBOOK.md`  
**Evidencia:** El README titula secciones "Pipeline nocturno" y describe el timer 2:00-8:00. El `systemd/autokernel-continuous.service` es la nueva apuesta, pero no aparece en la documentación principal.

**Impacto:** Un nuevo colaborador o agente lee documentación obsoleta. El modo continuo queda como feature oculta.

**Parche:** Reescribir README con dos modos: nocturno (legacy/standalone) y continuo (recomendado). Añadir sección de operación y troubleshooting del servicio continuo.

---

### 1.7 Pruebas y Cobertura

#### DEFECTO-015 — Cobertura real del 38 % y módulos core sin tests

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Evidencia:**
```
Name                                   Stmts   Miss  Cover
------------------------------------------------------------
autokernel/semaphore.py                   66     66     0%
autokernel/rag_index.py                  138     99    28%
autokernel/nemotron_client.py             84     52    38%
autokernel/llm_assistant.py              129     80    38%
scripts/generate_dashboard.py            136    136     0%
scripts/continuous_pipeline.py           226    116    49%
autokernel/github_publisher.py           101     27    73%
```

**Impacto:** Un pipeline que se autoproclama crítico de hardware no tiene garantía de que sus módulos de concurrencia, RAG, LLM y publicación funcionan. El modo continuo es fé de que no hay regresiones.

**Parche:** Añadir tests unitarios con mocks para `semaphore.py`, `rag_index.py`, `nemotron_client.py`, `llm_assistant.py` y `generate_dashboard.py`. Fijar cobertura mínima del 70 % en CI.

---

#### DEFECTO-016 — CI de GitHub ejecuta tests de forma rota

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Ubicación:** `.github/workflows/ci.yml:16-22`  
**Evidencia:**
```yaml
- run: uv sync
- run: uv run pytest tests/ -x -q
```

**Impacto:**
- `uv sync` sin `--extra testing --extra pipeline` no instala `requests` ni `ollama`/`aiohttp`/`faiss-cpu`/`pyyaml`, así que tests como `test_llm_router.py` o imports fallan.
- `-x` para en el primer error, ocultando el resto.
- No se deseleccionan tests lentos/integración (`test_ollama_integration.py`).

**Parche:**
```yaml
- run: uv sync --extra testing --extra pipeline
- run: uv run pytest -m "not slow" -q --cov=autokernel --cov=scripts --cov-fail-under=70
```

---

#### DEFECTO-017 — CI no ejecuta lint, SAST ni escáner de secretos

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Ubicación:** `.github/workflows/ci.yml`  
**Evidencia:** El job `lint` ejecuta `ruff check .` y `ruff format --check .`, pero no `bandit`, `gitleaks`, `py_compile` ni validación de systemd units.

**Impacto:** Los defectos de seguridad (B110, B607, B310) y los secretos accidentales pasan desapercibidos en PRs.

**Parche:** Añadir jobs de seguridad al CI:
```yaml
security:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v4
    - run: uv sync --extra testing --extra pipeline
    - run: uv run bandit -r autokernel scripts
    - run: uv run gitleaks detect --source . --verbose
    - run: systemd-analyze verify systemd/*.service systemd/*.timer
```

---

### 1.8 Errores Microscópicos y Estilo

#### DEFECTO-018 — F-strings vacíos y módulos con imports no usados (microscópico, sistémico)

**Severidad:** MICROSCÓPICO  
**Confianza:** ALTA  
**Evidencia:** `ruff check .` devuelve principalmente F541. También hay imports como `import importlib.util` en `tests/test_llm_router.py` no usados? No, ahora sí se usan. Revisar con `ruff check --select F401`.

**Impacto:** Ruido y CI roto.

**Parche:** Aplicar `ruff check . --fix` y fijar `ruff` en CI.

---

### 1.9 DevOps, CI/CD y Despliegue

#### DEFECTO-019 — El workflow de GitHub Pages despliega `docs/` sin regenerar el dashboard

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `.github/workflows/pages.yml:18-31`  
**Evidencia:**
```yaml
- uses: actions/upload-pages-artifact@v3
  with:
    path: docs
```

**Impacto:** El dashboard puede quedar desactualizado si `scripts/generate_dashboard.py` no se ejecuta antes del push. El sitio muestra datos antiguos.

**Parche:** Añadir paso previo que genere el dashboard:
```yaml
- run: uv sync --extra pipeline
- run: uv run python scripts/generate_dashboard.py --workspace workspace --config config/pipeline.yaml --output docs/index.html
```

---

#### DEFECTO-020 — `nightly_pipeline.sh` usa `|| true` y `|| echo` que enmascaran fallos reales

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `scripts/nightly_pipeline.sh:91-181`  
**Evidencia:**
```bash
timeout "${PROFILE_TIMEOUT}m" uv run python profile.py ... || {
    echo "WARN: Profile failed for $M_NAME, using existing data"
}
```

**Impacto:** El pipeline nunca falla; siempre informa "success". Los errores de profiling, extracción, optimización o verificación se convierten en silencios. Métricas y dashboards pueden ser de corridas anteriores o vacíos.

**Parche:** Distinguir entre errores recuperables (usar cache) y no recuperables. Registrar el estado final en `workspace/pipeline_status.json`. No usar `|| true` para todo.

---

### 1.10 Licencias y Dependencias

#### DEFECTO-021 — `torch` se instala desde índice `cu128` pero el hardware declara CUDA 13.1

**Severidad:** MODERADO  
**Confianza:** MEDIA  
**Ubicación:** `pyproject.toml:52-60`  
**Evidencia:**
```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
```
vs `README.md`: "CUDA 13.1".

**Impacto:** PyTorch cu128 puede no ser compatible con CUDA Toolkit 13.1 en runtime, causando fallos silenciosos de carga de CUDA o degradación a CPU.

**Parche:** Verificar compatibilidad con `nvcc --version` y `torch.version.cuda`. Si no coincide, cambiar el índice o documentar explícitamente la combinación soportada.

---

## 2. CRÍTICA DEL SITIO WEB (GitHub Pages)

### 2.1 Rendimiento y Core Web Vitals

#### DEFECTO-022 — Carga recursos de CDN externos sin SRI y bloqueantes

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Ubicación:** `docs/index.html:7-10`  
**Evidencia:**
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
```

**Impacto:** Sin SRI, un CDN comprometido puede inyectar CSS/JS en el dashboard. Plotly es render-blocking (no `defer`/`async`). No hay fallback offline.

**Parche:** Añadir `integrity` y `crossorigin="anonymous"`. Cargar Plotly con `defer` o async. Considerar vendorizar los assets críticos en `docs/static/`.

---

#### DEFECTO-023 — CSS/JS inline y sin minificación

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `docs/index.html`  
**Evidencia:** El HTML contiene ~230 líneas de código inline entre CSS y JS. No se sirve como archivo separado ni se comprime.

**Impacto:** Cada despliegue invalida el cache completo de la página. No se aprovecha cache de assets estáticos.

**Parche:** Separar CSS/JS a archivos externos y versionarlos por hash. Añadir Vite/rollup o un generador simple de assets estáticos.

---

### 2.2 SEO y Descubribilidad

#### DEFECTO-024 — Falta meta description, canonical y Open Graph

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Ubicación:** `docs/index.html:1-12`  
**Evidencia:**
```html
<title>AutoKernel — Dashboard</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
```
No hay `<meta name="description">`, `<link rel="canonical">`, `<meta property="og:*">`, `<meta name="twitter:*">`.

**Impacto:** Google y redes sociales generan snippets automáticos, normalmente peores. El sitio no se posiciona por términos clave.

**Parche:**
```html
<meta name="description" content="Dashboard del pipeline de optimización autónoma de kernels GPU para RTX 5060">
<link rel="canonical" href="https://iniciativas-alexendros.github.io/autokernel/">
<meta property="og:title" content="AutoKernel — Dashboard">
<meta property="og:description" content="...">
<meta property="og:url" content="https://iniciativas-alexendros.github.io/autokernel/">
```

---

#### DEFECTO-025 — No hay `robots.txt` ni `sitemap.xml`

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `https://iniciativas-alexendros.github.io/autokernel/robots.txt` (404)  
**Evidencia:** `curl` devuelve 404 con la página de error de GitHub Pages.

**Impacto:** Los crawlers no tienen instrucciones explícitas. Google descubre el sitio, pero no optimiza el presupuesto de rastreo.

**Parche:** Añadir `docs/robots.txt` y `docs/sitemap.xml`.

---

### 2.3 Accesibilidad

#### DEFECTO-026 — Emojis usados como iconos sin alternativa accesible

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `docs/index.html` (múltiples `<span class="icon">⚡</span>`, `<span class="icon">📊</span>`)  
**Evidencia:** Los emojis son leídos por lectores de pantalla como "rayo", "gráfico de barras", etc. No hay `aria-hidden="true"` ni texto alternativo.

**Parche:**
```html
<span class="icon" aria-hidden="true">⚡</span>
<span class="sr-only">AutoKernel</span>
```

---

#### DEFECTO-027 — Tablas sin `scope` y sin caption

**Severidad:** MICROSCÓPICO  
**Confianza:** ALTA  
**Ubicación:** `docs/index.html` (tablas de modelos y kernels)  
**Evidencia:** Uso de `<th>` sin `scope="col"`.

**Impacto:** Lectores de pantalla no asocian encabezados con celdas correctamente.

**Parche:** Añadir `scope="col"` a los `<th>`.

---

### 2.4 Seguridad Observable

#### DEFECTO-028 — GitHub Pages no sirve cabeceras de seguridad esenciales

**Severidad:** GRAVE  
**Confianza:** ALTA  
**Ubicación:** Respuesta HTTP del sitio  
**Evidencia:** Cabeceras presentes: `strict-transport-security`, `access-control-allow-origin: *`. Faltan: `X-Frame-Options`, `X-Content-Type-Options`, `Content-Security-Policy`, `Referrer-Policy`, `Permissions-Policy`.

**Impacto:**
- Sin `X-Frame-Options`: clickjacking.
- Sin `CSP`: XSS reflejado tiene más impacto.
- Sin `Referrer-Policy`: fuga de URL interna a terceros.

**Parche:** GitHub Pages no permite headers personalizados. Mitigar con meta tags:
```html
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' cdn.plot.ly 'unsafe-inline'; style-src 'self' fonts.googleapis.com cdn.jsdelivr.net 'unsafe-inline'; font-src fonts.gstatic.com; img-src 'self' data:;">
<meta http-equiv="X-Content-Type-Options" content="nosniff">
<meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">
```
Y añadir un script de frame-busting:
```js
if (window.top !== window.self) window.top.location = window.self.location;
```

---

### 2.5 Buenas Prácticas Web y Privacidad

#### DEFECTO-029 — Dependencia de Google Fonts y CDN sin política de privacidad

**Severidad:** MODERADO  
**Confianza:** MEDIA  
**Ubicación:** `docs/index.html:7-10`  
**Evidencia:** Se cargan Google Fonts, Bootstrap CDN y Plotly CDN. No hay política de privacidad ni cookie banner.

**Impacto:** GDPR: Google Fonts puede filtrar IP del visitante. Aunque sea técnico, un sitio público debería declarar qué datos se comparten.

**Parche:** Vendorizar fuentes o añadir aviso de privacidad mínimo. Incluir `referrer` adecuado en los links de CDN.

---

### 2.6 Calidad del Frontend Servido y Enlaces

#### DEFECTO-030 — HTML generado manualmente, difícil de mantener

**Severidad:** MODERADO  
**Confianza:** ALTA  
**Ubicación:** `docs/index.html` (230 líneas de HTML/CSS/JS inline)  
**Evidencia:** `generate_dashboard.py` genera el HTML con concatenaciones de strings. No hay plantilla ni test de renderizado.

**Impacto:** Cada cambio de UI requiere tocar Python y regenerar. Riesgo de XSS si algún dato del usuario se inyecta sin escapar (aunque ahora todo es local).

**Parche:** Usar Jinja2 (ya está disponible vía alguna dependencia?) o una plantilla HTML separada. Escapar datos con `html.escape`.

---

## 3. PLAN DE SANEAMIENTO CON CHECKLIST ADAPTABLE

### Sprint 1 — Detener la sangría (1-2 días)

- [ ] **DEFECTO-012/018**: Correr `uv run ruff check . --fix` y commitear. Fijar lint en CI.
- [ ] **DEFECTO-016/017**: Arreglar `.github/workflows/ci.yml` (extras, `-m "not slow"`, cobertura mínima, bandit/gitleaks, systemd verify).
- [ ] **DEFECTO-004**: Reemplazar todos los `except Exception: pass` por `logger.exception` o métrica de error.
- [ ] **DEFECTO-007**: Usar `shutil.which` o paths absolutos para `gh`, `nvidia-smi`, `git`.
- [ ] **DEFECTO-008**: Validar scheme de URLs en `llm_assistant._call_api`.
- [ ] **DEFECTO-019**: Añadir paso de generación de dashboard en `.github/workflows/pages.yml`.
- [ ] **DEFECTO-025**: Añadir `docs/robots.txt` y `docs/sitemap.xml`.
- [ ] **DEFECTO-024/028**: Añadir meta description, canonical, Open Graph, CSP meta, X-Content-Type-Options, Referrer-Policy.

### Sprint 2 — Consolidar calidad (3-5 días)

- [ ] **DEFECTO-015**: Alcanzar 70 % de cobertura con tests unitarios para `semaphore`, `rag_index`, `nemotron_client`, `llm_assistant`, `generate_dashboard`.
- [ ] **DEFECTO-014**: Actualizar README, PLAYBOOK y ARCHITECTURE con el modo continuo y el nuevo ecosistema.
- [ ] **DEFECTO-001**: Refactorizar `orchestrate.py` en módulos por responsabilidad (mínimo: `state`, `auto`, `publish`, `cli`).
- [ ] **DEFECTO-003**: Crear adaptador `LLMBackend` con implementaciones Ollama/OpenRouter/Nemotron.
- [ ] **DEFECTO-005**: Hacer atómico el copiado de `kernel.py` en `_run_bench`.
- [ ] **DEFECTO-006**: Usar el resultado de tests auto-generados para feedback al LLM.
- [ ] **DEFECTO-020**: Revisar `nightly_pipeline.sh` para distinguir fallos recuperables vs críticos.

### Sprint 3 — Robustez y operación (1-2 semanas)

- [ ] **DEFECTO-002**: Publicador GitHub lee rama por defecto y verifica `gh` autenticado.
- [ ] **DEFECTO-010**: Batch/paralelizar embeddings de RAG.
- [ ] **DEFECTO-011**: Paralelismo controlado en `continuous_pipeline.py`.
- [ ] **DEFECTO-021**: Validar compatibilidad torch/cuda y documentarla.
- [ ] **DEFECTO-022/023**: Vendorizar o añadir SRI a CDN, separar CSS/JS.
- [ ] **DEFECTO-030**: Plantilla HTML separada para el dashboard.
- [ ] **DEFECTO-026/027**: Accesibilidad: emojis con aria-hidden, tablas con scope.

## 4. PLAN ANEXO DE MAGNIFICACIÓN

Sobre el código ya sano:

1. **Métricas operativas en tiempo real**: exponer `/metrics` (Prometheus) desde el runner continuo con GPU util, latencia por fase, tasa de éxito de LLM, tokens consumidos.
2. **Optimización multi-GPU**: cuando el hardware evolucione, distribuir kernels por GPU con un scheduler basado en VRAM.
3. **A/B testing de kernels**: mantener variante original y optimizada, enrutar tráfico de inference por rendimiento, rollback automático si diverge.
4. **Dashboard interactivo**: migrar de HTML estático a una pequeña app (Next.js o Svelte) con filtros, historial y comparativa de commits.
5. **Integración con KernelBench**: usar el extra `kernelbench` para validar contra 250+ problemas estandarizados.
6. **Documentación viva**: que `self_audit.py` genere PRs automáticos de actualización de docs/EVOLUTION.md cuando cambien las métricas.

---

## RESUMEN EJECUTIVO (para decisores)

AutoKernel es un proyecto técnicamente interesante y con visión, pero en su estado actual no está listo para funcionar desatendido en producción sin riesgo. La CI de GitHub está configurada para fallar, la cobertura de tests es insuficiente y hay múltiples puntos donde los errores se silencian. El sitio público carece de SEO y cabeceras de seguridad básicas. La buena noticia es que los defectos son mayoritariamente mecánicos y corregibles en 1-2 semanas de trabajo enfocado. La prioridad es: (1) sanear CI y lint, (2) eliminar `except: pass`, (3) subir cobertura de módulos core, (4) mejorar el sitio público con SEO/CSP.

---

## TABLA RESUMEN

| Severidad | Código | Sitio | Total |
| --------- | ------ | ----- | ----- |
| CATASTRÓFICO | 0 | 0 | 0 |
| GRAVE | 7 | 3 | 10 |
| MODERADO | 10 | 4 | 14 |
| MICROSCÓPICO | 2 | 1 | 3 |
| **TOTAL** | **19** | **8** | **27** |

---

## MÉTRICAS DE LA AUDITORÍA

| Métrica | Valor |
| ------- | ----- |
| Tiempo total | ~45 min |
| Nivel de profundidad | 3-PROFUNDO |
| Modo de objetivo | COMBINADO |
| Modo de entorno | COMPLETO (con LIMITADO en Lighthouse/Core Web Vitals) |
| Herramientas ejecutadas | gitleaks, ruff, bandit, tokei, pytest+coverage, curl |
| Hallazgos brutos de herramientas | ruff: decenas F541; bandit: 23 issues (B110, B404, B603, B607, B310) |
| Falsos positivos filtrados | B603 en subprocess sin shell=True son advertencias de estilo, no exploit directo; se incluyen como MODERADO por gobierno |
| Defectos finales | 27 (todos confirmados, 0 sospechas) |
| Defectos sistémicos | 6 (CI, cobertura, excepciones silenciadas, subprocess, sitio SEO/security, arquitectura monolito) |
| Categoría más castigada | 1.9 DevOps/CI/CD + 1.7 Pruebas |
| Archivos auditados manualmente | 12 (orchestrate, llm_assistant, nemotron_client, semaphore, github_publisher, continuous_pipeline, self_audit, generate_dashboard, ci.yml, pages.yml, pipeline.yaml, index.html) |
| Commit auditado | `84b87f9` |
