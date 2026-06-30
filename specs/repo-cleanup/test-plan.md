# Test Plan: Limpieza del Repo AutoKernel

## Estrategia

Tests de verificación post-limpieza. No se escriben tests unitarios nuevos, sino que se verifican los criterios de aceptación de la spec mediante comandos shell y checks de integridad.

## Tests por módulo

### T1: Verificación de archivos eliminados

| Test | Comando                | Esperado                    |
| ---- | ---------------------- | --------------------------- |
| T1.1 | `ls examples/ 2>&1`    | "No such file or directory" |
| T1.2 | `ls kernelbench/ 2>&1` | "No such file or directory" |
| T1.3 | `ls CHANGELOG.md 2>&1` | "No such file or directory" |
| T1.4 | `ls SUMMARY.txt 2>&1`  | "No such file or directory" |
| T1.5 | `ls LICENSE 2>&1`      | "No such file or directory" |
| T1.6 | `ls progress.png 2>&1` | "No such file or directory" |

### T2: Verificación de archivos renombrados

| Test | Comando                   | Esperado                    |
| ---- | ------------------------- | --------------------------- |
| T2.1 | `ls docs/ARCHITECTURE.md` | Exitoso                     |
| T2.2 | `ls docs/PLAYBOOK.md`     | Exitoso                     |
| T2.3 | `ls cuda-samples/`        | Exitoso                     |
| T2.4 | `ls PROPOSAL.md 2>&1`     | "No such file or directory" |
| T2.5 | `ls program.md 2>&1`      | "No such file or directory" |
| T2.6 | `ls cuda-lab/ 2>&1`       | "No such file or directory" |

### T3: Verificación de README

| Test | Comando                                         | Esperado              |
| ---- | ----------------------------------------------- | --------------------- |
| T3.1 | `grep "RTX 5060" README.md`                     | Exitoso               |
| T3.2 | `grep "pipeline nocturno" README.md`            | Exitoso               |
| T3.3 | `grep -i "rightnowai\|h100\|discord" README.md` | Falla (no encontrado) |
| T3.4 | `grep "Iniciativas-Alexendros" README.md`       | Exitoso               |

### T4: Verificación de archivos mantenidos

| Test  | Comando                                 | Esperado |
| ----- | --------------------------------------- | -------- |
| T4.1  | `ls export_hf.py`                       | Exitoso  |
| T4.2  | `ls config/pipeline.yaml`               | Exitoso  |
| T4.3  | `ls scripts/nightly_pipeline.sh`        | Exitoso  |
| T4.4  | `ls scripts/generate_dashboard.py`      | Exitoso  |
| T4.5  | `ls systemd/autokernel-nightly.service` | Exitoso  |
| T4.6  | `ls systemd/autokernel-nightly.timer`   | Exitoso  |
| T4.7  | `ls models/phi3_mini.py`                | Exitoso  |
| T4.8  | `ls orchestrate.py`                     | Exitoso  |
| T4.9  | `ls verify.py`                          | Exitoso  |
| T4.10 | `ls extract.py`                         | Exitoso  |

### T5: Verificación de .gitignore

| Test | Comando                          | Esperado                |
| ---- | -------------------------------- | ----------------------- |
| T5.1 | `grep "cuda-samples" .gitignore` | Exitoso (si hay reglas) |
| T5.2 | `grep "cuda-lab" .gitignore`     | Falla (ya no aplica)    |

### T6: Verificación de integridad Python

| Test | Comando                                              | Esperado |
| ---- | ---------------------------------------------------- | -------- |
| T6.1 | `python -m py_compile orchestrate.py`                | Exitoso  |
| T6.2 | `python -m py_compile verify.py`                     | Exitoso  |
| T6.3 | `python -m py_compile extract.py`                    | Exitoso  |
| T6.4 | `python -m py_compile scripts/generate_dashboard.py` | Exitoso  |
| T6.5 | `zsh -n scripts/nightly_pipeline.sh`                 | Exitoso  |

### T7: Verificación de tests existentes

| Test | Comando                                          | Esperado                                            |
| ---- | ------------------------------------------------ | --------------------------------------------------- |
| T7.1 | `grep -r "kernelbench\|examples" tests/`         | Sin resultados (no referencian archivos eliminados) |
| T7.2 | `python -m py_compile tests/test_orchestrate.py` | Exitoso                                             |

### T8: Verificación de Git

| Test | Comando                | Esperado                                          |
| ---- | ---------------------- | ------------------------------------------------- |
| T8.1 | `git status`           | "nothing to commit, working tree clean"           |
| T8.2 | `git log --oneline -1` | Mensaje de commit contiene "limpieza" o "cleanup" |

### T9: Verificación de GitHub Pages

| Test | Comando                                                                    | Esperado     |
| ---- | -------------------------------------------------------------------------- | ------------ |
| T9.1 | `curl -sI https://iniciativas-alexendros.github.io/autokernel/ \| head -1` | "HTTP/2 200" |

## Orden de ejecución

1. **T1** — Verificar archivos eliminados
2. **T2** — Verificar archivos renombrados
3. **T3** — Verificar README
4. **T4** — Verificar archivos mantenidos
5. **T5** — Verificar .gitignore
6. **T6** — Verificar integridad Python
7. **T7** — Verificar tests existentes
8. **T8** — Verificar Git
9. **T9** — Verificar GitHub Pages

## Cobertura

| Categoría            | Tests  | Cobertura |
| -------------------- | ------ | --------- |
| Archivos eliminados  | 6      | 100%      |
| Archivos renombrados | 6      | 100%      |
| README               | 4      | 100%      |
| Archivos mantenidos  | 10     | 100%      |
| .gitignore           | 2      | 100%      |
| Integridad Python    | 5      | 100%      |
| Tests existentes     | 2      | 100%      |
| Git                  | 2      | 100%      |
| GitHub Pages         | 1      | 100%      |
| **Total**            | **38** | **100%**  |
