# Scenarios: Limpieza del Repo AutoKernel

## Happy Paths

### HP1: Limpieza completa exitosa

- **Dado**: Repo con todo el contenido upstream
- **Cuando**: Se ejecuta `rm -rf examples/ kernelbench/ CHANGELOG.md SUMMARY.txt LICENSE progress.png`
- **Entonces**: Los archivos eliminados ya no existen
- **Y**: `git status` muestra los cambios

### HP2: Renombrado de archivos exitoso

- **Dado**: Repo limpio de archivos upstream
- **Cuando**: Se ejecuta `mv PROPOSAL.md docs/ARCHITECTURE.md && mv program.md docs/PLAYBOOK.md && mv cuda-lab/ cuda-samples/`
- **Entonces**: Los archivos originales no existen
- **Y**: Los archivos renombrados existen en las nuevas ubicaciones

### HP3: README en español creado

- **Dado**: Repo con archivos eliminados y renombrados
- **Cuando**: Se crea `README.md` con contenido en español
- **Entonces**: `grep "RTX 5060" README.md` retorna 0
- **Y**: `grep "pipeline nocturno" README.md` retorna 0

### HP4: Tests adaptados

- **Dado**: Repo limpio
- **Cuando**: Se ejecutan los tests
- **Entonces**: Todos los tests pasan
- **Y**: Ningún test importa archivos eliminados

### HP5: Commit y push exitoso

- **Dado**: Todos los cambios realizados
- **Cuando**: Se ejecuta `git add -A && git commit -m "..." && git push --force-with-lease origin main`
- **Entonces**: El push es exitoso
- **Y**: `git status` muestra "up to date"

### HP6: Dashboard sigue accesible

- **Dado**: Repo actualizado en GitHub
- **Cuando**: Se accede a `https://iniciativas-alexendros.github.io/autokernel/`
- **Entonces**: Retorna HTTP 200
- **Y**: El HTML contiene "AutoKernel"

## Edge Cases

### EC1: Archivos eliminados referenciados en imports

- **Dado**: Un archivo Python que importa de `examples/` o `kernelbench/`
- **Cuando**: Se eliminan esos directorios
- **Entonces**: El import falla con `ModuleNotFoundError`
- **Acción**: Verificar que ningún archivo Python importe de los directorios eliminados

### EC2: Tests que dependen de archivos eliminados

- **Dado**: Tests que importan de `kernelbench/`
- **Cuando**: Se elimina `kernelbench/`
- **Entonces**: Los tests fallan con `ModuleNotFoundError`
- **Acción**: Adaptar o eliminar esos tests

### EC3: `program.md` referenciado por orchestrate.py

- **Dado**: `orchestrate.py` que lee `program.md`
- **Cuando**: Se renombra a `docs/PLAYBOOK.md`
- **Entonces**: `orchestrate.py` no encuentra el archivo
- **Acción**: Verificar referencias en `orchestrate.py` antes de renombrar

### EC4: `cuda-lab/` referenciado en .gitignore

- **Dado**: `.gitignore` con entradas para `cuda-lab/`
- **Cuando**: Se renombra a `cuda-samples/`
- **Entonces**: Las reglas de gitignore no aplican
- **Acción**: Actualizar `.gitignore` con las nuevas rutas

### EC5: GitHub Pages workflow usa archivos eliminados

- **Dado**: `.github/workflows/pages.yml` que sube `docs/`
- **Cuando**: Se eliminan archivos de `docs/`
- **Entonces**: El workflow falla
- **Acción**: Verificar que `docs/index.html` y `docs/ARCHITECTURE.md` existan

### EC6: README contiene referencias a upstream

- **Dado**: README que menciona "RightNowAI", "H100", "Discord"
- **Cuando**: Se revisa el contenido
- **Entonces**: Esas referencias no deben existir
- **Acción**: Revisar y eliminar referencias upstream

## Errores esperados

### EE1: `git push` falla por conflicto

- **Causa**: El remote tiene cambios que el local no tiene
- **Solución**: `git pull --rebase origin main` antes de push

### EE2: `mv` falla porque el destino ya existe

- **Causa**: `docs/ARCHITECTURE.md` ya existe
- **Solución**: Verificar que el destino no exista antes de renombrar

### EE3: Tests fallan tras limpieza

- **Causa**: Test importa archivo eliminado
- **Solución**: Adaptar test o eliminarlo

### EE4: Dashboard no genera HTML válido

- **Causa**: `generate_dashboard.py` falla
- **Solución**: Verificar que el script funcione con la nueva estructura
