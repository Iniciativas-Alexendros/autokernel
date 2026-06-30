#!/usr/bin/env bash
# Test suite: Verificación post-limpieza del repo
# Ejecutar ANTES de la limpieza (debe fallar) y DESPUÉS (debe pasar)
set -euo pipefail

REPO="/home/alexendros/repositorios/org-iniciativas-alexendros/autokernel"
PASS=0
FAIL=0
TOTAL=0

check() {
  local desc="$1"
  shift
  TOTAL=$((TOTAL + 1))
  if "$@" >/dev/null 2>&1; then
    echo "  ✅ $desc"
    PASS=$((PASS + 1))
  else
    echo "  ❌ $desc"
    FAIL=$((FAIL + 1))
  fi
}

check_fail() {
  local desc="$1"
  shift
  TOTAL=$((TOTAL + 1))
  if "$@" >/dev/null 2>&1; then
    echo "  ❌ $desc (debería haber fallado)"
    FAIL=$((FAIL + 1))
  else
    echo "  ✅ $desc (correctamente eliminado)"
    PASS=$((PASS + 1))
  fi
}

cd "$REPO"

echo "=== T1: Archivos eliminados ==="
check_fail "examples/ no existe" test -d examples
check_fail "kernelbench/ no existe" test -d kernelbench
check_fail "CHANGELOG.md no existe" test -f CHANGELOG.md
check_fail "SUMMARY.txt no existe" test -f SUMMARY.txt
check_fail "LICENSE no existe" test -f LICENSE
check_fail "progress.png no existe" test -f progress.png

echo ""
echo "=== T2: Archivos renombrados ==="
check "docs/ARCHITECTURE.md existe" test -f docs/ARCHITECTURE.md
check "docs/PLAYBOOK.md existe" test -f docs/PLAYBOOK.md
check "cuda-samples/ existe" test -d cuda-samples
check_fail "PROPOSAL.md no existe" test -f PROPOSAL.md
check_fail "program.md no existe" test -f program.md
check_fail "cuda-lab/ no existe" test -d cuda-lab

echo ""
echo "=== T3: README ==="
check "README contiene RTX 5060" grep -q "RTX 5060" README.md
check "README contiene pipeline nocturno" grep -qi "pipeline nocturno" README.md
check "README no contiene upstream" bash -c '! grep -qi "rightnowai\|h100\|discord" README.md'
check "README contiene Iniciativas-Alexendros" grep -q "Iniciativas-Alexendros" README.md

echo ""
echo "=== T4: Archivos mantenidos ==="
check "export_hf.py existe" test -f export_hf.py
check "config/pipeline.yaml existe" test -f config/pipeline.yaml
check "scripts/nightly_pipeline.sh existe" test -f scripts/nightly_pipeline.sh
check "scripts/generate_dashboard.py existe" test -f scripts/generate_dashboard.py
check "systemd/autokernel-nightly.service existe" test -f systemd/autokernel-nightly.service
check "systemd/autokernel-nightly.timer existe" test -f systemd/autokernel-nightly.timer
check "models/phi3_mini.py existe" test -f models/phi3_mini.py
check "orchestrate.py existe" test -f orchestrate.py
check "verify.py existe" test -f verify.py
check "extract.py existe" test -f extract.py

echo ""
echo "=== T5: .gitignore ==="
check "gitignore menciona cuda-samples" grep -q "cuda-samples" .gitignore

echo ""
echo "=== T6: Integridad Python ==="
check "orchestrate.py compila" python -m py_compile orchestrate.py
check "verify.py compila" python -m py_compile verify.py
check "extract.py compila" python -m py_compile extract.py
check "generate_dashboard.py compila" python -m py_compile scripts/generate_dashboard.py
check "nightly_pipeline.sh syntax OK" zsh -n scripts/nightly_pipeline.sh

echo ""
echo "=== T7: Tests existentes ==="
check "tests no referencian archivos eliminados" bash -c '! grep -r "kernelbench\|examples\|CHANGELOG\|SUMMARY\|LICENSE" tests/ --include="*.py" 2>/dev/null'

echo ""
echo "=== T8: Git ==="
check "working tree limpio" bash -c 'cd '"$REPO"' && [ -z "$(git status --porcelain)" ]'

echo ""
echo "=== T9: GitHub Pages ==="
check "Dashboard accesible" bash -c 'curl -sI https://iniciativas-alexendros.github.io/autokernel/ | head -1 | grep -q "200"'

echo ""
echo "=============================="
echo "RESULTADOS: $PASS/$TOTAL passed, $FAIL failed"
echo "=============================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
