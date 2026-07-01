#!/usr/bin/env zsh
# AutoKernel Nightly Pipeline
# Runs 2AM-8AM via systemd timer
# Profile → Extract → Auto-optimize top kernels → Verify → CUDA migrate → Report → Commit → Notify

set -euo pipefail

REPO_DIR="/home/alexendros/repositorios/org-iniciativas-alexendros/autokernel"
CONFIG="${REPO_DIR}/config/pipeline.yaml"
LOG_DIR="/home/alexendros/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nightly_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

cd "$REPO_DIR"

PIPELINE_START=$(date +%s)
PIPELINE_STATUS="success"
PIPELINE_MODELS=0
PIPELINE_KERNELS=0
PIPELINE_REGRESSIONS=0

echo "=============================================="
echo "  AutoKernel Nightly Pipeline"
echo "  Started: $(date)"
echo "=============================================="

# Read config via yq
WEBHOOK_URL=$(yq -r '.pipeline.webhook_url // ""' "$CONFIG")
MAX_PARALLEL=$(yq -r '.pipeline.max_parallel_kernels // 2' "$CONFIG")
MAX_DURATION_H=$(yq -r '.pipeline.max_duration_hours // 6' "$CONFIG")
MAX_DURATION_S=$((MAX_DURATION_H * 3600))
LLM_PLANNER=$(yq -r '.models.planner // "ornith:9b"' "$CONFIG")
LLM_CODER=$(yq -r '.models.coder // "qwen2.5-coder:7b"' "$CONFIG")
OPT_ITERATIONS=$(yq -r '.pipeline.phases.optimize.iterations_per_kernel // 5' "$CONFIG")
OPT_TIMEOUT=$(yq -r '.pipeline.phases.optimize.timeout_per_iteration_sec // 1800' "$CONFIG")
PROFILE_TIMEOUT=$(yq -r '.pipeline.phases.profile.timeout_min // 30' "$CONFIG")
EXTRACT_TOP_K=$(yq -r '.pipeline.phases.extract.top_k // 5' "$CONFIG")
EXTRACT_BACKEND=$(yq -r '.pipeline.phases.extract.backend // "triton"' "$CONFIG")
REPORT_DIR=$(yq -r '.pipeline.phases.report.dashboard_dir // "docs"' "$CONFIG")
CUDA_KERNELS=($(yq -r '.pipeline.phases.cuda_migrate.kernels[] // "matmul"' "$CONFIG"))

# Read enabled models from config
MODELS=()
while IFS= read -r line; do
  MODELS+=("$line")
done < <(yq -r '.pipeline.target_models[] | select(.enabled == true) | [.name, .path, .class, .shape, .dtype] | join(":")' "$CONFIG")

echo "  Models: ${#MODELS[@]} enabled"
echo "  Planner: $LLM_PLANNER | Coder: $LLM_CODER"
echo "  Max duration: ${MAX_DURATION_H}h | Max parallel: $MAX_PARALLEL"
echo "  Webhook: ${WEBHOOK_URL:-none}"

# [1/8] Build RAG index if missing
if [[ ! -f workspace/rag/faiss.index ]]; then
  echo ""
  echo "[1/8] Building RAG index..."
  uv run python -m autokernel.rag_index 2>&1 || echo "WARN: RAG index build failed"
fi

# [2-6/8] Process each model
for model_spec in "${MODELS[@]}"; do
  IFS=':' read -r M_NAME M_PATH M_CLASS M_SHAPE M_DTYPE <<<"$model_spec"
  PIPELINE_MODELS=$((PIPELINE_MODELS + 1))

  ELAPSED=$(($(date +%s) - PIPELINE_START))
  if [[ $ELAPSED -ge $MAX_DURATION_S ]]; then
    echo "TIMEOUT: ${MAX_DURATION_H}h exceeded, stopping."
    PIPELINE_STATUS="timeout"
    break
  fi

  echo ""
  echo "=============================================="
  echo "  [Model $PIPELINE_MODELS] $M_NAME ($M_CLASS, shape=[$M_SHAPE], $M_DTYPE)"
  echo "=============================================="

  MODEL_WS="workspace/${M_NAME}"
  mkdir -p "$MODEL_WS"

  # [2/8] Profile
  PROFILE_DIR="${MODEL_WS}/profile"
  mkdir -p "$PROFILE_DIR"
  echo ""
  echo "[2/8] Profiling $M_NAME..."
  timeout "${PROFILE_TIMEOUT}m" uv run python profile.py \
    --model "$M_PATH" \
    --class-name "$M_CLASS" \
    --input-shape "$M_SHAPE" \
    --dtype "$M_DTYPE" \
    --output "$PROFILE_DIR" 2>&1 || {
    echo "WARN: Profile failed for $M_NAME, using existing data"
  }

  # [3/8] Extract top kernels
  echo ""
  echo "[3/8] Extracting top-${EXTRACT_TOP_K} kernels for $M_NAME..."
  uv run python extract.py \
    --top "$EXTRACT_TOP_K" \
    --backend "$EXTRACT_BACKEND" \
    --report "${PROFILE_DIR}/profile_report.json" 2>&1 || {
    echo "WARN: Extract failed for $M_NAME, using existing kernels"
  }

  # [4/8] Auto-optimize top kernels (parallel with resource semaphores)
  echo ""
  echo "[4/8] Auto-optimizing kernels for $M_NAME..."

  KERNEL_TYPES=()
  if [[ -f "${MODEL_WS}/optimization_plan.json" ]]; then
    KERNEL_TYPES=($(uv run python -c "
import json
with open('${MODEL_WS}/optimization_plan.json') as f:
    plan = json.load(f)
kernels = plan.get('kernels_to_optimize', plan.get('kernels', []))
seen = set()
for k in kernels[:5]:
    op = k.get('op_type', '')
    if op and op not in seen:
        seen.add(op)
        print(op)
" 2>/dev/null))
  fi

  if [[ ${#KERNEL_TYPES[@]} -eq 0 ]]; then
    KERNEL_TYPES=(matmul flash_attention softmax rmsnorm elementwise)
  fi

  echo "  Kernel types: ${KERNEL_TYPES[*]}"

  running=0
  for kt in "${KERNEL_TYPES[@]}"; do
    ELAPSED=$(($(date +%s) - PIPELINE_START))
    if [[ $ELAPSED -ge $MAX_DURATION_S ]]; then
      echo "TIMEOUT: skipping remaining kernels for $M_NAME"
      break
    fi

    echo "  Optimizing: $kt"
    timeout "$OPT_TIMEOUT"s uv run python orchestrate.py --workspace "$MODEL_WS" auto \
      --kernel "$kt" \
      --llm-planner "$LLM_PLANNER" \
      --llm-coder "$LLM_CODER" \
      --iterations "$OPT_ITERATIONS" \
      --timeout "$OPT_TIMEOUT" &

    running=$((running + 1))
    PIPELINE_KERNELS=$((PIPELINE_KERNELS + 1))
    if [[ $running -ge $MAX_PARALLEL ]]; then
      wait -n 2>/dev/null || wait
      running=$((running - 1))
    fi
  done
  wait

  # [5/8] Verify end-to-end
  echo ""
  echo "[5/8] Verifying end-to-end for $M_NAME..."
  VERIFY_OUT="${MODEL_WS}/verification_$(date +%Y%m%d).json"
  uv run python verify.py \
    --model "$M_PATH" \
    --class-name "$M_CLASS" \
    --input-shape "$M_SHAPE" \
    --dtype "$M_DTYPE" \
    --workspace "$MODEL_WS" \
    --json "$VERIFY_OUT" 2>&1 || {
    echo "WARN: Verification issues for $M_NAME"
    PIPELINE_REGRESSIONS=$((PIPELINE_REGRESSIONS + 1))
  }

  # [6/8] CUDA migration for critical kernels
  echo ""
  echo "[6/8] CUDA migration for $M_NAME..."
  for kt in "${CUDA_KERNELS[@]}"; do
    if find "$MODEL_WS" -maxdepth 1 -name "kernel_${kt}_*optimized*.py" -print -quit 2>/dev/null | grep -q . ||
      [[ -f "${MODEL_WS}/kernel_${kt}_optimized.py" ]]; then
      echo "  Migrating $kt to CUDA..."
      uv run python orchestrate.py --workspace "$MODEL_WS" migrate-cuda \
        --kernel "$kt" 2>&1 || {
        echo "  WARN: CUDA migration failed for $kt"
      }
    fi
  done
done

# [7/8] Generate combined report + HTML dashboard
echo ""
echo "[7/8] Generating reports..."
uv run python orchestrate.py --workspace workspace report-extended >"workspace/nightly_report_$(date +%Y%m%d).md" 2>&1 || true

mkdir -p "$REPORT_DIR"
uv run python scripts/generate_dashboard.py \
  --workspace workspace \
  --config "$CONFIG" \
  --output "${REPORT_DIR}/index.html" 2>&1 || {
  echo "WARN: HTML dashboard generation failed"
}

# [8/8] Git commit + push if changes
echo ""
echo "[8/8] Git sync..."
if ! git diff --quiet docs/ 2>/dev/null || [[ -n "$(git ls-files --others --exclude-standard docs/ 2>/dev/null)" ]]; then
  git add docs/
  git commit -m "nightly: $(date +%Y%m%d) ${PIPELINE_MODELS} models, ${PIPELINE_KERNELS} kernels" || true
  git pull --rebase origin main 2>/dev/null || true
  git push origin main 2>&1 || echo "WARN: git push failed"
  echo "Changes committed and pushed."
else
  echo "No changes to commit."
fi

# Webhook notification
PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
if [[ -n "$WEBHOOK_URL" ]]; then
  echo ""
  echo "Sending webhook notification..."
  curl -s -H "Content-Type: application/json" \
    -d "{\"status\":\"${PIPELINE_STATUS}\",\"duration_s\":${PIPELINE_DURATION},\"models\":${PIPELINE_MODELS},\"kernels\":${PIPELINE_KERNELS},\"regressions\":${PIPELINE_REGRESSIONS},\"date\":\"$(date +%Y%m%d)\"}" \
    "$WEBHOOK_URL" 2>/dev/null || echo "WARN: webhook failed"
fi

echo ""
echo "=============================================="
echo "  Pipeline ${PIPELINE_STATUS}: $(date)"
echo "  Duration: $((PIPELINE_DURATION / 60))m"
echo "  Models: $PIPELINE_MODELS | Kernels: $PIPELINE_KERNELS | Regressions: $PIPELINE_REGRESSIONS"
echo "  Log: $LOG"
echo "=============================================="
