#!/usr/bin/env bash
# Post-fix baseline rerun queue (see results/baselines/README_INVALID.md).
# Runs each requested baseline sequentially on one GPU, nohup-safe and
# IDEMPOTENT: a run is skipped if a post-fix result JSON already exists for
# its (method, benchmark, language) — pre-fix JSONs live in
# pre_fix_prompt_chain/ and don't match. Safe to relaunch after a crash or
# agent disconnect; launch with:
#   setsid nohup bash scripts/rerun_baselines_queue.sh <gpu_id> <queue_name> \
#     <method:benchmark:language> [...] >> logs/baselines/<queue_name>.log 2>&1 &
#
# method ∈ {latentmas, thoughtcomm}; e.g. latentmas:mgsm:en latentmas:belebele:tha_Thai
set -u
GPU=$1
QUEUE=$2
shift 2

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/baselines

log() { echo "[$(date -u +%FT%TZ)] [$QUEUE] $*"; }

log "queue start gpu=$GPU specs=$*"
RC_ALL=0
for SPEC in "$@"; do
  IFS=: read -r METHOD BENCH LANG <<< "$SPEC"
  MODULE="latent_coordination.baselines.run_${METHOD}"
  OUT_DIR="results/baselines/${METHOD}"
  if compgen -G "${OUT_DIR}/${METHOD}_${BENCH}_${LANG}_*.json" > /dev/null; then
    log "SKIP $SPEC (post-fix result already exists)"
    continue
  fi
  log "START $SPEC"
  nohup python -m "$MODULE" \
    --benchmark "$BENCH" --language "$LANG" --n 200 \
    --device cuda:0 --load_in_8bit --output_dir "$OUT_DIR"
  RC=$?
  log "END $SPEC exit=$RC"
  [ "$RC" -ne 0 ] && RC_ALL=1
done
log "queue done rc=$RC_ALL"
exit $RC_ALL
