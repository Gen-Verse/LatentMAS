#!/usr/bin/env bash
# The first attempt at export_geo_profiles.py (from the GPU7 chain watcher)
# crashed in 3s: it passed 'en' as a target language, but 'en' is the
# anchor/pivot language, not a valid target (fixed in
# watch_and_launch_gpu7_chain.sh for future chains). This retries the
# corrected command once a full GPU is free (needs ~8-9GB for the 8-bit
# model; current per-GPU headroom is ~6-7GB, not enough to share).
set -u
cd "$(dirname "$0")/.."
LOG=logs/baselines/geo_profiles_watcher.log
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
mkdir -p logs/baselines
export LOG
log() { echo "[$(date -u +%FT%TZ)] [geo_profiles] $*" >> "$LOG"; }

try_claim_and_launch() (
  set -u
  FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) {print $1; exit}}')
  [ -z "$FREE_GPU" ] && return 1

  log "claimed gpu=$FREE_GPU"
  CUDA_VISIBLE_DEVICES=$FREE_GPU PYTHONPATH=src python scripts/export_geo_profiles.py \
    --model aisingapore/Llama-SEA-LION-v3-8B-IT \
    --languages th,my,km,lo,am,sw,bn,te \
    --n-samples 64 \
    --output results/mechanistic/geo_profiles.json \
    >> logs/baselines/export_geo_profiles.log 2>&1
  log "export_geo_profiles.py exit=$?"
  return 0
)

log "watching for 1 idle GPU (<500MiB used)"
while true; do
  if flock "$LOCK" bash -c "$(declare -f try_claim_and_launch log); try_claim_and_launch"; then
    break
  fi
  sleep 300
done
