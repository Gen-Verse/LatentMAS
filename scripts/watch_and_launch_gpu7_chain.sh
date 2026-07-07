#!/usr/bin/env bash
# Waits for the current GPU7 job (PID $1, LatentMAS Belebele SEA-LION rerun)
# to finish, then runs a short GPU7 chain:
#   1. export_geo_profiles.py -- cheap prerequisite artifact that unblocks
#      staircase-ablation rows 3-6 (they fail fast without it).
#   2. ThoughtComm Belebele baseline queue (9 languages) -- fills the
#      ThoughtComm reference-point gap in tab:latentmas (paper 1).
# Idempotent: export_geo_profiles.py overwrite is harmless to rerun, and
# rerun_baselines_queue.sh skips specs whose result already exists.
set -u
WAIT_PID=$1
cd "$(dirname "$0")/.."
LOG=logs/baselines/gpu7_chain_watcher.log
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

log "watching pid=$WAIT_PID (GPU7 latentmas belebele SEA-LION rerun)"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
log "pid=$WAIT_PID gone -- GPU7 free"

log "step 1/2: export_geo_profiles.py (unblocks staircase rows 3-6)"
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python scripts/export_geo_profiles.py \
  --model aisingapore/Llama-SEA-LION-v3-8B-IT \
  --languages th,my,km,lo,am,sw,bn,te \
  --n-samples 64 \
  --output results/mechanistic/geo_profiles.json \
  >> logs/baselines/export_geo_profiles.log 2>&1
RC=$?
log "step 1/2 done exit=$RC"

log "step 2/2: launching ThoughtComm Belebele queue on GPU7"
MODEL="aisingapore/Llama-SEA-LION-v3-8B-IT"
setsid nohup bash scripts/rerun_baselines_queue.sh 7 gpu7_thoughtcomm_belebele \
  "thoughtcomm:belebele:eng_Latn:$MODEL" \
  "thoughtcomm:belebele:tha_Thai:$MODEL" \
  "thoughtcomm:belebele:mya_Mymr:$MODEL" \
  "thoughtcomm:belebele:khm_Khmr:$MODEL" \
  "thoughtcomm:belebele:lao_Laoo:$MODEL" \
  "thoughtcomm:belebele:amh_Ethi:$MODEL" \
  "thoughtcomm:belebele:swh_Latn:$MODEL" \
  "thoughtcomm:belebele:ben_Beng:$MODEL" \
  "thoughtcomm:belebele:tel_Telu:$MODEL" \
  >> logs/baselines/rerun_gpu7_thoughtcomm_belebele.log 2>&1 < /dev/null &
disown -a
log "launched thoughtcomm belebele queue"
