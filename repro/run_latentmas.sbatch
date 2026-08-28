#!/bin/bash
#SBATCH --job-name=latentmas
#SBATCH --account=ynagaraj
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=repro/logs/%x-%j.out
#SBATCH --error=repro/logs/%x-%j.err
# free-* partitions preempt. --requeue puts the job straight back in the queue and
# --checkpoint (below) makes the restart resume instead of redoing 3 hours of work.
# append mode so a requeued attempt does not truncate the previous attempt's log.
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Usage:
#   sbatch --export=ALL,METHOD=latent_mas,MODEL=Qwen/Qwen3-14B,TASK=mbppplus,\
# PROMPT=sequential,LSTEPS=0,NSAMP=-1,BS=20 repro/run_latentmas.sbatch
#
# GPU selection cheat sheet (bf16 is HARDCODED in models.py -> never use V100/sm_70):
#   Qwen3-4B  (~8 GB)  : --partition=free-gpu   --gres=gpu:A30:1
#   Qwen3-8B  (~16 GB) : --partition=free-gpu   --gres=gpu:A30:1     (tight) or L40S
#   Qwen3-14B (~29 GB) : --partition=free-gpu32 --gres=gpu:L40S:1 | gpu:RTX6000:1
#                        or --partition=free-gpu --gres=gpu:A100:1
# free-* partitions are preemptible. Swap to gpu / gpu32 (billed) for long runs.

set -euo pipefail

METHOD=${METHOD:-latent_mas}
MODEL=${MODEL:-Qwen/Qwen3-14B}
TASK=${TASK:-mbppplus}
PROMPT=${PROMPT:-sequential}
LSTEPS=${LSTEPS:-0}
NSAMP=${NSAMP:--1}
BS=${BS:-15}
MAXNEW=${MAXNEW:-4096}
REALIGN=${REALIGN:-0}
# --think appends a literal <think> to the assistant turn. BOTH released reference
# logs were produced with think:True, so this defaults ON to match the paper.
THINK=${THINK:-1}
# Preemption on this cluster is CANCEL, not requeue: the free-gpu32 L40S nodes
# (k54-06/07) are also in the billed gpu32 partition, so a paying job evicts us
# with "CANCELLED ... DUE to SIGNAL Terminated" and --requeue never fires. Four
# jobs died that way on 2026-08-28. So the script resubmits ITSELF on TERM and on
# any non-zero exit; --checkpoint makes each new attempt resume where it stopped.
CHAIN=${CHAIN:-1}
CHAIN_N=${CHAIN_N:-0}
CHAIN_MAX=${CHAIN_MAX:-20}

export HF_HOME=/pub/$USER/hf
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}   # cache still serves hits; offline=1 makes any id mismatch fatal
export HF_DATASETS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
# ~/.local/lib/python3.10/site-packages leaks into every python3.10 env on this
# account and shadows torch's CUDA libs. Never let it onto sys.path.
export PYTHONNOUSERSITE=1
# latent_mas OOM'd on a 24 GB A30 at bs=15 with 4.33 GiB reserved-but-unallocated.
# The judger inherits the concatenated KV of all prior agents, so peak memory is
# much higher than baseline/text_mas at the same batch size.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Use the env interpreter directly. `conda activate` proved fragile here: five
# jobs died instantly with "EnvironmentLocationNotFound" while the env dir was
# being rewritten. An absolute path has no such race.
PY_BIN=/pub/$USER/envs/latentmas/bin/python
[ -x "$PY_BIN" ] || { echo "FATAL: $PY_BIN missing"; exit 1; }
export PATH=/pub/$USER/envs/latentmas/bin:$PATH

cd "$SLURM_SUBMIT_DIR"
mkdir -p repro/logs repro/results

TAG="${METHOD}_$(basename "$MODEL")_${TASK}_${PROMPT}_ls${LSTEPS}_bs${BS}_think${THINK}"
OUT="repro/results/${TAG}_${SLURM_JOB_ID}.log"
# Checkpoint path is keyed on the CONFIG, not the job id, so a requeued attempt
# (which keeps its job id) and a hand-resubmitted job both pick the work back up.
CKPT="repro/results/${TAG}.ckpt.jsonl"

ckpt_lines() { [ -f "$CKPT" ] && wc -l < "$CKPT" || echo 0; }

resubmit() {
  local why="$1"
  if [ "$CHAIN" != "1" ]; then echo "[chain] $why; chaining disabled"; return; fi
  if [ "$CHAIN_N" -ge "$CHAIN_MAX" ]; then echo "[chain] $why; hit CHAIN_MAX=$CHAIN_MAX"; return; fi
  # A crash that solved nothing will crash again. Only chain if this attempt made
  # progress, otherwise 20 attempts just replay the same traceback into the queue.
  if [ "$(ckpt_lines)" -le "$START_LINES" ]; then
    echo "[chain] $why; but checkpoint still at $START_LINES problems -- no progress, not chaining"
    return
  fi
  local next=$((CHAIN_N + 1))
  echo "[chain] $why; resubmitting as attempt $next/$CHAIN_MAX"
  sbatch --partition="$SLURM_JOB_PARTITION" --gres=gpu:${GPUMODEL}:1 \
    --mem="${JOBMEM}" --time="${JOBTIME}" \
    --export=ALL,METHOD=$METHOD,MODEL=$MODEL,TASK=$TASK,PROMPT=$PROMPT,LSTEPS=$LSTEPS,\
NSAMP=$NSAMP,BS=$BS,MAXNEW=$MAXNEW,REALIGN=$REALIGN,THINK=$THINK,\
CHAIN=$CHAIN,CHAIN_N=$next,CHAIN_MAX=$CHAIN_MAX,GPUMODEL=$GPUMODEL,JOBMEM=$JOBMEM,JOBTIME=$JOBTIME \
    repro/run_latentmas.sbatch
}

# GPU model is pinned across every arm on purpose: wall-clock is half the paper's
# claim, so a comparison that straddles L40S and A30 measures the cluster, not the method.
GPUMODEL=${GPUMODEL:-L40S}
JOBMEM=${JOBMEM:-32G}
JOBTIME=${JOBTIME:-12:00:00}

# Preemption arrives as SIGTERM to this script. Chain first, then die.
trap 'resubmit "caught SIGTERM (preemption)"; exit 143' TERM

EXTRA=()
[ "$REALIGN" = "1" ] && EXTRA+=(--latent_space_realign)
[ "$THINK" = "1" ] && EXTRA+=(--think)

START_LINES=$(ckpt_lines)

nvidia-smi
echo "=== $TAG (attempt ${SLURM_RESTART_COUNT:-0}) ==="
if [ -f "$CKPT" ]; then echo "resuming from $(wc -l < "$CKPT") completed problems in $CKPT"; fi

set +e
"$PY_BIN" -u run.py \
  --method "$METHOD" \
  --model_name "$MODEL" \
  --task "$TASK" \
  --prompt "$PROMPT" \
  --latent_steps "$LSTEPS" \
  --max_samples "$NSAMP" \
  --max_new_tokens "$MAXNEW" \
  --generate_bs "$BS" \
  --device cuda:0 \
  --checkpoint "$CKPT" \
  "${EXTRA[@]}" 2>&1 | tee -a "$OUT"
rc=${PIPESTATUS[0]}
set -e

echo "--- final line ---"
tail -1 "$OUT"
if [ "$rc" -ne 0 ]; then
  resubmit "run.py exited $rc"
else
  echo "[chain] completed cleanly after $CHAIN_N resubmission(s)"
fi
exit "$rc"
