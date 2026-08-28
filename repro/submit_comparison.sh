#!/bin/bash
# The paper's claim is COMPARATIVE (same accuracy, far fewer tokens, much faster),
# so LatentMAS alone proves nothing. This submits all three arms on identical data.
#
# Defaults reproduce the released-log hyperparameters (think=1, max_new_tokens=4096,
# temp 0.6, top_p 0.95, seed 42) on a GSM8K subset with Qwen3-4B.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-4B}
TASK=${TASK:-gsm8k}
PROMPT=${PROMPT:-sequential}
NSAMP=${NSAMP:-120}
BS=${BS:-15}
LSTEPS=${LSTEPS:-10}
PART=${PART:-free-gpu}
GRES=${GRES:-gpu:A30:1}

sub () {  # sub <method> <lsteps> <timelimit>
  sbatch --partition="$PART" --gres="$GRES" --time="$3" \
    --export=ALL,MODEL=$MODEL,TASK=$TASK,METHOD=$1,PROMPT=$PROMPT,LSTEPS=$2,NSAMP=$NSAMP,BS=$BS,MAXNEW=4096,THINK=1 \
    repro/run_latentmas.sbatch
}

# latent_mas: only the judger generates -> fastest
sub latent_mas "$LSTEPS" 4:00:00
# baseline: one generation, no agents
sub baseline   0          4:00:00
# text_mas: all four agents generate -> ~4x the wall clock
sub text_mas   0          8:00:00

echo
squeue -u "$USER" -o "%.10i %.12P %.24j %.8T %.10M %.6D %R"
