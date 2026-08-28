#!/bin/bash
# Three-arm MBPP+ comparison, Qwen3-4B, bs=15, think=1, max_new_tokens=4096.
#
# Partition: free-gpu32. The billed gpu/gpu32 partitions are NOT usable by this
# account -- slurm_job_submit rejects them with "without specifying a GPU Account",
# and `sacctmgr show assoc user=ynagaraj` lists only the plain `ynagaraj` account.
# So preemption is unavoidable; run.py --checkpoint + #SBATCH --requeue is what
# makes it survivable. Jobs 55644121/55644122 previously lost 2.5 GPU-hours to it.
#
# All arms pinned to gpu:L40S so wall-clock is comparable across them (the paper's
# speed claim is half its thesis, so arms must not straddle GPU models).
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON=(--partition=free-gpu32 --gres=gpu:L40S:1 --mem=32G --time=12:00:00)
BASE="MODEL=Qwen/Qwen3-4B,TASK=mbppplus,PROMPT=sequential,NSAMP=-1,BS=15"

# baseline               -- single agent, the arm LatentMAS must beat on cost
sbatch "${COMMON[@]}" --export=ALL,$BASE,METHOD=baseline,LSTEPS=0    repro/run_latentmas.sbatch
# latent_mas ls=0        -- the released-log config; degenerate, judger gets past_kv=None
sbatch "${COMMON[@]}" --export=ALL,$BASE,METHOD=latent_mas,LSTEPS=0  repro/run_latentmas.sbatch
# latent_mas ls=10       -- the latent channel actually switched on
sbatch "${COMMON[@]}" --export=ALL,$BASE,METHOD=latent_mas,LSTEPS=10 repro/run_latentmas.sbatch
