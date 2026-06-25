#!/usr/bin/env bash
# Smoke test: runs all three pipelines end-to-end with minimal data to verify all code paths.
#
# Usage:
#   bash scripts/smoke_test.sh                          # default: Llama-SEA-LION on cuda:0
#   bash scripts/smoke_test.sh --model SeaLLMs/SeaLLMs-v3-7B-Chat
#   bash scripts/smoke_test.sh --model sail/Sailor2-8B-Chat --device cuda:1
#   bash scripts/smoke_test.sh --mechanistic-only
#   bash scripts/smoke_test.sh --coordination-only
#   bash scripts/smoke_test.sh --surgical-only
#   bash scripts/smoke_test.sh --dry-run     # import check only, no inference
#
# Expected runtime: 30–60 min on 1× V100-16GB with Llama-SEA-LION-v3-8B-IT.
# The model is loaded once per pipeline (not per stage), so model-load overhead (~5 min)
# is paid twice: once for Mechanistic Disentanglement, once for Latent Coordination.
#
# Outputs (checkpointed — use --resume to restart from last stage):
#   results/smoketest/mechanistic/<timestamp>/   — Mechanistic Disentanglement stage outputs and plots
#   results/smoketest/coordination/<timestamp>/  — Latent Coordination stage outputs and plots
#   .cache/checkpoints/smoketest_acl/    — Mechanistic Disentanglement stage checkpoints
#   .cache/checkpoints/smoketest_aaai/   — Latent Coordination stage checkpoints
#   logs/smoketest/                      — plain-text logs for both pipelines

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL="aisingapore/Llama-SEA-LION-v3-8B-IT"
DEVICE="cuda:0"
DRY_RUN=""
RUN_MECHANISTIC=true
RUN_COORDINATION=true
RUN_SURGICAL=true
RESUME=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";   shift 2 ;;
        --device)     DEVICE="$2";  shift 2 ;;
        --dry-run)    DRY_RUN="--dry-run"; shift ;;
        --mechanistic-only)   RUN_COORDINATION=false; RUN_SURGICAL=false; shift ;;
        --coordination-only)  RUN_MECHANISTIC=false; RUN_SURGICAL=false; shift ;;
        --surgical-only)      RUN_MECHANISTIC=false; RUN_COORDINATION=false; shift ;;
        --resume)     RESUME="--resume"; shift ;;
        *)            echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "============================================================"
echo "  Smoke Test — $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Model  : $MODEL"
echo "  Device : $DEVICE"
echo "  Repo   : $REPO_ROOT"
echo "============================================================"

# Validate GPU is accessible
if [[ "$DEVICE" != "cpu" ]]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; \
               d=int('$DEVICE'.replace('cuda:','') or 0); \
               print(f'GPU {d}: {torch.cuda.get_device_name(d)}, ' \
                     f'{torch.cuda.get_device_properties(d).total_memory // 1024**3} GB')" || {
        echo "[WARN] Could not verify GPU — continuing anyway"
    }
fi

PASS=0
FAIL=0
ERRORS=()

# ---------------------------------------------------------------------------
# Mechanistic Disentanglement Pipeline
# ---------------------------------------------------------------------------
if $RUN_MECHANISTIC; then
    echo ""
    echo "---- Mechanistic Disentanglement Pipeline (Stages A–H) ----"
    echo "Config : configs/mechanistic_smoketest.yaml"
    echo "Stages : A=Lexicon+Data, B=Activations, C=SVD, D=Isomorphism,"
    echo "         E=SteeringVectors, F=Benchmark, G=Ablation, H=Viz+Report"
    echo ""

    MECH_START=$(date +%s)

    python -m scripts.run_mechanistic_pipeline \
        --config configs/mechanistic_smoketest.yaml \
        --model "$MODEL" \
        --device "$DEVICE" \
        $DRY_RUN \
        $RESUME \
        && MECH_EXIT=0 || MECH_EXIT=$?

    MECH_END=$(date +%s)
    MECH_ELAPSED=$(( MECH_END - MECH_START ))

    if [[ $MECH_EXIT -eq 0 ]]; then
        echo "[OK]  Mechanistic Disentanglement pipeline completed in ${MECH_ELAPSED}s"
        PASS=$(( PASS + 1 ))
    else
        echo "[FAIL] Mechanistic Disentanglement pipeline failed (exit $MECH_EXIT) after ${MECH_ELAPSED}s"
        FAIL=$(( FAIL + 1 ))
        ERRORS+=("Mechanistic Disentanglement pipeline exited $MECH_EXIT")
    fi
fi

# ---------------------------------------------------------------------------
# Latent Coordination Pipeline
# ---------------------------------------------------------------------------
if $RUN_COORDINATION; then
    echo ""
    echo "---- Latent Coordination Pipeline (Stages A–H) ----"
    echo "Config : configs/coordination_smoketest.yaml"
    echo "Stages : A=AdapterTraining, B=CVAETraining, C=IntentCentroids,"
    echo "         D=Benchmark, E=CommModeAblation, F=Scalability,"
    echo "         G=SafetyEval, H=Viz+Report"
    echo ""

    COORD_START=$(date +%s)

    python -m scripts.run_coordination_pipeline \
        --config configs/coordination_smoketest.yaml \
        --agents "orchestrator.device=$DEVICE,translation_agent.device=$DEVICE,reasoning_agent.device=$DEVICE,safety_agent.device=$DEVICE" \
        $DRY_RUN \
        $RESUME \
        && COORD_EXIT=0 || COORD_EXIT=$?

    COORD_END=$(date +%s)
    COORD_ELAPSED=$(( COORD_END - COORD_START ))

    if [[ $COORD_EXIT -eq 0 ]]; then
        echo "[OK]  Latent Coordination pipeline completed in ${COORD_ELAPSED}s"
        PASS=$(( PASS + 1 ))
    else
        echo "[FAIL] Latent Coordination pipeline failed (exit $COORD_EXIT) after ${COORD_ELAPSED}s"
        FAIL=$(( FAIL + 1 ))
        ERRORS+=("Latent Coordination pipeline exited $COORD_EXIT")
    fi
fi

# ---------------------------------------------------------------------------
# Surgical MRRE Pipeline
# ---------------------------------------------------------------------------
if $RUN_SURGICAL; then
    echo ""
    echo "---- Surgical MRRE Pipeline (Stages A–D) ----"
    echo "Config : configs/surgical_smoketest.yaml"
    echo "Stages : A=HiddenStateMapping, B=FitSurgicalMRRE,"
    echo "         C=IFLEval(baseline vs steered + DSL), D=Report"
    echo ""

    SURG_START=$(date +%s)

    python -m scripts.run_surgical_pipeline \
        --config configs/surgical_smoketest.yaml \
        --model "$MODEL" \
        --device "$DEVICE" \
        $DRY_RUN \
        $RESUME \
        && SURG_EXIT=0 || SURG_EXIT=$?

    SURG_END=$(date +%s)
    SURG_ELAPSED=$(( SURG_END - SURG_START ))

    if [[ $SURG_EXIT -eq 0 ]]; then
        echo "[OK]  Surgical MRRE pipeline completed in ${SURG_ELAPSED}s"
        PASS=$(( PASS + 1 ))
    else
        echo "[FAIL] Surgical MRRE pipeline failed (exit $SURG_EXIT) after ${SURG_ELAPSED}s"
        FAIL=$(( FAIL + 1 ))
        ERRORS+=("Surgical MRRE pipeline exited $SURG_EXIT")
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Smoke Test Summary"
echo "  Passed : $PASS"
echo "  Failed : $FAIL"
if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo "  Errors :"
    for e in "${ERRORS[@]}"; do
        echo "    - $e"
    done
fi
echo "============================================================"

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
