#!/usr/bin/env bash
set -euo pipefail

# Compare the original-style prompts on MGSM:
#   - target-language MGSM questions via --mgsm_lang
#   - English/original prompt templates via --plain_prompts
#   - no translated role prompts, no target-language directive, no localized think prefill

MODEL="${MODEL:-Qwen/Qwen3-4B}"
LANGUAGES="${LANGUAGES:-bn,de,en,es,fr,ja,ru,sw,te,th,zh}"
METHODS="${METHODS:-baseline,text_mas,latent_mas}"
PROMPT="${PROMPT:-sequential}"
DEVICE="${DEVICE:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GENERATE_BS="${GENERATE_BS:-1}"
LATENT_STEPS="${LATENT_STEPS:-3}"
TEXT_MAS_CONTEXT_LENGTH="${TEXT_MAS_CONTEXT_LENGTH:--1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-mgsm_first${MAX_SAMPLES}_plain_original_compare}"
OUT_ROOT="${OUT_ROOT:-results/mgsm_plain_original_compare}"

MODEL_SAFE="${MODEL//\//_}"
OUT_DIR="${OUT_ROOT}/${MODEL_SAFE}/${RUN_NAME}"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
SUMMARY_JSONL="${OUT_DIR}/summary.jsonl"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${OUT_DIR}" logs

echo "================ MGSM plain/original comparison ================"
echo "  model      : ${MODEL}"
echo "  languages  : ${LANGUAGES}"
echo "  methods    : ${METHODS}"
echo "  prompt     : ${PROMPT}"
echo "  device     : ${DEVICE}"
echo "  max samples: ${MAX_SAMPLES}"
echo "  max tokens : ${MAX_NEW_TOKENS}"
echo "  run_name   : ${RUN_NAME}"
echo "  out_dir    : ${OUT_DIR}"
echo "================================================================"

printf "lang,method,status,accuracy,correct,max_samples,total_time_sec,time_per_sample_sec,log_path\n" > "${SUMMARY_CSV}"
: > "${SUMMARY_JSONL}"

IFS=',' read -r -a LANG_ARRAY <<< "${LANGUAGES}"
IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"

for METHOD in "${METHOD_ARRAY[@]}"; do
  for L in "${LANG_ARRAY[@]}"; do
    LOG_PATH="${OUT_DIR}/${METHOD}_${L}.log"
    echo "=== ${METHOD} ${L} ==="

    CMD=(
      python run.py
      --method "${METHOD}"
      --model_name "${MODEL}"
      --task mgsm
      --mgsm_lang "${L}"
      --split test
      --prompt "${PROMPT}"
      --device "${DEVICE}"
      --max_samples "${MAX_SAMPLES}"
      --generate_bs "${GENERATE_BS}"
      --max_new_tokens "${MAX_NEW_TOKENS}"
      --temperature "${TEMPERATURE}"
      --top_p "${TOP_P}"
      --seed "${SEED}"
      --plain_prompts
    )

    if [[ "${METHOD}" == "latent_mas" ]]; then
      CMD+=(--latent_steps "${LATENT_STEPS}")
    fi

    if [[ "${METHOD}" == "text_mas" ]]; then
      CMD+=(--text_mas_context_length "${TEXT_MAS_CONTEXT_LENGTH}")
    fi

    set +e
    "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
    STATUS="${PIPESTATUS[0]}"
    set -e

    python - "${LOG_PATH}" "${L}" "${METHOD}" "${STATUS}" "${SUMMARY_CSV}" "${SUMMARY_JSONL}" <<'PY'
import csv
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
lang = sys.argv[2]
method = sys.argv[3]
status = int(sys.argv[4])
summary_csv = Path(sys.argv[5])
summary_jsonl = Path(sys.argv[6])

result = None
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if obj.get("method") == method:
        result = obj

result = result or {}
row = {
    "lang": lang,
    "method": method,
    "status": "ok" if status == 0 else f"exit_{status}",
    "accuracy": result.get("accuracy", ""),
    "correct": result.get("correct", ""),
    "max_samples": result.get("max_samples", ""),
    "total_time_sec": result.get("total_time_sec", ""),
    "time_per_sample_sec": result.get("time_per_sample_sec", ""),
    "log_path": str(log_path),
}

with summary_csv.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writerow(row)

with summary_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
  done
done

echo "[OK] wrote ${OUT_DIR}"
