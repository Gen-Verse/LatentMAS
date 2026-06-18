#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
LANGUAGES="${LANGUAGES:-bn,de,en,es,fr,ja,ru,sw,te,th,zh}"
DEVICE="${DEVICE:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
GENERATE_BS="${GENERATE_BS:-20}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-mgsm_baseline_all_target_lang}"
OUT_ROOT="${OUT_ROOT:-results/mgsm_baseline}"

MODEL_SAFE="${MODEL//\//_}"
OUT_DIR="${OUT_ROOT}/${MODEL_SAFE}/${RUN_NAME}"
SUMMARY_JSONL="${OUT_DIR}/summary.jsonl"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${OUT_DIR}" logs

echo "================ MGSM single-agent baseline ================"
echo "  model      : ${MODEL}"
echo "  languages  : ${LANGUAGES}"
echo "  device     : ${DEVICE}"
echo "  max samples: ${MAX_SAMPLES}"
echo "  max tokens : ${MAX_NEW_TOKENS}"
echo "  batch size : ${GENERATE_BS}"
echo "  run_name   : ${RUN_NAME}"
echo "  out_dir    : ${OUT_DIR}"
echo "============================================================"

if [ ! -f "${SUMMARY_CSV}" ]; then
  printf "lang,status,method,model,split,seed,max_samples,accuracy,correct,total_time_sec,time_per_sample_sec,log_path\n" > "${SUMMARY_CSV}"
fi

IFS=',' read -r -a LANG_ARRAY <<< "${LANGUAGES}"

for L in "${LANG_ARRAY[@]}"; do
  LOG_PATH="${OUT_DIR}/${L}.log"
  echo "=== ${L} ==="

  set +e
  python run.py \
    --method baseline \
    --model_name "${MODEL}" \
    --task mgsm \
    --mgsm_lang "${L}" \
    --split test \
    --device "${DEVICE}" \
    --max_samples "${MAX_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --generate_bs "${GENERATE_BS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_PATH}"
  STATUS="${PIPESTATUS[0]}"
  set -e

  python - "${LOG_PATH}" "${L}" "${STATUS}" "${SUMMARY_JSONL}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
lang = sys.argv[2]
status = int(sys.argv[3])
summary_jsonl = Path(sys.argv[4])
summary_csv = Path(sys.argv[5])

result = None
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if obj.get("method") == "baseline":
        result = obj

if result is None:
    result = {
        "method": "baseline",
        "model": "",
        "split": "",
        "seed": "",
        "max_samples": "",
        "accuracy": "",
        "correct": "",
        "total_time_sec": "",
        "time_per_sample_sec": "",
    }

row = {
    "lang": lang,
    "status": "ok" if status == 0 else f"exit_{status}",
    "method": result.get("method", "baseline"),
    "model": result.get("model", ""),
    "split": result.get("split", ""),
    "seed": result.get("seed", ""),
    "max_samples": result.get("max_samples", ""),
    "accuracy": result.get("accuracy", ""),
    "correct": result.get("correct", ""),
    "total_time_sec": result.get("total_time_sec", ""),
    "time_per_sample_sec": result.get("time_per_sample_sec", ""),
    "log_path": str(log_path),
}

with summary_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")

with summary_csv.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writerow(row)

print("[summary]", json.dumps(row, ensure_ascii=False))
PY

  if [ "${STATUS}" -ne 0 ]; then
    echo "[error] ${L} failed with status ${STATUS}; see ${LOG_PATH}"
    exit "${STATUS}"
  fi
done

echo "[OK] wrote ${OUT_DIR}"
