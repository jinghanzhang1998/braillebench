#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

if [[ -f activate.sh ]]; then
  source activate.sh
elif [[ -f /scratch/jinghaz/braillebench/activate.sh ]]; then
  source /scratch/jinghaz/braillebench/activate.sh
fi

mkdir -p data/results/local logs

echo "== Qwen3-1.5B BrailleBench local inference =="
date
echo "Project: $(pwd)"

ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
GPU_COUNT="${BRAILLE_GPU_COUNT:-}"
if [[ -z "$GPU_COUNT" ]]; then
  GPU_COUNT="$(python - <<'PY'
try:
    import torch
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    count = 0
print(count or 1)
PY
)"
fi

if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || [[ "$GPU_COUNT" -lt 1 ]]; then
  GPU_COUNT=1
fi

DEVICE_IDS=()
if [[ -n "$ORIGINAL_CUDA_VISIBLE_DEVICES" ]]; then
  IFS=',' read -r -a DEVICE_IDS <<< "$ORIGINAL_CUDA_VISIBLE_DEVICES"
fi
if [[ "${#DEVICE_IDS[@]}" -lt "$GPU_COUNT" ]]; then
  DEVICE_IDS=()
  for worker in $(seq 0 $((GPU_COUNT - 1))); do
    DEVICE_IDS+=("${worker}")
  done
fi

python - <<'PY'
import os
import louis
import torch

print("python ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("louis", louis.translateString(["en-ueb-g2.ctb"], "Hello world"))
print("LOUIS_TABLEPATH", os.environ.get("LOUIS_TABLEPATH"))
PY
echo "GPU workers: ${GPU_COUNT}"
echo "Device assignment: ${DEVICE_IDS[*]}"

run_one_worker() {
  local worker="$1"
  local total="$2"
  local device_id="${DEVICE_IDS[$worker]}"
  local log_file="logs/qwen15b_worker_${worker}.log"

  {
    echo "== Worker ${worker}/${total} started =="
    date
    echo "CUDA_VISIBLE_DEVICES=${device_id}"
    CUDA_VISIBLE_DEVICES="${device_id}" python scripts/run_qwen15b_worker.py \
      --model qwen3-1.5b \
      --output-dir data/results/local \
      --num-workers "${total}" \
      --worker-index "${worker}"
    echo "== Worker ${worker}/${total} done =="
    date
  } 2>&1 | tee "${log_file}"
}

if [[ "$GPU_COUNT" -eq 1 ]]; then
  echo "== Single-GPU run =="
  run_one_worker 0 1
else
  echo "== Multi-GPU sharded run (${GPU_COUNT} workers) =="
  pids=()
  for worker in $(seq 0 $((GPU_COUNT - 1))); do
    run_one_worker "${worker}" "${GPU_COUNT}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one worker failed. Check logs/qwen15b_worker_*.log"
    exit 1
  fi
fi

echo "== Done =="
date
