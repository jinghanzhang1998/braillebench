#!/bin/bash
# After current tasks complete, run qwen3-32b G1-EN braille eval
# This script waits for current processes to finish, then starts qwen3-32b

cd /Users/jinghanz/research01/braille-benchmark
export LOUIS_TABLEPATH=/Users/jinghanz/local/share/liblouis/tables

echo "Waiting for current evaluate processes to finish..."
while pgrep -f "evaluate.py|evaluate_braille.py" > /dev/null 2>&1; do
    sleep 60
done

echo "Current tasks done. Starting qwen3-32b G1-EN..."

# First finish any remaining baseline
python3 src/run_full_eval.py > data/results/baseline_full_run13.log 2>&1

# Then run qwen3-32b G1-EN braille eval
python3 src/run_braille_eval.py --configs G1-EN --formats ascii --models qwen3-32b > data/results/braille_eval_qwen32b_g1en.log 2>&1

echo "Done!"
