#!/bin/bash
# Full visual-seed AR + eval in sequence. Logs each stage.
set -o pipefail
cd "C:/Users/lang/Desktop/construct violation"

echo "=== [$(date)] STAGE 1: visual-seed AR ==="
.venv-webapp/Scripts/python.exe scripts/visual_seed_from_disk.py \
    --project AR --threshold 0.6 \
    > logs_ar_visualseed.log 2>&1
echo "visual-seed exit: $?"

echo
echo "=== [$(date)] STAGE 2: 100-photo eval ==="
.venv-webapp/Scripts/python.exe scripts/evaluate_rag.py --n 100 --seed 42 \
    > logs_ar_visualeval.log 2>&1
echo "eval exit: $?"

echo "=== [$(date)] pipeline done ==="
