#!/bin/bash
# RUVN scrape + auto-seed + eval.
set -o pipefail

LOG_DIR="C:/Users/lang/Desktop/construct violation"
cd "$LOG_DIR"

echo "=== [$(date)] STAGE 1: scrape RUVN ==="
cd "C:/Users/lang/Desktop/construct bot"
echo "y" | .venv/Scripts/python.exe scripts/run_issues_scrape.py \
    --project RUVN --max 500 \
    > "$LOG_DIR/logs_ruvn_scrape.log" 2>&1
echo "scrape exit: $?"

echo
echo "=== [$(date)] STAGE 2: auto-seed RUVN ==="
cd "$LOG_DIR"
.venv-webapp/Scripts/python.exe scripts/auto_seed_from_disk.py --project RUVN \
    > "$LOG_DIR/logs_ruvn_autoseed.log" 2>&1
echo "auto-seed exit: $?"

echo
echo "=== [$(date)] STAGE 3: 100-photo eval ==="
.venv-webapp/Scripts/python.exe scripts/evaluate_rag.py --n 100 --seed 42 \
    > "$LOG_DIR/logs_ruvn_eval.log" 2>&1
echo "eval exit: $?"

echo "=== [$(date)] pipeline done ==="
