#!/bin/bash
set -o pipefail
LOG="C:/Users/lang/Desktop/construct violation"

echo "=== [$(date)] STAGE 1: scrape H9 ==="
cd "C:/Users/lang/Desktop/construct bot"
echo "y" | .venv/Scripts/python.exe scripts/run_issues_scrape.py --project H9 --max 500 \
    > "$LOG/logs_h9_scrape.log" 2>&1
echo "scrape exit: $?"

echo
echo "=== [$(date)] STAGE 2: auto-seed H9 (DTag only, FK-safe) ==="
cd "$LOG"
.venv-webapp/Scripts/python.exe scripts/auto_seed_from_disk.py --project H9 --no-titles \
    > "$LOG/logs_h9_autoseed.log" 2>&1
echo "auto-seed exit: $?"

echo
echo "=== [$(date)] STAGE 3: 100-photo eval ==="
.venv-webapp/Scripts/python.exe scripts/evaluate_rag.py --n 100 --seed 42 \
    > "$LOG/logs_h9_eval.log" 2>&1
echo "eval exit: $?"
echo "=== [$(date)] done ==="
