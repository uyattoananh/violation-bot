#!/bin/bash
# Overnight AR pipeline: full scrape -> auto-seed (LLM-skipping paperwork) ->
# 100-photo eval -> shutdown PC with 5-min cancel window.
#
# Logs each stage to its own file so failure points are obvious when you're
# back at the PC.
#
# Cancel shutdown: run `shutdown /a` within 5 minutes of the final stage.

set -o pipefail

LOG_DIR="C:/Users/lang/Desktop/construct violation"
SCRAPE_LOG="$LOG_DIR/logs_ar_scrape.log"
SEED_LOG="$LOG_DIR/logs_ar_autoseed.log"
EVAL_LOG="$LOG_DIR/logs_ar_eval.log"
ORCH_LOG="$LOG_DIR/logs_orchestrator.log"

exec > "$ORCH_LOG" 2>&1

echo "=== [$(date)] STAGE 1: scrape AR up to 500 issues ==="
cd "C:/Users/lang/Desktop/construct bot"
echo "y" | .venv/Scripts/python.exe scripts/run_issues_scrape.py --project AR --max 500 \
    > "$SCRAPE_LOG" 2>&1
scrape_rc=$?
echo "scrape exit: $scrape_rc"
if [ $scrape_rc -ne 0 ]; then
    echo "Scrape failed, aborting remaining stages (but still shutting down)"
fi

echo
echo "=== [$(date)] STAGE 2: auto-seed AR into pgvector ==="
cd "C:/Users/lang/Desktop/construct violation"
.venv-webapp/Scripts/python.exe scripts/auto_seed_from_disk.py --project AR \
    > "$SEED_LOG" 2>&1
seed_rc=$?
echo "auto-seed exit: $seed_rc"

echo
echo "=== [$(date)] STAGE 3: 100-photo eval ==="
.venv-webapp/Scripts/python.exe scripts/evaluate_rag.py --n 100 --seed 42 \
    > "$EVAL_LOG" 2>&1
eval_rc=$?
echo "eval exit: $eval_rc"

echo
echo "=== [$(date)] STAGE 4: shutting down PC in 5 minutes ==="
echo "Run 'shutdown /a' to cancel."
shutdown /s /t 300 /c "Violation AI pipeline complete. Auto-shutdown in 5 min."
echo "=== pipeline done ==="
