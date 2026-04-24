#!/bin/bash
# Wait for orphan RUVN scrape to finish, then auto-seed DTag-only + eval.
set -o pipefail
LOG="C:/Users/lang/Desktop/construct violation"

# Poll until scrape log shows "Done." (scrape completed naturally)
echo "[$(date)] waiting for scrape to finish..."
while true; do
    if grep -aq "INFO    Done\. issues=" "$LOG/logs_ruvn_scrape.log" 2>/dev/null; then
        echo "[$(date)] scrape done"
        break
    fi
    sleep 15
done

cd "$LOG"
echo
echo "=== [$(date)] STAGE: auto-seed RUVN (DTag only) ==="
.venv-webapp/Scripts/python.exe scripts/auto_seed_from_disk.py \
    --project RUVN --no-titles \
    > "$LOG/logs_ruvn_autoseed_dtag.log" 2>&1
echo "auto-seed exit: $?"

echo
echo "=== [$(date)] STAGE: 100-photo eval ==="
.venv-webapp/Scripts/python.exe scripts/evaluate_rag.py --n 100 --seed 42 \
    > "$LOG/logs_ruvn_eval_dtag.log" 2>&1
echo "eval exit: $?"

echo "=== [$(date)] done ==="
