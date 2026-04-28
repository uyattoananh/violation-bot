#!/usr/bin/env bash
# Clean restart of webapp + worker on a Linux VPS.
# Idempotent: safe to run repeatedly. Always leaves exactly one of each.
#
# Usage:
#   cd ~/violation-bot
#   bash scripts/restart-vps.sh
#
# Optional env vars:
#   PORT     (default 8765)
#   HOST     (default 0.0.0.0)
#   APP_DIR  (default $PWD)
#   VENV     (default $APP_DIR/.venv-webapp)

set -euo pipefail

PORT="${PORT:-8765}"
HOST="${HOST:-0.0.0.0}"
APP_DIR="${APP_DIR:-$(pwd)}"
VENV="${VENV:-$APP_DIR/.venv-webapp}"
LOG_DIR="$APP_DIR/logs"

echo "==> APP_DIR = $APP_DIR"
echo "==> VENV    = $VENV"
echo "==> PORT    = $PORT"

if [ ! -d "$VENV" ]; then
  echo "ERROR: venv not found at $VENV — run BLOCK 2 first." >&2
  exit 1
fi
if [ ! -f "$APP_DIR/.env" ]; then
  echo "ERROR: .env not found at $APP_DIR/.env — run BLOCK 3 first." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

# 1. Kill everything bound to the port + any orphan workers.
echo "==> stopping any existing instances..."
fuser -k "${PORT}/tcp" 2>/dev/null || true
pkill -f "uvicorn webapp" 2>/dev/null || true
pkill -f "webapp.worker" 2>/dev/null || true
sleep 1

# 2. Confirm port is free.
if ss -tlnp 2>/dev/null | grep -q ":${PORT}\\b"; then
  echo "ERROR: port $PORT still occupied:" >&2
  ss -tlnp | grep ":${PORT}\\b" >&2
  exit 1
fi
echo "==> port $PORT is clear"

# 3. Start webapp.
cd "$APP_DIR"
PYTHON="$VENV/bin/python"
nohup "$PYTHON" -m uvicorn webapp.app:app --host "$HOST" --port "$PORT" \
    > "$LOG_DIR/webapp.log" 2>&1 &
echo $! > "$LOG_DIR/webapp.pid"
echo "==> started webapp pid=$(cat $LOG_DIR/webapp.pid)"

# 4. Start worker.
nohup "$PYTHON" -m webapp.worker \
    > "$LOG_DIR/worker.log" 2>&1 &
echo $! > "$LOG_DIR/worker.pid"
echo "==> started worker  pid=$(cat $LOG_DIR/worker.pid)"

# 5. Health check.
echo "==> waiting for /healthz..."
for i in 1 2 3 4 5 6 7 8; do
  if curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null; then
    echo "==> healthy ✓"
    break
  fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null; then
  echo "==> health check FAILED. Last 30 lines of webapp.log:"
  tail -30 "$LOG_DIR/webapp.log"
  exit 1
fi

# 6. Show what's running.
echo "==> running processes:"
ps -o pid=,cmd= -p "$(cat $LOG_DIR/webapp.pid)" "$(cat $LOG_DIR/worker.pid)" 2>/dev/null || true

echo ""
echo "Logs:"
echo "  tail -f $LOG_DIR/webapp.log"
echo "  tail -f $LOG_DIR/worker.log"
