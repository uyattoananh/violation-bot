#!/usr/bin/env bash
# Install / refresh the proper systemd units for the live /root/ deployment.
#
# Migrates from the old single misconfigured 'violation-bot.service' (which
# was pointing at /var/www/violation-bot/ and serving a stale instance) to
# two correctly-pointed units — one for the webapp, one for the classify
# worker — both rooted at /root/violation-bot/ with restart-on-crash and
# boot survival.
#
# Idempotent: safe to re-run after each git pull. Just stops the running
# units, copies the latest unit files, reloads daemon, restarts.
#
# Usage:
#   sudo bash deploy/install_systemd.sh
#
# Run from /root/violation-bot/ (or wherever the repo is checked out).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITS_DIR="${REPO_DIR}/deploy"
SYSTEMD_DIR="/etc/systemd/system"

if [[ "$EUID" -ne 0 ]]; then
  echo "ERROR: must run as root (use sudo)" >&2
  exit 1
fi

# Sanity: confirm the repo dir looks right
if [[ ! -f "${REPO_DIR}/.venv-webapp/bin/python" ]]; then
  echo "ERROR: ${REPO_DIR}/.venv-webapp/bin/python not found." >&2
  echo "Edit the WorkingDirectory + ExecStart in deploy/*.service if the repo lives elsewhere." >&2
  exit 1
fi

echo "==> Installing units from ${UNITS_DIR} to ${SYSTEMD_DIR}"
cp "${UNITS_DIR}/violation-webapp.service" "${SYSTEMD_DIR}/"
cp "${UNITS_DIR}/violation-worker.service" "${SYSTEMD_DIR}/"

echo "==> Reloading systemd daemon"
systemctl daemon-reload

# Stop the misconfigured legacy unit if it's still around (was pointing
# at /var/www/violation-bot/). Keep the disable so it doesn't come back
# on reboot. Don't touch its on-disk file — user can rm if they want.
if systemctl list-unit-files | grep -q '^violation-bot\.service'; then
  echo "==> Stopping + disabling legacy violation-bot.service"
  systemctl stop violation-bot.service 2>/dev/null || true
  systemctl disable violation-bot.service 2>/dev/null || true
fi

# Kill any orphan processes that were launched manually via nohup/disown,
# so the new systemd units have a clean port to bind to.
echo "==> Killing any orphaned webapp/worker processes"
pkill -f '/root/violation-bot/\.venv-webapp/bin/python -m uvicorn webapp\.app:app' || true
pkill -f '/root/violation-bot/\.venv-webapp/bin/python -m webapp\.worker' || true
sleep 1

echo "==> Enabling + starting the new units"
systemctl enable --now violation-webapp.service
systemctl enable --now violation-worker.service

sleep 2
echo
echo "==> Status check (top 8 lines each)"
systemctl --no-pager status violation-webapp.service | head -8 || true
echo
systemctl --no-pager status violation-worker.service | head -8 || true

echo
echo "Done. Tail logs with:"
echo "  sudo journalctl -u violation-webapp -f"
echo "  sudo journalctl -u violation-worker -f"
