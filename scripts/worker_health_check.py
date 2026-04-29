"""Cron-friendly health check for the violation-worker + webapp.

Pings webapp /healthz, checks classify_jobs error rate, and posts to a
Slack/Discord/Telegram webhook on regression. Catches silent failures
the systemd-restart-loop wouldn't surface (e.g. classifier hitting a
quota limit and erroring on every job).

Designed to run every 5 minutes via cron. Idempotent — uses a small
state file to avoid alert-storming on the same condition repeatedly.

Checks:
  1. /healthz responds 200 within 5s
  2. error rate in classify_jobs over the last hour is below threshold
  3. there are no jobs stuck in "running" status for >10 minutes (worker
     died mid-job)

Configure via env (typically in /root/violation-bot/.env):
  HEALTHCHECK_WEBAPP_URL=http://127.0.0.1:8765/healthz   (default)
  HEALTHCHECK_WEBHOOK_URL=https://hooks.slack.com/...    (or Discord, etc.)
  HEALTHCHECK_ERROR_RATE_THRESHOLD=0.20                  (0-1, default 0.20)
  HEALTHCHECK_STUCK_RUNNING_MINUTES=10                   (default 10)
  HEALTHCHECK_STATE_FILE=/var/lib/violation-bot/healthcheck.state
                                                         (default /tmp/violation_healthcheck.state)

Cron suggestion (every 5 minutes):
  */5 * * * * cd /root/violation-bot && .venv-webapp/bin/python scripts/worker_health_check.py >/dev/null 2>&1
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import request as urlreq, error as urlerr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
log = logging.getLogger("healthcheck")


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:  # noqa: BLE001
        return default


WEBAPP_URL = os.environ.get("HEALTHCHECK_WEBAPP_URL", "http://127.0.0.1:8765/healthz")
WEBHOOK_URL = os.environ.get("HEALTHCHECK_WEBHOOK_URL", "")
ERROR_RATE_THRESHOLD = _get_env_float("HEALTHCHECK_ERROR_RATE_THRESHOLD", 0.20)
STUCK_MINUTES = int(_get_env_float("HEALTHCHECK_STUCK_RUNNING_MINUTES", 10))
STATE_FILE = Path(os.environ.get(
    "HEALTHCHECK_STATE_FILE", "/tmp/violation_healthcheck.state"))
HOSTNAME = socket.gethostname()


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("could not write state file: %s", e)


def _post_alert(message: str) -> None:
    """Best-effort notify. Auto-detects webhook style by URL pattern.
    Slack and Discord both accept JSON; Telegram needs a different
    URL form (handled below)."""
    if not WEBHOOK_URL:
        log.warning("ALERT (no webhook configured): %s", message)
        return
    body = json.dumps({"text": message, "content": message}).encode("utf-8")
    req = urlreq.Request(
        WEBHOOK_URL, data=body,
        headers={"Content-Type": "application/json", "User-Agent": "violation-bot-healthcheck/1"},
        method="POST",
    )
    try:
        with urlreq.urlopen(req, timeout=10) as r:
            if r.status >= 300:
                log.warning("webhook returned HTTP %s", r.status)
            else:
                log.info("alert sent (HTTP %s)", r.status)
    except urlerr.URLError as e:
        log.error("webhook POST failed: %s", e)


def check_webapp() -> tuple[bool, str]:
    try:
        req = urlreq.Request(WEBAPP_URL, headers={"User-Agent": "healthcheck/1"})
        t0 = time.time()
        with urlreq.urlopen(req, timeout=5) as r:
            elapsed_ms = int((time.time() - t0) * 1000)
            if r.status != 200:
                return False, f"HTTP {r.status} from {WEBAPP_URL}"
            return True, f"healthy ({elapsed_ms}ms)"
    except Exception as e:  # noqa: BLE001
        return False, f"failed: {e}"


def check_classify_health() -> tuple[bool, str]:
    """Look at the last hour of classify_jobs:
       - error rate above threshold -> alert
       - any "running" job older than STUCK_MINUTES -> alert
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        return True, "skip (no Supabase creds)"
    from supabase import create_client
    db = create_client(url, key)

    one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    try:
        recent = (
            db.table("classify_jobs").select("status,updated_at")
              .gte("created_at", one_hour_ago)
              .limit(5000).execute().data or []
        )
    except Exception as e:  # noqa: BLE001
        return False, f"DB query failed: {e}"

    if not recent:
        return True, "no recent jobs (idle)"
    n = len(recent)
    n_err = sum(1 for r in recent if (r.get("status") or "") == "error")
    n_done = sum(1 for r in recent if (r.get("status") or "") == "done")
    rate = n_err / max(n, 1)
    msgs = [f"last hour: {n_done} done, {n_err} error ({rate * 100:.0f}%)"]

    # Stuck running
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STUCK_MINUTES)
    try:
        stuck = (
            db.table("classify_jobs").select("id,updated_at")
              .eq("status", "running").lte("updated_at", cutoff.isoformat())
              .limit(50).execute().data or []
        )
    except Exception:  # noqa: BLE001
        stuck = []

    problems = []
    if rate > ERROR_RATE_THRESHOLD:
        problems.append(f"error rate {rate * 100:.0f}% > threshold {ERROR_RATE_THRESHOLD * 100:.0f}%")
    if stuck:
        problems.append(f"{len(stuck)} jobs stuck in 'running' >{STUCK_MINUTES} min")
        msgs.append(f"stuck: {len(stuck)}")

    summary = "; ".join(msgs)
    if problems:
        return False, summary + " — " + "; ".join(problems)
    return True, summary


def main() -> int:
    state = _load_state()
    fired = []

    web_ok, web_msg = check_webapp()
    log.info("webapp: %s", web_msg)
    cls_ok, cls_msg = check_classify_health()
    log.info("classify: %s", cls_msg)

    # Per-check alert dedupe — don't re-fire if the same condition was
    # already alerted in the last 30 minutes.
    now_ts = int(time.time())
    cooldown = 30 * 60

    def maybe_fire(key: str, msg: str):
        last = state.get(key, 0)
        if now_ts - last < cooldown:
            log.info("(suppressing %s alert — in cooldown)", key)
            return
        fired.append(msg)
        state[key] = now_ts

    # Recovery notice — if a check that was failing now succeeds, send a
    # one-shot "recovered" message and clear the cooldown.
    def maybe_recover(key: str, msg: str):
        if state.get(key):
            fired.append(msg)
            state.pop(key, None)

    if not web_ok:
        maybe_fire("webapp_down", f"[{HOSTNAME}] webapp DOWN: {web_msg}")
    else:
        maybe_recover("webapp_down", f"[{HOSTNAME}] webapp recovered: {web_msg}")

    if not cls_ok:
        maybe_fire("classify_unhealthy", f"[{HOSTNAME}] classify unhealthy: {cls_msg}")
    else:
        maybe_recover("classify_unhealthy", f"[{HOSTNAME}] classify recovered: {cls_msg}")

    if fired:
        _post_alert("\n".join(fired))

    _save_state(state)
    return 0 if (web_ok and cls_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
