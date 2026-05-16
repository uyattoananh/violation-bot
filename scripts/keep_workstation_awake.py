"""Prevent the Windows workstation from sleeping while the seed
ingestion loop runs. Calls SetThreadExecutionState with
ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED to tell
Windows we're doing real work and the system shouldn't idle out.

The flag is reset to ES_CONTINUOUS only on a clean exit so the
default sleep policy returns.

Run from a terminal window — leave it open as long as you want the
machine to stay awake. Ctrl+C exits cleanly.

Usage:
    ./.venv-webapp/Scripts/python.exe scripts/keep_workstation_awake.py
    ./.venv-webapp/Scripts/python.exe scripts/keep_workstation_awake.py --no-display
"""
from __future__ import annotations
import argparse
import ctypes
import signal
import sys
import time

ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040


def _set(flags: int) -> int:
    r = ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_ulong(flags))
    return int(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-display", action="store_true",
                    help="allow display to sleep; just keep the system "
                         "awake (useful for headless long ingest runs)")
    ap.add_argument("--away-mode", action="store_true",
                    help="use ES_AWAYMODE_REQUIRED — Windows behaves "
                         "as if the user stepped away. Some background "
                         "tasks need this on locked sessions.")
    ap.add_argument("--heartbeat-seconds", type=int, default=60,
                    help="re-issue the state every N seconds (Windows "
                         "respects the call as long as the thread is "
                         "alive, but renewing is safe).")
    args = ap.parse_args()

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if not args.no_display:
        flags |= ES_DISPLAY_REQUIRED
    if args.away_mode:
        flags |= ES_AWAYMODE_REQUIRED

    sys.stdout.write(
        "[keep-awake] holding system awake "
        f"(flags=0x{flags:08X}; display={not args.no_display}; "
        f"away_mode={args.away_mode}). Ctrl+C to release.\n"
    )
    sys.stdout.flush()

    # Apply once
    prior = _set(flags)
    if prior == 0:
        sys.stderr.write("ERROR: SetThreadExecutionState returned 0 "
                         "(invalid flags or not on Windows?)\n")
        return 2

    stopping = False
    def _on_sig(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    n_beats = 0
    while not stopping:
        time.sleep(args.heartbeat_seconds)
        n_beats += 1
        # Refresh — cheap, idempotent
        _set(flags)
        if n_beats % 60 == 0:
            sys.stdout.write(f"[keep-awake] still holding "
                             f"({n_beats * args.heartbeat_seconds / 60:.0f} min)\n")
            sys.stdout.flush()

    # Release
    _set(ES_CONTINUOUS)
    sys.stdout.write("[keep-awake] released — default sleep policy restored.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
