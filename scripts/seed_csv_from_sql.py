"""Rebuild result_after_query.csv from the raw SQL dumps WITHOUT
the IssueActionID = 1 filter.

The CSV handed over by the AECIS DB manager came from:

    SELECT i.*, ia.*, ip.*
    FROM PM.Issue i
    JOIN PM.IssueActivity ia ON i.IssueID = ia.IssueID
    JOIN PM.IssuePhoto    ip ON ia.IssueActivityID = ip.IssueActivityID
    WHERE i.IssueActionID = 1 AND i.IsDeleted = 0

The `IssueActionID = 1` clause restricts to photos attached to the
issue-CREATION event only. Photos uploaded during update / resolve /
evidence-of-fix / close are dropped, costing us ~245,000 rows out
of 284,975 in PM.IssuePhoto.

This script does the same join in Python directly on the .sql dumps
and writes a FULL CSV (no action-type filter). Output is column-
identical to the original CSV so the existing downloader works
without changes — just point `_AECIS_CSV_REL_PATH` at the new file.

Inputs (UTF-16-LE BOM-prefixed Microsoft .sql exports):
  Issue_Gen/Issue_Gen/PM.Issue.sql        (632 MB)
  Issue_Gen/Issue_Gen/PM.IssueActivity.sql (478 MB)
  Issue_Gen/Issue_Gen/PM.IssuePhoto.sql   (304 MB, 284,975 rows)

Output:
  Issue_Gen/Issue_Gen/result_after_query.full.csv

Usage:
  ./.venv-webapp/Scripts/python.exe scripts/seed_csv_from_sql.py
  ./.venv-webapp/Scripts/python.exe scripts/seed_csv_from_sql.py --include-deleted
"""
from __future__ import annotations
import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "Issue_Gen" / "Issue_Gen"
PM_ISSUE = SRC_DIR / "PM.Issue.sql"
PM_ACTIVITY = SRC_DIR / "PM.IssueActivity.sql"
PM_PHOTO = SRC_DIR / "PM.IssuePhoto.sql"
OUT_CSV = SRC_DIR / "result_after_query.full.csv"


# MSSQL `INSERT [PM].[Table] (col, col, …) VALUES (v, v, NULL, 'text', …)`
# We extract the VALUES tuple. Strings are 'single-quoted' with '' for
# embedded apostrophes; NULL is literal; numbers and timestamps are bare.
_VALUES_RE = re.compile(r"VALUES\s*\((.*?)\)\s*\Z", re.DOTALL)


def _open_utf16le(path: Path):
    f = path.open("rb")
    head = f.read(2)
    if head not in (b"\xff\xfe", b"\xfe\xff"):
        f.seek(0)
    enc = "utf-16-le" if head == b"\xff\xfe" else (
          "utf-16-be" if head == b"\xfe\xff" else "utf-8")
    return io.TextIOWrapper(f, encoding=enc, errors="replace", newline="")


def _split_sql_values(payload: str) -> list[str]:
    """Split a `v1, v2, …` SQL tuple respecting 'quoted strings' that
    may contain commas, doubled apostrophes ('' = literal '), and
    N'unicode prefix' literals."""
    out: list[str] = []
    i = 0
    n = len(payload)
    buf = []
    in_str = False
    while i < n:
        c = payload[i]
        if not in_str:
            if c == "," :
                out.append("".join(buf).strip())
                buf = []
                i += 1; continue
            if c == "N" and i + 1 < n and payload[i+1] == "'":
                in_str = True
                buf.append("'")
                i += 2; continue
            if c == "'":
                in_str = True
                buf.append("'")
                i += 1; continue
            buf.append(c)
            i += 1
        else:
            if c == "'":
                if i + 1 < n and payload[i+1] == "'":
                    buf.append("'")
                    i += 2; continue
                in_str = False
                buf.append("'")
                i += 1; continue
            buf.append(c)
            i += 1
    out.append("".join(buf).strip())
    return out


def _unquote(v: str) -> str:
    """Strip leading/trailing single-quote and unescape doubled '."""
    if v == "NULL":
        return ""
    if len(v) >= 2 and v.startswith("'") and v.endswith("'"):
        inner = v[1:-1].replace("''", "'")
        return inner
    return v


def iter_insert_rows(path: Path):
    """Yield each row's value list. Streams the file, accumulates
    multi-line VALUES tuples, and only fully-parses VALUES blocks."""
    with _open_utf16le(path) as f:
        buf = []
        inserting = False
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not inserting:
                if "INSERT" in s.upper() and "VALUES" in s.upper():
                    buf = [s]
                    inserting = True
                    if s.endswith(")"):
                        # Full row on one line
                        full = " ".join(buf)
                        inserting = False
                        m = _VALUES_RE.search(full)
                        if m:
                            yield [_unquote(v) for v in _split_sql_values(m.group(1))]
                        buf = []
                continue
            buf.append(s)
            if s.endswith(")"):
                full = " ".join(buf)
                inserting = False
                m = _VALUES_RE.search(full)
                if m:
                    yield [_unquote(v) for v in _split_sql_values(m.group(1))]
                buf = []


def _table_columns(path: Path) -> list[str]:
    """Pull the column ordering from the CREATE TABLE block (in order)."""
    with _open_utf16le(path) as f:
        in_create = False
        cols: list[str] = []
        for line in f:
            s = line.strip()
            if not in_create:
                if s.upper().startswith("CREATE TABLE"):
                    in_create = True
                continue
            if s.startswith(")"):
                break
            m = re.match(r"\[([^\]]+)\]\s+\[", s)
            if m:
                cols.append(m.group(1))
    return cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-deleted", action="store_true",
                    help="keep rows where IsDeleted = 1 (default: drop)")
    ap.add_argument("--limit-photos", type=int, default=0,
                    help="stop after N photo rows (debug; 0 = all)")
    args = ap.parse_args()

    for p in (PM_ISSUE, PM_ACTIVITY, PM_PHOTO):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing {p}\n")
            return 2

    t0 = time.perf_counter()

    # 1. PM.Issue → {IssueID: row}
    issue_cols = _table_columns(PM_ISSUE)
    print(f"PM.Issue columns: {len(issue_cols)} ({issue_cols[:6]}...)")
    idx_issueid = issue_cols.index("IssueID")
    try:
        idx_is_deleted = issue_cols.index("IsDeleted")
    except ValueError:
        idx_is_deleted = None
    issue_by_id: dict[str, list[str]] = {}
    n = 0
    for row in iter_insert_rows(PM_ISSUE):
        n += 1
        if len(row) < len(issue_cols):
            continue
        if not args.include_deleted and idx_is_deleted is not None:
            if (row[idx_is_deleted] or "0").strip() not in ("0", ""):
                continue
        issue_by_id[row[idx_issueid]] = row
        if n % 50000 == 0:
            print(f"  Issue: {n} parsed, {len(issue_by_id)} kept")
    print(f"PM.Issue: {n} parsed → {len(issue_by_id)} kept in {time.perf_counter()-t0:.1f}s")

    # 2. PM.IssueActivity → {IssueActivityID: row}
    act_cols = _table_columns(PM_ACTIVITY)
    print(f"PM.IssueActivity columns: {len(act_cols)} ({act_cols[:6]}...)")
    idx_actid = act_cols.index("IssueActivityID")
    idx_act_issueid = act_cols.index("IssueID")
    try:
        idx_act_is_deleted = act_cols.index("IsDeleted")
    except ValueError:
        idx_act_is_deleted = None
    activity_by_id: dict[str, list[str]] = {}
    n = 0
    for row in iter_insert_rows(PM_ACTIVITY):
        n += 1
        if len(row) < len(act_cols):
            continue
        if not args.include_deleted and idx_act_is_deleted is not None:
            if (row[idx_act_is_deleted] or "0").strip() not in ("0", ""):
                continue
        if row[idx_act_issueid] not in issue_by_id:
            continue
        activity_by_id[row[idx_actid]] = row
        if n % 50000 == 0:
            print(f"  Activity: {n} parsed, {len(activity_by_id)} kept")
    print(f"PM.IssueActivity: {n} parsed → {len(activity_by_id)} kept in {time.perf_counter()-t0:.1f}s")

    # 3. Stream PM.IssuePhoto and emit joined rows
    photo_cols = _table_columns(PM_PHOTO)
    idx_photo_actid = photo_cols.index("IssueActivityID")
    try:
        idx_photo_is_deleted = photo_cols.index("IsDeleted")
    except ValueError:
        idx_photo_is_deleted = None
    idx_photo_filepath = photo_cols.index("FilePath")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = issue_cols + act_cols + photo_cols
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as outf:
        w = csv.writer(outf)
        w.writerow(header)
        n_total = n_kept = 0
        for row in iter_insert_rows(PM_PHOTO):
            n_total += 1
            if len(row) < len(photo_cols):
                continue
            if not args.include_deleted and idx_photo_is_deleted is not None:
                if (row[idx_photo_is_deleted] or "0").strip() not in ("0", ""):
                    continue
            actid = row[idx_photo_actid]
            act = activity_by_id.get(actid)
            if not act:
                continue
            issue_id = act[idx_act_issueid]
            issue = issue_by_id.get(issue_id)
            if not issue:
                continue
            if not (row[idx_photo_filepath] or "").strip():
                continue
            # AECIS internal code (method.txt §3.1) normalizes backslashes
            # to forward slashes before signing URLs. MSSQL stores paths
            # with raw '\' on Windows-uploaded photos; do the same fix
            # here so the rebuilt CSV is drop-in compatible with the
            # downloader and the existing 39k-row CSV.
            row[idx_photo_filepath] = row[idx_photo_filepath].replace("\\", "/")
            w.writerow(issue + act + row)
            n_kept += 1
            if args.limit_photos and n_kept >= args.limit_photos:
                break
            if n_total % 25000 == 0:
                print(f"  Photo: {n_total} parsed, {n_kept} kept "
                      f"(elapsed {time.perf_counter()-t0:.0f}s)")
    print(f"PM.IssuePhoto: {n_total} parsed → {n_kept} kept "
          f"in {time.perf_counter()-t0:.1f}s")
    print(f"OUT: {OUT_CSV} ({OUT_CSV.stat().st_size//1024//1024} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
