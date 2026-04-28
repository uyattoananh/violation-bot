"""Audit taxonomy_merges.json for duplicate source-slug mappings.

The eval scripts build a `source_slug -> consolidated_slug` dict by
iterating `hse_type_clusters` in file order, where each later mapping
overwrites earlier ones. So when a source slug appears in multiple
clusters' `absorbs[]` list, its ground-truth label depends on file
order — non-deterministic from a data-modeling standpoint.

This script:
  1. Finds every duplicate source slug
  2. Counts how many photos in the local dataset use each duplicate
     (so we know the real-world impact of resolving one way or the other)
  3. Proposes a resolution using the "specific over general" heuristic:
     the cluster with FEWER total absorbed slugs is preferred (it's the
     more focused category — e.g., Smoking_area_unsafe over
     Housekeeping_general for cigarette-butt photos)
  4. Prints a proposed resolution table and flags ambiguous cases

Does NOT modify taxonomy_merges.json. Pipe the suggested resolutions
into a follow-up script once a policy is approved.

Usage:
  python scripts/audit_taxonomy_merges.py
  python scripts/audit_taxonomy_merges.py --root /path/to/dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None,
                    help="Dataset root for impact counting (defaults to "
                         "~/Desktop/aecis-violations)")
    args = ap.parse_args()

    merges_path = REPO_ROOT / "taxonomy_merges.json"
    merges = json.loads(merges_path.read_text(encoding="utf-8"))

    # Cluster size = how many source slugs absorb into this cluster
    cluster_size = {c["slug"]: len(c["absorbs"]) for c in merges["hse_type_clusters"]}

    # Map each source slug to the list of clusters claiming it
    src_to_clusters: dict[str, list[str]] = defaultdict(list)
    for c in merges["hse_type_clusters"]:
        for s in c["absorbs"]:
            src_to_clusters[s].append(c["slug"])

    duplicates = {s: cs for s, cs in src_to_clusters.items() if len(cs) > 1}

    # Optional: count how many photos use each source slug. The dataset
    # layout is `<root>/<source_hse_slug>/<issue_id>/<photos>` so the
    # source slug = top-level directory name.
    impact: dict[str, int] = defaultdict(int)
    root = Path(args.root).expanduser().resolve() if args.root else (
        Path.home() / "Desktop" / "aecis-violations"
    )
    if root.exists():
        for d in root.iterdir():
            if not d.is_dir():
                continue
            n_photos = sum(
                1
                for issue_dir in d.iterdir()
                if issue_dir.is_dir()
                for _ in issue_dir.glob("*.jp*g")
            )
            impact[d.name] = n_photos
    else:
        print(f"(dataset root not found at {root} — skipping impact counts)\n")

    print(f"Found {len(duplicates)} duplicate source-slug mappings in "
          f"taxonomy_merges.json.\n")
    print("Resolution heuristic: 'specific over general' = pick the cluster "
          "with FEWER absorbs (= more focused category).\n")

    # Header
    print(f"{'Source slug':<60}  {'Photos':>7}  {'Clusters (size)':<60}  "
          f"{'Suggested':<32}  {'Reason'}")
    print("-" * 200)

    auto_resolvable = 0
    ambiguous = 0
    proposed: dict[str, str] = {}

    for s in sorted(duplicates):
        cs = duplicates[s]
        sizes = [(c, cluster_size.get(c, 0)) for c in cs]
        sizes_str = ", ".join(f"{c}({n})" for c, n in sizes)
        n_photos = impact.get(s, 0)

        # Suggested: smaller cluster wins (= more specific). If sizes equal,
        # mark ambiguous.
        sizes_sorted = sorted(sizes, key=lambda kv: kv[1])
        smallest_size = sizes_sorted[0][1]
        winners = [c for c, sz in sizes_sorted if sz == smallest_size]
        if len(winners) == 1:
            sug = winners[0]
            reason = "smallest cluster (most specific)"
            auto_resolvable += 1
        else:
            sug = " or ".join(winners) + " (TIE)"
            reason = "tied size — needs human pick"
            ambiguous += 1

        proposed[s] = sug if " or " not in sug else ""
        truncated_s = s if len(s) <= 60 else s[:57] + "..."
        print(f"{truncated_s:<60}  {n_photos:>7}  {sizes_str:<60}  "
              f"{sug:<32}  {reason}")

    print()
    print(f"Auto-resolvable (clear specificity winner): {auto_resolvable}")
    print(f"Ambiguous (tied size — human decision needed): {ambiguous}")
    if impact:
        total_photos_in_dups = sum(impact[s] for s in duplicates)
        print(f"Total photos affected by ambiguity: {total_photos_in_dups}")

    # Special-case audit: flag any cluster that's ALMOST entirely duplicates
    # — these are candidates for outright removal.
    print()
    print("=" * 80)
    print("Cluster-level health check (clusters with high % of duplicated sources):")
    print("=" * 80)
    for c in merges["hse_type_clusters"]:
        n_total = len(c["absorbs"])
        if n_total == 0:
            continue
        n_dup = sum(1 for s in c["absorbs"] if s in duplicates)
        if n_dup == 0:
            continue
        pct = n_dup / n_total * 100
        if pct >= 50:
            note = "  ⚠  may be redundant — most of its sources are claimed elsewhere"
        else:
            note = ""
        print(f"  {c['slug']:<40}  {n_dup}/{n_total} sources duplicated  "
              f"({pct:>5.1f}%){note}")

    # Print machine-readable proposed resolution at the end so a follow-up
    # script can consume it.
    print()
    print("=" * 80)
    print("Proposed resolution (machine-readable JSON below):")
    print("=" * 80)
    print(json.dumps({
        "policy": "specific over general (smallest cluster wins)",
        "duplicates_total": len(duplicates),
        "auto_resolvable": auto_resolvable,
        "ambiguous": ambiguous,
        "resolutions": proposed,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
