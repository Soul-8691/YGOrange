#!/usr/bin/env python3
"""
Add or update a ``git`` field on each object in chimeratech.json from an id→caption map.

The nogibjj Yu-Gi-Oh project
(https://github.com/nogibjj/Generating-Yu-Gi-Oh-Monsters-From-Archetypes)
trains GIT on official card text; it does **not** ship a single committed file that lists
every passcode with a model-generated GIT caption. Captions are produced by running the
GIT notebooks / model. Export your own mapping (JSON or CSV), then run this script.

Mapping formats
---------------
* JSON object:  { "89631139": "a dragon on a card ...", ... }
* JSON array:   [ { "id": 89631139, "git": "..." }, ... ]

CSV (header row):
  id,git
  89631139,"..."

Examples
--------
  python scripts/merge_git_into_chimeratech.py --git-json my_git_captions.json
  python scripts/merge_git_into_chimeratech.py --git-csv captions.csv --git-column git_caption
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def load_git_mapping_json(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            try:
                key = int(str(k).strip())
            except ValueError:
                continue
            out[key] = str(v).strip()
        return out
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            cid = row.get("id")
            g = row.get("git") or row.get("caption") or row.get("git_caption")
            if cid is None or g is None:
                continue
            try:
                key = int(cid)
            except (TypeError, ValueError):
                continue
            s = str(g).strip()
            if s:
                out[key] = s
        return out
    print("git JSON must be an object or an array of objects", file=sys.stderr)
    sys.exit(1)


def load_git_mapping_csv(
    path: Path, id_col: str, git_col: str
) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if id_col not in (r.fieldnames or []) or git_col not in (r.fieldnames or []):
            print(
                f"CSV must have columns {id_col!r} and {git_col!r}; got {r.fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)
        for row in r:
            try:
                key = int(str(row[id_col]).strip())
            except (KeyError, ValueError):
                continue
            g = (row.get(git_col) or "").strip()
            if g:
                out[key] = g
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--chimeratech",
        type=Path,
        default=Path("chimeratech.json"),
        help="Input chimeratech-style JSON array",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite --chimeratech)",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--git-json",
        type=Path,
        help="JSON dict id→str or list of {id, git}",
    )
    g.add_argument(
        "--git-csv",
        type=Path,
        help="CSV with id and caption columns",
    )
    ap.add_argument(
        "--csv-id-column",
        default="id",
        help="CSV column for passcode (default: id)",
    )
    ap.add_argument(
        "--git-column",
        default="git",
        help="CSV column for GIT caption (default: git)",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite existing non-empty git values",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write",
    )
    ap.add_argument(
        "--ensure-git-key",
        action="store_true",
        help='Add "git": "" on rows missing git (after merge)',
    )
    args = ap.parse_args()

    out_path = args.output or args.chimeratech

    try:
        ch = json.loads(args.chimeratech.read_text(encoding="utf-8"))
    except OSError as e:
        print(f"Cannot read {args.chimeratech}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(ch, list):
        print("chimeratech.json must be a JSON array", file=sys.stderr)
        sys.exit(1)

    if args.git_json:
        mapping = load_git_mapping_json(args.git_json)
    else:
        mapping = load_git_mapping_csv(
            args.git_csv, args.csv_id_column, args.git_column
        )

    if not mapping:
        print("No entries in git mapping.", file=sys.stderr)
        sys.exit(1)

    matched = 0
    updated = 0
    skipped_existing = 0
    for row in ch:
        if not isinstance(row, dict):
            continue
        cid = row.get("id")
        try:
            key = int(cid)
        except (TypeError, ValueError):
            continue
        if key not in mapping:
            continue
        matched += 1
        new_g = mapping[key]
        old = row.get("git", "")
        if isinstance(old, str) and old.strip() and not args.replace:
            skipped_existing += 1
            continue
        row["git"] = new_g
        updated += 1

    if args.ensure_git_key:
        for row in ch:
            if isinstance(row, dict) and "git" not in row:
                row["git"] = ""

    print(
        f"Mapping size: {len(mapping)} | chimeratech rows: {len(ch)} | "
        f"ids matched: {matched} | git fields set: {updated} | "
        f"skipped (already had git): {skipped_existing}",
        file=sys.stderr,
    )

    if args.dry_run:
        return

    try:
        out_path.write_text(
            json.dumps(ch, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        print(f"Cannot write {out_path}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
