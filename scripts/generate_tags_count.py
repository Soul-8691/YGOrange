#!/usr/bin/env python3
"""
Count how many species list each tag in a pokemon_tags-style JSON file.

Input: JSON array of { "pokemon": str, "tags": [str, ...] }.
Output: JSON object mapping tag -> count (each species counts at most once per tag).
Keys are sorted by descending count, then tag name (case-insensitive).

Example:
  python scripts/generate_tags_count.py -i pokemon_tags_retagged.json -o tags_count.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        default="pokemon_tags.json",
        help="Input: list of { pokemon, tags }",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="tags_count.json",
        help="Output: object tag -> species count",
    )
    args = ap.parse_args()

    try:
        with open(args.input, encoding="utf-8") as f:
            rows = json.load(f)
    except OSError as e:
        print(f"Cannot read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(rows, list):
        print("Input must be a JSON array", file=sys.stderr)
        sys.exit(1)

    c: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for t in row.get("tags") or []:
            if isinstance(t, str) and (s := t.strip()):
                c[s] += 1

    out: dict[str, int] = dict(
        sorted(c.items(), key=lambda kv: (-kv[1], kv[0].casefold()))
    )

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Cannot write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Wrote {args.output}: {len(out)} unique tags, {sum(c.values())} tag instances",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
