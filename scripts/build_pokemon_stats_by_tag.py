#!/usr/bin/env python3
"""
Build per-trait-tag aggregates matching pokemon_stats.json bucket shape:

For each string in tags_condensed_removals.json, collect species that carry that tag in
pokemon_tags_retagged.json, map display names to pokedex slugs, then compute the same
percentiles (p25/p50/p75) for base stats + BST and Gen 9 move frequencies as
build_pokemon_stats.py.

Outputs pokemon_stats_by_tag.json with:
  - by_trait_tag: { tag: { species_count, stats, bst, moves } }
  - fully_evolved.by_trait_tag: same without moves (fully-evolved species only)

Optional --merge-into pokemon_stats.json adds those two keys into an existing stats file.

Inputs (defaults relative to cwd):
  tags_condensed_removals.json, pokemon_tags_retagged.json, pokedex.ts, learnsets.json

Install: numpy (same as build_pokemon_stats.py)

Example::

  python scripts/build_pokemon_stats_by_tag.py
  python scripts/build_pokemon_stats_by_tag.py --merge-into pokemon_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from build_pokemon_stats import (
    load_gen9_learnsets,
    moves_summary,
    parse_pokedex_ts,
    stats_summary,
)


def load_json_array(path: Path) -> list[Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected JSON array")
    return raw


def load_condensed_tags(path: Path) -> list[str]:
    rows = load_json_array(path)
    seen: set[str] = set()
    out: list[str] = []
    for t in rows:
        s = str(t).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_retagged(path: Path) -> list[dict[str, Any]]:
    rows = load_json_array(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and row.get("pokemon"):
            out.append(row)
    return out


def name_to_slug_map(dex: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Display name -> slug (last wins if duplicate names)."""
    m: dict[str, str] = {}
    for slug, row in dex.items():
        name = str(row.get("name", "")).strip()
        if name:
            m[name] = slug
    return m


def slugs_for_tag(
    tag: str,
    condensed_set: set[str],
    retagged: list[dict[str, Any]],
    name_map: dict[str, str],
) -> list[str]:
    found: set[str] = set()
    for row in retagged:
        name = str(row.get("pokemon", "")).strip()
        slug = name_map.get(name)
        if not slug:
            continue
        tags = row.get("tags", [])
        if not isinstance(tags, list):
            continue
        for t in tags:
            if str(t).strip() == tag and tag in condensed_set:
                found.add(slug)
                break
    return sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tags",
        type=Path,
        default=Path("tags_condensed_removals.json"),
        help="Allowed tag strings (order preserved in output keys)",
    )
    ap.add_argument(
        "--pokemon-tags",
        type=Path,
        default=Path("pokemon_tags_retagged.json"),
        help="Per-species tag assignments",
    )
    ap.add_argument("--pokedex", type=Path, default=Path("pokedex.ts"))
    ap.add_argument("--learnsets", type=Path, default=Path("learnsets.json"))
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pokemon_stats_by_tag.json"),
    )
    ap.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="If set, read this JSON and add by_trait_tag + fully_evolved.by_trait_tag",
    )
    ap.add_argument("--min-num", type=int, default=1)
    args = ap.parse_args()

    condensed = load_condensed_tags(args.tags)
    condensed_set = set(condensed)
    retagged = load_retagged(args.pokemon_tags)

    dex = parse_pokedex_ts(str(args.pokedex), args.min_num)
    learn = load_gen9_learnsets(str(args.learnsets))
    nmap = name_to_slug_map(dex)
    fully_slugs = {s for s, r in dex.items() if r.get("fully_evolved")}

    by_trait: dict[str, Any] = {}
    fe_by_trait: dict[str, Any] = {}

    for tag in condensed:
        slugs = slugs_for_tag(tag, condensed_set, retagged, nmap)
        block = stats_summary(slugs, dex)
        block["moves"] = moves_summary(slugs, learn)
        by_trait[tag] = block

        fe_list = [s for s in slugs if s in fully_slugs]
        fe_block = stats_summary(fe_list, dex)
        fe_by_trait[tag] = fe_block

    unmatched = sum(
        1
        for row in retagged
        if (nm := str(row.get("pokemon", "")).strip()) and nm not in nmap
    )

    payload: dict[str, Any] = {
        "meta": {
            "kind": "by_trait_tag",
            "tags_source": str(args.tags),
            "assignments_source": str(args.pokemon_tags),
            "pokedex_entries_used": len(dex),
            "learnset_generation": "9",
            "trait_tags_emitted": len(condensed),
            "retagged_rows_unmatched_to_pokedex": unmatched,
            "fully_evolved_definition": "Species entries without a prevo field",
            "note": "Species are included in a tag bucket if pokemon_tags_retagged lists that tag.",
        },
        "by_trait_tag": by_trait,
        "fully_evolved": {"by_trait_tag": fe_by_trait},
    }

    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({len(by_trait)} tag buckets)", file=sys.stderr)

    if args.merge_into is not None:
        try:
            base = json.loads(args.merge_into.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Merge failed: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(base, dict):
            print("merge-into file must be a JSON object", file=sys.stderr)
            sys.exit(1)
        base["by_trait_tag"] = by_trait
        fe = base.get("fully_evolved")
        if not isinstance(fe, dict):
            fe = {}
            base["fully_evolved"] = fe
        fe["by_trait_tag"] = fe_by_trait
        args.merge_into.write_text(
            json.dumps(base, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Merged by_trait_tag into {args.merge_into}", file=sys.stderr)


if __name__ == "__main__":
    main()
