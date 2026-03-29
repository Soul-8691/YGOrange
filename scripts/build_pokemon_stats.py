#!/usr/bin/env python3
"""
Build pokemon_stats.json: aggregate base stat percentiles and Gen 9 learnset moves by:
  - single Pokémon type
  - single egg group
  - habitat (species may appear in multiple habitat buckets)

Also emits fully_evolved (no `prevo` in pokedex) stat percentiles only for the same groupings.

Inputs:
  pokedex.ts          — baseStats, types, eggGroups, prevo, name, num
  learnsets.json      — object \"9\" → species slug → learnset
  habitats.json       — habitat name → [slug, ...]

Output: pokemon_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np

STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def find_matching_brace(s: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    while i < len(s):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def parse_ts_string_list(inner: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r'"([^"]*)"', inner):
        out.append(m.group(1))
    return out


def parse_pokedex_ts(path: str, min_num: int) -> dict[str, dict[str, Any]]:
    text = open(path, encoding="utf-8").read()
    species_pat = re.compile(r"^\t([a-z][a-z0-9]*):\s*\{", re.MULTILINE)
    out: dict[str, dict[str, Any]] = {}
    for m in species_pat.finditer(text):
        slug = m.group(1)
        open_brace = m.end() - 1
        if open_brace < 0 or text[open_brace] != "{":
            continue
        close = find_matching_brace(text, open_brace)
        if close < 0:
            continue
        body = text[open_brace + 1 : close]

        nm = re.search(r"^\t\tname:\s*\"([^\"]*)\"", body, re.MULTILINE)
        if not nm:
            continue
        name = nm.group(1)

        num_m = re.search(r"^\t\tnum:\s*(-?\d+)", body, re.MULTILINE)
        if not num_m:
            continue
        num = int(num_m.group(1))
        if num < min_num:
            continue

        tm = re.search(r"^\t\ttypes:\s*\[([^\]]*)\]", body, re.MULTILINE)
        if not tm:
            continue
        types = parse_ts_string_list(tm.group(1))

        em = re.search(r"^\t\teggGroups:\s*\[([^\]]*)\]", body, re.MULTILINE)
        if not em:
            continue
        egg_groups = parse_ts_string_list(em.group(1))

        bm = re.search(
            r"^\t\tbaseStats:\s*\{\s*hp:\s*(\d+),\s*atk:\s*(\d+),\s*def:\s*(\d+),\s*spa:\s*(\d+),\s*spd:\s*(\d+),\s*spe:\s*(\d+)\s*\}",
            body,
            re.MULTILINE,
        )
        if not bm:
            continue
        base_stats = {
            "hp": int(bm.group(1)),
            "atk": int(bm.group(2)),
            "def": int(bm.group(3)),
            "spa": int(bm.group(4)),
            "spd": int(bm.group(5)),
            "spe": int(bm.group(6)),
        }

        has_prevo = bool(re.search(r"^\t\tprevo:\s*\"", body, re.MULTILINE))

        out[slug] = {
            "name": name,
            "num": num,
            "types": types,
            "eggGroups": egg_groups,
            "baseStats": base_stats,
            "fully_evolved": not has_prevo,
        }
    return out


def load_gen9_learnsets(path: str) -> dict[str, set[str]]:
    data = json.load(open(path, encoding="utf-8"))
    g9 = data.get("9")
    if not isinstance(g9, dict):
        raise SystemExit('learnsets.json: missing top-level "9" object')
    moves_by_slug: dict[str, set[str]] = {}
    for slug, entry in g9.items():
        if not isinstance(entry, dict):
            continue
        ls = entry.get("learnset")
        if isinstance(ls, dict):
            moves_by_slug[slug] = set(ls.keys())
        else:
            moves_by_slug[slug] = set()
    return moves_by_slug


def load_habitat_membership(path: str) -> dict[str, list[str]]:
    """slug (lower) -> list of habitat names."""
    h = json.load(open(path, encoding="utf-8"))
    if not isinstance(h, dict):
        raise SystemExit("habitats.json must be an object")
    slug_to_habs: dict[str, list[str]] = defaultdict(list)
    for hab_name, slugs in h.items():
        if not isinstance(slugs, list):
            continue
        for s in slugs:
            if isinstance(s, str) and s.strip():
                slug_to_habs[s.strip().lower()].append(str(hab_name))
    return dict(slug_to_habs)


def percentile_triplet(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0}
    arr = np.array(values, dtype=np.float64)
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    return {
        "p25": round(float(p25), 2),
        "p50": round(float(p50), 2),
        "p75": round(float(p75), 2),
    }


def stats_summary(slugs: list[str], dex: dict[str, Any]) -> dict[str, Any]:
    rows = [dex[s]["baseStats"] for s in slugs if s in dex]
    if not rows:
        return {
            "species_count": 0,
            "stats": {k: {"p25": 0.0, "p50": 0.0, "p75": 0.0} for k in STAT_KEYS},
            "bst": {"p25": 0.0, "p50": 0.0, "p75": 0.0},
        }
    stats_block: dict[str, dict[str, float]] = {}
    for k in STAT_KEYS:
        stats_block[k] = percentile_triplet([float(r[k]) for r in rows])
    bst_vals = [sum(float(r[x]) for x in STAT_KEYS) for r in rows]
    return {
        "species_count": len(rows),
        "stats": stats_block,
        "bst": percentile_triplet(bst_vals),
    }


def moves_summary(slugs: list[str], learn: dict[str, set[str]]) -> list[dict[str, Any]]:
    c: Counter[str] = Counter()
    for s in slugs:
        for mv in learn.get(s, ()):
            c[mv] += 1
    return [{"move": m, "species_count": n} for m, n in c.most_common()]


def aggregate_by_buckets(
    dex: dict[str, dict[str, Any]],
    learn: dict[str, set[str]],
    buckets: dict[str, list[str]],
    *,
    include_moves: bool,
    slug_filter: set[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, slugs in sorted(buckets.items(), key=lambda x: x[0].casefold()):
        uniq = sorted(set(slugs))
        if slug_filter is not None:
            uniq = [s for s in uniq if s in slug_filter]
        if not uniq:
            continue
        block: dict[str, Any] = stats_summary(uniq, dex)
        if include_moves:
            block["moves"] = moves_summary(uniq, learn)
        result[label] = block
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pokedex", default="pokedex.ts")
    ap.add_argument("--learnsets", default="learnsets.json")
    ap.add_argument("--habitats", default="habitats.json")
    ap.add_argument("-o", "--output", default="pokemon_stats.json")
    ap.add_argument(
        "--min-num",
        type=int,
        default=1,
        help="Skip species with num below this (excludes most fake/Pokéstar entries)",
    )
    args = ap.parse_args()

    dex = parse_pokedex_ts(args.pokedex, args.min_num)
    learn = load_gen9_learnsets(args.learnsets)
    hab_mem = load_habitat_membership(args.habitats)

    by_type: dict[str, list[str]] = defaultdict(list)
    by_egg: dict[str, list[str]] = defaultdict(list)
    by_hab: dict[str, list[str]] = defaultdict(list)

    for slug, row in dex.items():
        types = row["types"]
        if len(types) == 1:
            by_type[types[0]].append(slug)
        eggs = row["eggGroups"]
        if len(eggs) == 1:
            by_egg[eggs[0]].append(slug)
        for hab in hab_mem.get(slug, ()):
            by_hab[hab].append(slug)

    fully_slugs = {s for s, r in dex.items() if r["fully_evolved"]}

    payload: dict[str, Any] = {
        "meta": {
            "pokedex_entries_used": len(dex),
            "learnset_generation": "9",
            "single_type_only": True,
            "single_egg_group_only": True,
            "habitat_note": "A species is counted in every habitat list it appears in.",
            "fully_evolved_definition": "Species entries without a prevo field",
        },
        "by_single_type": aggregate_by_buckets(
            dex, learn, dict(by_type), include_moves=True
        ),
        "by_single_egg_group": aggregate_by_buckets(
            dex, learn, dict(by_egg), include_moves=True
        ),
        "by_habitat": aggregate_by_buckets(
            dex, learn, dict(by_hab), include_moves=True
        ),
        "fully_evolved": {
            "by_single_type": aggregate_by_buckets(
                dex,
                learn,
                dict(by_type),
                include_moves=False,
                slug_filter=fully_slugs,
            ),
            "by_single_egg_group": aggregate_by_buckets(
                dex,
                learn,
                dict(by_egg),
                include_moves=False,
                slug_filter=fully_slugs,
            ),
            "by_habitat": aggregate_by_buckets(
                dex,
                learn,
                dict(by_hab),
                include_moves=False,
                slug_filter=fully_slugs,
            ),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {args.output} ({len(dex)} species, "
        f"{len(payload['by_single_type'])} type buckets, "
        f"{len(payload['by_habitat'])} habitat buckets)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
