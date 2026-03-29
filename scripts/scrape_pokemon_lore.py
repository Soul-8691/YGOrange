#!/usr/bin/env python3
"""
Scrape Bulbapedia (biology, forms, dex) + The World of Pokémon (research arrays, taxonomy).
Output JSON: list of { id, num, name, biology, forms, dex, research, taxonomy } (all str).

Post-processing (always applied before write): strip ===Forms=== and below from biology; drop
every species that has a baseSpecies in pokedex.ts (megas, regional formes, G-Max, etc.); merge
each removed row's taxonomy into that baseSpecies entry, comma-separated (deduped).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bulbapedia often returns 403 for Chrome/Chromium-style UAs from urllib; Firefox works.
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"
BULB_API = "https://bulbapedia.bulbagarden.net/w/api.php"
WOP_BASE = "https://www.theworldofpokemon.com"


@dataclass
class SpeciesRow:
    id: str
    num: int
    name: str
    base_species: str | None


def http_get(url: str, timeout: int = 90) -> str:
    headers = {
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
    }
    if "theworldofpokemon.com" in url:
        headers["Referer"] = "https://www.theworldofpokemon.com/"
        headers["Origin"] = "https://www.theworldofpokemon.com"
    elif "bulbagarden.net" in url:
        headers["Referer"] = "https://bulbapedia.bulbagarden.net/"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def http_get_json(url: str, timeout: int = 90) -> dict[str, Any]:
    return json.loads(http_get(url, timeout))


def decode_ts_string_literal(inner: str) -> str:
    """Decode contents of a TS/JS double-quoted string (\\n, \\uXXXX, etc.)."""
    try:
        return json.loads('"' + inner.replace("\r", "") + '"')
    except json.JSONDecodeError:
        return inner.encode("utf-8").decode("unicode_escape")


def parse_pokedex_ts(path: str) -> list[SpeciesRow]:
    text = open(path, encoding="utf-8").read()
    # Split on species keys at tab indent
    parts = re.split(r"(?m)^\t(\w+):\s*\{\s*$", text)
    rows: list[SpeciesRow] = []
    if len(parts) < 2:
        return rows
    # parts[0] is preamble; then pairs (id, block), ...
    it = iter(parts[1:])
    for sid, block in zip(it, it):
        m_num = re.search(r"(?m)^\t\tnum:\s*(-?\d+)", block)
        m_name = re.search(
            r'(?m)^\t\tname:\s*"((?:\\.|[^"\\])*)"', block
        )
        if not m_num or not m_name:
            continue
        num = int(m_num.group(1))
        name = decode_ts_string_literal(m_name.group(1))
        m_bs = re.search(
            r'(?m)^\t\tbaseSpecies:\s*"((?:\\.|[^"\\])*)"', block
        )
        base = decode_ts_string_literal(m_bs.group(1)) if m_bs else None
        rows.append(SpeciesRow(id=sid, num=num, name=name, base_species=base))
    return rows


def extract_js_array_literal(source: str, var_name: str) -> str | None:
    for needle in (var_name + "=", var_name + " ="):
        i = source.find(needle)
        if i >= 0:
            i += len(needle)
            break
    else:
        return None
    while i < len(source) and source[i] in " \t\n\r":
        i += 1
    if i >= len(source) or source[i] != "[":
        return None
    depth = 0
    start = i
    while i < len(source):
        c = source[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        elif c in "\"'":
            q = c
            i += 1
            while i < len(source):
                if source[i] == "\\" and i + 1 < len(source):
                    i += 2
                    continue
                if source[i] == q:
                    i += 1
                    break
                i += 1
            continue
        i += 1
    return None


def parse_js_string_array(array_literal: str) -> list[str]:
    """Parse ['a','b'] or ["a","b"] — only strings and commas/whitespace."""
    s = array_literal.strip()
    if not s.startswith("[") or not s.endswith("]"):
        return []
    inner = s[1:-1]
    out: list[str] = []
    i = 0
    n = len(inner)
    while i < n:
        while i < n and inner[i] in ", \t\n\r":
            i += 1
        if i >= n:
            break
        if inner[i] not in "\"'":
            # skip unknown token
            break
        q = inner[i]
        i += 1
        chunk: list[str] = []
        while i < n:
            if inner[i] == "\\" and i + 1 < n:
                chunk.append(inner[i + 1])
                i += 2
                continue
            if inner[i] == q:
                i += 1
                break
            chunk.append(inner[i])
            i += 1
        out.append("".join(chunk))
    return out


def wiki_page_title(species_name: str) -> str:
    return species_name.replace(" ", "_") + "_(Pokémon)"


def normalize_section_title(line: str) -> str:
    t = re.sub(r"<[^>]+>", "", line)
    return html.unescape(t).strip()


def bulbapedia_fetch_sections(title: str) -> list[dict[str, Any]]:
    params = {
        "action": "parse",
        "page": title,
        "prop": "sections",
        "format": "json",
        "formatversion": "2",
    }
    url = BULB_API + "?" + urllib.parse.urlencode(params)
    data = http_get_json(url)
    if "error" in data:
        return []
    return data.get("parse", {}).get("sections", []) or []


def bulbapedia_section_wikitext(title: str, section_index: str) -> str:
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "section": section_index,
        "format": "json",
        "formatversion": "2",
    }
    url = BULB_API + "?" + urllib.parse.urlencode(params)
    data = http_get_json(url)
    if "error" in data:
        return ""
    return data.get("parse", {}).get("wikitext", "") or ""


def bulbapedia_section_html_text(title: str, section_index: str) -> str:
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "section": section_index,
        "format": "json",
        "formatversion": "2",
    }
    url = BULB_API + "?" + urllib.parse.urlencode(params)
    data = http_get_json(url)
    if "error" in data:
        return ""
    html_snip = data.get("parse", {}).get("text", "") or ""
    return strip_html_to_text(html_snip)


def strip_html_to_text(fragment: str) -> str:
    t = re.sub(r"(?is)<script.*?</script>", "", fragment)
    t = re.sub(r"(?is)<style.*?</style>", "", t)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"</p\s*>", "\n\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def simple_wikitext_to_readable(wt: str) -> str:
    t = wt
    t = re.sub(r"(?s)<ref[^>]*>.*?</ref>", "", t, flags=re.I)
    t = re.sub(r"<ref\s[^>]*/>", "", t, flags=re.I)
    t = re.sub(r"\{\{[^}]+\}\}", "", t)
    t = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", t)
    t = re.sub(r"\[\[([^\]]+)\]\]", r"\1", t)
    t = re.sub(r"''+", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def find_section_index(
    sections: list[dict[str, Any]], *candidates: str
) -> str | None:
    want = {c.lower() for c in candidates}
    for sec in sections:
        line = normalize_section_title(sec.get("line", ""))
        if line.lower() in want:
            return str(sec.get("index", ""))
    for sec in sections:
        line = normalize_section_title(sec.get("line", ""))
        low = line.lower()
        for w in want:
            if w in low:
                return str(sec.get("index", ""))
    return None


def find_dex_section_index(sections: list[dict[str, Any]]) -> str | None:
    """Prefer a section titled like 'Pokédex entries' (not the whole 'Game data' block)."""
    best_i: str | None = None
    best_lvl = -1
    for sec in sections:
        line = normalize_section_title(sec.get("line", "")).lower()
        if "pokédex" not in line and "pokedex" not in line:
            continue
        if "entr" not in line and "entry" not in line:
            continue
        try:
            lvl = int(sec.get("toclevel", 0))
        except (TypeError, ValueError):
            lvl = 0
        if lvl >= best_lvl:
            best_lvl = lvl
            best_i = str(sec.get("index", ""))
    return best_i


FORMS_SUBSECTION_RE = re.compile(r"\n={2,4}\s*Forms\s*={2,4}", re.IGNORECASE)


def is_derived_species_form(row: SpeciesRow) -> bool:
    """True when pokedex.ts gives baseSpecies and this row is not the canonical species name."""
    b = (row.base_species or "").strip()
    return bool(b and b != row.name)


def strip_biology_after_forms_heading(biology: str) -> str:
    """Remove the ===Forms=== wikitext subsection and everything after it from Biology text."""
    if not biology:
        return biology
    m = FORMS_SUBSECTION_RE.search(biology)
    if m:
        return biology[: m.start()].rstrip()
    return biology


def consolidate_base_species_lore(
    rows: list[SpeciesRow],
    records: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Keep only base species (no baseSpecies, or name equals baseSpecies); strip Forms from biology;
    merge taxonomy from every derived form (baseSpecies set, name differs) into the base row
    (comma-separated, deduped, pokedex order for extras).
    """
    by_id: dict[str, dict[str, str]] = {r["id"]: dict(r) for r in records}

    for rec in by_id.values():
        rec["biology"] = strip_biology_after_forms_heading(rec.get("biology", ""))

    extra_taxa: dict[str, list[str]] = {}
    for row in rows:
        if not is_derived_species_form(row):
            continue
        base = (row.base_species or "").strip()
        if not base:
            continue
        rec = by_id.get(row.id)
        if not rec:
            continue
        tax = (rec.get("taxonomy") or "").strip()
        if tax:
            extra_taxa.setdefault(base, []).append(tax)

    out: list[dict[str, str]] = []
    for row in rows:
        if is_derived_species_form(row):
            continue
        rec = by_id.get(row.id)
        if rec is None:
            continue
        new_r = dict(rec)
        extras = extra_taxa.get(row.name, [])
        parts: list[str] = []
        bt = (new_r.get("taxonomy") or "").strip()
        if bt:
            parts.append(bt)
        for t in extras:
            t = t.strip()
            if t and t not in parts:
                parts.append(t)
        new_r["taxonomy"] = ", ".join(parts)
        out.append(new_r)
    return out


def taxonomy_candidates_for_name(name: str) -> list[str]:
    """Generate display names that might appear in taxonomytree.js."""
    seen: list[str] = []

    def add(x: str) -> None:
        if x and x not in seen:
            seen.append(x)

    add(name)
    if "-Alola" in name:
        add(f"Alolan {name.replace('-Alola', '')}")
    if "-Galar" in name:
        add(f"Galarian {name.replace('-Galar', '')}")
    if "-Hisui" in name:
        add(f"Hisuian {name.replace('-Hisui', '')}")
    if name == "Tauros-Paldea-Combat":
        add("Paldean Tauros (Combat Breed)")
    elif name == "Tauros-Paldea-Blaze":
        add("Paldean Tauros (Blaze Breed)")
    elif name == "Tauros-Paldea-Aqua":
        add("Paldean Tauros (Aqua Breed)")
    elif "-Paldea" in name:
        add(f"Paldean {re.sub(r'-Paldea.*', '', name)}")

    m = re.match(r"^(.+)-(Mega|Primal)(?:-([XY]))?$", name)
    if m:
        base, kind, xy = m.group(1), m.group(2), m.group(3)
        if kind == "Mega" and xy:
            add(f"Mega {base} {xy}")
        elif kind == "Mega":
            add(f"Mega {base}")
        elif kind == "Primal":
            add(f"Primal {base}")

    return seen


def load_taxonomy_maps(tree_js: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns:
      child_to_parent: taxonomy node id -> parent id
      label_to_node: Pokémon display label (2nd column) -> node id (1st column) for Species/Subspecies rows
    """
    lit = extract_js_array_literal(tree_js, "var taxonomyTree")
    if not lit:
        lit = extract_js_array_literal(tree_js, "taxonomyTree")
    if not lit:
        return {}, {}
    rows = parse_js_string_array(lit)
    child_to_parent: dict[str, str] = {}
    label_to_node: dict[str, str] = {}
    for row in rows:
        parts = row.split("|")
        if len(parts) != 3:
            continue
        parent, children_csv, rank = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if rank in ("Species", "Subspecies"):
            label_to_node[children_csv] = parent
            continue
        for ch in children_csv.split(","):
            ch = ch.strip()
            if ch:
                child_to_parent[ch] = parent
    return child_to_parent, label_to_node


def taxonomy_path_string(
    label: str,
    child_to_parent: dict[str, str],
    label_to_node: dict[str, str],
) -> str:
    node = label_to_node.get(label)
    if not node:
        return ""
    chain: list[str] = []
    cur: str | None = node
    guard = 0
    while cur and guard < 200:
        chain.append(cur)
        cur = child_to_parent.get(cur)
        guard += 1
    chain.reverse()
    chain.append(label)
    return " > ".join(chain)


def research_string(
    n: int,
    overview_research: list[str],
    overview_diet: list[str],
    overview_caution: list[str],
    overview_care: list[str],
) -> str:
    if n < 0:
        return ""
    parts: list[str] = []

    def pick(arr: list[str]) -> str:
        if n < len(arr):
            return arr[n] or ""
        return ""

    r, d, c, ca = pick(overview_research), pick(overview_diet), pick(
        overview_caution
    ), pick(overview_care)
    if r:
        parts.append("=== Classification / research ===\n" + r)
    if d:
        parts.append("=== Diet ===\n" + d)
    if ca:
        parts.append("=== Care ===\n" + ca)
    if c:
        parts.append("=== Caution ===\n" + c)
    return "\n\n".join(parts)


def scrape_species_bulbapedia(
    row: SpeciesRow,
    delay: float,
    cache: dict[str, tuple[str, str, str]],
) -> tuple[str, str, str]:
    base = row.base_species or row.name
    title = wiki_page_title(base)
    if title in cache:
        return cache[title]

    time.sleep(delay)
    try:
        sections = bulbapedia_fetch_sections(title)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        cache[title] = ("", "", "")
        return cache[title]

    if not sections:
        cache[title] = ("", "", "")
        return cache[title]

    bio_i = find_section_index(sections, "Biology")
    forms_i = find_section_index(sections, "Forms", "Form data")
    dex_i = find_dex_section_index(sections)
    if not dex_i:
        dex_i = find_section_index(
            sections, "Pokédex entries", "Pokedex entries"
        )

    biology = ""
    if bio_i:
        time.sleep(delay)
        try:
            wt = bulbapedia_section_wikitext(title, bio_i)
            biology = simple_wikitext_to_readable(wt)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            pass

    forms = ""
    if forms_i:
        time.sleep(delay)
        try:
            wt = bulbapedia_section_wikitext(title, forms_i)
            forms = simple_wikitext_to_readable(wt)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            pass

    dex = ""
    if dex_i:
        time.sleep(delay)
        try:
            dex = bulbapedia_section_html_text(title, dex_i)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            pass

    cache[title] = (biology, forms, dex)
    return cache[title]


def load_wop_file(url: str, cache_dir: Path | None, filename: str) -> str:
    if cache_dir and (cache_dir / filename).is_file():
        return (cache_dir / filename).read_text(encoding="utf-8", errors="replace")
    return http_get(url)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pokedex",
        default="pokedex.ts",
        help="Path to pokedex.ts",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="pokemon_lore.json",
        help="Output JSON path",
    )
    ap.add_argument(
        "--postprocess-input",
        default="",
        help=(
            "If set, skip scraping: load this lore JSON, apply Forms-strip + regional merge from "
            "--pokedex, and write to --output"
        ),
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="Seconds between Bulbapedia requests",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only first N species (for testing)",
    )
    ap.add_argument(
        "--skip-bulbapedia",
        action="store_true",
        help="Only fetch WoP data (taxonomy + research)",
    )
    ap.add_argument(
        "--wop-cache",
        type=str,
        default="",
        help=(
            "Directory with pre-downloaded WoP files (optional): "
            "pkmnDBNotes.js, pkmnDBDiet.js, pkmnDBCaution.js, pkmnDBCare.js, taxonomytree.js"
        ),
    )
    args = ap.parse_args()

    rows = parse_pokedex_ts(args.pokedex)
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Parsed {len(rows)} species from pokedex.ts", file=sys.stderr)

    if args.postprocess_input:
        try:
            with open(args.postprocess_input, encoding="utf-8") as f:
                existing = json.load(f)
        except OSError as e:
            print(f"Cannot read {args.postprocess_input}: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(existing, list):
            print("postprocess input must be a JSON array", file=sys.stderr)
            sys.exit(1)
        out = consolidate_base_species_lore(rows, existing)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(
            f"Post-processed {args.postprocess_input} -> {args.output} ({len(out)} records)",
            file=sys.stderr,
        )
        return

    wop_dir = Path(args.wop_cache).resolve() if args.wop_cache else None
    if wop_dir and not wop_dir.is_dir():
        print(f"Warning: --wop-cache is not a directory: {wop_dir}", file=sys.stderr)
        wop_dir = None
    env_wop = os.environ.get("WOP_CACHE_DIR", "").strip()
    if not wop_dir and env_wop:
        p = Path(env_wop).resolve()
        if p.is_dir():
            wop_dir = p

    # --- The World of Pokémon: one-time downloads ---
    print("Loading WoP JS bundles...", file=sys.stderr)

    def fetch_wop(name: str, url: str) -> str:
        try:
            return load_wop_file(url, wop_dir, name)
        except urllib.error.HTTPError as e:
            print(
                f"  {name}: HTTP {e.code} (use --wop-cache with saved copy)",
                file=sys.stderr,
            )
            return ""
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  {name}: {e}", file=sys.stderr)
            return ""

    notes_js = fetch_wop("pkmnDBNotes.js", WOP_BASE + "/data/pkmnDBNotes.js")
    diet_js = fetch_wop("pkmnDBDiet.js", WOP_BASE + "/data/pkmnDBDiet.js")
    caution_js = fetch_wop("pkmnDBCaution.js", WOP_BASE + "/data/pkmnDBCaution.js")
    care_js = fetch_wop("pkmnDBCare.js", WOP_BASE + "/data/pkmnDBCare.js")
    tree_js = fetch_wop("taxonomytree.js", WOP_BASE + "/taxonomytree.js")

    overview_research = parse_js_string_array(
        extract_js_array_literal(notes_js, "var overviewResearch") or "[]"
    )
    overview_diet = parse_js_string_array(
        extract_js_array_literal(diet_js, "var overviewDiet") or "[]"
    )
    overview_caution = parse_js_string_array(
        extract_js_array_literal(caution_js, "var overviewCaution") or "[]"
    )
    overview_care = parse_js_string_array(
        extract_js_array_literal(care_js, "var overviewCare") or "[]"
    )

    c2p, l2n = load_taxonomy_maps(tree_js) if tree_js else ({}, {})
    print(
        f"Taxonomy: {len(l2n)} labeled species/subspecies nodes",
        file=sys.stderr,
    )

    bulba_cache: dict[str, tuple[str, str, str]] = {}
    out: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        research = research_string(
            row.num,
            overview_research,
            overview_diet,
            overview_caution,
            overview_care,
        )

        tax = ""
        for cand in taxonomy_candidates_for_name(row.name):
            tax = taxonomy_path_string(cand, c2p, l2n)
            if tax:
                break

        if args.skip_bulbapedia:
            biology, forms, dex = "", "", ""
        else:
            biology, forms, dex = scrape_species_bulbapedia(
                row, args.delay, bulba_cache
            )

        out.append(
            {
                "id": row.id,
                "num": str(row.num),
                "name": row.name,
                "biology": biology,
                "forms": forms,
                "dex": dex,
                "research": research,
                "taxonomy": tax,
            }
        )
        if (idx + 1) % 50 == 0:
            print(f"  ... {idx + 1}/{len(rows)}", file=sys.stderr)

    out = consolidate_base_species_lore(rows, out)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output} ({len(out)} records)", file=sys.stderr)


if __name__ == "__main__":
    main()
