#!/usr/bin/env python3
"""
Copy pokemon_lore_merged.json and add string field "classification": lead/summary from English
Wikipedia for the deepest taxonomy label that resolves to an article.

Taxonomy strings use " > " chains; merged regional paths are comma-separated. The Pokémon name
is stripped from the end of each chain when it matches the record's "name". Within each chain, segments are walked right-to-left. For a multi-word segment, the genus
(first word) is tried before the full string so fictitious trinomials resolve to real genera
(e.g. "Rattus normalus kantus" → Rattus). Terminal labels like "Alolan Rattata" are dropped
when the last word matches the species name, so comma-merged paths do not query Pokémon
display names on Wikipedia.

Each comma-separated path is resolved independently; distinct extracts are joined with a blank
line and "---" separator. Identical extracts are deduped.

With --commas: only rows whose taxonomy string contains "," are re-resolved; for those, the
taxonomy passed to Wikipedia is the substring before the first comma (split once). All other
rows are copied unchanged (including classification). The full input length is always written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia blocks or rate-limits unidentified clients; include a clear UA per policy.
UA = (
    "YGOrange-classification/1.0 (Pokémon taxonomy lore; "
    "+https://github.com/) Python-urllib"
)

# Lead paragraphs shorter than this are usually stubs or noise; keep low for edge cases.
MIN_EXTRACT_CHARS = 30


def http_get_json(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise


def expand_segment_to_titles(segment: str) -> list[str]:
    """Wikipedia title attempts for one taxonomy segment; prefer real genus over fake epithets."""
    s = segment.strip()
    if not s:
        return []
    words = s.split()
    if len(words) >= 2:
        # Genus first: avoids bogus pages and matches how users expect (e.g. Rattus).
        return [words[0], s]
    return [s]


def _terminal_matches_species(terminal: str, species_name: str) -> bool:
    """True if this chain tail is the species label (incl. 'Alolan Rattata' when species is Rattata)."""
    t = terminal.strip()
    n = species_name.strip()
    if not t or not n:
        return False
    tl, nl = t.casefold(), n.casefold()
    if tl == nl:
        return True
    toks = t.split()
    if toks and toks[-1].casefold() == nl:
        return True
    if tl.endswith(nl) and (
        len(tl) == len(nl) or tl[-len(nl) - 1].isspace() or tl[-len(nl) - 1] == "-"
    ):
        return True
    return False


def strip_trailing_pokemon_name(parts: list[str], species_name: str) -> list[str]:
    while parts:
        if not _terminal_matches_species(parts[-1], species_name):
            break
        parts = parts[:-1]
    return parts


def _extract_ok(ex: str) -> str | None:
    ex = ex.strip()
    if len(ex) < MIN_EXTRACT_CHARS:
        return None
    low = ex.lower()
    if "may refer to:" in low or low.startswith("may refer to:"):
        return None
    return ex


def wikipedia_lead_extract(title: str, timeout: int) -> str | None:
    """Return plain-text lead section, or None if missing / disambig-only / error."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": "true",
        "explaintext": "true",
        "redirects": "true",
        "format": "json",
        "formatversion": "2",
    }
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    try:
        data = http_get_json(url, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    pages = (data.get("query") or {}).get("pages") or []
    # With redirects, page order is not guaranteed; never assume pages[0] has the extract.
    for p in pages:
        if p.get("missing"):
            continue
        ex = p.get("extract")
        if not isinstance(ex, str):
            continue
        ok = _extract_ok(ex)
        if ok:
            return ok
    return None


def wikipedia_opensearch_first_title(query: str, timeout: int) -> str | None:
    """First enwiki title from OpenSearch, or None."""
    params = {
        "action": "opensearch",
        "search": query,
        "limit": "5",
        "namespace": "0",
        "format": "json",
    }
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    try:
        data = http_get_json(url, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list) or len(data) < 2:
        return None
    titles = data[1]
    if not isinstance(titles, list) or not titles:
        return None
    t0 = titles[0]
    return str(t0) if t0 else None


def wikipedia_resolve_title(title: str, timeout: int) -> str | None:
    """Lead extract for this title, with OpenSearch fallback if direct query yields nothing."""
    ex = wikipedia_lead_extract(title, timeout)
    if ex:
        return ex
    # Fuzzy match (helps if normalization or disambiguation title differs slightly).
    alt = wikipedia_opensearch_first_title(title, timeout)
    if alt and alt.casefold() != title.casefold():
        return wikipedia_lead_extract(alt, timeout)
    return None


def normalize_taxonomy_separators(s: str) -> str:
    """Use ASCII > and , so chains parse even if data used fullwidth punctuation."""
    for a, b in (
        ("\uff1e", ">"),
        ("\u203a", ">"),
        ("\uff0c", ","),
        ("\u201a", ","),
    ):
        s = s.replace(a, b)
    return s


def taxonomy_before_first_comma(taxonomy: str) -> str:
    """Index-0 segment: split only on the first comma (after normalizing comma chars)."""
    s = normalize_taxonomy_separators(taxonomy.strip())
    if "," not in s:
        return s.strip()
    return s.split(",", 1)[0].strip()


def ordered_titles_for_chain(chain: str, species_name: str) -> list[str]:
    chain = normalize_taxonomy_separators(chain)
    parts = [p.strip() for p in chain.split(">") if p.strip()]
    parts = strip_trailing_pokemon_name(parts, species_name)
    seen: set[str] = set()
    out: list[str] = []
    for seg in reversed(parts):
        for t in expand_segment_to_titles(seg):
            key = t.casefold()
            if key not in seen:
                seen.add(key)
                out.append(t)
    return out


def split_taxonomy_paths(taxonomy: str) -> list[str]:
    if not taxonomy or not taxonomy.strip():
        return []
    s = normalize_taxonomy_separators(taxonomy.strip())
    return [p.strip() for p in re.split(r",\s*", s) if p.strip()]


def resolve_chain(
    chain: str,
    species_name: str,
    cache: dict[str, str | None],
    delay: float,
    timeout: int,
) -> tuple[str, str]:
    """First successful extract for this single chain, plus Wikipedia title used."""
    for t in ordered_titles_for_chain(chain, species_name):
        ck = t.casefold()
        if ck not in cache:
            time.sleep(delay)
            cache[ck] = wikipedia_resolve_title(t, timeout)
        ex = cache[ck]
        if ex:
            return ex, t
    return "", ""


def build_classification(
    taxonomy: str,
    species_name: str,
    cache: dict[str, str | None],
    delay: float,
    timeout: int,
) -> tuple[str, str]:
    """
    Returns (classification_text, first_resolved_title for logging).
    """
    paths = split_taxonomy_paths(taxonomy)
    if not paths:
        return "", ""

    chunks: list[str] = []
    seen_body: set[str] = set()
    first_title = ""

    for path in paths:
        text, title = resolve_chain(path, species_name, cache, delay, timeout)
        if not text:
            continue
        key = text.casefold()
        if key in seen_body:
            continue
        seen_body.add(key)
        chunks.append(text)
        if not first_title:
            first_title = title

    if not chunks:
        return "", ""
    if len(chunks) == 1:
        return chunks[0], first_title
    return "\n\n---\n\n".join(chunks), first_title


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        default="pokemon_lore_merged.json",
        help="Source lore JSON",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="pokemon_lore_classified.json",
        help="Output JSON (copy of input + classification field)",
    )
    ap.add_argument("--delay", type=float, default=0.35, help="Seconds between Wikipedia API calls")
    ap.add_argument("--timeout", type=int, default=45, help="HTTP timeout per request")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process and output the first N species (for testing)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Merge with existing output: keep non-empty classification for matching ids",
    )
    ap.add_argument(
        "--commas",
        action="store_true",
        help=(
            "Only re-fetch classification for species whose taxonomy contains a comma; use "
            "taxonomy.split(',', 1)[0] after normalizing comma chars for lookups. Other rows "
            "are copied as-is. Writes "
            "every record (ignores --limit for output size)."
        ),
    )
    args = ap.parse_args()

    try:
        with open(args.input, encoding="utf-8") as f:
            rows_in = json.load(f)
    except OSError as e:
        print(f"Cannot read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(rows_in, list):
        print("Input must be a JSON array", file=sys.stderr)
        sys.exit(1)

    existing_by_id: dict[str, dict[str, Any]] = {}
    if args.skip_existing:
        try:
            with open(args.output, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, list):
                for r in prev:
                    if isinstance(r, dict) and "id" in r:
                        existing_by_id[str(r["id"])] = r
        except OSError:
            pass

    total = len(rows_in)
    n = total if args.limit <= 0 else min(total, args.limit)

    cache: dict[str, str | None] = {}
    out: list[dict[str, Any]] = []

    if args.commas:
        comma_total = sum(
            1
            for r in rows_in
            if isinstance(r, dict)
            and "," in str(r.get("taxonomy", "") or "")
        )
        print(
            f"--commas: will re-resolve {comma_total} species with comma in taxonomy",
            file=sys.stderr,
        )
        for idx in range(total):
            row = rows_in[idx]
            if not isinstance(row, dict):
                out.append(row)
                continue

            taxonomy = str(row.get("taxonomy", "") or "")
            species_name = str(row.get("name", ""))
            new_row = dict(row)

            if "," not in taxonomy:
                out.append(new_row)
                continue

            effective = taxonomy_before_first_comma(taxonomy)
            if not effective:
                new_row["classification"] = ""
                out.append(new_row)
                print(
                    f"  [{idx + 1}/{total}] {species_name}: (empty before first comma)",
                    file=sys.stderr,
                )
                continue

            text, wiki_title = build_classification(
                effective, species_name, cache, args.delay, args.timeout
            )
            new_row["classification"] = text
            out.append(new_row)
            if wiki_title:
                print(
                    f"  [{idx + 1}/{total}] {species_name}: {wiki_title} "
                    f"(taxonomy prefix: {effective[:60]}{'…' if len(effective) > 60 else ''})",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  [{idx + 1}/{total}] {species_name}: (no Wikipedia hit)",
                    file=sys.stderr,
                )

            if (idx + 1) % 50 == 0:
                print(f"  ... row {idx + 1}/{total}", file=sys.stderr)
    else:
        for idx in range(n):
            row = rows_in[idx]
            if not isinstance(row, dict):
                out.append(row)
                continue

            rid = str(row.get("id", idx))
            species_name = str(row.get("name", ""))
            taxonomy = str(row.get("taxonomy", "") or "")

            new_row = dict(row)

            if rid in existing_by_id and args.skip_existing:
                prev_c = existing_by_id[rid].get("classification")
                if isinstance(prev_c, str) and prev_c.strip():
                    new_row["classification"] = prev_c
                    out.append(new_row)
                    continue

            if not taxonomy.strip():
                new_row["classification"] = ""
                out.append(new_row)
            else:
                text, wiki_title = build_classification(
                    taxonomy, species_name, cache, args.delay, args.timeout
                )
                new_row["classification"] = text
                out.append(new_row)
                if wiki_title:
                    print(f"  [{idx + 1}/{n}] {species_name}: {wiki_title}", file=sys.stderr)
                else:
                    print(
                        f"  [{idx + 1}/{n}] {species_name}: (no Wikipedia hit)",
                        file=sys.stderr,
                    )

            if (idx + 1) % 50 == 0:
                print(f"  ... {idx + 1}/{n}", file=sys.stderr)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Cannot write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {args.output} ({len(out)} records)", file=sys.stderr)


if __name__ == "__main__":
    main()
