#!/usr/bin/env python3
"""
Re-tag Pokémon using sentence embeddings + cosine similarity.

Reads pokemon_lore.json and tags_condensed.json, embeds lore snippets and tag labels with
sentence-transformers, scores tags with sklearn's cosine_similarity, and assigns 3–8 tags
per species. Writes pokemon_tags_retagged.json: [{ "pokemon": str, "tags": [str, ...] }, ...].

Install: pip install -r requirements-retag.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def lore_document(row: dict[str, Any], budget: int) -> str:
    """Single text blob per species for embedding (aligned with tag-generation lore)."""
    per = max(400, budget // 6)
    name = str(row.get("name", "Pokémon"))
    parts = [
        f"Pokémon: {name}",
        f"num: {row.get('num', '')}",
        f"id: {row.get('id', '')}",
        f"taxonomy: {truncate(str(row.get('taxonomy', '')), per)}",
        f"biology: {truncate(str(row.get('biology', '')), per)}",
        f"forms: {truncate(str(row.get('forms', '')), per // 2)}",
        f"dex: {truncate(str(row.get('dex', '')), per)}",
        f"research: {truncate(str(row.get('research', '')), per)}",
        f"classification: {truncate(str(row.get('classification', '')), per // 2)}",
    ]
    text = "\n".join(parts)
    if len(text) > budget:
        return text[: budget - 3] + "..."
    return text


def tag_to_phrase(tag: str) -> str:
    """Hyphenated slugs -> short natural phrases for embedding."""
    t = tag.strip().lower().replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return f"Pokémon trait or behavior: {t}"


def pick_tags_for_row(
    sims: np.ndarray,
    tag_names: list[str],
    *,
    min_tags: int,
    max_tags: int,
    sim_floor: float,
) -> list[str]:
    order = np.argsort(-sims)
    picked: list[str] = []
    for idx in order:
        if len(picked) >= max_tags:
            break
        if sims[idx] < sim_floor and len(picked) >= min_tags:
            break
        name = tag_names[int(idx)]
        if name not in picked:
            picked.append(name)
    if len(picked) < min_tags:
        for idx in order:
            name = tag_names[int(idx)]
            if name not in picked:
                picked.append(name)
            if len(picked) >= min_tags:
                break
    return picked[:max_tags]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="pokemon_lore.json", help="Lore JSON array")
    ap.add_argument(
        "--tags",
        default="tags_condensed.json",
        help="JSON array of allowed tag strings",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="pokemon_tags_retagged.json",
        help="Output: list of { pokemon, tags }",
    )
    ap.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (see SBERT docs)",
    )
    ap.add_argument("--batch-size", type=int, default=32, help="Encode batch size")
    ap.add_argument(
        "--context-chars",
        type=int,
        default=12000,
        help="Max characters of lore per species",
    )
    ap.add_argument(
        "--min-tags",
        type=int,
        default=3,
        help="Minimum tags per species (clamped to <= max-tags)",
    )
    ap.add_argument(
        "--max-tags",
        type=int,
        default=8,
        help="Maximum tags per species",
    )
    ap.add_argument(
        "--sim-floor",
        type=float,
        default=0.22,
        help="Stop adding tags once cosine similarity drops below this (after min-tags)",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, only first N species")
    ap.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. cuda or cpu (default: auto)",
    )
    args = ap.parse_args()

    min_t = max(1, min(args.min_tags, args.max_tags))
    max_t = max(min_t, args.max_tags)

    try:
        with open(args.input, encoding="utf-8") as f:
            rows = json.load(f)
    except OSError as e:
        print(f"Cannot read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(rows, list):
        print("Lore input must be a JSON array", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.tags, encoding="utf-8") as f:
            tag_names = json.load(f)
    except OSError as e:
        print(f"Cannot read {args.tags}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(tag_names, list) or not tag_names:
        print("Tags file must be a non-empty JSON array of strings", file=sys.stderr)
        sys.exit(1)
    tag_names = [str(t).strip() for t in tag_names if str(t).strip()]
    tag_labels = [tag_to_phrase(t) for t in tag_names]

    print(f"Loading model {args.model!r}…", file=sys.stderr)
    model = SentenceTransformer(args.model, device=args.device)
    print(f"Encoding {len(tag_names)} tags…", file=sys.stderr)
    tag_emb = model.encode(
        tag_labels,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    n = len(rows) if args.limit <= 0 else min(len(rows), args.limit)
    docs: list[str] = []
    names: list[str] = []
    for i in range(n):
        row = rows[i]
        if not isinstance(row, dict):
            continue
        pname = str(row.get("name", "")).strip()
        if not pname:
            continue
        names.append(pname)
        docs.append(lore_document(row, args.context_chars))

    if not docs:
        print("No species with names to process", file=sys.stderr)
        sys.exit(1)

    print(f"Encoding {len(docs)} species lore blobs…", file=sys.stderr)
    doc_emb = model.encode(
        docs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Scoring tags (cosine similarity)…", file=sys.stderr)
    sims = cosine_similarity(doc_emb, tag_emb)

    out: list[dict[str, Any]] = []
    for i, pname in enumerate(names):
        row_tags = pick_tags_for_row(
            sims[i],
            tag_names,
            min_tags=min_t,
            max_tags=max_t,
            sim_floor=args.sim_floor,
        )
        out.append({"pokemon": pname, "tags": row_tags})

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Cannot write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Wrote {args.output} ({len(out)} species), "
        f"{min_t}–{max_t} tags each, sim_floor={args.sim_floor}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
