#!/usr/bin/env python3
"""
Re-tag Pokémon using sentence embeddings + cosine similarity.

Reads pokemon_lore.json and tags_condensed.json, embeds lore snippets and tag labels with
sentence-transformers, scores tags with sklearn's cosine_similarity, and assigns 3–8 tags
per species. Optional MMR (--mmr-lambda) and/or an embedding-neighborhood cap
(--max-similar-picked + --similar-pair-threshold) reduce redundant near-synonym tags.
Writes pokemon_tags_retagged.json: [{ "pokemon": str, "tags": [str, ...] }, ...].

Install: pip install -r requirements-retag.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Optional

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
    tag_tag_sim: Optional[np.ndarray],
    *,
    min_tags: int,
    max_tags: int,
    sim_floor: float,
    mmr_lambda: float,
    max_similar_picked: int,
    similar_pair_threshold: float,
) -> list[str]:
    """
    Select tags by relevance (cosine to lore) with optional MMR and a cap on
    near-duplicate tags (embedding cosine ≥ similar_pair_threshold).
    """
    n = len(tag_names)
    order = np.argsort(-sims)
    use_mmr = mmr_lambda < 1.0 - 1e-9
    use_cap = max_similar_picked > 0 and tag_tag_sim is not None

    if not use_mmr and not use_cap:
        picked_idx: list[int] = []
        for idx in order:
            if len(picked_idx) >= max_tags:
                break
            if sims[idx] < sim_floor and len(picked_idx) >= min_tags:
                break
            ii = int(idx)
            if ii not in picked_idx:
                picked_idx.append(ii)
        if len(picked_idx) < min_tags:
            for idx in order:
                ii = int(idx)
                if ii not in picked_idx:
                    picked_idx.append(ii)
                if len(picked_idx) >= min_tags:
                    break
        return [tag_names[i] for i in picked_idx[:max_tags]]

    picked_idx = []
    picked_set: set[int] = set()

    while len(picked_idx) < max_tags:
        best_i: Optional[int] = None
        best_score = -1e9
        for i in range(n):
            if i in picked_set:
                continue
            rel = float(sims[i])
            if len(picked_idx) >= min_tags and rel < sim_floor:
                continue
            if use_cap:
                near = sum(
                    1 for p in picked_idx if tag_tag_sim[i, p] >= similar_pair_threshold
                )
                if near >= max_similar_picked:
                    continue
            if use_mmr:
                max_sim_to_picked = (
                    max(float(tag_tag_sim[i, p]) for p in picked_idx)
                    if picked_idx
                    else 0.0
                )
                score = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim_to_picked
            else:
                score = rel
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        picked_idx.append(best_i)
        picked_set.add(best_i)

    if len(picked_idx) < min_tags:
        for idx in order:
            ii = int(idx)
            if ii in picked_set:
                continue
            picked_idx.append(ii)
            picked_set.add(ii)
            if len(picked_idx) >= min_tags:
                break

    return [tag_names[i] for i in picked_idx[:max_tags]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="pokemon_lore.json", help="Lore JSON array")
    ap.add_argument(
        "--tags",
        default="tags_condensed_removals.json",
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
        default=5,
        help="Maximum tags per species",
    )
    ap.add_argument(
        "--sim-floor",
        type=float,
        default=0.55,
        help="Stop adding tags once cosine similarity drops below this (after min-tags)",
    )
    ap.add_argument(
        "--mmr-lambda",
        type=float,
        default=1.0,
        help=(
            "MMR tradeoff: 1.0 = pure lore relevance (default). "
            "Lower (e.g. 0.6–0.75) penalizes tags too similar to tags already chosen."
        ),
    )
    ap.add_argument(
        "--max-similar-picked",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Cap tags that sit in the same embedding neighborhood: refuse a candidate "
            "if it already has N picked tags with pairwise cosine ≥ --similar-pair-threshold. "
            "0 = disabled (default). Use 1 to allow at most one near-duplicate cluster member."
        ),
    )
    ap.add_argument(
        "--similar-pair-threshold",
        type=float,
        default=0.75,
        help="Cosine between tag embeddings for --max-similar-picked (default: 0.88)",
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
    mmr_lambda = float(np.clip(args.mmr_lambda, 0.0, 1.0))
    use_pairwise = mmr_lambda < 1.0 - 1e-9 or args.max_similar_picked > 0
    tag_tag_sim: Optional[np.ndarray]
    if use_pairwise:
        tag_tag_sim = np.asarray(tag_emb @ tag_emb.T, dtype=np.float64)
    else:
        tag_tag_sim = None

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
            tag_tag_sim,
            min_tags=min_t,
            max_tags=max_t,
            sim_floor=args.sim_floor,
            mmr_lambda=mmr_lambda,
            max_similar_picked=max(0, args.max_similar_picked),
            similar_pair_threshold=float(args.similar_pair_threshold),
        )
        out.append({"pokemon": pname, "tags": row_tags})

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Cannot write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    extra = f", sim_floor={args.sim_floor}"
    if mmr_lambda < 1.0 - 1e-9:
        extra += f", mmr_lambda={mmr_lambda}"
    if args.max_similar_picked > 0:
        extra += (
            f", max_similar_picked={args.max_similar_picked}"
            f"(≥{args.similar_pair_threshold})"
        )
    print(
        f"Wrote {args.output} ({len(out)} species), {min_t}–{max_t} tags each{extra}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
