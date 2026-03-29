#!/usr/bin/env python3
"""
Tag Yu-Gi-Oh cards with 1–4 strings from a fixed JSON tag list (default
``tags_condensed_removals.json``) using **only** card-side text:

* GIT captions (e.g. ``git_subset.json``), and
* Konami flavor from ``chimeratech.json`` when present.

Each passcode in ``--cards`` becomes one document; sentence-transformers embeds that text and
each tag phrase (``tag_to_phrase`` from ``retag_pokemon_embeddings``). Tags are chosen by cosine
similarity with the same optional MMR / near-duplicate cap as the Pokémon retagger — **no
Pokémon lore and no separate classifier training**.

Install::

  pip install -r requirements-ygo-pokemon-tags.txt

Example::

  python scripts/tag_ygo_cards_text.py \\
    --git-json git_subset.json --chimeratech chimeratech.json \\
    --tags tags_condensed_removals.json --cards cards.txt \\
    -o ygo_card_tags.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from retag_pokemon_embeddings import pick_tags_for_row, tag_to_phrase, truncate


def _norm_passcode(s: str) -> int:
    s = str(s).strip()
    if not s:
        raise ValueError("empty passcode")
    return int(s, 10)


def load_git_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object passcode -> caption")
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            pid = int(str(k).strip(), 10)
        except (TypeError, ValueError):
            continue
        if isinstance(v, str) and v.strip():
            out[pid] = v.strip()
    return out


def load_chimeratech_desc(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return {}
    out: dict[int, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        cid = row.get("id")
        desc = row.get("desc", "")
        if cid is None:
            continue
        try:
            key = int(cid)
        except (TypeError, ValueError):
            continue
        if isinstance(desc, str) and desc.strip():
            out[key] = desc.strip()
    return out


def load_cards_list(path: Path) -> list[int]:
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        out.append(_norm_passcode(line))
    return out


def load_tag_vocabulary(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must be a non-empty JSON array of tag strings")
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        s = str(t).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def ygo_card_document(
    passcode: int,
    caption: str,
    desc: str,
    *,
    cap_budget: int,
    desc_budget: int,
) -> str:
    parts = [f"Yu-Gi-Oh trading card (passcode {passcode})."]
    if caption.strip():
        parts.append(f"Image caption: {truncate(caption, cap_budget)}")
    if desc.strip():
        parts.append(f"Official flavor text: {truncate(desc, desc_budget)}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--git-json", type=Path, default=Path("git_subset.json"))
    ap.add_argument("--chimeratech", type=Path, default=Path("chimeratech.json"))
    ap.add_argument(
        "--tags",
        type=Path,
        default=Path("tags_condensed_removals.json"),
        help="JSON array of allowed tag strings",
    )
    ap.add_argument("--cards", type=Path, default=Path("cards.txt"))
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("ygo_card_tags.json"),
    )
    ap.add_argument("--st-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--encode-batch-size", type=int, default=32)
    ap.add_argument("--min-tags", type=int, default=1)
    ap.add_argument("--max-tags", type=int, default=5)
    ap.add_argument(
        "--sim-floor",
        type=float,
        default=0.3,
        help="Min cosine to keep a tag after min-tags are filled (tune for YGO text)",
    )
    ap.add_argument(
        "--mmr-lambda",
        type=float,
        default=1,
        help="1.0 = relevance only; lower adds MMR diversity vs tag–tag similarity",
    )
    ap.add_argument(
        "--max-similar-picked",
        type=int,
        default=1,
        metavar="N",
        help="Cap near-duplicate tags (0=off)",
    )
    ap.add_argument(
        "--similar-pair-threshold",
        type=float,
        default=0.75,
        help="Tag–tag cosine threshold for --max-similar-picked",
    )
    ap.add_argument("--caption-chars", type=int, default=400)
    ap.add_argument("--desc-chars", type=int, default=800)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    min_t = max(1, min(args.min_tags, args.max_tags))
    max_t = max(min_t, args.max_tags)
    mmr_lambda = float(np.clip(args.mmr_lambda, 0.0, 1.0))

    tag_names = load_tag_vocabulary(args.tags)
    cards = load_cards_list(args.cards)
    if not cards:
        print("No passcodes in --cards.", file=sys.stderr)
        sys.exit(1)

    git_map = load_git_map(args.git_json) if args.git_json.is_file() else {}
    desc_map = load_chimeratech_desc(args.chimeratech) if args.chimeratech.is_file() else {}

    docs: list[str] = []
    for pid in cards:
        docs.append(
            ygo_card_document(
                pid,
                git_map.get(pid, ""),
                desc_map.get(pid, ""),
                cap_budget=args.caption_chars,
                desc_budget=args.desc_chars,
            )
        )

    device_kw: dict[str, Any] = {}
    if args.device:
        device_kw["device"] = args.device
    st = SentenceTransformer(args.st_model, **device_kw)

    tag_labels = [tag_to_phrase(t) for t in tag_names]
    print(f"Encoding {len(tag_names)} tags…", file=sys.stderr)
    tag_emb = st.encode(
        tag_labels,
        batch_size=args.encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    tag_emb = np.asarray(tag_emb, dtype=np.float32)
    use_pairwise = mmr_lambda < 1.0 - 1e-9 or args.max_similar_picked > 0
    if use_pairwise:
        tag_tag_sim = np.asarray(tag_emb @ tag_emb.T, dtype=np.float64)
    else:
        tag_tag_sim = None

    print(f"Encoding {len(docs)} card texts…", file=sys.stderr)
    doc_emb = st.encode(
        docs,
        batch_size=args.encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    doc_emb = np.asarray(doc_emb, dtype=np.float32)
    sims_all = np.asarray(doc_emb @ tag_emb.T, dtype=np.float64)

    out: list[dict[str, Any]] = []
    tag_to_i = {t: i for i, t in enumerate(tag_names)}
    for j, pid in enumerate(cards):
        row_sims = sims_all[j]
        picked = pick_tags_for_row(
            row_sims,
            tag_names,
            tag_tag_sim,
            min_tags=min_t,
            max_tags=max_t,
            sim_floor=float(args.sim_floor),
            mmr_lambda=mmr_lambda,
            max_similar_picked=max(0, args.max_similar_picked),
            similar_pair_threshold=float(args.similar_pair_threshold),
        )
        out.append(
            {
                "passcode": pid,
                "tags": picked,
                "scores": {t: float(row_sims[tag_to_i[t]]) for t in picked},
            }
        )

    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(out)} cards)", file=sys.stderr)


if __name__ == "__main__":
    main()
