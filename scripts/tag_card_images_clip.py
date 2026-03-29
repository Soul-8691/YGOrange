#!/usr/bin/env python3
"""
Tag Yu-Gi-Oh (or other) card images with a fixed vocabulary using CLIP (local weights only).

Walks a local image folder, loads tags from JSON (e.g. tags_condensed_removals.json), and
scores each tag with CLIP image–text similarity. If chimeratech.json is provided and the image
filename is a numeric passcode with a non-empty ``desc`` for that id, tag scores blend image
similarity with description–tag similarity (see --desc-weight).

Defaults: 2–5 tags, Yu-Gi-Oh-oriented prompts, optional description fusion.

Large tag lists (e.g. ~6000 strings) are slow mainly because CLIP must encode every tag
× every prompt template once per run; 20 images is cheap after that. Use ``--no-prompt-ensemble``
or a smaller JSON to speed up. ``--hf-token`` / ``HF_TOKEN`` only helps Hub download/auth, not
local scoring.

Examples:
  python scripts/tag_card_images_clip.py -i chimeratech --tags tags_condensed_removals.json
  # Only passcodes listed in cards.txt (one per line), e.g. same format as chimeratech.txt
  python scripts/tag_card_images_clip.py -i chimeratech -c cards.txt --tags tags_condensed_removals.json -o subset.json
  # Quick try: 20 random cards, save aside, tune flags
  python scripts/tag_card_images_clip.py -i chimeratech --shuffle-seed 1 --limit 20 -o card_image_tags_try.json
  # Next slice + merge into one JSON
  python scripts/tag_card_images_clip.py -i chimeratech --shuffle-seed 1 --offset 20 --limit 20 -o card_image_tags_try.json --merge

Install: pip install -r requirements-card-image-tags.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from card_image_passcodes import (
    filter_paths_by_passcodes,
    load_cards_allowlist,
    passcode_from_image_stem,
)

# Averaged CLIP prompts (standard zero-shot trick) — more visually grounded than one template.
DEFAULT_PROMPT_TEMPLATES: list[str] = [
    "Yu-Gi-Oh trading card artwork showing: {}",
    "fantasy trading card illustration of: {}",
    "a photo of printed card art depicting: {}",
    "illustration with visible theme or character: {}",
]


def tag_to_visual_phrase(tag: str, template: str) -> str:
    t = tag.strip().lower().replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    try:
        return template.format(t)
    except (KeyError, IndexError, ValueError):
        return template.replace("{}", t) if "{}" in template else f"{template} {t}"


def _tag_emb_dots_to_picked(
    tag_emb: np.ndarray, i: int, picked: list[int]
) -> np.ndarray:
    """Cosine sims from tag i to each picked index (rows of tag_emb are L2-normalized)."""
    if not picked:
        return np.array([], dtype=np.float32)
    return tag_emb[picked] @ tag_emb[i]


def pick_tags_for_row(
    sims: np.ndarray,
    tag_names: list[str],
    tag_emb: Optional[np.ndarray],
    *,
    min_tags: int,
    max_tags: int,
    sim_floor: float,
    top_margin: float,
    mmr_lambda: float,
    max_similar_picked: int,
    similar_pair_threshold: float,
) -> list[str]:
    n = len(tag_names)
    order = np.argsort(-sims)
    use_mmr = mmr_lambda < 1.0 - 1e-9 and tag_emb is not None
    use_cap = max_similar_picked > 0 and tag_emb is not None
    best = float(np.max(sims)) if n else 0.0
    thresh = (
        max(sim_floor, best - top_margin) if top_margin > 0 else sim_floor
    )

    if not use_mmr and not use_cap:
        picked_idx: list[int] = []
        for idx in order:
            if len(picked_idx) >= max_tags:
                break
            if sims[idx] < thresh and len(picked_idx) >= min_tags:
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

    picked_idx: list[int] = []
    picked_set: set[int] = set()

    while len(picked_idx) < max_tags:
        best_i: Optional[int] = None
        best_score = -1e9
        for i in range(n):
            if i in picked_set:
                continue
            rel = float(sims[i])
            if len(picked_idx) >= min_tags and rel < thresh:
                continue
            if use_cap:
                dots = _tag_emb_dots_to_picked(tag_emb, i, picked_idx)
                if int(np.sum(dots >= similar_pair_threshold)) >= max_similar_picked:
                    continue
            if use_mmr:
                dots_m = _tag_emb_dots_to_picked(tag_emb, i, picked_idx)
                max_sim_to_picked = float(np.max(dots_m)) if len(dots_m) else 0.0
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


def load_image_rgb(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im


def _as_feature_tensor(x: Any, *, tower: str) -> torch.Tensor:
    """CLIP towers should return a 2D float tensor; some transformers builds return ModelOutput."""
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        t = getattr(x, attr, None)
        if isinstance(t, torch.Tensor):
            return t
    raise TypeError(
        f"Expected a Tensor from CLIP {tower} features; got {type(x).__name__}. "
        "Try upgrading transformers or use openai/clip-vit-base-patch32."
    )


def list_images(root: Path, *, recursive: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    it = root.rglob("*") if recursive else root.iterdir()
    out = [p for p in it if p.is_file() and p.suffix.lower() in exts]
    return sorted(out)


def load_chimeratech_desc_map(path: Path) -> dict[int, str]:
    """id -> stripped non-empty desc from chimeratech.json [{ id, desc }, ...]."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
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


def encode_texts_normalized(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    device: Any,
) -> torch.Tensor:
    """Return [N, D] L2-normalized CLIP text embeddings."""
    if not texts:
        return torch.empty(0, 0, device=device)
    enc = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(processor.tokenizer, "model_max_length", 77) or 77,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    tf = model.get_text_features(input_ids=input_ids, attention_mask=attn)
    tf = _as_feature_tensor(tf, tower="text")
    tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        default=Path("chimeratech"),
        help=(
            "Local folder of card images (absolute or relative to cwd; can be outside the repo)"
        ),
    )
    ap.add_argument(
        "--tags",
        type=Path,
        default=Path("tags_condensed_removals.json"),
        help="JSON array of tag strings",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("card_image_tags.json"),
        help="Output JSON: list of { file, tags [, scores] }",
    )
    ap.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="HF model id for CLIP (local weights after first download)",
    )
    ap.add_argument(
        "--prompt-template",
        default="Yu-Gi-Oh trading card artwork suggesting: {}",
        help=(
            "Single-tag prompt when --no-prompt-ensemble; must include {} for the spaced phrase"
        ),
    )
    ap.add_argument(
        "--no-prompt-ensemble",
        action="store_true",
        help="Use only --prompt-template instead of averaging several CLIP prompts per tag",
    )
    ap.add_argument(
        "--top-margin",
        type=float,
        default=0.048,
        help=(
            "Per image: after min-tags, drop tags weaker than (best_tag_score - this). "
            "Reduces unrelated Pokémon-trait noise on Yu-Gi-Oh art. 0 disables."
        ),
    )
    ap.add_argument(
        "--chimeratech",
        type=Path,
        default=Path("chimeratech.json"),
        help="Optional JSON array of { id, desc }; merge desc into scoring when stem is passcode",
    )
    ap.add_argument(
        "--desc-weight",
        type=float,
        default=0.35,
        help=(
            "When desc is present: score = (1-w)*image_tag_sim + w*desc_tag_sim. "
            "0 = image only; 1 = text only"
        ),
    )
    ap.add_argument(
        "--desc-prefix",
        default="Yu-Gi-Oh card flavor text: ",
        help="Prepended to each non-empty desc before CLIP text encoding",
    )
    ap.add_argument("--recursive", action="store_true", help="Include subfolders")
    ap.add_argument("--batch-size", type=int, default=8, help="Images per forward pass")
    ap.add_argument(
        "--text-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="CLIP text encoder batch size (raise for GPU; large tag lists = many batches)",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face token for gated/private Hub models (falls back to HF_TOKEN env). "
            "Does not speed up scoring; only affects download/auth."
        ),
    )
    ap.add_argument("--min-tags", type=int, default=2)
    ap.add_argument("--max-tags", type=int, default=5)
    ap.add_argument(
        "--sim-floor",
        type=float,
        default=0.195,
        help="Minimum cosine (with --top-margin, effective floor is max of this and best-margin)",
    )
    ap.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.72,
        help="1.0 = relevance only; lower adds MMR diversity vs tag–tag similarity",
    )
    ap.add_argument(
        "--max-similar-picked",
        type=int,
        default=1,
        metavar="N",
        help="Cap near-duplicate tags (0=off); default 1",
    )
    ap.add_argument(
        "--similar-pair-threshold",
        type=float,
        default=0.88,
        help="Tag–tag cosine threshold for --max-similar-picked",
    )
    ap.add_argument(
        "--with-scores",
        action="store_true",
        help="Include per-tag cosine scores in output",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process this many images after --offset (e.g. 20 for quick tests)",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many images after sort/shuffle (batch testing next chunk)",
    )
    ap.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="If set, shuffle image list with this seed before offset/limit (reproducible samples)",
    )
    ap.add_argument(
        "-c",
        "--cards",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Text file of numeric passcodes, one per line (# comments OK). "
            "Only matching images (e.g. 89631139.jpg) under -i are processed."
        ),
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="Load existing --output JSON (if any) and upsert by file path, then write merged list",
    )
    ap.add_argument("--device", default=None, help="cuda, cpu, or empty for auto")
    args = ap.parse_args()

    min_t = max(1, min(args.min_tags, args.max_tags))
    max_t = max(min_t, args.max_tags)
    mmr_lambda = float(np.clip(args.mmr_lambda, 0.0, 1.0))
    desc_w = float(np.clip(args.desc_weight, 0.0, 1.0))

    if not args.image_dir.is_dir():
        print(f"Image directory not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)
    try:
        tag_names = json.loads(args.tags.read_text(encoding="utf-8"))
    except OSError as e:
        print(f"Cannot read tags: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(tag_names, list) or not tag_names:
        print("Tags file must be a non-empty JSON array of strings", file=sys.stderr)
        sys.exit(1)
    tag_names = [str(t).strip() for t in tag_names if str(t).strip()]

    desc_map: dict[int, str] = {}
    if args.chimeratech.is_file():
        desc_map = load_chimeratech_desc_map(args.chimeratech)
        print(
            f"Loaded {len(desc_map)} non-empty desc entries from {args.chimeratech}",
            file=sys.stderr,
        )
    paths = list_images(args.image_dir, recursive=args.recursive)
    if args.cards is not None:
        try:
            allow = load_cards_allowlist(args.cards)
        except (OSError, ValueError) as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        paths, missing = filter_paths_by_passcodes(paths, allow)
        if missing:
            sample = sorted(missing)
            if len(sample) > 30:
                sample = sample[:30] + ["…"]
            print(
                f"Warning: no image in -i for {len(missing)} passcode(s): {sample}",
                file=sys.stderr,
            )
        print(
            f"Filtered to {len(paths)} image(s) from {args.cards} ({len(allow)} passcode(s))",
            file=sys.stderr,
        )
    if args.shuffle_seed is not None:
        rng = np.random.default_rng(int(args.shuffle_seed))
        rng.shuffle(paths)
    off = max(0, int(args.offset))
    if off:
        paths = paths[off:]
    if args.limit > 0:
        paths = paths[: int(args.limit)]
    if not paths:
        print(f"No images under {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw_tok = (args.hf_token or os.environ.get("HF_TOKEN") or "").strip()
    hf_tok: Optional[str] = raw_tok or None
    print(f"Device: {device}, model: {args.model}", file=sys.stderr)
    print(f"Loading CLIP…", file=sys.stderr)
    load_kw: dict[str, Any] = {}
    if hf_tok:
        load_kw["token"] = hf_tok
    model = CLIPModel.from_pretrained(args.model, **load_kw).to(device)
    processor = CLIPProcessor.from_pretrained(args.model, **load_kw)
    model.eval()

    templates = (
        [args.prompt_template]
        if args.no_prompt_ensemble
        else list(DEFAULT_PROMPT_TEMPLATES)
    )
    n_tags = len(tag_names)
    n_tmpl = len(templates)
    phrases: list[str] = []
    for t in tag_names:
        for tmpl in templates:
            phrases.append(tag_to_visual_phrase(t, tmpl))
    text_features_list: list[torch.Tensor] = []
    text_bs = max(1, int(args.text_batch_size))
    n_phrase = len(phrases)
    print(
        f"Encoding {n_phrase} tag prompts ({n_tags} tags) — this dominates startup when the list is huge…",
        file=sys.stderr,
    )
    with torch.no_grad():
        for start in range(0, len(phrases), text_bs):
            batch = phrases[start : start + text_bs]
            tf = encode_texts_normalized(model, processor, batch, device)
            text_features_list.append(tf)
    stacked = torch.cat(text_features_list, dim=0)
    # [n_tags * n_tmpl, D] -> mean over templates -> normalize
    d = stacked.shape[1]
    stacked = stacked.view(n_tags, n_tmpl, d).mean(dim=1)
    text_features = stacked / stacked.norm(dim=-1, keepdim=True)
    print(
        f"Tag text embeddings: {n_tags} tags × {n_tmpl} prompt(s) (ensemble={not args.no_prompt_ensemble})",
        file=sys.stderr,
    )

    need_tag_emb = mmr_lambda < 1.0 - 1e-9 or args.max_similar_picked > 0
    tag_emb_np: Optional[np.ndarray] = None
    if need_tag_emb:
        tag_emb_np = text_features.detach().float().cpu().numpy().astype(
            np.float32, copy=False
        )

    out: list[dict[str, Any]] = []
    bs = max(1, args.batch_size)

    with torch.no_grad():
        for start in range(0, len(paths), bs):
            batch_paths = paths[start : start + bs]
            images = []
            ok_paths: list[Path] = []
            for p in batch_paths:
                try:
                    images.append(load_image_rgb(p))
                    ok_paths.append(p)
                except OSError as e:
                    print(f"Skip unreadable {p}: {e}", file=sys.stderr)

            if not images:
                continue

            inp = processor(images=images, return_tensors="pt", padding=True)
            pixels = inp["pixel_values"].to(device)
            img_f = model.get_image_features(pixel_values=pixels)
            img_f = _as_feature_tensor(img_f, tower="image")
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            sims_b = (img_f @ text_features.T).float().cpu().numpy()

            desc_rows: list[int] = []
            desc_strings: list[str] = []
            for row, pth in enumerate(ok_paths):
                if desc_w <= 0 or not desc_map:
                    continue
                cid = passcode_from_image_stem(pth)
                if cid is None:
                    continue
                d = desc_map.get(cid)
                if not d:
                    continue
                desc_rows.append(row)
                desc_strings.append(f"{args.desc_prefix}{d}")

            sims_desc_subset: Optional[np.ndarray] = None
            row_to_desc_idx: dict[int, int] = {}
            if desc_strings:
                d_emb = encode_texts_normalized(
                    model, processor, desc_strings, device
                )
                sims_desc_subset = (d_emb @ text_features.T).float().cpu().numpy()
                row_to_desc_idx = {r: j for j, r in enumerate(desc_rows)}

            for row, pth in enumerate(ok_paths):
                sims = sims_b[row].copy()
                if (
                    sims_desc_subset is not None
                    and row in row_to_desc_idx
                    and desc_w > 0
                ):
                    di = row_to_desc_idx[row]
                    sims = (1.0 - desc_w) * sims + desc_w * sims_desc_subset[di]
                tags = pick_tags_for_row(
                    sims,
                    tag_names,
                    tag_emb_np,
                    min_tags=min_t,
                    max_tags=max_t,
                    sim_floor=args.sim_floor,
                    top_margin=max(0.0, float(args.top_margin)),
                    mmr_lambda=mmr_lambda,
                    max_similar_picked=max(0, args.max_similar_picked),
                    similar_pair_threshold=float(args.similar_pair_threshold),
                )
                try:
                    rel = pth.resolve().relative_to(Path.cwd().resolve())
                    file_key = rel.as_posix()
                except ValueError:
                    file_key = pth.resolve().as_posix()
                rec: dict[str, Any] = {
                    "file": file_key,
                    "tags": tags,
                }
                if args.with_scores:
                    idx_map = {tag_names[j]: float(sims[j]) for j in range(len(tag_names))}
                    rec["scores"] = {t: idx_map[t] for t in tags}
                out.append(rec)

            print(f"  {min(start + len(ok_paths), len(paths))}/{len(paths)}", file=sys.stderr)

    merged_n = 0
    if args.merge and args.output.is_file():
        try:
            prev = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prev = []
        if isinstance(prev, list):
            by_file: dict[str, dict[str, Any]] = {}
            for row in prev:
                if isinstance(row, dict) and row.get("file"):
                    by_file[str(row["file"])] = row
            merged_n = len(by_file)
            for row in out:
                by_file[str(row["file"])] = row
            out = sorted(by_file.values(), key=lambda r: str(r.get("file", "")))

    try:
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"Cannot write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    msg = f"Wrote {args.output} ({len(out)} records"
    if args.merge and merged_n:
        msg += f", merged from prior file"
    msg += ")"
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
