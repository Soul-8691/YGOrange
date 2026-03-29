#!/usr/bin/env python3
"""
Tag card images with a fixed vocabulary using CLIP (local Hugging Face weights only).

Walks an image folder, loads tags from JSON (same format as tags_condensed_removals.json),
embeds each tag as text and each file as an image, ranks by cosine similarity, and writes
3–10 tags per image. No Inference API: model downloads once from the Hub, then runs offline.

Typical:  python scripts/tag_card_images_clip.py -i chimeratech --tags tags_condensed_removals.json

Install: pip install -r requirements-card-image-tags.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def tag_to_visual_phrase(tag: str, template: str) -> str:
    t = tag.strip().lower().replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    try:
        return template.format(t)
    except (KeyError, IndexError, ValueError):
        return template.replace("{}", t) if "{}" in template else f"{template} {t}"


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

    picked_idx: list[int] = []
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


def load_image_rgb(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im


def list_images(root: Path, *, recursive: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    it = root.rglob("*") if recursive else root.iterdir()
    out = [p for p in it if p.is_file() and p.suffix.lower() in exts]
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        default=Path("chimeratech"),
        help="Folder of card images",
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
        default="Pokémon trading card artwork suggesting: {}",
        help="Text prompt for each tag; must include {} for the spaced tag phrase",
    )
    ap.add_argument("--recursive", action="store_true", help="Include subfolders")
    ap.add_argument("--batch-size", type=int, default=8, help="Images per forward pass")
    ap.add_argument("--min-tags", type=int, default=3)
    ap.add_argument("--max-tags", type=int, default=10)
    ap.add_argument(
        "--sim-floor",
        type=float,
        default=0.14,
        help="Stop after min-tags when image–text cosine falls below this (CLIP scale)",
    )
    ap.add_argument(
        "--mmr-lambda",
        type=float,
        default=1.0,
        help="1.0 = relevance only; lower adds MMR diversity vs tag–tag similarity",
    )
    ap.add_argument(
        "--max-similar-picked",
        type=int,
        default=0,
        metavar="N",
        help="Cap near-duplicate tags (0=off); 1 recommended with CLIP",
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
    ap.add_argument("--limit", type=int, default=0, help="If >0, only first N images")
    ap.add_argument("--device", default=None, help="cuda, cpu, or empty for auto")
    args = ap.parse_args()

    min_t = max(1, min(args.min_tags, args.max_tags))
    max_t = max(min_t, args.max_tags)
    mmr_lambda = float(np.clip(args.mmr_lambda, 0.0, 1.0))

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

    paths = list_images(args.image_dir, recursive=args.recursive)
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        print(f"No images under {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, model: {args.model}", file=sys.stderr)
    print(f"Loading CLIP…", file=sys.stderr)
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    phrases = [tag_to_visual_phrase(t, args.prompt_template) for t in tag_names]
    text_features_list: list[torch.Tensor] = []
    text_bs = 64
    with torch.no_grad():
        for start in range(0, len(phrases), text_bs):
            batch = phrases[start : start + text_bs]
            t_in = processor(text=batch, return_tensors="pt", padding=True).to(device)
            tf = model.get_text_features(**t_in)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            text_features_list.append(tf)
    text_features = torch.cat(text_features_list, dim=0)

    use_pairwise = mmr_lambda < 1.0 - 1e-9 or args.max_similar_picked > 0
    tag_tag_sim: Optional[np.ndarray]
    if use_pairwise:
        tag_tag_sim = (
            (text_features @ text_features.T).detach().float().cpu().numpy()
        )
    else:
        tag_tag_sim = None

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

            inp = processor(images=images, return_tensors="pt", padding=True).to(device)
            img_f = model.get_image_features(pixel_values=inp["pixel_values"])
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            sims_b = (img_f @ text_features.T).float().cpu().numpy()

            for row, pth in enumerate(ok_paths):
                sims = sims_b[row]
                tags = pick_tags_for_row(
                    sims,
                    tag_names,
                    tag_tag_sim,
                    min_tags=min_t,
                    max_tags=max_t,
                    sim_floor=args.sim_floor,
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

    try:
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"Cannot write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {args.output} ({len(out)} images)", file=sys.stderr)


if __name__ == "__main__":
    main()
