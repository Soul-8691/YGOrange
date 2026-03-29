#!/usr/bin/env python3
"""
Open-ended visual tags for card images using a local vision–language model (BLIP-2).

Unlike tag_card_images_clip.py, there is no fixed tag list: the model generates text from
the image + prompt (default: comma-separated tags for Yu-Gi-Oh-style card art).

Subset controls match the CLIP script: --limit, --offset, --shuffle-seed, --recursive.

Default model: Salesforce/blip2-opt-2.7b (~5.4 GB weights). GPU strongly recommended; CPU is
very slow and may need a smaller --model.

Install: pip install -r requirements-card-vlm.txt

Examples:
  python scripts/tag_card_images_vlm.py -i chimeratech --limit 5 -o vlm_tags_try.json
  python scripts/tag_card_images_vlm.py -i chimeratech --shuffle-seed 1 --offset 20 --limit 10 -o out.json --merge
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
from transformers import Blip2ForConditionalGeneration, Blip2Processor


def list_images(root: Path, *, recursive: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    it = root.rglob("*") if recursive else root.iterdir()
    out = [p for p in it if p.is_file() and p.suffix.lower() in exts]
    return sorted(out)


def load_image_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def split_tags_from_text(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = re.split(r"[,;]", s)
    out: list[str] = []
    for p in parts:
        t = p.strip().strip("\"'").strip()
        if t and len(t) < 120:
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        default=Path("chimeratech"),
        help="Folder of images (absolute or relative to cwd)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("card_image_tags_vlm.json"),
        help="Output JSON: list of { file, raw [, tags] }",
    )
    ap.add_argument(
        "--model",
        default="Salesforce/blip2-opt-2.7b",
        help="HF model id (BLIP-2 family)",
    )
    ap.add_argument(
        "--prompt",
        default=(
            "Question: List 3 to 8 short comma-separated visual tags for this "
            "Yu-Gi-Oh trading card illustration (characters, colors, setting, objects). "
            "Answer with tags only, no full sentences. Answer:"
        ),
        help="Prompt prepended to generation (BLIP-2 VQA style works well)",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Generation length cap",
    )
    ap.add_argument(
        "--num-beams",
        type=int,
        default=3,
        help="Beam search width (1 = greedy)",
    )
    ap.add_argument("--recursive", action="store_true", help="Include subfolders")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only this many images after offset (e.g. 10–20 for trials)",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many images after sort/shuffle",
    )
    ap.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Shuffle image order with this seed before offset/limit",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="Upsert into existing --output JSON by file path",
    )
    ap.add_argument(
        "--no-parse-tags",
        action="store_true",
        help="Do not split raw text into a tags array",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help="HF token (or HF_TOKEN env) for gated/private Hub models",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cuda, cpu, or empty for auto (prefers CUDA if available)",
    )
    args = ap.parse_args()

    if not args.image_dir.is_dir():
        print(f"Image directory not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    paths = list_images(args.image_dir, recursive=args.recursive)
    if args.shuffle_seed is not None:
        rng = np.random.default_rng(int(args.shuffle_seed))
        rng.shuffle(paths)
    off = max(0, int(args.offset))
    if off:
        paths = paths[off:]
    if args.limit > 0:
        paths = paths[: int(args.limit)]
    if not paths:
        print("No images to process (check folder, offset, limit).", file=sys.stderr)
        sys.exit(1)

    raw_tok = (args.hf_token or os.environ.get("HF_TOKEN") or "").strip()
    hf_tok: Optional[str] = raw_tok or None
    load_kw: dict[str, Any] = {}
    if hf_tok:
        load_kw["token"] = hf_tok

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda = device.type == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Device: {device}, dtype: {dtype}, model: {args.model}", file=sys.stderr)
    print("Loading processor and BLIP-2 (first run downloads weights)…", file=sys.stderr)

    processor = Blip2Processor.from_pretrained(args.model, **load_kw)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        **load_kw,
    ).to(device)
    model.eval()

    out: list[dict[str, Any]] = []
    prompt = args.prompt

    with torch.inference_mode():
        for idx, pth in enumerate(paths, start=1):
            try:
                image = load_image_rgb(pth)
            except OSError as e:
                print(f"Skip unreadable {pth}: {e}", file=sys.stderr)
                continue

            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                num_beams=max(1, int(args.num_beams)),
            )
            raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            try:
                rel = pth.resolve().relative_to(Path.cwd().resolve())
                file_key = rel.as_posix()
            except ValueError:
                file_key = pth.resolve().as_posix()

            rec: dict[str, Any] = {"file": file_key, "raw": raw}
            if not args.no_parse_tags:
                rec["tags"] = split_tags_from_text(raw)
            out.append(rec)
            print(f"  {idx}/{len(paths)} {file_key}", file=sys.stderr)

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
        msg += ", merged"
    msg += ")"
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
