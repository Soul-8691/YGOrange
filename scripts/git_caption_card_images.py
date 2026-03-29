#!/usr/bin/env python3
"""
Run **Microsoft GIT** image captioning on Yu-Gi-Oh-style card crops (same model family as
nogibjj/Generating-Yu-Gi-Oh-Monsters-From-Archetypes GIT notebooks: ``microsoft/git-base``).

This script is **inference only** (load pretrained or your fine-tuned checkpoint, generate
one caption per image). It does **not** run their full Colab training loop.

**Generic junk captions** (e.g. ``digital art selected for the #``): COCO-pretrained GIT
often does this on out-of-distribution art (YGO crops). Defaults use beam search +
repetition penalty and **retry** with sampling if the line matches known bad patterns.
Fine-tuning on card data (as in nogibjj) fixes this properly; these heuristics only help a bit.

Rough runtime (order-of-magnitude)
------------------------------------
* **This script (inference), GPU:** often about **0.05–0.4 s per image** for ``git-base``
  (plus one-time model load). ~2k images might be **~2–15 minutes** depending on GPU and
  batch size.
* **This script, CPU:** often **~1–6 s per image** — large folders get slow quickly.
* **Their fine-tuning notebooks (training):** highly variable — small archetype splits with
  few epochs can be **tens of minutes to a few GPU-hours**; larger data / more epochs scales
  up. You need their notebooks + a labeled dataset + Trainer setup for that path.

Output: JSON object ``{ "passcode": "caption", ... }`` suitable for
``scripts/merge_git_into_chimeratech.py --git-json``.

Examples::

  python scripts/git_caption_card_images.py -i chimeratech -o git_captions.json --limit 50
  python scripts/git_caption_card_images.py -i chimeratech -c cards.txt -o git_subset.json
  python scripts/git_caption_card_images.py -i chimeratech -o git.json --batch-size 4 --device cuda

Install::

  pip install torch transformers pillow accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from card_image_passcodes import (
    filter_paths_by_passcodes,
    load_cards_allowlist,
    passcode_from_image_stem,
)


def list_images(root: Path, *, recursive: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    it = root.rglob("*") if recursive else root.iterdir()
    out = [p for p in it if p.is_file() and p.suffix.lower() in exts]
    return sorted(out)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# Stock phrases GIT often emits on non-COCO / stylized game art.
_GARBAGE_FRAGMENTS = (
    "digital art selected for the",
    "selected for the #",
    "stock photo",
    "getty images",
)


def looks_like_git_junk_caption(text: str) -> bool:
    t = (text or "").lower().strip()
    if len(t) < 6:
        return True
    return any(g in t for g in _GARBAGE_FRAGMENTS)


def build_gen_kwargs(
    *,
    max_new_tokens: int,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "num_beams": max(1, int(num_beams)),
        "repetition_penalty": float(repetition_penalty),
    }
    if no_repeat_ngram_size > 0:
        kw["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    if num_beams > 1:
        kw["early_stopping"] = True
    if do_sample:
        kw["do_sample"] = True
        kw["temperature"] = float(temperature)
        kw["top_p"] = float(top_p)
    return kw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        default=Path("chimeratech"),
        help="Folder of passcode-named images (e.g. 89631139.jpg)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("git_captions.json"),
        help="Output JSON: { \"89631139\": \"caption\", ... }",
    )
    ap.add_argument(
        "--model",
        default="microsoft/git-base",
        help="HF model id or local path (nogibjj notebooks use microsoft/git-base)",
    )
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        metavar="SEED",
    )
    ap.add_argument(
        "-c",
        "--cards",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Text file of numeric passcodes, one per line (# comments OK). "
            "Only matching images under -i are captioned."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Max generated tokens per caption (notebooks often use ~50)",
    )
    ap.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Beam search width (1 = greedy; higher often reduces junk on GIT)",
    )
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.25,
        help=">1 discourages repeated/generic token loops",
    )
    ap.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Block repeating n-grams; 0 disables",
    )
    ap.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling (usually worse alone; combined with retry can escape junk)",
    )
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument(
        "--retry-garbage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If caption looks like stock GIT junk, retry with sampling (default: on)",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model weights dtype (auto: float16 on CUDA, else float32)",
    )
    args = ap.parse_args()

    if not args.image_dir.is_dir():
        print(f"Not a directory: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

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
        print(
            "No images to process (check folder, --cards, offset, limit).",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_tok = (args.hf_token or os.environ.get("HF_TOKEN") or "").strip()
    load_kw: dict[str, Any] = {"token": raw_tok} if raw_tok else {}

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dtype == "auto":
        use_fp16 = device.type == "cuda"
        torch_dtype = torch.float16 if use_fp16 else torch.float32
    else:
        torch_dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model!r} on {device} ({torch_dtype})…", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(args.model, **load_kw)
    mp_kw = dict(load_kw)
    if device.type == "cuda":
        mp_kw["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **mp_kw)
    model = model.to(device)
    model.eval()

    gen_primary = build_gen_kwargs(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    gen_retry = build_gen_kwargs(
        max_new_tokens=args.max_new_tokens,
        num_beams=1,
        repetition_penalty=min(1.5, args.repetition_penalty + 0.2),
        no_repeat_ngram_size=max(args.no_repeat_ngram_size, 3),
        do_sample=True,
        temperature=0.88,
        top_p=0.92,
    )

    def run_generate(pixel_values: torch.Tensor) -> str:
        with torch.inference_mode():
            g = model.generate(pixel_values=pixel_values, **gen_primary)
        return processor.batch_decode(g, skip_special_tokens=True)[0].strip()

    def run_generate_retry(pixel_values: torch.Tensor) -> str:
        with torch.inference_mode():
            g = model.generate(pixel_values=pixel_values, **gen_retry)
        return processor.batch_decode(g, skip_special_tokens=True)[0].strip()

    def caption_one_image(im: Image.Image) -> str:
        inp = processor(images=[im], return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        if device.type == "cuda":
            inp["pixel_values"] = inp["pixel_values"].to(dtype=torch_dtype)
        pv = inp["pixel_values"]
        cap = run_generate(pv)
        if args.retry_garbage and looks_like_git_junk_caption(cap):
            alt = run_generate_retry(pv)
            if not looks_like_git_junk_caption(alt):
                return alt
        return cap

    bs = max(1, int(args.batch_size))
    out_map: dict[str, str] = {}
    n_ok = n_skip = 0

    for start in range(0, len(paths), bs):
        chunk = paths[start : start + bs]
        images: list[Image.Image] = []
        ids: list[int] = []
        for p in chunk:
            pid = passcode_from_image_stem(p)
            if pid is None:
                n_skip += 1
                continue
            try:
                images.append(load_rgb(p))
                ids.append(pid)
            except OSError as e:
                print(f"Skip {p}: {e}", file=sys.stderr)
                n_skip += 1

        if not images:
            continue

        caps: list[str] = []
        if bs == 1:
            for im in images:
                try:
                    caps.append(caption_one_image(im))
                except Exception as e:
                    print(f"Generation failed: {e!r}", file=sys.stderr)
                    caps.append("")
        else:
            try:
                inputs = processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if device.type == "cuda" and inputs["pixel_values"].dtype != torch_dtype:
                    inputs["pixel_values"] = inputs["pixel_values"].to(
                        dtype=torch_dtype
                    )
                with torch.inference_mode():
                    gen = model.generate(
                        pixel_values=inputs["pixel_values"],
                        **gen_primary,
                    )
                caps = [
                    t.strip() for t in processor.batch_decode(gen, skip_special_tokens=True)
                ]
            except Exception as e:
                print(
                    f"Batch failed ({e!r}); falling back to one-by-one.",
                    file=sys.stderr,
                )
                caps = []
                for im in images:
                    try:
                        caps.append(caption_one_image(im))
                    except Exception as e2:
                        print(f"Skip image: {e2!r}", file=sys.stderr)
                        caps.append("")

            if args.retry_garbage:
                for j, (im, c) in enumerate(zip(images, caps)):
                    if not looks_like_git_junk_caption(c):
                        continue
                    try:
                        caps[j] = caption_one_image(im)
                    except Exception as e2:
                        print(f"Retry failed: {e2!r}", file=sys.stderr)

        for pid, cap in zip(ids, caps):
            cap = (cap or "").strip()
            if not cap:
                continue
            out_map[str(pid)] = cap
            n_ok += 1

        done = min(start + len(chunk), len(paths))
        print(f"  {done}/{len(paths)} images ({n_ok} captions)", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} ({len(out_map)} captions, skipped {n_skip} paths)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
