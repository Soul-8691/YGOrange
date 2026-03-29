#!/usr/bin/env python3
"""
Download a Hugging Face CLIP model into the local HF cache with one tqdm progress bar.

Uses the same cache layout as transformers / tag_card_images_clip.py, so after this runs,
from_pretrained() can load without waiting on the network.

  python scripts/download_clip_model.py
  python scripts/download_clip_model.py --model openai/clip-vit-large-patch14

Install: pip install huggingface_hub tqdm
(transformers installs huggingface_hub; tqdm is commonly already present)
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="HF model repo id (CLIP weights + processor files)",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help="Optional branch / tag / commit (default: main)",
    )
    args = ap.parse_args()

    api = HfApi()
    try:
        files = api.list_repo_files(
            args.model,
            repo_type="model",
            revision=args.revision,
        )
    except Exception as e:
        print(f"Could not list repo {args.model}: {e}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("Repo returned no files.", file=sys.stderr)
        sys.exit(1)

    bar = tqdm(
        files,
        desc=f"Downloading {args.model}",
        unit="file",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}]",
        file=sys.stderr,
    )
    for rel_path in bar:
        bar.set_postfix_str(rel_path[:48] + ("…" if len(rel_path) > 48 else ""))
        try:
            hf_hub_download(
                args.model,
                filename=rel_path,
                repo_type="model",
                revision=args.revision,
            )
        except Exception as e:
            print(f"\nFailed on {rel_path}: {e}", file=sys.stderr)
            sys.exit(1)

    bar.close()
    print(f"Done. Model cached for transformers: {args.model}", file=sys.stderr)


if __name__ == "__main__":
    main()
