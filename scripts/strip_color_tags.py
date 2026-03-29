#!/usr/bin/env python3
"""
Filter tags_condensed-style JSON by dropping any tag whose hyphen-separated tokens
include a chroma / metal / common color modifier word (e.g. blue-eyes, black-stripes).

Matching is token-based (split on \"-\") so substrings like \"red\" inside \"predator\"
do not count.

Example:
  python scripts/strip_color_tags.py \\
    -i tags_condensed.json -o tags_condensed_no_colors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import FrozenSet

# Lowercase tokens only. Any token equal to one of these removes the whole tag.
_COLOR_TOKENS: FrozenSet[str] = frozenset(
    {
        # Basic
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
        "grey",
        # Common variants / hues
        "crimson",
        "scarlet",
        "vermilion",
        "maroon",
        "burgundy",
        "azure",
        "cyan",
        "teal",
        "turquoise",
        "navy",
        "indigo",
        "violet",
        "magenta",
        "lavender",
        "lilac",
        "mauve",
        "beige",
        "tan",
        "ivory",
        "cream",
        "coral",
        "salmon",
        "peach",
        "amber",
        "gold",
        "golden",
        "silver",
        "silvery",
        "bronze",
        "copper",
        "charcoal",
        "ebony",
        "jade",
        "emerald",
        "ruby",
        "sapphire",
        "pearly",
        "pearl",
        "ochre",
        "russet",
        "aqua",
        "fuchsia",
        "plum",
        "rose",
        "cerulean",
        "ultramarine",
        "chartreuse",
        "sepia",
        "taupe",
        "mahogany",
        "wine",
        # Adjective-ish stems as whole tokens
        "reddish",
        "bluish",
        "greenish",
        "yellowish",
        "whitish",
        "blackish",
        "pinkish",
        "purplish",
        "brownish",
        "grayish",
        "greyish",
        # Shimmer / paint words often used as color descriptors
        "iridescent",
        "multicolored",
        "multicoloured",
    }
)


def tag_has_color_token(tag: str) -> bool:
    parts = tag.lower().strip().split("-")
    return any(p in _COLOR_TOKENS for p in parts if p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        default="tags_condensed.json",
        help="JSON array of tag strings",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="tags_condensed_no_colors.json",
        help="Filtered JSON array (tag names only)",
    )
    args = ap.parse_args()

    try:
        with open(args.input, encoding="utf-8") as f:
            tags = json.load(f)
    except OSError as e:
        print(f"Cannot read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(tags, list):
        print("Input must be a JSON array of strings", file=sys.stderr)
        sys.exit(1)

    original = [str(t).strip() for t in tags if str(t).strip()]
    kept = [t for t in original if not tag_has_color_token(t)]
    removed = len(original) - len(kept)

    out = sorted(kept, key=str.casefold)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Cannot write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Wrote {args.output}: {len(out)} tags kept, {removed} removed (color tokens)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
