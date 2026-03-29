"""Filter card image runs by numeric passcode (filename stem, e.g. 89631139.jpg)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def passcode_from_image_stem(path: Path) -> Optional[int]:
    stem = path.stem.strip()
    return int(stem, 10) if stem.isdigit() else None


def parse_cards_file(path: Path) -> list[int]:
    """One passcode per line; # starts a comment; blank lines skipped."""
    text = path.read_text(encoding="utf-8")
    out: list[int] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.isdigit():
            out.append(int(line, 10))
    return out


def load_cards_allowlist(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Cards list not found: {path}")
    ids = parse_cards_file(path)
    if not ids:
        raise ValueError(f"No numeric passcodes parsed from {path}")
    return set(ids)


def filter_paths_by_passcodes(
    paths: list[Path], allow: set[int]
) -> tuple[list[Path], set[int]]:
    """Keep paths whose stem is a passcode in allow. Returns (filtered, missing_from_disk)."""
    filtered: list[Path] = []
    found: set[int] = set()
    for p in paths:
        c = passcode_from_image_stem(p)
        if c is not None and c in allow:
            filtered.append(p)
            found.add(c)
    return filtered, allow - found
