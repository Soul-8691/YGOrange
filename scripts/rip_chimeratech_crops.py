#!/usr/bin/env python3
"""Download YGOProDeck card_cropped JPGs for each passcode listed in chimeratech.txt."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://images.ygoprodeck.com/images/cards_cropped/{id}.jpg"
UA = "Mozilla/5.0 (compatible; YGOrange-chimeratech-crops/1.0)"


def parse_passcodes(path: Path) -> list[int]:
    codes: list[int] = []
    seen: set[int] = set()
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        s = line.strip()
        if not s or not s.isdigit():
            continue
        n = int(s, 10)
        if n not in seen:
            seen.add(n)
            codes.append(n)
    return codes


def download_one(url: str, dest: Path, timeout: int) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept": "image/*,*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest.write_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("chimeratech.txt"),
        help="Text file: one passcode per line",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("chimeratech"),
        help="Folder for {id}.jpg files",
    )
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--delay", type=float, default=0.05, help="Seconds between requests")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download if {id}.jpg already exists",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, only first N passcodes")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    codes = parse_passcodes(args.input)
    if args.limit > 0:
        codes = codes[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0
    for idx, cid in enumerate(codes, start=1):
        dest = args.output_dir / f"{cid}.jpg"
        if args.skip_existing and dest.is_file() and dest.stat().st_size > 0:
            skip += 1
            continue
        url = BASE_URL.format(id=cid)
        try:
            download_one(url, dest, args.timeout)
            ok += 1
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"  fail {cid}: {e}", file=sys.stderr)
            fail += 1
        if args.delay > 0 and idx < len(codes):
            time.sleep(args.delay)
        if idx % 200 == 0:
            print(f"  ... {idx}/{len(codes)}", file=sys.stderr)

    print(
        f"Done: {ok} downloaded, {skip} skipped, {fail} failed -> {args.output_dir.resolve()}",
        file=sys.stderr,
    )
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
