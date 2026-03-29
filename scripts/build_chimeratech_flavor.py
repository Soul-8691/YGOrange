#!/usr/bin/env python3
"""Download YGOProDeck cardinfo, save tab-indented cardinfo.json, build chimeratech.json from chimeratech.txt."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

CARDINFO_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
NORMAL_TYPE = "Normal Monster"


def http_get_json(url: str, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; YGOrange-chimeratech/1.0)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_card_index(payload: dict) -> dict[int, dict]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("cardinfo payload missing list 'data'")
    out: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        cid = item.get("id")
        if isinstance(cid, int):
            out[cid] = item
        elif isinstance(cid, str) and cid.strip().isdigit():
            out[int(cid)] = item
    return out


def parse_id_line(line: str) -> int | None:
    s = line.strip()
    if not s or not s.isdigit():
        return None
    return int(s, 10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cardinfo-url",
        default=CARDINFO_URL,
        help="YGOProDeck cardinfo API URL",
    )
    ap.add_argument(
        "-c",
        "--cardinfo-out",
        default="cardinfo.json",
        help="Path to write full cardinfo (tab-indented JSON)",
    )
    ap.add_argument(
        "--cardinfo-in",
        default="",
        help="If set, load this file instead of downloading (skip fetch)",
    )
    ap.add_argument(
        "-i",
        "--ids",
        default="chimeratech.txt",
        help="Text file: one passcode/id per line",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="chimeratech.json",
        help="Output JSON: ordered list of {id, desc}",
    )
    ap.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    args = ap.parse_args()

    if args.cardinfo_in:
        try:
            with open(args.cardinfo_in, encoding="utf-8") as f:
                payload = json.load(f)
        except OSError as e:
            print(f"Cannot read {args.cardinfo_in}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            payload = http_get_json(args.cardinfo_url, args.timeout)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
            print(f"Failed to fetch cardinfo: {e}", file=sys.stderr)
            sys.exit(1)
        try:
            with open(args.cardinfo_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent="\t")
                f.write("\n")
        except OSError as e:
            print(f"Cannot write {args.cardinfo_out}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Wrote {args.cardinfo_out}", file=sys.stderr)

    try:
        index = load_card_index(payload)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.ids, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Cannot read {args.ids}: {e}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, object]] = []
    skipped = 0
    missing = 0
    for line in lines:
        cid = parse_id_line(line)
        if cid is None:
            skipped += 1
            continue
        card = index.get(cid)
        if card is None:
            missing += 1
            rows.append({"id": cid, "desc": ""})
            continue
        ctype = card.get("type")
        if ctype == NORMAL_TYPE:
            desc = card.get("desc")
            rows.append({"id": cid, "desc": desc if isinstance(desc, str) else ""})
        else:
            rows.append({"id": cid, "desc": ""})

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent="\t")
            f.write("\n")
    except OSError as e:
        print(f"Cannot write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    nonempty = sum(1 for r in rows if r.get("desc"))
    print(
        f"Wrote {args.output}: {len(rows)} rows, {nonempty} non-empty desc, "
        f"missing_ids={missing}, skipped_lines={skipped}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
