#!/usr/bin/env python3
"""Fetch Yu-Gi-Oh! card konamiCode values from Format Library API (paginated) into chimeratech.txt."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://formatlibrary.com/api/cards"
DEFAULT_FILTER = "tcg:eq:true,tcgDate:lte:2007-01-27"


def fetch_page(page: int, limit: int, sort: str, filter_expr: str, timeout: int) -> list[dict]:
    params = {
        "limit": str(limit),
        "page": str(page),
        "sort": sort,
        "filter": filter_expr,
    }
    url = API_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; YGOrange-konami-rip/1.0)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array from page {page}, got {type(data).__name__}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        default="chimeratech.txt",
        help="Output path (one konamiCode per line)",
    )
    ap.add_argument("--pages", type=int, default=22, help="Number of pages to fetch")
    ap.add_argument("--limit", type=int, default=100, help="Cards per page")
    ap.add_argument("--sort", default="name:asc", help="API sort parameter")
    ap.add_argument("--filter", default=DEFAULT_FILTER, help="API filter parameter")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    args = ap.parse_args()

    codes: list[str] = []
    for page in range(1, args.pages + 1):
        try:
            cards = fetch_page(
                page, args.limit, args.sort, args.filter, args.timeout
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            print(f"Error on page {page}: {e}", file=sys.stderr)
            sys.exit(1)
        for card in cards:
            if not isinstance(card, dict):
                continue
            code = card.get("konamiCode")
            if code is not None and str(code).strip() != "":
                codes.append(str(code).strip())
        print(f"  page {page}/{args.pages}: {len(cards)} cards", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(codes))
        if codes:
            f.write("\n")

    print(f"Wrote {len(codes)} konamiCode lines to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
