#!/usr/bin/env python3
"""
Generate 3–10 descriptive tags per Pokémon from pokemon_lore.json via the Hugging Face
Inference API (OpenAI-compatible chat completions on the HF router).

Authentication: Hugging Face token only—``--api-key hf_...`` or ``HF_TOKEN`` /
``HUGGING_FACE_HUB_TOKEN`` (needs permission to call Inference).

Default ``--chat-url`` is ``https://router.huggingface.co/v1/chat/completions``.

The router picks an Inference Provider per model. Some models are served via Together; that
path can return a Cloudflare HTML page (HTTP 403) even though you never called
``api.together.xyz`` yourself. Pin a provider with ``--inference-provider`` (default
``novita``), or put the suffix on the model id (e.g. ``Qwen/Qwen3.5-27B:novita``).

If you use **Groq** (``--inference-provider groq``), its edge (``api.groq.com``) may return
Cloudflare **1010** for some client signatures. This script sends a normal browser
``User-Agent`` by default; override with ``HF_CHAT_USER_AGENT`` or ``--user-agent``.

Writes:
  - pokemon_tags.json: [{ "pokemon": "<Name>", "tags": ["a-b", "c", ...] }, ...]
  - tags.json: every unique tag, sorted alphabetically
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Qwen3.5 chat model (instruction-tuned); Novita serves this on the HF router (not Qwen2.5-32B-Instruct).
DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

# Groq Cloudflare 1010 often targets Python-urllib / python-requests default User-Agents.
DEFAULT_CHAT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def resolve_router_model(model: str, inference_provider: str) -> str:
    """Pin HF Inference provider so the router does not forward to Together (Cloudflare 403)."""
    m = (model or "").strip()
    if not m or ":" in m:
        return m
    p = (inference_provider or "").strip().lower()
    if p in ("", "auto", "none"):
        return m
    return f"{m}:{p}"


def resolve_user_agent(cli_override: str) -> str:
    o = (cli_override or "").strip()
    if o:
        return o
    env = (os.environ.get("HF_CHAT_USER_AGENT") or "").strip()
    if env:
        return env
    return DEFAULT_CHAT_USER_AGENT


def _groq_cf_browser_ban(body: str) -> bool:
    if not body:
        return False
    if "browser_signature_banned" in body.lower():
        return True
    try:
        j = json.loads(body)
        if isinstance(j, dict):
            if j.get("error_name") == "browser_signature_banned":
                return True
            if j.get("error_code") == 1010 and j.get("zone") == "api.groq.com":
                return True
    except json.JSONDecodeError:
        pass
    return False


class ChatHttpError(Exception):
    def __init__(self, code: int, body: str) -> None:
        self.code = code
        self.body = body
        super().__init__(f"HTTP {code}: {body[:600]}")


SYSTEM_PROMPT = """You label fictional Pokémon creatures for search and filtering.
Rules:
- Your entire reply must be ONE JSON array of 3 to 10 strings and nothing else. No markdown fences, no prose before or after.
- Each string is one or two English words; if two words, join with a single hyphen (e.g. photosynthesis, back-bulb, poison-powder).
- Use lowercase letters and digits only inside tags (no spaces except you must use hyphen for two-word tags).
- Tags describe physiology, behavior, habitat, diet, temperament, or notable traits from the text—not the Pokémon's proper name unless unavoidable.
Example (format only): ["photosynthesis", "back-bulb", "poison-powder"]
"""


def truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def lore_brief(row: dict[str, Any], budget: int) -> str:
    """Pack main lore fields into a bounded string for the prompt."""
    per = max(400, budget // 6)
    parts = [
        f"num: {row.get('num', '')}",
        f"name: {row.get('name', '')}",
        f"id: {row.get('id', '')}",
        f"taxonomy: {truncate(str(row.get('taxonomy', '')), per)}",
        f"biology: {truncate(str(row.get('biology', '')), per)}",
        f"forms: {truncate(str(row.get('forms', '')), per // 2)}",
        f"dex: {truncate(str(row.get('dex', '')), per)}",
        f"research: {truncate(str(row.get('research', '')), per)}",
        f"classification: {truncate(str(row.get('classification', '')), per // 2)}",
    ]
    text = "\n".join(parts)
    if len(text) > budget:
        return text[: budget - 3] + "..."
    return text


def _chat_headers(api_key: str, user_agent: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.9",
    }


def _post_chat_json(
    chat_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    prefer_requests: bool,
) -> dict[str, Any]:
    use_requests = prefer_requests
    if use_requests:
        try:
            import requests
        except ImportError:
            use_requests = False
    if use_requests:
        import requests

        r = requests.post(chat_url, json=payload, headers=headers, timeout=timeout)
        if r.status_code >= 400:
            raise ChatHttpError(r.status_code, r.text or "")
        return r.json()

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(chat_url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise ChatHttpError(e.code, err_body) from e


def chat_complete(
    api_key: str,
    chat_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int,
    *,
    user_agent: str,
    prefer_requests: bool = False,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = _chat_headers(api_key, user_agent)
    data: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            data = _post_chat_json(
                chat_url, payload, headers, timeout, prefer_requests
            )
            break
        except ChatHttpError as e:
            if e.code in (429, 503) and attempt < 2:
                time.sleep(2.0 * (attempt + 1) + random.uniform(0, 0.5))
                continue
            body_low = (e.body or "").lower()
            if "together.xyz" in body_low and "cloudflare" in body_low:
                raise RuntimeError(
                    "Hugging Face routed this request to Together, which blocked it "
                    "(Cloudflare). Pin another provider, e.g. "
                    "--inference-provider novita (default) or "
                    "--model your/hub-model:groq — see HF Inference Providers docs."
                ) from e
            if _groq_cf_browser_ban(e.body or ""):
                raise RuntimeError(
                    "Groq (behind HF Inference) blocked the HTTP client (Cloudflare 1010: "
                    "browser/user-agent signature). This script already uses a non-Python "
                    "default User-Agent; try --user-agent with your real browser’s string, "
                    "set HF_CHAT_USER_AGENT, use another network/VPN, or use a non-Groq "
                    "provider (this script defaults to --inference-provider novita)."
                ) from e
            raise RuntimeError(str(e)) from e
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(str(e)) from e
    if data is None:
        raise RuntimeError("No response from chat API after retries")

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in API response: {json.dumps(data)[:800]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected message content: {msg!r}")
    return content.strip()


def _strip_markdown_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


# Qwen3.5 chat template closes reasoning with </think> then emits the user-visible answer.
_QWEN_THINK_END = "".join((chr(60), chr(47), "think", chr(62)))


def _strip_thinking_blocks(text: str) -> str:
    """Keep only the segment after the Qwen </think> delimiter so JSON can be parsed."""
    t = text.strip()
    if _QWEN_THINK_END in t:
        t = t.rsplit(_QWEN_THINK_END, 1)[-1].strip()
    return t


def _json_balanced_span(s: str, start: int, open_c: str, close_c: str) -> str | None:
    """Extract a JSON object or array starting at start; respects quoted strings."""
    if start >= len(s) or s[start] != open_c:
        return None
    depth = 0
    i = start
    in_str = False
    esc = False
    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
        i += 1
    return None


def _try_parse_json_list(s: str) -> list[Any] | None:
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, list) else None


def _try_parse_tags_object(s: str) -> list[Any] | None:
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(v, dict):
        return None
    tags = v.get("tags")
    return tags if isinstance(tags, list) else None


def _line_based_tag_fallback(text: str) -> list[Any] | None:
    """Bullets or numbered lines when the model ignores JSON."""
    tags: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(
            r"^[-*•]\s*[`\"']?([a-zA-Z0-9][a-zA-Z0-9\s-]{0,48})[`\"']?\s*$",
            ln,
        )
        if m:
            tags.append(m.group(1).strip().lower().replace(" ", "-"))
            continue
        m2 = re.match(r"^\d+[.)]\s*[`\"']?(.+)$", ln)
        if m2:
            inner = m2.group(1).strip().strip('`"\'')
            if inner:
                tags.append(inner.lower().replace(" ", "-"))
    return tags if len(tags) >= 3 else None


def extract_json_array(text: str) -> list[Any]:
    raw = text or ""
    t = _strip_thinking_blocks(raw)
    t = _strip_markdown_fences(t)
    t = t.strip()
    if not t:
        raise ValueError(f"No JSON array in model output (empty): {raw[:200]!r}...")

    # Whole string is JSON
    if t[0] == "[":
        span = _json_balanced_span(t, 0, "[", "]")
        if span:
            got = _try_parse_json_list(span)
            if got is not None:
                return got
    if t[0] == "{":
        span = _json_balanced_span(t, 0, "{", "}")
        if span:
            got = _try_parse_tags_object(span)
            if got is not None:
                return got
            got2 = _try_parse_json_list(span)
            if got2 is not None:
                return got2

    # First balanced [...] or {...} anywhere
    for i, ch in enumerate(t):
        if ch == "[":
            span = _json_balanced_span(t, i, "[", "]")
            if span:
                got = _try_parse_json_list(span)
                if got is not None:
                    return got
        elif ch == "{":
            span = _json_balanced_span(t, i, "{", "}")
            if span:
                got = _try_parse_tags_object(span)
                if got is not None:
                    return got

    lb = _line_based_tag_fallback(t)
    if lb is not None:
        return lb

    raise ValueError(f"No JSON array in model output: {t[:280]!r}...")


_TAG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)?$")


def normalize_tags(raw: list[Any]) -> list[str]:
    out: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue
        t = x.strip().lower().replace(" ", "-")
        t = re.sub(r"[^a-z0-9-]", "", t)
        t = re.sub(r"-+", "-", t).strip("-")
        if not t:
            continue
        if not _TAG_RE.match(t):
            continue
        if t not in out:
            out.append(t)
    if len(out) < 3:
        raise ValueError(f"Too few valid tags after normalize: {raw!r} -> {out}")
    if len(out) > 10:
        out = out[:10]
    return out


def generate_tags_for_species(
    row: dict[str, Any],
    api_key: str,
    chat_url: str,
    model: str,
    context_chars: int,
    max_tokens: int,
    temperature: float,
    timeout: int,
    *,
    user_agent: str,
    prefer_requests: bool = False,
    json_repair: bool = True,
) -> list[str]:
    name = str(row.get("name", "Pokémon"))
    user = (
        f"Here is lore for the Pokémon {name}.\n\n"
        f"{lore_brief(row, context_chars)}\n\n"
        f"Return the JSON array of 3–10 tags for {name}."
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    def parse_tags_from_content(content: str) -> list[str]:
        arr = extract_json_array(content)
        return normalize_tags(arr)

    content = chat_complete(
        api_key,
        chat_url,
        model,
        messages,
        max_tokens,
        temperature,
        timeout,
        user_agent=user_agent,
        prefer_requests=prefer_requests,
    )
    try:
        return parse_tags_from_content(content)
    except ValueError:
        if not json_repair:
            raise
        snippet = (content or "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        repair_user = (
            "That reply was not usable. Send ONE line only: a JSON array of 3–10 strings "
            '(lowercase tags, hyphen for two words), e.g. ["electric-mane","zebra-stripes","hooves"]. '
            "No markdown, no thinking blocks, no explanation.\n\n"
            f"Your broken reply was:\n{snippet}"
        )
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": repair_user})
        content2 = chat_complete(
            api_key,
            chat_url,
            model,
            messages,
            max_tokens,
            min(0.2, temperature),
            timeout,
            user_agent=user_agent,
            prefer_requests=prefer_requests,
        )
        return parse_tags_from_content(content2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--api-key",
        default="",
        help="Hugging Face token (hf_...); or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN",
    )
    ap.add_argument(
        "-i",
        "--input",
        default="pokemon_lore.json",
        help="Input lore JSON (array of species objects)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="pokemon_tags.json",
        help="Output: list of { pokemon, tags }",
    )
    ap.add_argument(
        "--tags-output",
        default="tags.json",
        help="Output: sorted unique tags (JSON array of strings)",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hub model id; combined with --inference-provider as model:provider if no ':'",
    )
    ap.add_argument(
        "--inference-provider",
        default="novita",
        help=(
            "HF Inference Providers suffix when model has no ':'. "
            "Default novita avoids Together and Groq edge issues. "
            "Use auto/none to let HF pick the provider."
        ),
    )
    ap.add_argument(
        "--chat-url",
        default=CHAT_URL,
        help="HF Inference chat URL (default: HF router only)",
    )
    ap.add_argument(
        "--prefer-requests",
        action="store_true",
        help="Send HTTP via requests library if installed",
    )
    ap.add_argument(
        "--user-agent",
        default="",
        help=(
            "HTTP User-Agent (some provider edges ban Python default clients). "
            "Default: browser-like string; override or set HF_CHAT_USER_AGENT."
        ),
    )
    ap.add_argument(
        "--context-chars",
        type=int,
        default=12000,
        help="Max characters of lore to send per species",
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.45)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--delay", type=float, default=0.25, help="Pause between API calls (seconds)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only first N species")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, skip species whose pokemon name already appears in it",
    )
    ap.add_argument(
        "--no-json-repair",
        action="store_true",
        help="Do not send a second API call when the model output is not parseable JSON",
    )
    args = ap.parse_args()

    cu = args.chat_url.strip().lower()
    if "together.xyz" in cu or "together.ai" in cu:
        print(
            "This script only calls the Hugging Face Inference router. "
            "You passed a Together URL (Cloudflare will block it without a Together account).",
            file=sys.stderr,
        )
        print(
            f"Omit --chat-url or set it explicitly to:\n  {CHAT_URL}",
            file=sys.stderr,
        )
        sys.exit(1)

    key = (
        (args.api_key or "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if not key:
        print(
            "Missing Hugging Face token: --api-key hf_... or HF_TOKEN / HUGGING_FACE_HUB_TOKEN",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved_model = resolve_router_model(args.model, args.inference_provider)
    user_agent = resolve_user_agent(args.user_agent)
    print(f"Using model: {resolved_model}", file=sys.stderr)

    try:
        with open(args.input, encoding="utf-8") as f:
            rows = json.load(f)
    except OSError as e:
        print(f"Cannot read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(rows, list):
        print("Input must be a JSON array", file=sys.stderr)
        sys.exit(1)

    done_names: set[str] = set()
    out_rows: list[dict[str, Any]] = []
    if args.resume and os.path.isfile(args.output):
        try:
            with open(args.output, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, list):
                for item in prev:
                    if isinstance(item, dict) and item.get("pokemon"):
                        done_names.add(str(item["pokemon"]))
                        out_rows.append(item)
        except (OSError, json.JSONDecodeError):
            pass

    n = len(rows) if args.limit <= 0 else min(len(rows), args.limit)
    failures = 0

    for idx in range(n):
        row = rows[idx]
        if not isinstance(row, dict):
            continue
        pname = str(row.get("name", "")).strip()
        if not pname:
            continue
        if pname in done_names:
            continue

        try:
            tags = generate_tags_for_species(
                row,
                key,
                args.chat_url,
                resolved_model,
                args.context_chars,
                args.max_tokens,
                args.temperature,
                args.timeout,
                user_agent=user_agent,
                prefer_requests=args.prefer_requests,
                json_repair=not args.no_json_repair,
            )
            out_rows.append({"pokemon": pname, "tags": tags})
            done_names.add(pname)
            print(f"  [{len(out_rows)}] {pname}: {tags}", file=sys.stderr)
        except (ValueError, RuntimeError, KeyError, TypeError) as e:
            print(f"  FAIL {pname}: {e}", file=sys.stderr)
            failures += 1

        if args.delay > 0 and idx < n - 1:
            time.sleep(args.delay)

        if (idx + 1) % 25 == 0:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(out_rows, f, ensure_ascii=False, indent=2)
            except OSError:
                pass

    all_tags: set[str] = set()
    for item in out_rows:
        for t in item.get("tags", []) or []:
            if isinstance(t, str) and t.strip():
                all_tags.add(t.strip())

    sorted_tags = sorted(all_tags, key=str.casefold)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)
        with open(args.tags_output, "w", encoding="utf-8") as f:
            json.dump(sorted_tags, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Cannot write output: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Wrote {args.output} ({len(out_rows)} species), {args.tags_output} ({len(sorted_tags)} unique tags)",
        file=sys.stderr,
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
