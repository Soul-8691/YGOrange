# Tagging pipelines (YGOrange)

This document summarizes the **text-tag** and **image-tag** workflows in this repo: how data moves between JSON files, which scripts to run, and what each optional stack installs. It also covers **aggregate Pokémon stats** (including per–trait-tag buckets) and the **Tkinter stats picker**.

---

## 1. Pokémon species tags (text / lore)

**Goal:** Assign a small set of hyphenated trait tags to each species using **written lore**, not card art.

| Step | Script | Input | Output |
|------|--------|--------|--------|
| Lore | `scripts/scrape_pokemon_lore.py` (and related) | (sources) | `pokemon_lore.json` |
| LLM tags | `scripts/generate_pokemon_tags_hf.py` | `pokemon_lore.json` | `pokemon_tags.json`, `tags.json` |
| Embedding retag | `scripts/retag_pokemon_embeddings.py` | `pokemon_lore.json`, tag list (e.g. `tags_condensed.json`) | `pokemon_tags_retagged.json` |
| Tag counts | `scripts/generate_tags_count.py` | retagged JSON | `tags_count.json` |
| Strip colors | `scripts/strip_color_tags.py` | tag list JSON | e.g. `tags_condensed_no_colors.json` |

**Retagger details:** Uses [sentence-transformers](https://www.sbert.net/) to embed lore and tag phrases, then cosine similarity. Supports optional **MMR** (`--mmr-lambda`) and a **near-duplicate cap** (`--max-similar-picked`, `--similar-pair-threshold`) so redundant synonyms do not fill the list.

**Install:** `pip install -r requirements-retag.txt`

---

## 2. Yu-Gi-Oh! card crops and flavor sidecar

| Asset | Role |
|--------|------|
| `chimeratech.txt` | Passcodes (one per line) for downloads |
| `scripts/rip_chimeratech_crops.py` | Downloads cropped JPGs from YGOProDeck |
| `chimeratech/` | `{passcode}.jpg` crops |
| `chimeratech.json` | Per-passcode records; **`desc`** = flavor / structure text when present |
| `scripts/build_chimeratech_flavor.py` | (related) flavor/build helpers for chimeratech data |
| `cardinfo.json` | Large local card DB (when used elsewhere) |

---

## 3. Card image tags with CLIP (fixed vocabulary)

**Goal:** Score each crop against a **JSON list of tag strings** (e.g. `tags_condensed_removals.json`) using **local** [OpenAI CLIP](https://github.com/openai/CLIP) weights via [Transformers](https://huggingface.co/docs/transformers)—no paid Inference API required.

**Script:** `scripts/tag_card_images_clip.py`

**Features (high level):**

- Any **local image folder** (`-i`); filenames `12345.jpg` tie into `chimeratech.json` for **`desc`** fusion when non-empty.
- **Prompt ensemble** (default): several Yu-Gi-Oh–oriented templates averaged per tag (disable with `--no-prompt-ensemble`).
- **Top-margin** filter: drops tags far below the best score on that image (reduces weak matches).
- **Subset runs:** `--limit`, `--offset`, `--shuffle-seed`; **`--merge`** to upsert into an existing output JSON by `file`.
- **Performance:** Large tag lists (e.g. thousands) cost time mostly in **one-time text encoding**; full N×N tag–tag matrices are avoided for MMR/cap (on-the-fly dots).
- **Model download helper:** `scripts/download_clip_model.py` (progress bar via `tqdm` / `huggingface_hub`).

**Install:** `pip install -r requirements-card-image-tags.txt`

**Passcode subset:** `-c` / `--cards FILE` — text file with one numeric passcode per line (`#` comments OK). Only matching stems under `-i` are processed (before shuffle/limit). Shared parsing lives in `scripts/card_image_passcodes.py` (also used by GIT).

**Typical commands:**

```bash
python scripts/tag_card_images_clip.py -i chimeratech --tags tags_condensed_removals.json -o card_image_tags.json
python scripts/tag_card_images_clip.py -i chimeratech -c cards.txt --tags tags_condensed_removals.json -o subset.json
python scripts/tag_card_images_clip.py -i chimeratech --shuffle-seed 1 --limit 20 -o card_image_tags_try.json
```

---

## 4. Card captions with Microsoft GIT (optional)

**Goal:** **Open-ended** one-line captions per crop (not the same as the fixed CLIP tag list).

**Script:** `scripts/git_caption_card_images.py`

- Default model: **`microsoft/git-base`** (same family as experiments in third-party YGO GIT notebooks).
- **Junk phrases** (e.g. “digital art selected for the #”) happen because COCO-pretrained GIT is a poor match for stylized game art; the script uses **beams, repetition penalty, n-gram blocking**, and optional **retry with sampling** when output matches known garbage patterns.
- Output: `git_captions.json` as `{ "passcode": "caption", ... }`.

**Passcode subset:** `-c` / `--cards FILE` — same one-passcode-per-line format as CLIP; only those images are captioned.

**Merge into chimeratech:**

```bash
python scripts/merge_git_into_chimeratech.py --git-json git_captions.json
python scripts/git_caption_card_images.py -i chimeratech -c cards.txt -o git_subset.json
```

**Install:** `pip install -r requirements-git-caption.txt`

---

## 5. Yu-Gi-Oh card text → condensed tags (sentence-transformers)

**Goal:** Assign **1–4** tags from `tags_condensed_removals.json` using **only** YGO text: GIT captions plus optional Konami **`desc`** from `chimeratech.json` — **no Pokémon lore**, no classifier training.

**Script:** `scripts/tag_ygo_cards_text.py`

- Embeds each card document and each tag phrase (`tag_to_phrase` from `retag_pokemon_embeddings.py`), then **`pick_tags_for_row`** (cosine similarity + optional MMR / near-duplicate cap).
- **Input:** `--git-json`, `--chimeratech`, `--tags`, `--cards` (passcode list), `-o` (default `ygo_card_tags.json`).

**Install:** `pip install -r requirements-ygo-pokemon-tags.txt`

```bash
python scripts/tag_ygo_cards_text.py \
  --git-json git_subset.json --chimeratech chimeratech.json \
  --tags tags_condensed_removals.json --cards cards.txt -o ygo_card_tags.json
```

---

## 6. VLM open tags (optional, heavy)

**Script:** `scripts/tag_card_images_vlm.py` — local **BLIP-2** (`Salesforce/blip2-opt-2.7b` by default), no fixed tag list. Large checkpoint; GPU recommended.

**Install:** `pip install -r requirements-card-vlm.txt`

---

## 7. Pokémon aggregate stats & trait-tag buckets

**Main aggregates** (`pokemon_stats.json`): `scripts/build_pokemon_stats.py` — percentiles (p25/p50/p75) for base stats and BST, plus Gen 9 move frequencies, grouped by single type, single egg group, and habitat; **`fully_evolved`** mirrors groupings **without** move lists.

**Per condensed trait tag** (`pokemon_stats_by_tag.json`): `scripts/build_pokemon_stats_by_tag.py` — for **each** string in `tags_condensed_removals.json`, finds species that have that tag in `pokemon_tags_retagged.json`, maps names to `pokedex.ts` slugs, and emits the **same bucket shape** (stats + BST + moves; fully evolved = stats only). Optional **`--merge-into pokemon_stats.json`** adds `by_trait_tag` and `fully_evolved.by_trait_tag` to an existing stats file (large).

```bash
python scripts/build_pokemon_stats_by_tag.py -o pokemon_stats_by_tag.json
python scripts/build_pokemon_stats_by_tag.py --merge-into pokemon_stats.json
```

**GUI:** `scripts/pokemon_stats_picker.py` — Tkinter viewer for any compatible JSON. **Group by** auto-detects available roots (`by_single_type`, `by_single_egg_group`, `by_habitat`, **`by_trait_tag`** when present). Shows species count, stat percentiles, BST, and move counts / % of group plus a simple bar column.

```bash
python scripts/pokemon_stats_picker.py
python scripts/pokemon_stats_picker.py pokemon_stats_by_tag.json
```

---

## 8. Other related scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_pokemon_stats.py` | Build `pokemon_stats.json` from dex / learnsets / habitats |
| `scripts/build_pokemon_stats_by_tag.py` | Build `pokemon_stats_by_tag.json` from condensed tags + retagged assignments |
| `scripts/pokemon_stats_picker.py` | Browse stats JSON in a desktop UI |
| `scripts/card_image_passcodes.py` | Shared passcode-file parsing for CLIP / GIT `-c` filters |

---

## Credits

Resources and projects referenced or used by these pipelines:

- **[Hugging Face](https://huggingface.co/)** — Hub, `transformers`, `huggingface_hub`, model hosting, and (for `generate_pokemon_tags_hf.py`) the **Inference Router** / OpenAI-compatible chat API.
- **[sentence-transformers](https://www.sbert.net/)** — embedding models (e.g. `all-MiniLM-L6-v2`) for `retag_pokemon_embeddings.py` and `tag_ygo_cards_text.py`.
- **[scikit-learn](https://scikit-learn.org/)** — cosine similarity utilities in the retagger.
- **[PyTorch](https://pytorch.org/)** — backend for CLIP, GIT, BLIP-2, and sentence-transformers.
- **[NumPy](https://numpy.org/)** — percentiles and arrays in `build_pokemon_stats.py` / `build_pokemon_stats_by_tag.py`.
- **Python [tkinter](https://docs.python.org/3/library/tkinter.html)** (stdlib) — `pokemon_stats_picker.py`.
- **[OpenAI CLIP](https://github.com/openai/CLIP)** — vision–language alignment; checkpoints such as `openai/clip-vit-base-patch32` via Transformers.
- **[Microsoft GIT](https://github.com/microsoft/GenerativeImage2Text)** / **`microsoft/git-base`** on the Hub — image captioning in `git_caption_card_images.py`.
- **[BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)** (Salesforce) — `Salesforce/blip2-opt-2.7b` in `tag_card_images_vlm.py`.
- **[YGOProDeck](https://ygoprodeck.com/)** — cropped card images (`images.ygoprodeck.com`) used by `rip_chimeratech_crops.py`.
- **[nogibjj / Generating-Yu-Gi-Oh-Monsters-From-Archetypes](https://github.com/nogibjj/Generating-Yu-Gi-Oh-Monsters-From-Archetypes)** — referenced for GIT-style YGO experiments; the repo does **not** ship a full passcode→GIT-caption table; local captioning uses Hub `microsoft/git-base` unless you add your own checkpoint.
- **Inference providers** (when using the HF chat router for Pokémon tagging): e.g. **[Novita](https://novita.ai/)**, **[Together AI](https://www.together.ai/)**, **[Groq](https://groq.com/)** — see `generate_pokemon_tags_hf.py` docstring for provider pinning and client quirks.
- **[Pokémon](https://www.pokemon.com/)** / **[Yu-Gi-Oh!](https://www.yugioh.com/)** — underlying franchises and card/game data; fan and research tooling in this repo is unofficial and should respect respective terms of use and IP.
