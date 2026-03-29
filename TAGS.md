# Tagging pipelines (YGOrange)

This document summarizes the **text-tag** and **image-tag** workflows in this repo: how data moves between JSON files, which scripts to run, and what each optional stack installs.

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

**Typical commands:**

```bash
python scripts/tag_card_images_clip.py -i chimeratech --tags tags_condensed_removals.json -o card_image_tags.json
python scripts/tag_card_images_clip.py -i chimeratech --shuffle-seed 1 --limit 20 -o card_image_tags_try.json
```

---

## 4. Card captions with Microsoft GIT (optional)

**Goal:** **Open-ended** one-line captions per crop (not the same as the fixed CLIP tag list).

**Script:** `scripts/git_caption_card_images.py`

- Default model: **`microsoft/git-base`** (same family as experiments in third-party YGO GIT notebooks).
- **Junk phrases** (e.g. “digital art selected for the #”) happen because COCO-pretrained GIT is a poor match for stylized game art; the script uses **beams, repetition penalty, n-gram blocking**, and optional **retry with sampling** when output matches known garbage patterns.
- Output: `git_captions.json` as `{ "passcode": "caption", ... }`.

**Merge into chimeratech:**

```bash
python scripts/merge_git_into_chimeratech.py --git-json git_captions.json
```

**Install:** `pip install -r requirements-git-caption.txt`

---

## 5. VLM open tags (optional, heavy)

**Script:** `scripts/tag_card_images_vlm.py` — local **BLIP-2** (`Salesforce/blip2-opt-2.7b` by default), no fixed tag list. Large checkpoint; GPU recommended.

**Install:** `pip install -r requirements-card-vlm.txt`

---

## 6. Other related script

| Script | Purpose |
|--------|---------|
| `scripts/build_pokemon_stats.py` | Stats from `pokedex.ts` / learnsets / habitats (separate from image tagging) |

---

## Credits

Resources and projects referenced or used by these pipelines:

- **[Hugging Face](https://huggingface.co/)** — Hub, `transformers`, `huggingface_hub`, model hosting, and (for `generate_pokemon_tags_hf.py`) the **Inference Router** / OpenAI-compatible chat API.
- **[sentence-transformers](https://www.sbert.net/)** — embedding models (e.g. `all-MiniLM-L6-v2`) for `retag_pokemon_embeddings.py`.
- **[scikit-learn](https://scikit-learn.org/)** — cosine similarity utilities in the retagger.
- **[PyTorch](https://pytorch.org/)** — backend for CLIP, GIT, BLIP-2, and training/inference stacks.
- **[OpenAI CLIP](https://github.com/openai/CLIP)** — vision–language alignment; checkpoints such as `openai/clip-vit-base-patch32` via Transformers.
- **[Microsoft GIT](https://github.com/microsoft/GenerativeImage2Text)** / **`microsoft/git-base`** on the Hub — image captioning in `git_caption_card_images.py`.
- **[BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)** (Salesforce) — `Salesforce/blip2-opt-2.7b` in `tag_card_images_vlm.py`.
- **[YGOProDeck](https://ygoprodeck.com/)** — cropped card images (`images.ygoprodeck.com`) used by `rip_chimeratech_crops.py`.
- **[nogibjj / Generating-Yu-Gi-Oh-Monsters-From-Archetypes](https://github.com/nogibjj/Generating-Yu-Gi-Oh-Monsters-From-Archetypes)** — referenced for GIT-style YGO experiments; the repo does **not** ship a full passcode→GIT-caption table; local captioning uses Hub `microsoft/git-base` unless you add your own checkpoint.
- **Inference providers** (when using the HF chat router for Pokémon tagging): e.g. **[Novita](https://novita.ai/)**, **[Together AI](https://www.together.ai/)**, **[Groq](https://groq.com/)** — see `generate_pokemon_tags_hf.py` docstring for provider pinning and client quirks.
- **[Pokémon](https://www.pokemon.com/)** / **[Yu-Gi-Oh!](https://www.yugioh.com/)** — underlying franchises and card/game data; fan and research tooling in this repo is unofficial and should respect respective terms of use and IP.
