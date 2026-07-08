"""Publish the **v3** chess-coach artifacts to the Hugging Face Hub — WITHOUT touching v2.

v3 = a QLoRA LoRA adapter on **Qwen3-32B** (deployable as base-4bit + adapter) trained on a
larger, faithfulness-filtered contrastive dataset (7,128 rows). The v2 publish already exists;
this script is **additive and idempotent**:

- **Model repo** ``qwen3-1.7b-chess-coach-mlx`` — uploads the v3 adapter into the subfolder
  ``v3-lora-qwen3-32b/`` (the v2 MLX model at the repo ROOT is left untouched) and inserts a
  v3 section into the model card (README.md), preserving all v2 content.
- **Dataset repo** ``chess-coach-move-review`` — adds a ``v3`` config (``v3/train.jsonl`` +
  ``v3/validation.jsonl``) alongside the existing v1/v2 configs and a v2→v3 card section.
  The default config stays ``v2`` (the shipped model's data).
- **Space** ``chess-coach-benchmark`` — inserts a self-contained, additive v3 ``<section>`` at
  the top of ``index.html`` (the existing 14-model tables/JS are left unchanged) + a card note.

Every card edit is guarded by a marker so re-running is safe. The WRITE token is read ONLY from
the environment (``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``) and is never printed.

Usage::

    set -a && source .env && set +a
    ~/.venvs/mlx/bin/python -m scripts.publish_hf_v3
    # subsets:
    ... --only model     # or dataset / space
    ... --skip-space
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# ---- Repos (existing v2 publish; DO NOT rename/recreate) --------------------- #
NAMESPACE = "khoilamalphaai"
MODEL_REPO = f"{NAMESPACE}/qwen3-1.7b-chess-coach-mlx"
DATASET_REPO = f"{NAMESPACE}/chess-coach-move-review"
SPACE_REPO = f"{NAMESPACE}/chess-coach-benchmark"

# ---- Local v3 artifacts ----------------------------------------------------- #
ADAPTER_DIR = _ROOT / "models" / "adapters" / "chess-coach-v3" / "adapter"
ADAPTER_SUBFOLDER = "v3-lora-qwen3-32b"
DS_TRAIN = _ROOT / "data" / "dataset" / "train_v3.jsonl"
DS_VALID = _ROOT / "data" / "dataset" / "valid_v3.jsonl"

# =========================================================================== #
# Content blocks (verbatim numbers from leaderboard.json / RESULTS_V3.md)
# =========================================================================== #

MODEL_CARD_V3_SECTION = """## \U0001F195 v3 (Qwen3-32B QLoRA adapter) — strongest *local* coach yet

> **Published alongside v2, not replacing it.** The v2 4-bit MLX model at this repo's **root is
> unchanged and remains the shipped model.** v3 is an **additional artifact**: a QLoRA LoRA adapter
> on **[`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B)** (a 20x larger base), trained on a
> larger, faithfulness-filtered contrastive dataset (**7,128 rows, 0% false labels**). The deployable
> form is **base 4-bit + this adapter** — exactly what the eval below measured.

**Adapter files:** [`v3-lora-qwen3-32b/`](./tree/main/v3-lora-qwen3-32b) — `adapter_config.json`,
`adapter_model.safetensors` (LoRA **r=32**), tokenizer + chat template. **Training data:**
[`chess-coach-move-review`, config `v3`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review)
(6,772 train / 356 valid).

### v3 result — 803-position benchmark, **15-model** field

Same airtight, zero-leakage 803-position gap set (each position coached at all 3 tiers with
byte-identical grounding), re-scored with a **15-model** blinded cross-family council (GPT-5.5 +
Claude Opus 4.8 + Gemini 3.1 Pro). Numbers are verbatim from the project's `leaderboard.json` /
`RESULTS_V3.md`.

| Model | tier-fit \u2191 (moat) | instr rank \u2193 (of 15) | top-1 \u2191 | fabrication \u2193 | balanced \u2191 | local |
|---|---:|---:|---:|---:|---:|:--:|
| **OURS-v3 (Qwen3-32B tuned)** | **53.2%** | **7.06** | **20.3%** | **5.4%** | **61.7 (2nd/15)** | yes |
| OURS-v2 (Qwen3-1.7B tuned) | 53.1% | 10.07 | 7.5% | 30.2% | 51.2 | yes |
| Qwen3-32B (untuned base of v3) | 36.9% | 9.07 | 0.0% | 6.1% | 53.6 | yes |
| GPT-5.5 | 43.1% | 3.35 | 24.4% | 3.3% | 62.4 (1st) | no |
| Claude Opus 4.8 | 45.8% | 4.71 | 19.2% | 4.7% | 55.8 | no |
| Gemini 3.1 Pro | 48.4% | 5.67 | 11.7% | 4.2% | 56.6 | no |
| GLM-5 (~355B) | 44.7% | 6.65 | 5.3% | 7.3% | 54.8 | no |

- **Balanced score 61.7 \u2192 2nd of 15**, behind only GPT-5.5 (62.4); ahead of Gemini, Claude, GLM-5,
  the untuned 32B, and v2. The **only near-frontier-balanced model that also runs locally & free.**
- **Instructiveness: best of every locally-runnable model** (council rank **7.06**, top-1 **20.3%**);
  vs v2 the rank improves **10.07 \u2192 7.06** and top-1 **7.5% \u2192 20.3%**.
- **Fabrication 30.2% \u2192 5.4%** (~6x lower) — the clean data + stronger base bring it to roughly the
  untuned 32B (6.1%) and the frontier APIs (3–5%).
- **Tier-fit (the moat) 53.2%, field-leading** (above GPT-5.5 40%, Claude 42%, Gemini 42%). vs the
  untuned Qwen3-32B it was tuned from: **+16.3 pts** — the specialist behavior is *trained in*, not emergent.

### Honest v3 tradeoffs (measured, not hidden)

- **Beginner move-calibration softened:** beginner tier-fit **47.9% \u2192 29.6%** (the 32B's stronger
  chess prior leans toward the engine-best move), while **advanced rose 60.9% \u2192 83.6%**. Net tier-fit
  ties v2 and still leads the field, but the *shape* of the win shifted from beginners to advanced.
- **~4–5% malformed raw outputs** (a spurious leading rating fragment / greedy-decode echo) put
  safety **94.4%** and no-jargon **95.6%** just below the strict 97% gate. **The blunder rate is only
  1.3%**; the malformed outputs are neutralized by the serve-time verifier + regeneration.
- **Does not beat the frontier on raw instructiveness** (GPT-5.5 3.35, Claude 4.71, Gemini 5.67 still
  out-coach 7.06). The claim is **"best local coach,"** not "beats GPT-5.5."

### Using v3 (base 4-bit + adapter)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "unsloth/Qwen3-32B-unsloth-bnb-4bit"   # 4-bit base the adapter was trained on
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(
    model, "khoilamalphaai/qwen3-1.7b-chess-coach-mlx", subfolder="v3-lora-qwen3-32b")
# then apply the tier-grounded chat prompt exactly as for v2
```

Full report & reproduction: `RESULTS_V3.md` and `RESULTS_FULL_EVAL_803_v3.md` in the project repo.

"""

ADAPTER_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3-32B
library_name: peft
pipeline_tag: text-generation
tags:
- chess
- coaching
- qwen3
- lora
- qlora
- peft
- sft
datasets:
- khoilamalphaai/chess-coach-move-review
---

# Chess Coach v3 — Qwen3-32B QLoRA adapter

A **LoRA adapter** (QLoRA, r=32) that turns
**[`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B)** into a rating-calibrated,
engine-grounded chess **move-review coach**. This is the **v3** artifact of
[`khoilamalphaai/qwen3-1.7b-chess-coach-mlx`](https://huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx):
the repo root holds the shipped **v2** 1.7B MLX model; this subfolder is the additional v3 adapter.

- **Base:** `unsloth/Qwen3-32B-unsloth-bnb-4bit` (4-bit). Deployable form = **base 4-bit + this adapter**.
- **LoRA:** r=32, alpha=32, dropout=0, on `q/k/v/o` + `gate/up/down` proj.
- **Data:** [`chess-coach-move-review`, config `v3`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review)
  — 7,128 faithfulness-filtered rows (0% false labels).

## Result (803-position, 15-model benchmark)

| Metric | OURS-v3 | vs v2 | vs untuned Qwen3-32B |
|---|---:|---:|---:|
| Tier-fit (moat) | **53.2%** (field-leading) | 53.1% | 36.9% (+16.3) |
| Instructiveness (council rank, of 15) | **7.06** (best local) | 10.07 | 9.07 |
| Top-1 win-rate | **20.3%** | 7.5% | 0.0% |
| Fabrication | **5.4%** | 30.2% | 6.1% |
| Balanced score | **61.7** (2nd of 15) | 51.2 | 53.6 |

Best locally-runnable coach; 2nd overall on the balanced score behind only GPT-5.5 (62.4).
Honest tradeoffs: beginner tier-fit softened (47.9%->29.6%, advanced 60.9%->83.6%) and ~4–5% malformed
raw outputs (blunder rate 1.3%; neutralized at serve time). Full detail: `RESULTS_V3.md`.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "khoilamalphaai/qwen3-1.7b-chess-coach-mlx",
                                  subfolder="v3-lora-qwen3-32b")
```

License: Apache-2.0 (base); coaching behavior distilled from GPT-5.5 for research/education.
"""

DATASET_V3_CONFIG_BLOCK = """- config_name: v3
  data_files:
  - split: train
    path: v3/train.jsonl
  - split: validation
    path: v3/validation.jsonl
"""

DATASET_V3_SECTION = """## v2 \u2192 v3 (largest, cleanest set — Qwen3-32B target)

`v3` is the **largest and cleanest** config: **7,128 kept rows** (6,772 train / 356 validation) built
from **2,423 curated contrastive positions x 3 tiers**, keeping the **faithfulness reject gate** — only
141 candidates dropped (140 false-fact + 1 engine-speak), i.e. **0% false labels**. It is the set used
to fine-tune the **Qwen3-32B** v3 coach (a 20x larger base than v2's 1.7B).

| | v2 | v3 |
|---|---:|---:|
| Contrastive positions (x 3 tiers) | 348 FENs | **2,423** |
| Teacher candidates | 2,628 | **7,269** |
| Kept after filter | 2,586 | **7,128** (141 dropped) |
| **Train / validation** | 2,457 / 129 | **6,772 / 356** |
| False labels in release | 0% | **0%** |
| Fine-tuned base | Qwen3-1.7B | **Qwen3-32B** |

**Measured effect** (definitive 803-position, 15-model benchmark; full detail in `RESULTS_V3.md`): the
model trained on this `v3` data is the **best locally-runnable coach and 2nd of 15 overall on the
balanced score (61.7, behind only GPT-5.5)**. vs v2: fabrication **30.2% \u2192 5.4%**, instructiveness
council rank **10.07 \u2192 7.06** (top-1 **7.5% \u2192 20.3%**); tier-fit holds field-leading at **53.2%**
(advanced 60.9% \u2192 **83.6%**). vs the untuned Qwen3-32B: tier-fit **+16.3 pts** — the specialist
behavior is *trained in*, not emergent. Honest tradeoff: beginner tier-fit softened (47.9% \u2192 29.6%).
Adapter: [`qwen3-1.7b-chess-coach-mlx` \u2192 `v3-lora-qwen3-32b/`](https://huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx/tree/main/v3-lora-qwen3-32b).

The row schema is **identical** to v1/v2 (a `messages` triple: `system` / `user` / `assistant`).

"""

SPACE_INDEX_V3_SECTION = """<!--V3-QWEN3-32B-->
<section id="v3" style="margin-top:20px">
<div class="hero" style="border-color:rgba(51,194,127,.6);background:linear-gradient(180deg,rgba(51,194,127,.10),rgba(139,123,255,.06))">
  <div class="hero-top">
    <span class="stakes-band">\U0001F195 v3 update &middot; Qwen3-32B</span>
    <span class="pill cur" style="margin:0"><b>803</b> held-out positions</span>
    <span class="pill" style="margin:0"><b>15</b> models &times; <b>3</b> tiers</span>
    <span class="pill" style="margin:0"><b>OURS-v3</b> &middot; best local &middot; 2nd of 15</span>
  </div>
  <h2 style="font-size:24px;margin:8px 0 4px">\U0001F195 v3 &mdash; Qwen3-32B QLoRA coach (strongest <i>local</i> coach)</h2>
  <p class="framing" style="max-width:92ch">Since the 14-model run below recommended <b>Qwen3-32B</b> as the best v3 base, we <b class="ts">trained exactly that</b>: a QLoRA adapter on Qwen3-32B over a larger, faithfulness-filtered contrastive set (7,128 rows). Re-scored on the same 803&times;3 positions against a <b>15-model</b> field, <b class="ts">OURS-v3 is 2nd of 15 on the balanced score (61.7, behind only GPT-5.5) and the best locally-runnable model.</b></p>
  <div class="kpis">
    <div class="kpi"><div class="k">OURS-v3 &middot; balanced</div><div class="v ours">61.7</div><div class="d">2nd of 15 &middot; behind only GPT-5.5 (62.4)</div></div>
    <div class="kpi"><div class="k">OURS-v3 &middot; tier-fit (moat)</div><div class="v good">53.2%</div><div class="d">field-leading &middot; advanced 83.6%</div></div>
    <div class="kpi"><div class="k">Fabrication (raw)</div><div class="v good">5.4%</div><div class="d">v2 was 30.2% &middot; ~6&times; lower</div></div>
    <div class="kpi"><div class="k">Instructiveness rank</div><div class="v">7.06</div><div class="d">best local &middot; v2 was 10.07 (of 15)</div></div>
  </div>
  <p class="hint" style="margin-top:10px">Numbers verbatim from <code>leaderboard.json</code> / <code>RESULTS_V3.md</code>. Deployable form = base 4-bit + the <a href="https://huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx/tree/main/v3-lora-qwen3-32b">v3 LoRA adapter</a>; training data = <a href="https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review">chess-coach-move-review</a> config <code>v3</code>. The 14-model section below is the earlier (pre-v3) run, preserved as-is.</p>
</div>

<h3 style="margin:20px 0 2px">v3 balanced leaderboard <span class="mut" style="font-weight:400">&mdash; 803&times;3, 15-model field (higher better)</span></h3>
<table>
<thead><tr><th>Model</th><th>tier-fit&uarr; (moat)</th><th>instr rank&darr; (of 15)</th><th>top-1&uarr;</th><th>fab&darr;</th><th>balanced&uarr;</th><th>local</th></tr></thead>
<tbody>
<tr class="ours"><td>OURS-v3 (Qwen3-32B tuned)</td><td>53.2%</td><td>7.06</td><td>20.3%</td><td>5.4%</td><td>61.7 (2nd)</td><td>yes</td></tr>
<tr><td>GPT-5.5</td><td>43.1%</td><td>3.35</td><td>24.4%</td><td>3.3%</td><td>62.4 (1st)</td><td>no</td></tr>
<tr><td>Gemini 3.1 Pro</td><td>48.4%</td><td>5.67</td><td>11.7%</td><td>4.2%</td><td>56.6</td><td>no</td></tr>
<tr><td>Claude Opus 4.8</td><td>45.8%</td><td>4.71</td><td>19.2%</td><td>4.7%</td><td>55.8</td><td>no</td></tr>
<tr><td>GLM-5 (~355B)</td><td>44.7%</td><td>6.65</td><td>5.3%</td><td>7.3%</td><td>54.8</td><td>no</td></tr>
<tr><td>Qwen3-32B (untuned base of v3)</td><td>36.9%</td><td>9.07</td><td>0.0%</td><td>6.1%</td><td>53.6</td><td>yes</td></tr>
<tr><td>OURS-v2 (Qwen3-1.7B tuned)</td><td>53.1%</td><td>10.07</td><td>7.5%</td><td>30.2%</td><td>51.2</td><td>yes</td></tr>
</tbody>
</table>
<p class="hint" style="margin:2px 0 12px">Honest tradeoffs: beginner tier-fit softened (47.9%&rarr;29.6%; advanced 60.9%&rarr;83.6%) and ~4&ndash;5% malformed raw outputs put safety 94.4% / no-jargon 95.6% just under the 97% gate (blunder rate only 1.3%, neutralized at serve time). v3 does <b>not</b> beat the frontier on raw instructiveness &mdash; the claim is &ldquo;best local coach.&rdquo;</p>
</section>
<!--/V3-QWEN3-32B-->

"""

SPACE_README_V3_SECTION = """## \U0001F195 v3 (Qwen3-32B) — strongest *local* coach (added on top of v2)

Since the 14-model run recommended **Qwen3-32B** as the best fine-tuning base, we trained exactly that:
a **QLoRA adapter on Qwen3-32B** over a larger, faithfulness-filtered contrastive set (**7,128 rows**).
Re-scored on the same **803x3** positions against a **15-model** field, **OURS-v3 is 2nd of 15 on the
balanced score (61.7, behind only GPT-5.5's 62.4) and the best locally-runnable model** —
instructiveness council rank **7.06** (best local; v2 10.07), fabrication **5.4%** (v2 30.2%), tier-fit
field-leading **53.2%** (advanced 83.6%). Honest tradeoffs: beginner tier-fit softened (47.9%->29.6%)
and ~4–5% malformed raw outputs (blunder rate 1.3%, neutralized at serve time). The `index.html`
dashboard now **leads with a v3 section**; the **14-model tables below it are the earlier pre-v3 run,
preserved unchanged**. Deployable form = base 4-bit + the
[v3 LoRA adapter](https://huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx/tree/main/v3-lora-qwen3-32b).
Detail: `RESULTS_V3.md`.

"""


# =========================================================================== #
# Helpers
# =========================================================================== #
def _download_text(api, repo_id: str, repo_type: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id, filename, repo_type=repo_type, token=api.token)
    return Path(p).read_text(encoding="utf-8")


def _upload_text(api, repo_id: str, repo_type: str, path_in_repo: str, text: str,
                 message: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        tmp = fh.name
    api.upload_file(path_or_fileobj=tmp, path_in_repo=path_in_repo, repo_id=repo_id,
                    repo_type=repo_type, commit_message=message)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        print(f"ERROR: {msg}", file=sys.stderr)
        raise SystemExit(3)


# =========================================================================== #
# Model repo
# =========================================================================== #
def publish_model(api) -> None:
    print(f"\n=== MODEL {MODEL_REPO} ===")
    _require(ADAPTER_DIR.is_dir(), f"adapter dir missing: {ADAPTER_DIR}")
    _require((ADAPTER_DIR / "adapter_model.safetensors").exists(),
             f"adapter weights missing in {ADAPTER_DIR}")

    # 1) Upload the v3 adapter into its own subfolder (v2 root files untouched).
    print(f"[model] uploading adapter -> {ADAPTER_SUBFOLDER}/ (skipping its auto README)")
    api.upload_folder(
        folder_path=str(ADAPTER_DIR), repo_id=MODEL_REPO, repo_type="model",
        path_in_repo=ADAPTER_SUBFOLDER,
        ignore_patterns=["README.md", ".cache/*", "*.lock"],
        commit_message="Add v3 QLoRA adapter (Qwen3-32B) under v3-lora-qwen3-32b/",
    )
    # 2) Upload a proper v3 adapter card as the subfolder README.
    _upload_text(api, MODEL_REPO, "model", f"{ADAPTER_SUBFOLDER}/README.md", ADAPTER_CARD,
                 "Add v3 adapter card")

    # 3) Additively update the ROOT model card.
    card = _download_text(api, MODEL_REPO, "model", "README.md")
    changed = False
    if "Qwen/Qwen3-32B" not in card:
        card = card.replace(
            "base_model: Qwen/Qwen3-1.7B\n",
            "base_model:\n- Qwen/Qwen3-1.7B\n- Qwen/Qwen3-32B\n", 1)
        changed = True
    if "v3 (Qwen3-32B QLoRA adapter)" not in card:
        anchor = "## Why fine-tune (not prompt)?"
        _require(anchor in card, "model card anchor not found (## Why fine-tune)")
        card = card.replace(anchor, MODEL_CARD_V3_SECTION + anchor, 1)
        changed = True
    if changed:
        _upload_text(api, MODEL_REPO, "model", "README.md", card,
                     "Add v3 (Qwen3-32B) section to model card")
        print("[model] model card updated with v3 section")
    else:
        print("[model] model card already has v3 section (skip)")
    print(f"[model] done -> https://huggingface.co/{MODEL_REPO}")


# =========================================================================== #
# Dataset repo
# =========================================================================== #
def publish_dataset(api) -> None:
    print(f"\n=== DATASET {DATASET_REPO} ===")
    _require(DS_TRAIN.exists() and DS_VALID.exists(), "v3 dataset shards missing")
    n_tr = sum(1 for _ in DS_TRAIN.open())
    n_va = sum(1 for _ in DS_VALID.open())
    print(f"[data] v3: {n_tr} train / {n_va} valid")

    # 1) Upload the v3 split files (HF picks up the config from the card).
    api.upload_file(path_or_fileobj=str(DS_TRAIN), path_in_repo="v3/train.jsonl",
                    repo_id=DATASET_REPO, repo_type="dataset",
                    commit_message="Add v3 train split")
    api.upload_file(path_or_fileobj=str(DS_VALID), path_in_repo="v3/validation.jsonl",
                    repo_id=DATASET_REPO, repo_type="dataset",
                    commit_message="Add v3 validation split")

    # 2) Additively update the dataset card.
    card = _download_text(api, DATASET_REPO, "dataset", "README.md")
    changed = False
    if "config_name: v3" not in card:
        card = card.replace("configs:\n- config_name: v2",
                            "configs:\n" + DATASET_V3_CONFIG_BLOCK + "- config_name: v2", 1)
        changed = True
    if "v2 default; v3" not in card:
        card = card.replace(
            "pretty_name: Chess Coach — Move-Review SFT (v2 current, with v1\u2192v2)",
            "pretty_name: Chess Coach — Move-Review SFT (v2 default; v3 = Qwen3-32B set)", 1)
        changed = True
    if "| **v3** (newest, largest) |" not in card:
        card = card.replace(
            "| **v2** (default) | **2,457** | **129** |",
            "| **v3** (newest, largest) | **6,772** | **356** |\n"
            "| **v2** (default) | **2,457** | **129** |", 1)
        changed = True
    if '"chess-coach-move-review", "v3"' not in card:
        card = card.replace(
            'ds   = load_dataset("khoilamalphaai/chess-coach-move-review")          # v2 (default)\n'
            'ds_v1 = load_dataset("khoilamalphaai/chess-coach-move-review", "v1")   # v1',
            'ds    = load_dataset("khoilamalphaai/chess-coach-move-review")          # v2 (default)\n'
            'ds_v3 = load_dataset("khoilamalphaai/chess-coach-move-review", "v3")   # v3 (Qwen3-32B set, 7,128 rows)\n'
            'ds_v1 = load_dataset("khoilamalphaai/chess-coach-move-review", "v1")   # v1', 1)
        changed = True
    if "## v2 \u2192 v3 (largest, cleanest set" not in card:
        anchor = "## v1 \u2192 v2"
        _require(anchor in card, "dataset card anchor not found (## v1 -> v2)")
        card = card.replace(anchor, DATASET_V3_SECTION + anchor, 1)
        changed = True
    if changed:
        _upload_text(api, DATASET_REPO, "dataset", "README.md", card,
                     "Add v3 config + v2->v3 section to dataset card")
        print("[data] dataset card updated with v3 config + section")
    else:
        print("[data] dataset card already has v3 (skip)")
    print(f"[data] done -> https://huggingface.co/datasets/{DATASET_REPO}")


# =========================================================================== #
# Space
# =========================================================================== #
def publish_space(api) -> None:
    print(f"\n=== SPACE {SPACE_REPO} ===")
    # 1) index.html — additive v3 section above the (unchanged) 14-model section.
    html = _download_text(api, SPACE_REPO, "space", "index.html")
    if "<!--V3-QWEN3-32B-->" in html:
        print("[space] index.html already has v3 section (skip)")
    else:
        anchor = ("<!-- ==================== DEFINITIVE 803-POSITION / 14-MODEL EVAL "
                  "==================== -->")
        _require(anchor in html, "space index anchor not found (DEFINITIVE 803 comment)")
        html = html.replace(anchor, SPACE_INDEX_V3_SECTION + anchor, 1)
        _upload_text(api, SPACE_REPO, "space", "index.html", html,
                     "Add additive v3 (Qwen3-32B) leaderboard section")
        print("[space] index.html updated with additive v3 section")

    # 2) README.md — v3 note + short_description.
    card = _download_text(api, SPACE_REPO, "space", "README.md")
    changed = False
    if "v3 (Qwen3-32B) — strongest" not in card:
        anchor = "## Definitive eval — 803 held-out positions, 14 models"
        _require(anchor in card, "space README anchor not found")
        card = card.replace(anchor, SPACE_README_V3_SECTION + anchor, 1)
        changed = True
    if "short_description: Definitive 803-pos, 14-model chess-coach eval" in card:
        card = card.replace(
            "short_description: Definitive 803-pos, 14-model chess-coach eval",
            "short_description: Chess-coach eval \u00b7 v3 (Qwen3-32B) best local, 2nd/15", 1)
        changed = True
    if changed:
        _upload_text(api, SPACE_REPO, "space", "README.md", card,
                     "Add v3 note to Space card")
        print("[space] README.md updated with v3 note")
    else:
        print("[space] README.md already has v3 note (skip)")
    print(f"[space] done -> https://huggingface.co/spaces/{SPACE_REPO}")


# =========================================================================== #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN")
                   or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    p.add_argument("--only", choices=["model", "dataset", "space"], default=None)
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--skip-space", action="store_true")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, get_token

    token = args.token or get_token()
    if not token:
        print("ERROR: no token. `set -a && source .env && set +a` first, or pass --token.",
              file=sys.stderr)
        return 2

    api = HfApi(token=token)
    who = api.whoami()
    print(f"[hf] authenticated as {who['name']} (type={who.get('type')})")
    _require(who["name"] == NAMESPACE, f"unexpected namespace {who['name']} (want {NAMESPACE})")

    do = {"model": True, "dataset": True, "space": True}
    if args.only:
        do = {k: (k == args.only) for k in do}
    if args.skip_model:
        do["model"] = False
    if args.skip_dataset:
        do["dataset"] = False
    if args.skip_space:
        do["space"] = False

    if do["model"]:
        publish_model(api)
    if do["dataset"]:
        publish_dataset(api)
    if do["space"]:
        publish_space(api)

    print("\n[hf] v3 publish done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
