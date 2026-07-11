"""Sync Hugging Face cards for the chess-coach submission (docs-only, idempotent).

HF-only helper. It does NOT touch data files, weights, or any other repo content.
It (1) makes a surgical, assertion-guarded fix to the shipped v4 model card so its
base-vs-tuned table and links are consistent with the committed docs
(RESULTS_HONEST_EVAL_V4.md / README.md / SUBMISSION.md), and (2) writes short,
honest cards for the three stretch adapters and the deep-verified v6 dataset that
were previously stale, template-default, or missing.

Numbers are taken verbatim from the committed source docs (RESULTS_STAGE4_CORRECTED.md,
RESULTS_FULL_EVAL_803.md, RESULTS_HONEST_EVAL_V4.md, README.md, SUBMISSION.md) and the
live v6 manifest. Nothing is fabricated. No emojis, no em dashes, no hype.

Usage::

    ~/.venvs/mlx/bin/python -m scripts.sync_hf_cards --dry-run   # print planned actions
    ~/.venvs/mlx/bin/python -m scripts.sync_hf_cards             # push to HF

The write token is read from the repo .env (HF_TOKEN / HUGGING_FACE_HUB_TOKEN) or the
ambient environment / cached CLI login.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

NS = "khoilamalphaai"
GITHUB = "https://github.com/Alpha-AI-Engineering-Khoi/chess-instructor-llm"

MODEL_V4 = f"{NS}/chess-coach-32b-v4-qlora"
MODEL_DPO = f"{NS}/chess-coach-32b-v6-dpo"
MODEL_DPO2 = f"{NS}/chess-coach-32b-v6-dpo2"
MODEL_DISTILL = f"{NS}/chess-coach-32b-v6-distill"
DATASET_V6 = f"{NS}/chess-coach-v6"

BASE = "unsloth/Qwen3-32B-unsloth-bnb-4bit"


def _hf(kind: str, name: str) -> str:
    if kind == "dataset":
        return f"https://huggingface.co/datasets/{NS}/{name}"
    if kind == "space":
        return f"https://huggingface.co/spaces/{NS}/{name}"
    return f"https://huggingface.co/{NS}/{name}"


# --- v4 card: two surgical, assertion-guarded replacements ---------------------- #
# Align the base-vs-tuned frontier row with the canonical best-frontier-on-tier-fit
# (Gemini 3.1 Pro 0.553 / coh-viol 0.292 / distinct 0.210, RESULTS_HONEST_EVAL_V4.md),
# matching the card's own prose ("0.767 vs 0.553") and README/SUBMISSION. And add the
# GitHub code link the submission requires.
V4_OLD_FRONTIER = "| best frontier reference (GPT-5.5) | 0.494 | 0.342 | 0.280 |"
V4_NEW_FRONTIER = "| best frontier on tier-fit (Gemini 3.1 Pro) | 0.553 | 0.292 | 0.210 |"

V4_OLD_LINKS = (
    "| Training dataset (SFT) | [chess-coach-move-review]"
    "(https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review) (`v4` config) |"
)
V4_NEW_LINKS = (
    f"| Code / GitHub repo | [Alpha-AI-Engineering-Khoi/chess-instructor-llm]({GITHUB}) |\n"
    "| Training dataset (SFT) | [chess-coach-move-review]"
    "(https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review) (`v4` config) |"
)


def v4_patch(current: str) -> str:
    out = current
    for old, new in ((V4_OLD_FRONTIER, V4_NEW_FRONTIER), (V4_OLD_LINKS, V4_NEW_LINKS)):
        if new in out and old not in out:
            continue  # already applied (idempotent)
        if out.count(old) != 1:
            raise SystemExit(
                f"ERROR: v4 card anchor not found exactly once (found {out.count(old)}):\n  {old}\n"
                "Refusing to push a blind edit. Inspect the live card."
            )
        out = out.replace(old, new)
    return out


# --- v6-dpo ---------------------------------------------------------------------- #
DPO_CARD = f"""---
license: apache-2.0
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
language:
- en
datasets:
- {NS}/chess-coach-v6
tags:
- chess
- coaching
- dpo
- lora
- peft
- trl
- qwen3
---

# Chess Coach 32B v6-dpo (DPO preference adapter, stretch)

A LoRA adapter that DPO-sharpens the shipped v4 chess coach on tier-appropriate move
selection (the project's one trained behavior), trained on the deep-verified v6
preference pairs. This is a research stretch result, not the shipped model: the
shipped coach is [chess-coach-32b-v4-qlora]({_hf('model', 'chess-coach-32b-v4-qlora')}).
The stronger [chess-coach-32b-v6-dpo2]({_hf('model', 'chess-coach-32b-v6-dpo2')})
supersedes this adapter as the queued drop-in successor.

## What it is

- Base: `{BASE}` (the same 4-bit base as v4).
- Init: a two-adapter TRL DPO setup where the policy and the frozen reference both
  initialize from the v4 LoRA, so the KL pressure holds soundness and format in place
  while DPO moves only the move choice (improve v4 without regressing it).
- LoRA: r=32, alpha=32, dropout=0 on `q,k,v,o,gate,up,down` (inherited from v4).

## Preference data

Built from [chess-coach-v6]({_hf('dataset', 'chess-coach-v6')}). For each pair the
`chosen` response recommends the deep-verified tier-appropriate move
(`provenance.canonical_uci`) and the `rejected` response is the same text with only
the move swapped to the off-tier or off-spec contrast move
(`provenance.dpo_rejected_uci`). Because chosen and rejected are byte-identical except
for the move, DPO learns move preference, not prose style (840 pairs, 280 per tier).

## Honest result (corrected v6 benchmark, 120 held-out TEST, grounded)

Re-scored in one controlled session against the deep-verified v6 labels, same strict
extractor as the shipped v4 eval (RESULTS_STAGE4_CORRECTED.md):

| Model (grounded) | tier-policy match | beginner | intermediate | advanced | move-sound | distinct |
|---|---:|---:|---:|---:|---:|---:|
| v4 (shipped baseline) | 0.861 | 0.858 | 0.750 | 0.975 | 0.983 | 0.987 |
| v6-dpo (this) | 0.881 | 0.858 | 0.808 | 0.975 | 0.983 | 0.987 |

The preference tune sharpens the moat with no regression: +0.0195 overall tier-policy
match, and the entire gain is the intermediate tier (0.808 vs 0.750, out of
distribution). Beginner, advanced, soundness, and distinct-moves are unchanged from
v4. v6-dpo2 (checkpoint step 200) later reached 0.892 overall and supersedes this
adapter. See the honest limits below.

## Honest limits

- The gain is confined to the intermediate tier; beginner and advanced already sit at
  their ceiling under grounding.
- Tier-policy match is agreement with the project's `select_tier_move` rule, which is a
  learnability metric, not certified best teaching.
- Grounding is required at inference. Without the per-tier Maia signal in the prompt the
  three tiers collapse to a single move; the behavior does not live in the weights.

## Use

Load the base 4-bit model and apply this adapter with PEFT (the tokenizer is the base
Qwen3 tokenizer). Run it with engine grounding in the prompt, the same move-review
format v4 was trained on.

## Links

| Resource | Link |
|---|---|
| Shipped model (v4) | [chess-coach-32b-v4-qlora]({_hf('model', 'chess-coach-32b-v4-qlora')}) |
| Stronger successor (best DPO) | [chess-coach-32b-v6-dpo2]({_hf('model', 'chess-coach-32b-v6-dpo2')}) |
| Preference dataset | [chess-coach-v6]({_hf('dataset', 'chess-coach-v6')}) |
| Demo | [chess-coach-studio]({_hf('space', 'chess-coach-studio')}) |
| Code / GitHub repo | [Alpha-AI-Engineering-Khoi/chess-instructor-llm]({GITHUB}) |

License Apache-2.0, inherited from Qwen3-32B. The preference data is CC-BY-NC-4.0.
"""


# --- v6-dpo2 --------------------------------------------------------------------- #
DPO2_CARD = f"""---
license: apache-2.0
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
language:
- en
datasets:
- {NS}/chess-coach-v6
tags:
- chess
- coaching
- dpo
- lora
- peft
- trl
- qwen3
---

# Chess Coach 32B v6-dpo2 (best DPO adapter, stretch)

The strongest DPO variant: a stronger, tier-targeted successor to
[chess-coach-32b-v6-dpo]({_hf('model', 'chess-coach-32b-v6-dpo')}). This is a research
stretch result and the queued drop-in successor to v4, not the shipped model: the
shipped coach is [chess-coach-32b-v4-qlora]({_hf('model', 'chess-coach-32b-v4-qlora')}).

## What it is

- Base: `{BASE}` (the same 4-bit base as v4).
- Init: a two-adapter TRL DPO setup initialized from the shipped v4 LoRA for BOTH the
  policy and the frozen reference, so it is a clean superset of v6-dpo with no
  DPO-on-DPO drift.
- Training: harder, tier-targeted, style-matched preference pairs. Beginner learns to
  prefer the human-findable move over the sharp engine move; advanced learns to prefer
  the sharpest engine-best move over the softer sound move; intermediate keeps the
  original v6 pairs to preserve its gain. Selected checkpoint step 200. Preference data
  from [chess-coach-v6]({_hf('dataset', 'chess-coach-v6')}).

## Honest result (corrected v6 benchmark, 120 held-out TEST, grounded)

Re-scored in one controlled session against the deep-verified v6 labels, same strict
extractor as the shipped v4 eval (RESULTS_STAGE4_CORRECTED.md):

| Model (grounded) | tier-policy match | beginner | intermediate | advanced | move-sound | distinct |
|---|---:|---:|---:|---:|---:|---:|
| v4 (shipped baseline) | 0.861 | 0.858 | 0.750 | 0.975 | 0.983 | 0.987 |
| v6-dpo | 0.881 | 0.858 | 0.808 | 0.975 | 0.983 | 0.987 |
| v6-dpo2 (this, best DPO) | 0.892 | 0.858 | 0.842 | 0.975 | 0.983 | 0.987 |

Overall tier-policy match 0.892 is the best DPO result: +0.031 vs v4 and +0.011 vs
v6-dpo. Honestly, the entire gain is the intermediate tier (0.842 vs v4 0.750, vs
v6-dpo 0.808). Beginner (0.858) and advanced (0.975) are byte-identical to v4 and
v6-dpo because both already sit at their ceiling under grounding, so this is a stronger
v6-dpo, not a beginner or advanced breakthrough. Soundness (0.983) and distinct-moves
(0.987) are unchanged and names-a-move is nominally higher (0.986). Format (0.925) is
marginally under v4 (0.939), a 256-token-cap prose-length artifact, not a move or
soundness regression.

## Honest limits

- The gain is confined to the intermediate tier; beginner and advanced are ceilinged.
- Tier-policy match is agreement with the project's `select_tier_move` rule, a
  learnability metric, not certified best teaching.
- Grounding is required at inference. Without the per-tier Maia signal the three tiers
  collapse to a single move; the behavior does not live in the weights.

## Use

Load the base 4-bit model and apply this adapter with PEFT (the tokenizer is the base
Qwen3 tokenizer). Run it with engine grounding in the prompt, the same move-review
format v4 was trained on.

## Links

| Resource | Link |
|---|---|
| Shipped model (v4) | [chess-coach-32b-v4-qlora]({_hf('model', 'chess-coach-32b-v4-qlora')}) |
| Earlier DPO adapter | [chess-coach-32b-v6-dpo]({_hf('model', 'chess-coach-32b-v6-dpo')}) |
| Preference dataset | [chess-coach-v6]({_hf('dataset', 'chess-coach-v6')}) |
| Demo | [chess-coach-studio]({_hf('space', 'chess-coach-studio')}) |
| Code / GitHub repo | [Alpha-AI-Engineering-Khoi/chess-instructor-llm]({GITHUB}) |

License Apache-2.0, inherited from Qwen3-32B. The preference data is CC-BY-NC-4.0.
"""


# --- v6-distill ------------------------------------------------------------------ #
DISTILL_CARD = f"""---
license: apache-2.0
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
language:
- en
datasets:
- {NS}/chess-coach-v6
tags:
- chess
- coaching
- distillation
- lora
- peft
- sft
- trl
- qwen3
---

# Chess Coach 32B v6-distill (engine-distilled adapter, stretch)

A LoRA adapter that distills the tier-selection rule into the model WEIGHTS and is
evaluated with NO grounding (no engine and no Maia in the prompt). This is the harder
behavior-in-weights research result, not the shipped model: the shipped coach is
[chess-coach-32b-v4-qlora]({_hf('model', 'chess-coach-32b-v4-qlora')}), which is
served grounded.

## What it is

- Base: `{BASE}` (the same 4-bit base as v4).
- Objective: SFT on the no-grounding move-review format, with targets drawn from the
  deep-verified engine-best and canonical tier moves in
  [chess-coach-v6]({_hf('dataset', 'chess-coach-v6')}). The goal is to test whether the
  tier policy can be recovered from the weights alone when the engine and human-move
  grounding are removed from the prompt.

## Honest result (corrected v6 benchmark, 120 held-out TEST, no grounding)

Re-scored in one controlled session against the deep-verified v6 labels, same strict
extractor as the shipped v4 eval (RESULTS_STAGE4_CORRECTED.md). Both rows are scored
WITHOUT grounding:

| Model (no grounding) | tier-policy match | names-a-move | move-sound | distinct |
|---|---:|---:|---:|---:|
| Qwen3-32B base (untuned) | 0.022 | 0.250 | 0.081 | 0.040 |
| v6-distill (this) | 0.325 | 0.983 | 0.653 | 0.461 |

Stripped of grounding the untuned base essentially cannot coach: it names a move only
25% of the time and fabricates illegal or unsound moves without a sound list. The
distilled adapter recovers the tier rule from its weights: tier-policy match 0.325
(+0.303, about 15x), names-a-move 0.983, move-soundness 0.653, distinct 0.461.

## Honest limits

- Advanced-tier limit: per-tier tier-policy match is beginner 0.358, intermediate
  0.400, advanced 0.217. The advanced target is the engine-best move, and reproducing
  it from the weights alone, without the engine grounding the condition removes, is
  genuinely the hardest, so advanced is the weakest tier.
- Grounding-free soundness (0.653) trails the deployable grounded soundness (about
  0.98). This is a behavior-in-weights proof, not a claim that grounding is unnecessary
  in production.
- Tier-policy match is agreement with the project's `select_tier_move` rule, a
  learnability metric, not certified best teaching.

## Use

Load the base 4-bit model and apply this adapter with PEFT (the tokenizer is the base
Qwen3 tokenizer). Unlike the shipped grounded coach, this adapter is evaluated with the
no-grounding prompt format described above.

## Links

| Resource | Link |
|---|---|
| Shipped model (v4) | [chess-coach-32b-v4-qlora]({_hf('model', 'chess-coach-32b-v4-qlora')}) |
| Training dataset | [chess-coach-v6]({_hf('dataset', 'chess-coach-v6')}) |
| Demo | [chess-coach-studio]({_hf('space', 'chess-coach-studio')}) |
| Code / GitHub repo | [Alpha-AI-Engineering-Khoi/chess-instructor-llm]({GITHUB}) |

License Apache-2.0, inherited from Qwen3-32B. The training data is CC-BY-NC-4.0.
"""


# --- chess-coach-v6 dataset ------------------------------------------------------ #
V6_DATASET_CARD = f"""---
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- en
tags:
- chess
- coaching
- distillation
- dpo
- sft
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train_v6.jsonl
  - split: validation
    path: valid_v6.jsonl
- config_name: scenarios
  data_files:
  - split: test
    path: scenarios_v6.jsonl
---

# Chess Coach v6 (deep-verified training labels)

The current data frontier for the chess-instructor-llm coach: a foundational,
data-first rebuild of the training LABELS (the move plus full provenance), deep-verified
with Stockfish 17 (a two-depth root search with agreement bands), Syzygy tablebases
(endgames of seven pieces or fewer), and Maia-2 human-likelihood. It feeds the
downstream preference (DPO) and engine-distillation retrains.

This dataset is NOT the shipped SFT set. The shipped coach
([chess-coach-32b-v4-qlora]({_hf('model', 'chess-coach-32b-v4-qlora')})) is trained on
[chess-coach-move-review]({_hf('dataset', 'chess-coach-move-review')}) (default config
v4), which is left untouched. v6 is the deeper-verified successor used to train the
stretch adapters below.

## Configs and splits

| Config | Split | File | Rows |
|---|---|---|---:|
| default | train | `train_v6.jsonl` | 6,768 |
| default | validation | `valid_v6.jsonl` | 363 |
| scenarios | test | `scenarios_v6.jsonl` | 2,409 |

```python
from datasets import load_dataset

ds = load_dataset("{DATASET_V6}")                 # train + validation (SFT rows)
bench = load_dataset("{DATASET_V6}", "scenarios") # benchmark scenarios
```

The validation split is a game-disjoint holdout. The benchmark scenarios keep the 120
held-out TEST board ids stable across the rebuild (360 val rows), so v6-labelled scores
are directly comparable to the shipped v4 headline.

## Row schema

- `train_v6.jsonl` and `valid_v6.jsonl`: each row is `{{"messages": [...], "provenance": {{...}}}}`.
  `messages` is an OpenAI-style chat triple (system, user, assistant) carrying the
  coaching label; `provenance` carries the auditable grounding for that label
  (`pos_id`, `fen`, `tier`, `phase`, `engine_best`, `canonical_uci` / `canonical_san`,
  `canonical_pool_rank`, `canonical_is_engine_best`, `maia_policy`, `severity`,
  `student`, `review_action`, `discriminating`, and more).
- `scenarios_v6.jsonl`: the benchmark labels, one row per position and tier, with the
  full deep-verified `sound_pool`, `rejected` moves, `engine_best`, `canonical` move,
  `maia` policy, and the engine settings used.

## What changed versus v4 (the audit fixes, in the data)

| Fix | Result |
|---|---|
| Deep, robust sound pool (SF17 depth 14 and 20, agreement bands, WDL, Syzygy) | 28.9% of the old benchmark sound-pool moves are rejected as not-actually-sound under deep search |
| Advanced tier fixed to the verified engine-best move | old benchmark advanced not-equal-to engine_best on 43 positions; v6 has 0 |
| Complete triads by construction | every board carries all three tiers, with no collapsed levels |
| Re-derived training labels | 45.5% of comparable training labels changed (3,307 of 7,269) |

Dataset stats (v6 manifest): 2,377 unique boards, 2,123 discriminating boards, rows
balanced across tiers (2,377 beginner / intermediate / advanced), sources reused 3,210
and freshly mined 3,921.

## Engine configuration

```
stockfish 17, root-search depths [14, 20], time caps [1.0s, 6.0s], MultiPV 10,
tolerance 120 cp, Maia = maia2-rapid at 1100 / 1500 / 1900,
Syzygy = Lichess tablebase API (<= 7 pieces)
```

## What the labels feed (stretch adapters)

| Adapter | Uses | Honest result (corrected 120 TEST) |
|---|---|---|
| [chess-coach-32b-v6-dpo]({_hf('model', 'chess-coach-32b-v6-dpo')}) | canonical (chosen) vs contrast (rejected) pairs | tier-policy 0.881 grounded, gain in the intermediate tier |
| [chess-coach-32b-v6-dpo2]({_hf('model', 'chess-coach-32b-v6-dpo2')}) | harder tier-targeted pairs | tier-policy 0.892 grounded, best DPO, gain in the intermediate tier |
| [chess-coach-32b-v6-distill]({_hf('model', 'chess-coach-32b-v6-distill')}) | verified engine-best targets | tier-policy 0.325 no-grounding, behavior-in-weights with an advanced-tier limit |

## Honest caveats

- Tier-policy match is agreement with the project's `select_tier_move` rule, a
  learnability metric, not certified best teaching.
- The advanced label is by construction the sharp engine-best move, so advanced rows
  largely mirror the engine; the distinctive leveling signal is strongest at the
  beginner and intermediate tiers.
- Move labels are engine-verified against the deep sound pool, but the coaching prose is
  distilled from a teacher model. Truthfulness of that prose is enforced at serving time
  by a separate non-LLM verifier, not by the data.
- Grounding is required at inference for the grounded adapters; without the per-tier
  Maia signal the three tiers collapse to a single move.

## Links and license

| Resource | Link |
|---|---|
| Shipped model (v4) | [chess-coach-32b-v4-qlora]({_hf('model', 'chess-coach-32b-v4-qlora')}) |
| Shipped SFT dataset (v4) | [chess-coach-move-review]({_hf('dataset', 'chess-coach-move-review')}) |
| Demo | [chess-coach-studio]({_hf('space', 'chess-coach-studio')}) |
| Code / GitHub repo | [Alpha-AI-Engineering-Khoi/chess-instructor-llm]({GITHUB}) |

Positions derive from the public CC0 Lichess Open Database; the coaching text is
distilled from GPT-5.5. Released for research and education under CC-BY-NC-4.0.
"""


def _load_token() -> str | None:
    env_file = _ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
        except Exception:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="print planned actions, do not push")
    p.add_argument("--token", default=None)
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download

    token = args.token or _load_token()
    if not token:
        print("ERROR: no HF token (set HF_TOKEN or run `hf auth login`).", file=sys.stderr)
        return 2
    api = HfApi(token=token)

    # 1) v4 card: surgical patch
    cur = Path(hf_hub_download(repo_id=MODEL_V4, filename="README.md", token=token)).read_text()
    patched = v4_patch(cur)
    v4_changed = patched != cur

    # 2..5) full-body cards
    full_uploads = [
        ("model", MODEL_DPO, DPO_CARD, "Honest stretch card: v6-dpo (corrected Stage-4 result)"),
        ("model", MODEL_DPO2, DPO2_CARD, "Honest stretch card: v6-dpo2 (best DPO, corrected Stage-4)"),
        ("model", MODEL_DISTILL, DISTILL_CARD, "Honest stretch card: v6-distill (behavior-in-weights)"),
        ("dataset", DATASET_V6, V6_DATASET_CARD, "Dataset card: deep-verified v6 labels + configs"),
    ]

    if args.dry_run:
        print(f"[dry-run] v4 card {MODEL_V4}: {'WOULD PATCH' if v4_changed else 'already consistent (no change)'}")
        for rtype, rid, body, _ in full_uploads:
            print(f"[dry-run] {rtype} {rid}: would upload README.md ({len(body)} chars)")
        return 0

    if v4_changed:
        api.upload_file(
            path_or_fileobj=patched.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=MODEL_V4,
            commit_message="Align base-vs-tuned frontier row with committed docs; add GitHub link",
        )
        print(f"[hf] v4 card patched -> {_hf('model', 'chess-coach-32b-v4-qlora')}")
    else:
        print(f"[hf] v4 card already consistent -> {_hf('model', 'chess-coach-32b-v4-qlora')}")

    for rtype, rid, body, msg in full_uploads:
        api.upload_file(
            path_or_fileobj=body.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=rid,
            repo_type=rtype,
            commit_message=msg,
        )
        url = _hf(rtype, rid.split("/")[-1])
        print(f"[hf] {rtype} card written -> {url}")

    print("[hf] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
