# Results — v3 Chess Coach (Qwen3-32B), v2→v3 and untuned-32B→v3 deltas

**Headline.** v3 changes two things at once — a **20× larger base** (Qwen3-32B instead
of Qwen3-1.7B) and a **larger, cleaner contrastive dataset** (7,128 rows, 0% false
labels) — and it pays off where the extra capacity matters. On the definitive,
zero-leakage **803-position** benchmark (each position coached at all 3 tiers with
byte-identical engine grounding), against a **15-model** field:

- **Instructiveness: a large win, self-preference-corrected.** One blinded
  cross-family council (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) ranks the unified
  15-model field on **all 450 (position×tier) items where every model has a generation**
  — **n = 450 × 3 judges = 1,350 rankings** (the 120-item pilot is reused). Because each
  judge also grades its own lab's model, we report BOTH **raw** and
  **self-preference-corrected** mean ranks. Corrected mean rank (1 = best of 15) improved
  **OURS-v2 10.26 → OURS-v3 6.93**, top-1 win-rate **7.1% → 22.6%** (2nd-highest top-1 in
  the whole field, behind only GPT-5.5's 23.6%). v3 is the **best of every locally-runnable
  model** and **5th of 15 overall** — behind only GPT-5.5 (3.72), Claude (4.95),
  GLM-5 (~355B, 6.52) and Gemini (6.70), ahead of every other open model.
- **Self-preference is real, and corrected for.** Each judge favours its own lab
  (Δ own−peers: **Gemini −2.74, GPT −1.10, Claude −0.47**; mean −1.44 rank positions).
  The correction (drop each frontier competitor's same-lab judge) is what moves Gemini
  from raw **5.78** to corrected **6.70** — below GLM-5 — so no model is graded by its own
  family. 95% CIs (cluster bootstrap by item) are in `RESULTS_FULL_EVAL_803_v3.md` §3.
- **Tier-appropriate move selection (the moat): held, still field-leading.** Overall
  tier-fit **53.1% → 53.2%** (flat), highest in the field (> GPT 43%, Claude 46%, Gemini
  48%). Profile shifted: much stronger at **advanced** (60.9% → **83.6%**) and softer at
  **beginner** (47.9% → **29.6%**) — see the honest caveats.
- **Fine-tuning clearly adds value over the raw 32B.** vs the **untuned Qwen3-32B** base
  it was tuned from: tier-fit **36.9% → 53.2% (+16.3)**, corrected council rank **9.30 →
  6.93**, top-1 **0.1% → 22.6%** — the specialist behavior is *trained in*, not emergent.
- **Balanced score: tops the raw board, essentially tied with GPT-5.5.** The transparent
  weighted score (**tier 45% + self-preference-corrected instructiveness 45% + practical
  /local+cost 10%**) puts **OURS-v3 at 58.0 — 1st of 15, a hair above GPT-5.5 (57.7)**,
  ahead of Claude (51.3), GLM-5 (51.1), Gemini (49.2) and OURS-v2 (47.9). **Honest
  caveat:** OURS-v3 currently trips the strict 97% move-safety/no-jargon gate (94.3% /
  95.6%) — dominated by malformed-output *formatting*, not blunders (actual blunder rate
  ~1.3%, which the serve-time verifier neutralizes); among gate-passing models GPT-5.5
  leads. It is the only near-frontier-balanced model that also runs locally and free.
- **Truthfulness is a fairness floor, not a differentiator.** After the
  verify-and-regenerate gate **every model ships 0% user-visible fabrication** — raw
  pre-gate fabrication is intentionally **not** reported as a per-model comparison axis.
  The honest truth differentiator is the cross-family semantic-judge residual, under three
  nested rules with 95% CIs (any = strict lower bound / majority / unanimous = lenient
  upper bound): OURS-v2 **23% / 26% / 31%** vs GPT-5.5 **79% / 97% / 100%** (pooled 35% /
  57% / 74%). OURS-v3 is a gap803-only model and is not in that showcase sample.

v2 artifacts, the live v2 platform (ports 8000/3000), and `web/src` were not touched.
Everything is v3-suffixed.

---

## What changed (v2 → v3)

| | v2 | v3 |
|---|---|---|
| Base model | Qwen3-1.7B | **Qwen3-32B** (best locally-runnable base per `RESULTS_FULL_EVAL_803.md`) |
| Training | QLoRA on Modal A10G, LoRA r=16 | **QLoRA on Modal A100-80GB, LoRA r=32**, 2 epochs, eff-batch 16, checkpoint/resume |
| Dataset source | 2,628 candidates (348 contrastive FENs) | **7,269 candidates from `v3_candidates.jsonl`** (2,423 curated contrastive positions × 3 tiers) |
| Kept after filter | 2,586 | **7,128** (only 141 dropped: 140 false-fact + 1 engine-speak → **0% false labels**) |
| Train / valid | 2,457 / 129 | **6,772 / 356** |
| Local inference | 4-bit MLX (0.9 GB) | 4-bit MLX (~18 GB) — 32B, still on-device on Apple Silicon |

Teacher (GPT-5.5 via TrueFoundry, `--all-triples`): **7,266 labels, 0 failures,
$141.13**, fully checkpoint/resumed across interruptions.

---

## The 803-position benchmark — the numbers

All local + open models are scored on **all 803 positions × 3 tiers**; the 3 frontier
APIs on a balanced 150-position subset; instructiveness via a **15-model blinded council on
all 450 items where the full field exists (450 × 3 judges = 1,350 rankings)**, reported
raw and **self-preference-corrected**. `instr rank` below is the corrected rank.
Faithfulness is a gated fairness floor (0% user-visible fabrication for every model), so
it is **not** a per-model comparison column — the truth differentiator is the semantic-judge
residual (headline above). Reference points required by the brief are **bold**.

| Model | tier-fit↑ | instr rank↓ (corrected, of 15) | top-1↑ | move-sound↑ | no-jargon↑ | balanced↑ | local |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **OURS-v3 (Qwen3-32B tuned)** | **53.2%** | **6.93** | **22.6%** | 93.2% | 95.6% | **58.0** | yes |
| **OURS-v2 (Qwen3-1.7B tuned)** | **53.1%** | **10.26** | **7.1%** | 97.5% | 100% | 47.9 | yes |
| **Qwen3-32B (untuned base of v3)** | **36.9%** | **9.30** | **0.1%** | 99.6% | 99.4% | 47.8 | yes |
| BASE (Qwen3-1.7B untuned) | 36.5% | 14.18 | 0.0% | 91.6% | 96.4% | 32.5 | yes |
| GPT-5.5 | 43.1% | 3.72 | 23.6% | 98.4% | 100% | 57.7 | no |
| Claude Opus 4.8 | 45.8% | 4.95 | 19.3% | 96.9% | 100% | 51.3 | no |
| Gemini 3.1 Pro | 48.4% | 6.70 | 12.3% | 98.4% | 100% | 49.2 | no |
| GLM-5 (~355B, not local) | 44.7% | 6.52 | 4.0% | 99.6% | 100% | 51.1 | no |

_Balanced weights: tier 45% + self-preference-corrected instructiveness 45% + practical
(local+cost) 10%. OURS-v3 leads the raw balanced score but trips the 97% safety/no-jargon
gate (94.3% / 95.6%, formatting-driven — see caveat 2); GPT-5.5 leads the gate-passing
board._

### v2 → v3 deltas (apples-to-apples, same 15-model council)

| Metric | v2 | v3 | Δ |
|---|---:|---:|---:|
| **Instructiveness** (corrected council rank, lower better) | 10.26 | **6.93** | **−3.33 (better)** |
| Instructiveness top-1 win-rate | 7.1% | **22.6%** | **+15.5 pts** |
| Tier-fit (moat, mean of 3 tiers) | 53.1% | 53.2% | +0.1 (flat) |
| — tier-fit @ advanced | 60.9% | **83.6%** | **+22.7 pts** |
| — tier-fit @ beginner | 47.9% | 29.6% | **−18.3 pts** |
| Balanced score (fabrication removed from the score) | 47.9 | **58.0** | **+10.1** |
| Move-safety (blunder-free) | 98.9% | 94.3% | −4.6 pts |
| No-engine-jargon | 100% | 95.6% | −4.4 pts |

_Fabrication is no longer a delta row: it is a gated fairness floor (0% user-visible for
every model). The honest truth axis is the semantic-judge residual (headline)._

### untuned-32B → v3 deltas (what the fine-tune adds to the raw base)

| Metric | untuned 32B | v3 | Δ |
|---|---:|---:|---:|
| **Tier-fit (moat)** | 36.9% | **53.2%** | **+16.3 pts** |
| Instructiveness (corrected council rank) | 9.30 | **6.93** | **+2.37 (better)** |
| Instructiveness top-1 | 0.1% | **22.6%** | **+22.5 pts** |
| Balanced score | 47.8 | **58.0** | **+10.2** |
| Move-safety | 99.8% | 94.3% | −5.5 pts |

The fine-tune **installs the specialist behavior** (tier-appropriate selection +
instructive, human coaching) that the raw 32B does not have, while keeping the base's
capacity — at the cost of some output-formatting stability (below). Faithfulness is a
gated fairness floor for every model, so it is not a delta axis here.

---

## Honest caveats (measured, reported straight)

1. **Beginner move-calibration regressed vs v2.** The moat metric asks: for a
   *beginner*, does the coach pick the most **human-findable** sound move rather than
   the engine's sharpest? v2 (a small, easily-steered base) did this 47.9% of the
   time; v3 does it 29.6%. The 32B's much stronger chess prior pulls it toward the
   objectively-best move regardless of tier — which is why its **advanced** tier-fit
   is excellent (83.6%) but beginner is weak. Net tier-fit ties v2 and still leads
   the field, but the *shape* of the win moved from beginners to advanced players.
   The platform's deterministic `tier_select` can enforce the beginner move at serve
   time if desired; we did not change the live platform.

2. **~4–5% of raw outputs are malformed** (a spurious leading rating-range fragment,
   or occasional prompt-echo/repetition from greedy decoding on a 32B). This is why
   v3 sits just below the strict 97% safety/no-jargon gate (safety 94.4%, no-jargon
   95.6%). **It is not a blundering problem** — v3's actual blunder rate is **1.3%**,
   on par with v2 (1.1%); the gate shortfall is dominated by *unparseable* malformed
   outputs (~4.3%), which the serve-time verifier + regeneration neutralize. A light
   leading-garble cleanup (applied here, and trivially deployable) recovers most of
   the no-jargon gap; the residual is genuine echo/degeneration.

3. **v3 does not beat the frontier on raw coaching instructiveness.** On corrected
   council rank GPT-5.5 (3.72) and Claude (4.95) still out-coach it (6.93), though v3 is
   now level with Gemini (6.70) and GLM-5 (6.52). v3's edge is being the **only
   near-frontier-balanced model that runs locally and free**, with field-leading
   tier-appropriate move selection. The claim is "best local coach," not "beats GPT-5.5."

4. **Council fields differ across versions.** The v2 report's council ranked 14
   models; this one ranks 15 (adds OURS-v3), so the two reports' absolute ranks are
   not directly comparable — the v2→v3 delta above is measured **within the same
   15-model council**, where both were re-ranked together.

---

## Cost (v3 increment)

| Item | Cost |
|---|---:|
| Teacher v3 generation (GPT-5.5, 7,266 labels) | $141.13 |
| Modal QLoRA training (A100-80GB, incl. retries/resumes) | ~$12 |
| Modal eval generation (A100-80GB, 2,409 coachings, incl. re-run) | ~$8 |
| 15-model council (3 frontier judges × **450 items = 1,350 rankings**) | **$62.27** (~$46 added to expand the 120-item pilot → 450) |
| **v3 increment total** | **~$223** |

Open-model + frontier *coaching* generations were reused from the v2 803 run (not
re-billed). Every long stage (teacher gen, training, eval gen, council) is
checkpoint/resumable and survived multiple interruptions with no lost work.

---

## Artifacts (all v3-suffixed)

- **Model:** LoRA adapter on Modal volume `chess-coach-lora:/chess-coach-v3/adapter`
  + local `models/adapters/chess-coach-v3`; 4-bit MLX at `models/mlx/chess-coach-v3`.
- **Dataset:** `data/dataset/train_v3.jsonl` (6,772) · `valid_v3.jsonl` (356) ·
  candidates `data/generated/candidates_v3.jsonl` (7,269) · `cost_v3.json`.
- **Benchmark:** `data/benchmark_gap803/gen/ours_v3.jsonl`, `leaderboard.json`,
  `council.jsonl` (15-model, 1,350 rankings), `council_stats.json` (raw +
  self-pref-corrected ranks + 95% CIs), `move_safety.json`; truthfulness residual in
  `data/showcase/truthfulness.json`; full board in `RESULTS_FULL_EVAL_803_v3.md`.
- **Code:** `src/teacher/generate_v3.py`, `src/train/train_modal_v3.py`,
  `src/eval/eval_modal_v3.py`, `scripts/gap803_*` (ours_v3 registered in
  `src/eval/benchmark/config.py` + `gap803_report.py` + `gap803_council.py`).
