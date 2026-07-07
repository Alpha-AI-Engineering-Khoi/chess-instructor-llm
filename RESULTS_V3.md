# Results — v3 Chess Coach (Qwen3-32B), v2→v3 and untuned-32B→v3 deltas

**Headline.** v3 changes two things at once — a **20× larger base** (Qwen3-32B instead
of Qwen3-1.7B) and a **larger, cleaner contrastive dataset** (7,128 rows, 0% false
labels) — and it pays off where the extra capacity matters. On the definitive,
zero-leakage **803-position** benchmark (each position coached at all 3 tiers with
byte-identical engine grounding), against a **15-model** field:

- **Instructiveness: a large win.** Blinded cross-family council (GPT-5.5 + Claude
  Opus 4.8 + Gemini 3.1 Pro) mean rank (1 = best of 15) improved **OURS-v2 10.07 →
  OURS-v3 7.06**, and top-1 win-rate **7.5% → 20.3%**. v3 is the **best of every
  locally-runnable model** and 5th of 15 overall — behind only the three frontier
  APIs and one ~355B open model (GLM-5), ahead of Kimi-K2.5, DeepSeek-V3.2/R1,
  Llama-3.3-70B, Qwen3-Next-80B, Gemma-3-27B, Mistral-Large-3.
- **Fabrication: a large win.** False-board-fact rate **30.2% → 5.4%** — the clean
  faithfulness-filtered data plus the stronger base cut fabrication ~6×, to roughly
  the level of the untuned 32B (6.1%) and the frontier APIs (3–5%).
- **Tier-appropriate move selection (the moat): held, and still field-leading.**
  Overall tier-fit **53.1% → 53.2%** (statistically flat), and OURS-v3 remains the
  **highest-tier-fit model in the entire field** (higher than GPT-5.5 40%, Claude
  42%, Gemini 42%). The profile shifted: much stronger at **advanced** (60.9% →
  **83.6%**) and softer at **beginner** (47.9% → **29.6%**) — see the honest caveats.
- **Fine-tuning clearly adds value over the raw 32B.** vs the **untuned Qwen3-32B**
  base it was tuned from: tier-fit **36.9% → 53.2% (+16.3)**, council rank **9.07 →
  7.06**, top-1 **0.0% → 20.3%** — i.e. the specialist behavior is *trained in*, not
  emergent in the base.
- **Balanced score: 2nd of 15, essentially tied with the best frontier model.**
  The transparent weighted score (tier 40% + instructiveness 40% + faithfulness 10%
  + practical/local+cost 10%) puts **OURS-v3 at 61.7 — behind only GPT-5.5 (62.4)**
  and ahead of Gemini (56.6), Claude (55.8), GLM-5 (54.8), the untuned 32B (53.6),
  and OURS-v2 (51.2). It is the only model near the frontier that also runs locally
  and free.

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
APIs on a balanced 150-position subset; instructiveness via a stratified ~120-item,
15-model blinded council (360 judge calls). Reference points required by the brief are
**bold**.

| Model | tier-fit↑ | instr rank↓ (of 15) | top-1↑ | fabrication↓ | move-sound↑ | no-jargon↑ | balanced↑ | local |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| **OURS-v3 (Qwen3-32B tuned)** | **53.2%** | **7.06** | **20.3%** | **5.4%** | 93.2% | 95.6% | **61.7** | yes |
| **OURS-v2 (Qwen3-1.7B tuned)** | **53.1%** | **10.07** | **7.5%** | **30.2%** | 97.5% | 100% | 51.2 | yes |
| **Qwen3-32B (untuned base of v3)** | **36.9%** | **9.07** | **0.0%** | **6.1%** | 99.6% | 99.4% | 53.6 | yes |
| BASE (Qwen3-1.7B untuned) | 36.5% | 14.16 | 0.0% | 14.5% | 91.6% | 96.4% | 38.1 | yes |
| GPT-5.5 | 43.1% | 3.35 | 24.4% | 3.3% | 98% | 100% | 62.4 | no |
| Claude Opus 4.8 | 45.8% | 4.71 | 19.2% | 4.7% | 97% | 100% | 55.8 | no |
| Gemini 3.1 Pro | 48.4% | 5.67 | 11.7% | 4.2% | 98% | 100% | 56.6 | no |
| GLM-5 (~355B, not local) | 44.7% | 6.65 | 5.3% | 7.3% | 99.6% | 100% | 54.8 | no |

### v2 → v3 deltas (apples-to-apples, same 15-model council)

| Metric | v2 | v3 | Δ |
|---|---:|---:|---:|
| **Instructiveness** (council rank, lower better) | 10.07 | **7.06** | **−3.0 (better)** |
| Instructiveness top-1 win-rate | 7.5% | **20.3%** | **+12.8 pts** |
| **Fabrication** | 30.2% | **5.4%** | **−24.8 pts** |
| Tier-fit (moat, mean of 3 tiers) | 53.1% | 53.2% | +0.1 (flat) |
| — tier-fit @ advanced | 60.9% | **83.6%** | **+22.7 pts** |
| — tier-fit @ beginner | 47.9% | 29.6% | **−18.3 pts** |
| Balanced score | 51.2 | **61.7** | **+10.5** |
| Move-safety (blunder-free) | 98.9% | 94.4% | −4.5 pts |
| No-engine-jargon | 100% | 95.6% | −4.4 pts |

### untuned-32B → v3 deltas (what the fine-tune adds to the raw base)

| Metric | untuned 32B | v3 | Δ |
|---|---:|---:|---:|
| **Tier-fit (moat)** | 36.9% | **53.2%** | **+16.3 pts** |
| Instructiveness (council rank) | 9.07 | **7.06** | **+2.0 (better)** |
| Instructiveness top-1 | 0.0% | **20.3%** | **+20.3 pts** |
| Fabrication | 6.1% | 5.4% | −0.7 (held) |
| Balanced score | 53.6 | **61.7** | **+8.1** |
| Move-safety | 99.8% | 94.4% | −5.4 pts |

The fine-tune **installs the specialist behavior** (tier-appropriate selection +
instructive, human coaching) that the raw 32B does not have, while keeping its
faithfulness — at the cost of some output-formatting stability (below).

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

3. **v3 does not beat the frontier on raw coaching instructiveness.** GPT-5.5 (3.35),
   Claude (4.71), and Gemini (5.67) still out-coach it (7.06). v3's edge is being the
   **only near-frontier-balanced model that runs locally and free**, with
   field-leading tier-appropriate move selection. The claim is "best local coach,"
   not "beats GPT-5.5."

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
| 15-model council (3 frontier judges × 120 items) | ~$20 |
| **v3 increment total** | **~$181** |

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
  `council.jsonl` (15-model), `move_safety.json`; full board in
  `RESULTS_FULL_EVAL_803_v3.md`.
- **Code:** `src/teacher/generate_v3.py`, `src/train/train_modal_v3.py`,
  `src/eval/eval_modal_v3.py`, `scripts/gap803_*` (ours_v3 registered in
  `src/eval/benchmark/config.py` + `gap803_report.py` + `gap803_council.py`).
