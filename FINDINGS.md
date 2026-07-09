# FINDINGS — chess-instructor-llm (consolidated, honest, sourced)

> **2026-07-09 honest reframe (read first) — supersedes this doc's earlier phrasing.** A converged audit reframed the submission into three separate claims. Canonical corrections that override anything below: (1) "tier-fit" is renamed **tier-policy exact match** (agreement with the `select_tier_move` rule — a PROJECT RULE, not validated pedagogy); (2) **"0% / zero user-visible fabrication" is retracted** to **"zero verifier-DETECTABLE mechanical violations after gating"** — the checker is high-precision / low-recall, so relational pawn-SAN claims, forks, threats, negations, and eval claims can still reach users; (3) the shipped model is **OURS-v4 (Qwen3-32B)** at tier-policy match **0.767**, distinct-moves **0.730** (73/100 canonical beginner!=advanced opportunities); the honest unbiased head-to-head is **56-28-12** over the 96 diverging positions, and **51-5-6** over 62 is only the v4-success-conditioned subset (NOT a general win rate); (4) deployment-necessity is **false as built** — `select_tier_move` computes the canonical move at ~1.0 by construction, so the model approximates a policy the product already produces. See [`BRAINLIFT.md`](BRAINLIFT.md) and [`SUBMISSION.md`](SUBMISSION.md) for the canonical treatment.

**What this is.** One place to read the whole project story end-to-end, in logical order,
with exact numbers and a pointer to the source doc behind each claim. It supersedes nothing —
every source file is still authoritative — it just stitches the scattered results, analysis, and
experiment reports into a single honest narrative.

**The one-line thesis (earned below, not asserted).**
For a stated Elo tier, *leveled, human-findable "teaching-move" coaching* is **not reliably
delivered by a prompted frontier model** — and **can** be trained into a small local model to run
reliably, cheaply, and privately. We do **not** beat the frontier at raw coaching instructiveness.
Our defensible wins are three: **tier-appropriateness** (the behavior the frontier skips),
**cost/locality** (local MLX, ~$0, private), and **verifier-guaranteed faithfulness** (0%
user-visible fabrication for *any* model behind the gate).

**Model naming.** `base` = `Qwen3-1.7B-4bit` (untuned) · `v1` = `chess-coach-v1` (1.7B QLoRA) ·
`v2` = `chess-coach-v2` (1.7B QLoRA, shipped). Judge/council is always **cross-family** (GPT-5.5 +
Claude Opus 4.8 + Gemini 3.1 Pro), never same-family as the model under test.

_Compiled 2026-07-07 (UTC), covering the v2 + open-model + verifier runs and the definitive
803-position gap eval (§7), which is now complete and folded in._

> **Fabrication-reporting policy (read this first).** Faithfulness is a **fairness floor, not a
> per-model differentiator**: after the verify-and-regenerate gate, **every model ships 0%
> user-visible fabrication** (§4). The current leaderboards therefore do **not** rank or compare
> models on raw pre-gate fabrication. Where models genuinely differ on truth is the **cross-family
> semantic-judge residual** (any / majority / unanimous truthful-rate + 95% CIs; §4). Any raw
> pre-gate fabrication numbers that appear in the historical v1→v2 / open-model sections below are
> *model-capacity signals from those dated experiments*, not a serve-time or current-comparison axis.

---

## 0. v3 update (Qwen3-32B) — the capacity bet, tested

The BrainLift predicted the small-model faithfulness deficit is a **capacity effect a bigger
base removes for free**, and that "small local" should move to a ~27–32B base. v3 tests it: a
QLoRA fine-tune of **Qwen3-32B** on the larger faithfulness-filtered contrastive set (7,128 rows,
**0% false labels**), evaluated on the same 803-position benchmark against a **15-model** field.

| Axis | OURS-v2 (1.7B) | untuned Qwen3-32B | **OURS-v3 (32B tuned)** |
|---|---:|---:|---:|
| Tier-fit (the moat) | 53.1% | 36.9% | **53.2%** (field-leading; adv 83.6%, beg 29.6%) |
| Instructiveness (**corrected** council rank, of 15; lower=better) | 10.26 | 9.30 | **6.93** (best local; top-1 22.6%) |
| User-visible fabrication (after gate) | 0% | 0% | **0%** (fairness floor, all models) |
| Balanced score (fab removed from score) | 47.9 | 47.8 | **58.0** (1st of 15 raw; see gate caveat) |

**Read honestly.** (1) The capacity bet paid off *pre-gate*: the 32B base essentially removes
the small-model board-tracking deficit on its own — but note **user-visible fabrication is
already 0% for every model** behind the verify-and-regenerate gate, so this is a pre-gate
capacity story, not a serve-time or ranking axis (raw pre-gate fabrication is no longer reported
as a per-model comparison). (2) Instructiveness (self-preference-corrected council rank on all
**450 items × 3 judges = 1,350 rankings**) jumped **10.26 → 6.93**; v3 is the **best
locally-runnable coach**, behind only GPT-5.5, Claude, GLM-5 (~355B) and Gemini, with the
2nd-highest top-1 in the field (22.6%). (3) Tier-fit (moat) held field-leading, but its *shape*
shifted — the 32B's stronger prior made it excellent at the advanced/sharpest move (83.6%) and
weaker at the beginner/human-findable move (29.6% vs v2's 47.9%); serve-time `tier_select` can
enforce the beginner move if wanted. (4) v3 **tops the raw balanced score (58.0, a hair above
GPT-5.5's 57.7)** but trips the strict 97% safety/no-jargon gate (94.3% / 95.6%) — ~4–5% of raw
outputs are malformed (leading rating-range fragment / prompt echo), *not* blunders (true blunder
rate **1.3%**, ≈ v2), and caught at serve time; among gate-passing models GPT-5.5 leads. Detail:
[`RESULTS_V3.md`](RESULTS_V3.md), [`RESULTS_FULL_EVAL_803_v3.md`](RESULTS_FULL_EVAL_803_v3.md).
The live platform still serves v2.

---

## 1. The gap is real — and it is dense

**Claim.** Before claiming a trained model can *fill* a behavior gap, prove a prompted frontier
model does **not** reliably do the behavior. It doesn't, and the positions where the behavior
matters are the majority of real games — not a rare corner case.

On **50 held-out positions** with grounding byte-identical to the app (Stockfish sound pool +
Maia + verified facts), average of the three frontier models:

| Frontier behavior (avg of GPT-5.5 / Claude Opus 4.8 / Gemini 3.1 Pro) | Rate |
|---|---:|
| **Tier-differentiation** (picks ≥1 different move across the 3 Elo tiers) | **22.7%** |
| **Engine-mirroring at *every* tier** (returns Stockfish #1 regardless of level) | **68.7%** |
| Beginner steered to the *findable* move (opportunity subset, engine-best ≠ most-findable) | 20.5% |

The canonical failure: serving a 1200 the 3000-Elo engine-best move in a GM-level line — sound,
but not *findable* and not *instructive* for that student.

**How often does this matter?** On a fresh sweep of **5,999** real rated Lichess positions
(4,816 decidable), **3,226 = 67.0%** are *discriminating* — the tier-appropriate move differs from
the engine's #1 for at least one tier (per tier: beginner 57.4%, intermediate 56.1%, advanced
55.5%). So the gap is not a rare edge case; it is present in **~two-thirds** of real decision
positions.

**Supporting foundation (why findable ≠ engine-best).** Lichess puzzle *solutions* are
engine-best by construction — solution == Stockfish #1 in **99.3%** of puzzles, median cp-loss 0 —
yet only **60.0%** are naturally findable for a beginner (Maia-1100 top or policy ≥ 0.25), and
12.7% are low-policy "hard-to-find" best moves. This is why external solutions are used as
*context, never labels*: reusing them as labels would re-teach "always play the engine move" and
destroy tier differentiation.

**Sources:** `data/analysis/GAP_REPORT.md` (frontier gap, 50 held-out) ·
`data/eval/GAP_POSITIONS_REPORT.md` (67% gap density, 5,999 analyzed) ·
`data/analysis/PUZZLES_REPORT.md` (findable ≠ engine-best).

---

## 2. base → v1 → v2 — what fine-tuning fixed (and what it couldn't)

**Claim.** Fine-tuning reliably delivers everything it can control by shaping the training
distribution (sound move, no jargon, length cap, tier register). The one thing data-shaping alone
could **not** fix in v1 — **truthfulness** — was measured flat and became the project's real thesis;
the v2 data intervention then moved faithfulness *and* corrected tier-differentiation.

### 2a. base → v1 (15 held-out scenarios, Claude-Opus rubric + deterministic checks)

| Objective check (% pass) | base | v1 |
|---|---:|---:|
| move_sound | 87% | **100%** |
| no_engine_speak (no jargon) | 33% | **100%** |
| ply_cap_ok | 67% | **100%** |

| Claude judge (0–2) | base | v1 |
|---|---:|---:|
| spec_adherence | 0.47 | **0.93** |
| level_calibration | 0.60 | **1.13** |
| no_engine_speak | 0.87 | **1.87** |
| **truthfulness** | 0.13 | 0.13 ← **flat** |
| task_quality | 0.13 | 0.27 |

v1 won decisively on style/format (the behaviors that *resisted prompting* on base) but the judge
was consistent and damning on truthfulness: the tuned model says the right *kind* of thing in the
right *voice* but **invents board facts** (piece squares, hanging pieces, captures). Root cause is
in the **data**: v1 labels were GPT-5.5 teacher outputs filtered for format/soundness but **not**
faithfulness (6.3% of labels were false), so the 1.7B faithfully learned the teacher's style *and*
its occasional fabrication.

### 2b. v1 → v2 (the data intervention: faithfulness gate + tier-aware teacher + contrastive tiers)

| Metric | v1 | v2 |
|---|---:|---:|
| Tier-differentiation (120 matched held-out, greedy) | 27.5% | **39.2%** |
| Direction correct (beginner more findable than advanced) | **No** | **Yes** |
| — mean pool-rank, beginner (higher = more findable) | 0.43 | **0.78** |
| — mean pool-rank, advanced (lower = sharper) | 0.62 | **0.45** |
| — beginner move == its Maia (human) top | 39.2% | **61.7%** |
| Fabrication, benchmark **grounded** (100 held-out) | 50% | **33%** |
| — avg false facts / answer | 0.62 | **0.46** |
| Fabrication, divergence harness (120, raw output) | 46.1% | **31.7%** |
| Council instructiveness rank, grounded (1=best of 5) | 4.13 | **3.68** |
| — top-1 win-rate / gap to best frontier | 2% / +2.22 | **8% / +1.60** |
| False labels in the training set | 6.3% → residual 1.6% | **0%** (reject gate) |
| Contrastive multi-tier FENs (same position, per-tier move) | 0 | **348** |

Continuity on the same Claude rubric that flagged flat truthfulness: **base 0.13 → v1 0.13 → v2
0.20**; the small 0–2 lift agrees in *direction* with the far more robust deterministic fabrication
cut (50% → 33%).

> **Honest caveat carried from v2:** the win is instructiveness *narrowed, not closed*; v2 still
> trails the frontier council. And v2 is **more** brittle *ungrounded* (fabrication 87% → 99%)
> because it teaches more concretely — but the product always runs grounded, so that is the
> expected trade, reported plainly, not the deployment path.

**Sources:** `RESULTS.md` (base→v1) · `RESULTS_V2.md` (v1→v2, dataset rebuild, deltas).

---

## 3. The benchmarks — 5 models, then 9 bigger open models

**Claim.** Grounding closes the *behavioral* gap (soundness, jargon); the frontier still
**out-instructs**; and the fabrication gap our 1.7B carries is a **size** problem, not a
data-intervention problem — bigger open models fabricate 1–8% where ours sits at 33–38%.

### 3a. 5-model 2×2 (100 held-out, grounded vs ungrounded)

Grounding is the lever: OURS move-soundness **41% → 98%** and fabrication **99% → 33%** once the
verified facts + sound pool are in the prompt — the board-tracking a 1.7B cannot supply itself.
Grounded objective (v2 run):

| Model (grounded) | fabrication ↓ | move_sound ↑ | no_engine_speak ↑ | council rank (1=best of 5) ↓ |
|---|---:|---:|---:|---:|
| OURS (v2, 1.7B) | 33% | 98% | 100% | 3.68 |
| BASE (1.7B) | 13% | 92% | 100% | 4.70 |
| GPT-5.5 | 3% | 99% | 100% | 2.09 |
| Claude Opus 4.8 | 3% | 98% | 100% | 2.09 |
| Gemini 3.1 Pro | 4% | 100% | 100% | 2.44 |

Read: grounded, everyone picks a sound move and nobody leaks jargon; the honest gap is
**instructiveness** (council) and **fabrication**, where a bigger model still wins.

### 3b. Bigger open models (same 100 positions, identical grounding, unified 10-way council)

| Model | family | grounded fabrication ↓ | council rank (of 10) ↓ |
|---|---|---:|---:|
| Gemma-3-27B-it | open | **1%** | 5.25 |
| Llama-3.3-70B | open | 1% | 5.35 |
| GPT-5.5 | frontier | 2% | **2.53** |
| Gemini 3.1 Pro | frontier | 2% | 3.79 |
| DeepSeek-R1 (reasoning) | open | 2% | — |
| GLM-5 | open | 4% | — |
| Claude Opus 4.8 | frontier | 6% | 2.91 |
| DeepSeek-V3.2 | open | 6% | 5.18 |
| Mistral-Large-3 (675B) | open | 6% | 6.17 |
| Kimi-K2.5 | open | 7% | — |
| Qwen3-32B | open | 8% | 6.68 |
| Qwen3-Next-80B-A3B | open | 8% | — |
| BASE (1.7B) | ours | 15% | 9.17 |
| **OURS (v2, 1.7B)** | ours | **38%** | **7.95** |

Takeaways:
- **The truthfulness gap is a size problem, essentially closed by size.** Every open model
  fabricates 1–8% vs OURS-v2 38%; Gemma-3-27B (1%) matches the frontier (~3% avg). Our *data*
  intervention was never going to give a 1.7B the board-tracking that parameters buy.
- **Bigger open coaches beat OURS-v2 but not the frontier.** Best open coach DeepSeek-V3.2 (5.18)
  and Gemma-3-27B (5.25) out-rank OURS-v2 (7.95) by ~1.3–2.8, yet trail GPT-5.5 (2.53) by ~2.7.
- **Raw size is not the coaching driver.** Mistral-Large-3 (675B) (6.17) is out-coached by the
  much smaller DeepSeek-V3.2 and Gemma-3-27B — training/quality beats parameter count here.
- **Best open base for v3: Gemma-3-27B-it** — lowest fabrication in the whole field (1%),
  essentially the top open coach, QLoRA-able and 4-bit-runnable locally on a 64 GB Mac. Scaling
  our *own* family (Qwen3-32B) cuts fabrication 38% → 8% but coaches worst of the open field (6.68).

> **Number reconciliation (honest):** OURS-v2 grounded fabrication reads **33%** in
> `RESULTS_BENCHMARK_v2.md` and **38%** in `RESULTS_OPEN_MODELS.md`. Same outputs, *stricter
> current verifier* in the open-model re-score — the ranking and the size effect are unaffected.
> Also note three benchmark files exist: `RESULTS_BENCHMARK.md` is the first 2×2 run; the matched
> **`RESULTS_BENCHMARK_v1.md` / `RESULTS_BENCHMARK_v2.md`** pair is the v1-vs-v2 comparison cited
> in §2 (v1 grounded fab 50% → v2 33%).

**Sources:** `RESULTS_BENCHMARK.md`, `RESULTS_BENCHMARK_v1.md`, `RESULTS_BENCHMARK_v2.md`
(5-model 2×2) · `RESULTS_OPEN_MODELS.md` (9 bigger open models).

---

## 4. The verifier — faithfulness becomes a guarantee, not a hope

**Claim.** The production verify-and-regenerate gate drives **user-visible fabrication to 0% for
every model** — a structural guarantee, not a statistical nudge. The only honest differentiator
left is the **fallback rate** (how often the gate must throw the model's answer away).

Same 50 held-out positions (fabrication-weighted: 33 where OURS-v2 fabricated + 17 clean), N=4
regenerate attempts, then a deterministic engine-derived fallback that is true by construction:

| Model | RAW fab | GATED fab | Fallback rate |
|---|---:|---:|---:|
| **OURS-v2 (1.7B)** | **40%** | **0%** | **10.0%** (most dependent on the net) |
| Mistral-Large-3 (675B) | 14% | 0% | 0.0% |
| Kimi-K2.5 | 8% | 0% | 0.0% |
| GLM-5 / DeepSeek-V3.2 / Qwen3-32B / Qwen3-Next-80B | 6% | 0% | 0.0% |
| Claude Opus 4.8 / Gemma-3-27B | 4% | 0% | 0.0% |
| GPT-5.5 | 2% | 0% | 2.0% |
| DeepSeek-R1 (reasoning) | 2% | 0% | 2.0% |
| BASE (1.7B) | 2% | 0% | 0.0% |
| Gemini 3.1 Pro / Llama-3.3-70B | 0% | 0% | 0.0% (most self-sufficient) |

- **All 14 models land at 0% GATED user-visible fabrication.** For OURS-v2 that is 40% → 0%
  (−40 pts): 15 of its 20 RAW fabrications were fixed by the model *regenerating clean within 4
  attempts*; 5 needed the verified fallback. 90% of finals are still the model's own prose.
- **Independent audit:** re-running `verify_text` from scratch on all **700** stored GATED outputs
  finds **0** fabrications and **0** empty outputs — a fresh-check result, not just what the gate
  logged.
- **Honest residual risk:** GATED fabrication is measured by the *same* checker the gate uses, so
  0% means "no leak the checker can see." The remaining risk is the checker's **coverage** (a false
  claim phrased in a way `verify_text` doesn't yet recognize) — a single shared blind spot to
  harden in the verifier, applying equally to all models, not a reason to trust any one model more.

**Why this reframes the project:** faithfulness is now a **commodity layer** any model can stand
on. It stops being a differentiator — which pushes the moat up to move selection (§6).

**Sources:** `data/experiments/VERIFIER_EVAL.md` (2-model precursor) ·
`data/experiments/VERIFIER_EVAL_ALL.md` (all 14 models + independent audit).

---

## 5. Rich-grounding A/B — structured input backfires for the fine-tune

**Claim.** Handing the model a *complete structured board state* (every square + castling +
en-passant + side-to-move + move number, with the pool/Maia as tables) does **not** reduce
fabrication for our fine-tuned 1.7B — it **increases** it, because the format is off-distribution
from what v2 was trained on.

50 held-out positions, everything identical between A and B except the grounding block
(A = current prose/ASCII grounding the model was trained on; B = rich structured enumeration):

| OURS-v2 slice | A (prose) fab | B (rich) fab | Δ |
|---|---:|---:|---:|
| overall | 40% | 56% | **+16 pts** |
| clean stratum (A already handled) | 24% | 65% | **+41 pts** |
| fabricated stratum | 48% | 52% | +3 pts |
| move soundness | 100% | 94% | −6 pts |

Paired on the same 50 positions: B *fixed* 10 of A's fabrications but *created* **18** new ones —
net **+8**. The clean-stratum blowup (+41 pts) is the smoking gun: structured grounding actively
broke positions the prose grounding already handled. The mechanism is off-distribution parsing —
v2's SFT rows render the position as the `Board:` ASCII grid with no structured block, so B
substitutes a layout the 1.7B never saw and it tracks the board *worse* (inventing a "knight on
g2", "rook on f1", etc.). The **frontier** (gpt-5.5), fine-tuned on no grounding format, is
format-agnostic and stays near-zero (0% → 7%, within noise on 15 items) — so structured grounding
is fine *in principle*; the regression is specific to the small fine-tune.

**v3 implication:** don't bolt a new prompt shape onto a model trained on a different one. If you
want structured grounding, *train it in* (regenerate SFT in that format). The residual-fabrication
lever is the **verifier** (§4), not the prompt.

**Source:** `data/experiments/RICH_GROUNDING_AB.md`.

---

## 6. The synthesized thesis (what to build for v3)

Putting the six findings together:

1. **Faithfulness is table-stakes, and it is now solved as a layer.** The verifier guarantees 0%
   user-visible fabrication for *any* model (§4). So truthfulness is no longer a moat — it is a
   commodity gate every deployment can stand on. A model's fabrication rate now only sets its
   *fallback rate* (cost/UX), not whether it can lie to a student.

2. **The moat is tier-appropriate move selection.** That is the behavior a prompted frontier model
   does **not** reliably do (22.7% tier-diff, 68.7% engine-mirror), and it matters in **67%** of
   real positions (§1). It is the one axis where a targeted data intervention moved the needle in
   the *right direction* (v2: 27.5% → 39.2%, direction corrected — §2), and it is *not* bought by
   raw size (Mistral-675B out-coached by 27B — §3).

3. **"Small local" should mean ≈ 27–32B for v3, not 1.7B.** A 1.7B lacks the board-fact tracking
   that parameters buy — which is exactly why it needed grounding + a verifier to be safe, and why
   structured grounding backfired on it (§5). The recommended v3 base is **Qwen3-32B ≈ Gemma-3-27B-it
   — a tie** on the definitive 803 eval (§7): both are QLoRA-able and 4-bit-runnable on a 64 GB Mac,
   Qwen3-32B with more capacity, Gemma-3-27B smaller and more faithful (fab 2% vs 6%). (The best
   *balanced* open coach, GLM-5, is not locally runnable, so it is not a base.) The v3 recipe:
   **fine-tune a ~27–32B open base on tier-aware contrastive data (the 67%-dense gap set), keep the
   engine grounding, keep the verifier.** Size commoditizes faithfulness; our data intervention
   supplies the tier behavior size alone won't — and on the 803 set OURS-v2 already **leads the whole
   field on tier-appropriate move selection** (§7), the clearest proof that the behavior is trained,
   not bought with parameters.

**What we honestly claim — and don't:**
- ❌ We do **not** beat frontier models at coaching instructiveness (they still out-rank us on the
  council, grounded and ungrounded).
- ✅ We **do** win on **tier-appropriateness** (leveled, human-findable move selection the frontier
  skips), **cost/locality/privacy** (local MLX, ~$0/query, no data leaves the machine), and
  **verifier-guaranteed faithfulness** (0% user-visible fabrication, structurally).

---

## 7. Definitive 803-position gap eval — the moat, confirmed at scale

**Claim.** On the airtight, zero-leakage **803-position** gap set (100% discriminating — every
position is one where the tier-appropriate move differs from the engine's #1 for at least one tier),
scored across the **full 15-model field** with byte-identical grounding, **OURS-v3 and OURS-v2 lead
the entire field on tier-appropriate move selection** (tier-fit ~53%), the moat. With fabrication
removed as a scoring axis (it is a gated fairness floor — 0% user-visible for every model, §4) and
instructiveness **self-preference-corrected**, **OURS-v3 tops the raw balanced score (58.0)** — a
hair above GPT-5.5 (57.7); it trips the strict 97% safety/no-jargon gate on *formatting* (not
blunders), so among gate-passing models GPT-5.5 leads. OURS-v2 (the shipped 1.7B) sits mid-pack,
held down only by the 1.7B instructiveness ceiling.

**Method (cost-smart, transparent weighting).** Deterministic metrics (tier-fit, tier-diff,
direction, move-safety, no-jargon) are computed on **all 803 × 3 tiers** for the 12 models with
full generations (OURS-v2, OURS-v3, BASE + 9 open); the 3 frontier references are measured on a
balanced **150-position** stratified subset × 3 tiers. Instructiveness is one blinded cross-family
council (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) over **all 450 items where every model has a
generation — 450 × 3 judges = 1,350 rankings** — reported raw and **self-preference-corrected**
with 95% CIs (cluster bootstrap by item; §3 of `RESULTS_FULL_EVAL_803_v3.md`). **Balanced score =
tier-appropriate move selection 45% + self-preference-corrected instructiveness 45% + practical
(local + cost) 10%**, with move-safety and no-jargon as pass/fail gates. **Fabrication is not a
scoring axis** — every model ships 0% user-visible fabrication behind the gate (a fairness floor);
the honest truth differentiator is the semantic-judge residual (§4). Council cost **$62.27**
(1,350 rankings); full gap803 eval spend (all generations + council) **$112.15**.

### 7a. Balanced leaderboard (all 15 models)

`tier-fit` is the moat metric; `instr rank` is the **self-preference-corrected** council mean rank
(lower = better, of 15). Faithfulness is a gated fairness floor (0% user-visible fabrication for
every model), so it is **not** a comparison column here — the truth differentiator is §4.

| # | Model | family | tier-fit ↑ (moat) | instr rank ↓ (corrected) | **balanced** ↑ | gate | local |
|---|---|---|---:|---:|---:|:--:|:--:|
| 1 | **OURS-v3 (Qwen3-32B tuned)** | ours | **53%** | 6.93 | **58.0** | **FAIL** | yes |
| 2 | GPT-5.5 | frontier | 43% | 3.72 | **57.7** | pass | no |
| 3 | Claude Opus 4.8 | frontier | 46% | 4.95 | **51.3** | pass | no |
| 4 | GLM-5 | open | 45% | 6.52 | **51.1** | pass | no |
| 5 | Gemini 3.1 Pro | frontier | 48% | 6.70 | **49.2** | pass | no |
| 6 | Kimi-K2.5 | open | 36% | 7.43 | **48.2** | pass | no |
| 7 | **OURS-v2 (1.7B tuned)** | ours | **53%** | 10.26 | **47.9** | pass | yes |
| 8 | Qwen3-32B (untuned v3 base) | open | 37% | 9.30 | **47.8** | pass | yes |
| 9 | DeepSeek-R1 | open | 44% | 7.98 | **46.3** | pass | no |
| 10 | Llama-3.3-70B | open | 40% | 8.39 | **45.9** | pass | tight |
| 11 | Gemma-3-27B-it | open | 35% | 8.88 | **45.5** | pass | yes |
| 12 | DeepSeek-V3.2 | open | 41% | 8.24 | **45.4** | pass | no |
| 13 | Qwen3-Next-80B-A3B | open | 32% | 8.53 | **44.5** | pass | tight |
| 14 | Mistral-Large-3 (675B) | open | 37% | 9.43 | **41.0** | pass | no |
| 15 | BASE (1.7B untuned) | base | 36% | 14.18 | **32.5** | **FAIL** | yes |

**Per-judge self-preference** (Δ own − peers rank; negative ⇒ judge favours its own family): Gemini
**−2.74**, GPT **−1.10**, Claude **−0.47** (mean −1.44). The correction drops Gemini from raw 5.78
to corrected 6.70 (below GLM-5), so no model is graded by its own lab.

### 7b. The moat — the two OURS models lead the field on tier-appropriate move selection

Ordered by `tier-fit` (pick == the canonical `select_tier_move` move, mean over the 3 tiers): the
**two OURS models tie at 53% for #1–2 of all 15** — above every frontier and every bigger open model
(Gemini 48% > Claude 46% > GLM-5 45% > DeepSeek-R1 44% > GPT-5.5 43% > …). The lead is widest exactly
where the moat matters most: **OURS-v2's beginner-tier fit (48%) and intermediate-tier fit (50%) are
the highest in the field** (steering a weaker student toward the *human-findable* move), while
**OURS-v3 dominates the advanced tier (84%)** (the sharpest move, which is correct there). Together
they own both ends of the tier gradient.

The whole field is weak on the mean — even the best sits at 53% and most cluster at 32–48% — because
tier-appropriate move selection is a **trained** behavior, not an emergent one. That is precisely
the §1/§2 thesis confirmed at scale: prompting alone (frontier included) does not reliably deliver
it, and the models trained *for* it lead.

### 7c. The deployed reading — why raw ≠ shipped

Faithfulness is no longer a scoring penalty: it is a gated fairness floor (0% user-visible
fabrication for **every** model, §4), so the balanced score no longer docks any model for raw
pre-gate fabrication. What remains for the shipped **OURS-v2** (balanced 47.9, mid-pack) is the one
honest gap:
- **Instructiveness (corrected council rank 10.26).** The 1.7B prose ceiling is real and is *not*
  removed by the verifier — this is the axis where OURS-v2 honestly trails. **OURS-v3** closes most
  of it (6.93, best local), at the cost of tripping the formatting gate.

So the **deployed** OURS-v2 is the field's **tier-selection leader (with OURS-v3), at 0%
user-visible fabrication, running free and locally** — trailing only on prose instructiveness. That
is the honest shape of the product: not the top of the raw balanced leaderboard, but the best at the
one behavior that is the moat, with faithfulness handed to the verifier and cost/privacy to locality.

### 7d. Recommendations (supersedes the 100-position picks in §3)

- **Best overall coach, any provider: GPT-5.5** (balanced 62.3) — the frontier still coaches best;
  it is the distillation-teacher benchmark, not a deployable base.
- **Best OPEN coach: GLM-5** (balanced 55.3, best open instructiveness) — but it far exceeds 64 GB
  and **cannot run locally**, so it is not a v3 base.
- **Best v3 *base*: Qwen3-32B ≈ Gemma-3-27B-it — a genuine tie** (re-weighted base-fit 72.6 vs 72.2,
  within noise). Qwen3-32B brings more capacity/tier-selection; **Gemma-3-27B is smaller and more
  faithful (fab 2% vs 6%)**. Either is a defensible 4-bit-local v3 base on a 64 GB Mac; prefer Gemma
  if faithfulness/size is paramount, Qwen3-32B if raw capacity is.
- **This supersedes §3's "Gemma-3-27B is the clear best base"** from the 100-position run: on the
  definitive 803 set the two are tied, and the *balanced* open winner (GLM-5) is a different model
  that simply is not locally runnable. The v3 recipe is unchanged — fine-tune a ~27–32B open base on
  the tier-aware contrastive gap data, keep the engine grounding, keep the verifier — because
  tier-appropriateness (where OURS-v2 already leads) is exactly what fine-tuning adds, while capacity
  + faithfulness + local-runnability are what the base must bring.

**Sources:** `RESULTS_FULL_EVAL_803.md` (definitive leaderboard, all 14) ·
`data/eval/GAP_POSITIONS_REPORT.md` (803-set derivation, zero-leakage) ·
`data/benchmark_gap803/` (positions, generations, objective, council, leaderboard).

---

## Source index

| # | Topic | Source doc |
|---|---|---|
| 1 | Frontier gap (50 held-out) | `data/analysis/GAP_REPORT.md` |
| 1 | Gap density 67% (5,999 analyzed) + 803-set derivation | `data/eval/GAP_POSITIONS_REPORT.md` |
| 1 | v1 tier-differentiation weak & mis-directed; 0 contrastive tiers | `data/analysis/DIVERGENCE_REPORT.md` |
| 1 | Puzzle solutions = engine-best, not findable (context ≠ labels) | `data/analysis/PUZZLES_REPORT.md` |
| 2 | base → v1 (objective + Claude rubric, flat truthfulness) | `RESULTS.md` |
| 2 | v1 → v2 (dataset rebuild + deltas) | `RESULTS_V2.md` |
| 3 | 5-model 2×2 grounded/ungrounded benchmark | `RESULTS_BENCHMARK.md`, `RESULTS_BENCHMARK_v1.md`, `RESULTS_BENCHMARK_v2.md` |
| 3 | 9 bigger open models vs OURS / frontier | `RESULTS_OPEN_MODELS.md` |
| 4 | Verifier → 0% user-visible fabrication (2-model) | `data/experiments/VERIFIER_EVAL.md` |
| 4 | Verifier → 0% for all 14 models + audit | `data/experiments/VERIFIER_EVAL_ALL.md` |
| 5 | Rich/structured grounding A/B (backfires) | `data/experiments/RICH_GROUNDING_AB.md` |
| 7 | Definitive 803-position gap eval (all 14 models) | `RESULTS_FULL_EVAL_803.md`; `data/eval/GAP_POSITIONS_REPORT.md` (derivation) |
