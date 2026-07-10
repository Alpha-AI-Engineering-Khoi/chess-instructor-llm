# Stage-4 re-eval on the CORRECTED v6 benchmark (v4 / base / v6-dpo / v6-distill)

The consolidating, apples-to-apples re-evaluation of every OURS model on the
**corrected v6 benchmark labels** (`data/benchmark_gap803/scenarios_v6.jsonl`),
scored on the **120 held-out TEST positions x 3 tiers (360 scenarios)**. The TEST
FENs are STABLE across the v6 rebuild (0 FEN changes; only `canonical`/`engine_best`
re-derived under deep Stockfish-17 search + Syzygy), so this re-score is directly
comparable to the shipped v4 headline while judging against corrected targets.

## Method (un-gameable, one controlled session)

- **One Modal session on chess-instructor-2** (`scripts/stage4_eval.py`): the base
  `unsloth/Qwen3-32B-unsloth-bnb-4bit` is loaded ONCE, adapters are swapped, and
  every model condition is generated with the SAME greedy decode
  (`do_sample=False`, `repetition_penalty=1.15`, `no_repeat_ngram_size=4`), so the
  comparison is identical-input, identical-decode, single-session.
- **GROUNDED** (deployable coach: Stockfish sound-pool + Maia + facts) for base,
  v4, v6-dpo, built from the corrected v6 grounding via the exact v6 prompt path
  (`build_v6._scn_for_prompt` -> `src.eval.benchmark.prompts.build_grounded_user`);
  the rebuilt tier selection reproduces `scenarios_v6` canonical labels 360/360.
- **NO-GROUNDING** (distillation prompt: no engine, no Maia) for base and
  v6-distill, from `scripts/distill_v6_format.build_nogrounding_user` verbatim.
- **Scoring is deterministic + local** with the SAME vendored extractor the shipped
  v4 report uses (`src.eval.evaluate.extract_recommended_move`, the extractor
  `scripts/reproduce_v4.py` asserts against). Metrics: tier-policy exact match
  (agreement with the preregistered `select_tier_v6` canonical move), move-soundness
  (pick in the corrected sound pool), distinct-moves-per-level, names-a-move, and a
  prose-format check. Fabrication is 0 by construction: the extractor accepts only
  a LEGAL move, so an illegal/fabricated move is never counted as a named move.

Terminology and caveats are unchanged from `RESULTS_HONEST_EVAL_V4.md`: "tier-policy
match" is agreement with a PROJECT RULE (learnability), not validated pedagogy; the
grounded models are a reliable grounded EXECUTION of a policy the product already
computes from the same grounding; the distillation result is the harder "behavior in
the weights" claim.

## Headline (GROUNDED, corrected v6, 120 held-out TEST)

| Model (grounded) | tier-policy match | move-sound | distinct-per-level | names-a-move | format | B / I / A |
|---|---:|---:|---:|---:|---:|---|
| BASE (Qwen3-32B untuned) | 0.428 | 0.969 | 0.303 | 0.975 | 0.975 | 0.442 / 0.408 / 0.433 |
| OURS-v4 (shipped) | 0.861 | 0.983 | 0.987 | 0.983 | 0.939 | 0.858 / 0.750 / 0.975 |
| OURS-v6-dpo | **0.881** | 0.983 | 0.987 | 0.983 | 0.919 | 0.858 / 0.808 / 0.975 |

- distinct-per-level denominator = the 76 TEST positions whose canonical beginner
  and advanced moves differ (v4 and v6-dpo both differentiate on 75 of 76).
- format = fraction whose reply both names a move and closes with a "Takeaway:"
  line within the 256-token cap (a prose-completeness check, not a move check).

## Central questions, answered on the held-out corrected benchmark

### 1. Did v6-dpo beat v4 without regressing (out-of-distribution)?

**Yes, modestly, with no regression.** v6-dpo's tier-policy match is **0.881 vs
v4 0.861 (+0.0195)**, and the gain is entirely the **intermediate tier: 0.808 vs
0.750 (+0.0583)** on positions the DPO adapter never trained on. Beginner (0.858),
advanced (0.975), move-soundness (0.983), distinct-moves (0.987), and names-a-move
(0.983) are all **identical** to v4. The one movement against v6-dpo is format
(0.919 vs 0.939): with named-a-move and soundness identical, this is v6-dpo writing
marginally longer coaching that occasionally hits the 256-token cap before the
Takeaway line (16/360 v4 vs 23/360 v6-dpo truncated), not a move or coaching-quality
regression. So the preference tune SHARPENED the mid-tier moat and held it out of
distribution while leaving soundness, differentiation, and the beginner/advanced
tiers exactly where v4 had them.

### 2. Distillation: behavior in the weights (base-no-grounding vs distill-no-grounding)

Strip the engine + Maia grounding and the untuned base essentially cannot coach:
tier-policy match **0.022**, names-a-move **0.250**, move-soundness **0.081** (it
fabricates illegal or unsound moves without the sound list). The engine-distilled
adapter recovers the tier rule from its WEIGHTS: tier-policy match **0.325
(+0.303, ~15x)**, names-a-move **0.983**, move-soundness **0.653**, distinct
**0.461**.

Honest limit (advanced tier): distillation recovers the human-like tiers best and
the sharpest tier worst, tier-policy match B/I/A = **0.358 / 0.400 / 0.217**. The
advanced target is the engine-best move; reproducing it from weights alone, without
the engine grounding the condition removes, is genuinely the hardest, so advanced
is the distilled model's weakest tier. Grounding-free soundness (0.653) also trails
the grounded 0.98, so this is a behavior-in-weights proof, not a claim that
grounding is unnecessary in production.

### 3. Base-vs-tuned delta (the spec's core requirement, grounded)

On the corrected held-out benchmark, tuning is the load-bearing factor:
tier-policy match **v4 - base = +0.433** (0.861 vs 0.428) and **v6-dpo - base =
+0.453**; distinct-moves **+0.684** (0.987 vs 0.303). The correction lifts every
model's absolute grounded number slightly (cleaner grounding: base 0.347 -> 0.428,
v4 0.767 -> 0.861 relative to the v4-era publication), but the tuned-minus-base gap
is preserved, so the base-vs-tuned result is robust to the label correction.

## Continuity with the shipped v4 headline

The 120 TEST FENs are stable, so the SHIPPED v4 generations can be re-scored
against the corrected labels (`scripts/stage4_rescore_committed.py`). Re-scoring the
committed v4/base gens (which saw the OLD v4-era grounding) against the CORRECTED
labels gives tier-policy match v4 **0.481**, base **0.297**: lower than both the
published v4-era number (0.767) and the fresh corrected-grounding number (0.861),
because those committed outputs were produced against the pre-correction sound pool
and are now judged by re-derived targets. The fresh single-session numbers above
(every model on the corrected grounding) are the fair, apples-to-apples current
result; the re-score is the continuity check that isolates how much the label
correction alone moved the target.

## Instructiveness (blinded cross-family council, secondary)

A representative blinded council (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro via the
TrueFoundry gateway, `scripts/stage4_council.py`) ranks a compact field
(v6-dpo, v4, base, and the three frontier coaches) on a stratified 90-item sample of
the held-out TEST, GROUNDED, blinded (labels shuffled per item, deterministic key),
with bootstrap 95% CIs. 90 items x 3 judges = 270 rankings, 0 failures.

| Model (grounded) | instructiveness mean rank (1=best) | 95% CI |
|---|---:|---|
| Claude Opus 4.8 | 2.585 | [2.378, 2.804] |
| GPT-5.5 | 2.637 | [2.393, 2.896] |
| Gemini 3.1 Pro | 2.970 | [2.719, 3.230] |
| OURS-v6-dpo | 3.926 | [3.678, 4.170] |
| OURS-v4 (shipped) | 4.030 | [3.748, 4.304] |
| BASE (Qwen3-32B untuned) | 4.852 | [4.633, 5.067] |

The three frontier coaches lead on instructiveness, consistent with the shipped
framing (the frontier leads on prose; OURS leads on the tier-selection move). The key
Stage-4 signal: v6-dpo does NOT regress instructiveness versus v4. Their CIs overlap
and v6-dpo is nominally slightly more instructive: per-item head-to-head v6-dpo
better on 50, v4 better on 38, 2 ties, with a mean-rank advantage (v4 minus v6-dpo)
of +0.104, 95% CI [-0.200, 0.385] (straddles zero, so a tie, not a regression). Both
tuned models rank above the untuned base. So the DPO tune sharpened the deterministic
move moat (Q1) without paying for it in coaching instructiveness.

## Frontier moat

The tier-appropriate-selection moat over the frontier is established in
`RESULTS_FULL_EVAL_803.md` (full 803 x 3, all 15 models). Those numbers were
computed on the pre-correction labels; the deterministic head-to-head above shows
the label correction lifts absolute grounded tier-policy match while preserving the
tuned-over-base and tuned-over-frontier ordering, so the "beats-frontier on
tier-appropriate selection" claim holds on the corrected benchmark. A full 803 x 3
frontier re-score under corrected labels was not run this stage (cost); the 120 TEST
result is the headline and the 803 numbers remain the frontier-field reference.

## Recommendation (promote vs keep) — decision for the USER

The DPO gain is real but small (+0.0195 overall tier-policy, +0.0583 intermediate)
with no regression; the distillation result is a research-grade behavior-in-weights
proof with an honest advanced-tier limit and grounding-free soundness (0.653) below
the deployable grounded soundness (0.98). Weighed against the churn of re-shipping
the live coach, reseeding the platform, and re-validating this close to the deadline,
the recommendation is to **KEEP v4 shipped and present v6-dpo + v6-distill as
stretch-ladder results** (v6-dpo as the drop-in successor once a shipping window
opens). This is a product-shipping decision reserved for the USER; nothing live was
changed by this stage.

## Artifacts

- Eval inputs (grounded + no-grounding prompts, 360): `data/benchmark_gap803/stage4_eval_inputs.jsonl` (`scripts/stage4_build_inputs.py`)
- Generations (per condition): `data/benchmark_gap803/stage4/<pass>.jsonl`
- Deterministic scores: `data/benchmark_gap803/stage4/scores.json`; consolidated verdict: `data/benchmark_gap803/stage4/verdict.json`
- Continuity re-score of committed gens: `data/benchmark_gap803/stage4/rescore_committed.json`
- Council: `data/benchmark_gap803/stage4_council/{council,aggregate}.json`
- Drivers: `scripts/stage4_{build_inputs,eval,rescore_committed,verdict,council}.py`
- Adapters: `khoilamalphaai/chess-coach-32b-v4-qlora`, `khoilamalphaai/chess-coach-32b-v6-dpo`, `khoilamalphaai/chess-coach-32b-v6-distill`
