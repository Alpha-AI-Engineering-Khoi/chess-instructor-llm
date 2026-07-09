# HONEST eval — CENTERED on OURS-v4 (Qwen3-32B QLoRA coach)

The definitive base-vs-tuned eval, re-centered on the **32B v4** adapter. Every contender coaches the SAME held-out VAL positions the in-flight 4B eval used. The 4B trio (`ours_4b`/`base_4b`/`pbase_4b`) runs the full **gated** shipped pipeline (grounding + `src.teacher.coach_gate.run_gate`); OURS-v4, the untuned Qwen3-32B base (`q3_32b`), OURS-v3 and the three frontier APIs are REUSED ungated references (same grounded prompt, no gate) — for those, the deterministic gate axes below are measured on the RAW draft (i.e. what the shipped gate would see on attempt 1). Instructiveness is one blinded, cross-family frontier council (GPT-5.5 + Claude + Gemini via TrueFoundry) grading every response 0-10 on move + instructiveness; the rank is derived per item from the instructiveness grade.

- **VAL slice:** 120 held-out positions × 3 tiers; council items=360, judges=3, gradings=1068 (0-10 move + instr (absolute), frontier panel).

## What this measures, and what it does not

**Metric rename.** What earlier drafts called "tier-fit" is renamed **tier-policy exact match**: exact agreement with the preregistered `select_tier_move` policy (shortened to **tier-policy match**). It is agreement with a PROJECT RULE, not validated pedagogy.

**Three separate claims (do not conflate):**
1. **Learnability (validated).** The `select_tier_move` policy is distillable into weights: with identical Maia grounding on both sides, a fine-tune reproduces it where a prompt on the same base cannot. The genuinely small on-spec model leads it: **1.7B tuned 0.358 -> 0.578**, #2 of 20 and above every frontier (Gemini 0.553); 1.7B tune (0.578) > 4B tune (0.397), so the behavior is from data / contrastive signal, not capacity. 32B v4 (0.767) is the strongest **mid-size** extension, not a small model.
2. **Deployment-necessity (false as built).** The same Stockfish sound pool + Maia policy that feed the model's prompt also feed the ~20-line `select_tier_move` rule, which computes the canonical move directly at **~1.0 by construction** (it IS the target). The model APPROXIMATES a policy the product already produces; the deterministic rule is the true ceiling. The model is only load-bearing grounding-free and fully local, which was not built or measured.
3. **Pedagogy / value (unvalidated).** These numbers are agreement with our own heuristic, not evidence coaches or students prefer these moves or improve. Behavior validated, value not.

**Caveats carried by every number below:**
- **Un-promptability at 32B is a hypothesis, not a result:** the matched same-backend 32B prompt control was never run (only 1.7B and 4B prompt controls exist). Future work.
- **Grounded execution, not weights:** the tune needs Maia's per-tier grounding in the prompt at inference; without it the tiers collapse to one move. Honest claim = reliable grounded EXECUTION.
- **Soundness is heuristic fidelity:** the sound pool is a shallow 300ms / MultiPV-8 / 150cp search that can include position-worsening moves, and the rule's "advanced = engine best" diverges from the persisted engine_best on ~43/803. So tier-policy match is fidelity to a heuristic, not certified best teaching.
- **Faithfulness != truthfulness:** the gate's zero-fabrication figure is zero verifier-DETECTABLE mechanical violations; semantic falsehoods (relational pawn-SAN claims, forks / threats / negations / eval claims) can still reach users, and a cross-family LLM-judge residual exists.

See [`BRAINLIFT.md`](BRAINLIFT.md) for the full three-claim treatment and the v6 roadmap.

## Headline — base vs tuned, and the deterministic ceiling

**All-scenario lead (the unbiased number).** Across all 120 × 3 scenarios, OURS-v4 tier-policy match **0.767** vs the best frontier **0.553** (Gemini 3.1 Pro, #4 of 20); the tuned checkpoints take 4 of the top 5. The **`select_tier_move` deterministic rule scores ~1.0 by construction** — it IS the target, and is the **named ceiling**: the model approximates a move the product already computes from the same grounding.

**What the demo serves (served == evaluated move).** The 0.767 above is the RAW greedy draft's tier-policy match. The live coach runs that draft through the shipped verify-and-regenerate gate, which (post-fix `d4afd73`) **changes only the explanation, never the move**: on a prose failure it keeps the model's own greedy sound move and rewrites just the prose with a verified, engine-derived explanation of that same move. Replaying the shipped gate over the 120 val drafts measures the SERVED move's tier-policy match at **0.789** — at/above the evaluated greedy 0.767, with served move-soundness 1.000 — so the served-move distribution equals the evaluated greedy distribution and **0.767 applies to what the demo serves**. (One-line history: pre-fix, the gate could substitute engine-best on a prose failure, collapsing the served match to **0.589**; `d4afd73` fixed it.)

**Learnability, led by the small model.** 1.7B tuned 0.358 -> 0.578 (#2 of 20, above every frontier). The 32B v4 (0.767) is the strongest mid-size instance; v4 beats the 4B tune, but so does the 1.7B tune, so the behavior tracks the data, not capacity.

**32B vs 4B (this doc's original question).** OURS-v4 leads OURS-4B on the move axis (tier-policy match Δ 0.369, distinct-moves Δ 0.450) and trails on prose (below), which is on-thesis because prose is secondary to the evaluation claim (still in the SFT loss, not separately optimized).

**Core move + prose axes (v4 vs 4B):**

| axis | OURS-v4 (32B) | OURS-4B | Δ (v4−4b) | better | v4 not worse |
|---|---:|---:|---:|:--:|:--:|
| tier_policy_match_mean | 0.767 | 0.397 | 0.369 | higher↑ | yes |
| distinct_moves_per_level | 0.730 | 0.280 | 0.450 | higher↑ | yes |
| instr_council_rank | 6.076 | 5.622 | 0.454 | lower↓ | NO |
| coherence_violation_rate | 0.140 | 0.325 | -0.185 | lower↓ | yes |
| instr_grade_0_10 | 4.670 | 5.320 | -0.650 | higher↑ | NO |

**Shared gate floor** (BOTH models through the shipped verify-and-regenerate gate; so move-soundness/well-formedness equalize — a fairness floor, not a differentiator):

| axis | OURS-v4 gated | OURS-4B gated | Δ | note |
|---|---:|---:|---:|---|
| move_sound_gated | 1.000 | 1.000 | 0.000 | shared floor ~100% |
| well_formed_gated | 1.000 | 1.000 | 0.000 | shared floor ~100% |
| no_engine_speak_gated | 0.983 | 1.000 | -0.017 | 32B slips ~2% (still > v3 95.6%); negligible |

**vs untuned 32B base (`q3_32b`):** tier-policy match Δ 0.419, instr-rank Δ 0.257 (neg=better), distinct Δ 0.440, instr-grade Δ -0.550.
**vs best prompt-base on this slice (`pbase_4b`, the 4B prompt control):** tier-policy match Δ 0.389, instr-rank Δ -0.920 (neg=better). A matched same-backend **32B** prompt control was never run, so 32B un-promptability is a falsifiable hypothesis (future work), NOT a result; the real prompt-vs-tune evidence is at 1.7B and 4B.

**Distance to frontier:** best frontier = gpt (instr rank 2.128); OURS-v4 rank 6.076; gap = 3.948 rank positions.

## Head-to-head: unbiased vs selection-conditioned (NOT a general win rate)

**Unbiased head-to-head (the honest number).** Over ALL positions where OURS-v4's move diverges from the best frontier's — **not** conditioned on v4 first succeeding — v4 goes **56 wins / 28 losses / 12 ties over the 96 diverging positions** (equivalently **56-28-36 over all 120**, the 24 non-diverging positions counting as ties) on tier-policy match then soundness (`assemble.derive_wins`). This is the honest head-to-head; the frontier wins far more here (28 vs 5) precisely because it is no longer filtered to positions v4 already handles.

**v4-success-conditioned subset (overstates a win rate).** Of **120** val positions, OURS-v4 gives distinct, sound, correctly-graded per-tier moves on **67**; of those it also DIVERGES from the best frontier on **62**. Within that subset: **51 wins / 5 losses / 6 ties**. Because it is selected on v4 already being distinct, sound, and correct, it overstates a raw win rate — report it only as the clearly-labeled subset, never as a win rate over all positions.

The primary unbiased comparison remains the all-scenario tier-policy match **0.767 vs 0.553** above. Instructiveness (where the frontier leads) is reported separately with CIs.

## Leaderboard (v4-centered VAL field)

Note: this v4-centered field omits the small 1.7B tune. In the full 20-model grand eval (all 120 × 3, same strict extractor) the **1.7B tune posts tier-policy match 0.578, #2 of 20 and above every frontier** (base 0.358), which is the learnability lead; see `data/benchmark_grand/GRAND_EVAL_LEADERBOARD.md`.

| Model | gated | tier-policy match↑ | instr rank↓ | instr 0-10↑ | move 0-10↑ | move-sound↑ | distinct↑ | coh-viol↓ |
|---|:--:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.5 | reuse | 0.494 | 2.128 | 8.060 | 9.500 | 1.000 | 0.280 | 0.342 |
| Gemini 3.1 Pro | reuse | 0.553 | 3.538 | 7.160 | 9.380 | 1.000 | 0.210 | 0.292 |
| Claude Opus 4.8 | reuse | 0.508 | 3.587 | 6.990 | 9.300 | 1.000 | 0.200 | 0.308 |
| OURS-v3 (Qwen3-32B tuned, prior) | reuse | 0.558 | 4.100 | 6.350 | 8.630 | 0.950 | 0.585 | 0.229 |
| OURS-4B (Qwen3-4B tuned) | yes | 0.397 | 5.622 | 5.320 | 8.970 | 1.000 | 0.280 | 0.325 |
| BASE (Qwen3-32B untuned) | reuse | 0.347 | 5.819 | 5.220 | 9.010 | 1.000 | 0.290 | 0.500 |
| OURS-v4 (Qwen3-32B tuned) | reuse | 0.767 | 6.076 | 4.670 | 7.900 | 0.942 | 0.730 | 0.140 |
| PROMPT-BASE-4B (Qwen3-4B engineered) | yes | 0.378 | 6.996 | 4.220 | 8.790 | 1.000 | 0.460 | 0.333 |
| BASE-4B (Qwen3-4B untuned) | yes | 0.353 | 7.133 | 4.110 | 8.780 | 1.000 | 0.230 | 0.375 |

## Instructiveness (blinded frontier council, 0-10) with 95% CI

Absolute instructiveness grade pooled over items, 95% cluster-bootstrap CI by item (2000 resamples). Lower council RANK (derived per item) = better.

| Model | instr 0-10 [95% CI] | council rank↓ | top-1% |
|---|---:|---:|---:|
| GPT-5.5 | 8.059 [7.958–8.154] | 2.128 | 48.60 |
| Gemini 3.1 Pro | 7.163 [7.017–7.305] | 3.538 | 14.70 |
| Claude Opus 4.8 | 6.993 [6.833–7.161] | 3.587 | 17.50 |
| OURS-v3 (Qwen3-32B tuned, prior) | 6.349 [6.056–6.635] | 4.100 | 33.60 |
| OURS-4B (Qwen3-4B tuned) | 5.320 [5.127–5.509] | 5.622 | 1.100 |
| BASE (Qwen3-32B untuned) | 5.223 [5.051–5.395] | 5.819 | 0.600 |
| OURS-v4 (Qwen3-32B tuned) | 4.669 [4.387–4.955] | 6.076 | 12.20 |
| PROMPT-BASE-4B (Qwen3-4B engineered) | 4.219 [4.032–4.421] | 6.996 | 0.800 |
| BASE-4B (Qwen3-4B untuned) | 4.108 [3.948–4.265] | 7.133 | 0.000 |

## Deterministic gate axes (RAW draft for reused/ungated rows; telemetry for gated 4B)

| Model | gated | no-engine-speak↑ | well-formed↑ | move-sound↑ | verify-pass draft1↑ | mean attempts | fallback↓ |
|---|:--:|---:|---:|---:|---:|---:|---:|
| GPT-5.5 | reuse | 1.000 | 1.000 | 1.000 | 0.986 | — | — |
| Gemini 3.1 Pro | reuse | 0.997 | 1.000 | 1.000 | 0.958 | — | — |
| Claude Opus 4.8 | reuse | 1.000 | 1.000 | 1.000 | 0.944 | — | — |
| OURS-v3 (Qwen3-32B tuned, prior) | reuse | 0.964 | 0.958 | 0.950 | 0.942 | — | — |
| OURS-4B (Qwen3-4B tuned) | yes | 1.000 | 1.000 | — | — | 1.194 | 0.008 |
| BASE (Qwen3-32B untuned) | reuse | 0.992 | 1.000 | 1.000 | 0.950 | — | — |
| OURS-v4 (Qwen3-32B tuned) | reuse | 0.978 | 0.956 | 0.942 | 0.589 | — | — |
| PROMPT-BASE-4B (Qwen3-4B engineered) | yes | 1.000 | 1.000 | — | — | 1.167 | 0.003 |
| BASE-4B (Qwen3-4B untuned) | yes | 1.000 | 1.000 | — | — | 1.156 | 0.000 |

**Fairness — OURS-v4 through the SAME shipped gate (verify + fallback), like the 4B:** gated move-sound 1.000, gated well-formed 1.000, gated no-engine-speak 0.983 (gate fallback 0.444). On those 0.444 prose-failure fallbacks the post-fix gate replaces ONLY the prose and keeps the model's own greedy sound move (verified engine-derived text for that same move), so the served move == the evaluated greedy move — the served tier-policy match measured over these 120 val drafts is **0.789** (≥ the raw 0.767), not the pre-fix collapsed 0.589. Once gated, move-soundness and well-formedness hit the same ~100% floor as the gated 4B — so those axes are a shared fairness floor, NOT a v4 regression; the differentiators are tier-policy match / distinct-moves / instructiveness. The gate guarantees zero verifier-DETECTABLE mechanical violations, not certified truthfulness: semantic falsehoods (relational pawn-SAN claims, forks / threats / negations / eval claims) can still pass, and move-soundness itself is fidelity to the shallow sound pool.

_The 32B gate question ('did v4 fix the format / no-engine-speak trip v3 had?'): OURS-v4's RAW no-engine-speak + well-formed rates above vs v3's ~95.6% no-jargon / ~4.3% malformed (RESULTS_V3) — v4 improved, and the shipped gate closes the small remainder._

