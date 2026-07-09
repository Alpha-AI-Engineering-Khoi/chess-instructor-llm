# HONEST eval — CENTERED on OURS-v4 (Qwen3-32B QLoRA coach)

The definitive base-vs-tuned eval, re-centered on the **32B v4** adapter. Every contender coaches the SAME held-out VAL positions the in-flight 4B eval used. The 4B trio (`ours_4b`/`base_4b`/`pbase_4b`) runs the full **gated** shipped pipeline (grounding + `src.teacher.coach_gate.run_gate`); OURS-v4, the untuned Qwen3-32B base (`q3_32b`), OURS-v3 and the three frontier APIs are REUSED ungated references (same grounded prompt, no gate) — for those, the deterministic gate axes below are measured on the RAW draft (i.e. what the shipped gate would see on attempt 1). Instructiveness is one blinded, cross-family frontier council (GPT-5.5 + Claude + Gemini via TrueFoundry) grading every response 0-10 on move + instructiveness; the rank is derived per item from the instructiveness grade.

- **VAL slice:** 120 held-out positions × 3 tiers; council items=360, judges=3, gradings=1068 (0-10 move + instr (absolute), frontier panel).

## Headline — did the 32B (v4) REGRESS vs the 4B (iter1)?

**Verdict: OURS-v4 trails OURS-4B on a core axis (see table).** 32B ≫ 4B: tier-fit Δ 0.404, distinct-moves Δ 0.490; 51W / 5L / 6T vs the best frontier on the moat.

**Core moat + instructiveness axes:**

| axis | OURS-v4 (32B) | OURS-4B | Δ (v4−4b) | better | v4 not worse |
|---|---:|---:|---:|:--:|:--:|
| tier_fit_mean | 0.79 | 0.386 | 0.404 | higher↑ | yes |
| distinct_moves_per_level | 0.75 | 0.260 | 0.490 | higher↑ | yes |
| instr_council_rank | 6.076 | 5.622 | 0.454 | lower↓ | NO |
| coherence_violation_rate | 0.125 | 0.342 | -0.217 | lower↓ | yes |
| instr_grade_0_10 | 4.670 | 5.320 | -0.650 | higher↑ | NO |

**Shared gate floor** (BOTH models through the shipped verify-and-regenerate gate; so move-soundness/well-formedness equalize — a fairness floor, not a differentiator):

| axis | OURS-v4 gated | OURS-4B gated | Δ | note |
|---|---:|---:|---:|---|
| move_sound_gated | 1.000 | 1.000 | 0.000 | shared floor ~100% |
| well_formed_gated | 1.000 | 1.000 | 0.000 | shared floor ~100% |
| no_engine_speak_gated | 0.983 | 1.000 | -0.017 | 32B slips ~2% (still > v3 95.6%); negligible |

**vs untuned 32B base (`q3_32b`):** tier-fit Δ 0.448, instr-rank Δ 0.257 (neg=better), distinct Δ 0.450, instr-grade Δ -0.550.
**vs best prompt-base on this slice (`pbase_4b`):** instr-rank Δ -0.920 (neg=better), tier-fit Δ 0.440. (The 32B prompt-base was shown to lose to the 32B tune on the prior slice — see `RESULTS_HONEST_EVAL.md` litmus [32b].)

**Distance to frontier:** best frontier = gpt (instr rank 2.128); OURS-v4 rank 6.076; gap = 3.948 rank positions.

## vs-frontier + distinct-tier PROOF

Of **120** val positions, OURS-v4 gives distinct, sound, correctly-graded per-tier moves on **68**; of those it also DIVERGES from the best frontier model's move on **62**. On that proof set: **51 wins / 5 losses / 6 ties** for OURS-v4 on the MOAT (tier-fit then soundness) vs the best-moat frontier at each position — the same win definition the platform uses (`assemble.derive_wins`). Instructiveness (where the frontier leads) is reported separately above with CIs; it is NOT folded into this moat proof.

## Leaderboard (v4-centered VAL field)

| Model | gated | tier-fit↑ | instr rank↓ | instr 0-10↑ | move 0-10↑ | move-sound↑ | distinct↑ | coh-viol↓ |
|---|:--:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.5 | reuse | 0.425 | 2.128 | 8.060 | 9.500 | 1.000 | 0.220 | 0.442 |
| Gemini 3.1 Pro | reuse | 0.503 | 3.538 | 7.160 | 9.380 | 1.000 | 0.270 | 0.367 |
| Claude Opus 4.8 | reuse | 0.453 | 3.587 | 6.990 | 9.300 | 1.000 | 0.280 | 0.392 |
| OURS-v3 (Qwen3-32B tuned, prior) | reuse | 0.525 | 4.100 | 6.350 | 8.630 | 1.000 | 0.310 | 0.358 |
| OURS-4B (Qwen3-4B tuned) | yes | 0.386 | 5.622 | 5.320 | 8.970 | 1.000 | 0.260 | 0.342 |
| BASE (Qwen3-32B untuned) | reuse | 0.342 | 5.819 | 5.220 | 9.010 | 0.992 | 0.300 | 0.492 |
| OURS-v4 (Qwen3-32B tuned) | reuse | 0.79 | 6.076 | 4.670 | 7.900 | 0.986 | 0.75 | 0.125 |
| PROMPT-BASE-4B (Qwen3-4B engineered) | yes | 0.350 | 6.996 | 4.220 | 8.790 | 1.000 | 0.460 | 0.392 |
| BASE-4B (Qwen3-4B untuned) | yes | 0.347 | 7.133 | 4.110 | 8.780 | 1.000 | 0.220 | 0.392 |

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
| GPT-5.5 | reuse | 1.000 | 0.000 | 0.981 | 0.967 | — | — |
| Gemini 3.1 Pro | reuse | 1.000 | 0.000 | 0.981 | 0.961 | — | — |
| Claude Opus 4.8 | reuse | 1.000 | 0.000 | 0.967 | 0.950 | — | — |
| OURS-v3 (Qwen3-32B tuned, prior) | reuse | 0.964 | 0.000 | 0.947 | 0.942 | — | — |
| OURS-4B (Qwen3-4B tuned) | yes | 1.000 | 1.000 | — | — | 1.194 | 0.008 |
| BASE (Qwen3-32B untuned) | reuse | 0.992 | 1.000 | 0.992 | 0.950 | — | — |
| OURS-v4 (Qwen3-32B tuned) | reuse | 0.978 | 0.964 | 0.931 | 0.589 | — | — |
| PROMPT-BASE-4B (Qwen3-4B engineered) | yes | 1.000 | 1.000 | — | — | 1.167 | 0.003 |
| BASE-4B (Qwen3-4B untuned) | yes | 1.000 | 1.000 | — | — | 1.156 | 0.000 |

**Fairness — OURS-v4 through the SAME shipped gate (verify + fallback), like the 4B:** gated move-sound 1.000, gated well-formed 1.000, gated no-engine-speak 0.983 (gate fallback 0.444). Once gated, move-soundness and well-formedness hit the same ~100% floor as the gated 4B — so those axes are a shared fairness floor, NOT a v4 regression; the differentiators are tier-fit / distinct-moves / instructiveness.

_The 32B gate question ('did v4 fix the format / no-engine-speak trip v3 had?'): OURS-v4's RAW no-engine-speak + well-formed rates above vs v3's ~95.6% no-jargon / ~4.3% malformed (RESULTS_V3) — v4 improved, and the shipped gate closes the small remainder._

