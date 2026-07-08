# HONEST 4B base-vs-tuned eval — Qwen3-4B chess coach (iter1)

Every gated 4B contender coaches the SAME held-out positions through the **identical shipped pipeline** (grounding + `src.teacher.coach_gate.run_gate`), so `base_4b` vs `ours_4b` differ ONLY in the LoRA weights and `pbase_4b` differs ONLY in its system prompt. `ours_4b` = `mlx-community/Qwen3-4B-Instruct-2507-4bit` + our iter1 LoRA fused into the identical MLX base. Frontier (GPT-5.5 / Claude Opus 4.8 / Gemini 3.1 Pro via TrueFoundry) + `ours_v3` (our 32B tuned) rows are REUSED ungated references.

- **Validation slice:** 120 positions x 3 tiers; council n_items=360, judges=3, rankings=1080.

## Headline

**A. Training as the only variable (4B, identical gated pipeline):**
- tier-fit (canonical tier move): ours_4b 0.386 vs base_4b 0.347 (**Δ 0.039**).
- instructiveness council mean rank (lower=better): ours_4b 4.701 vs base_4b 5.896 (**Δ -1.195**).
- instructiveness rubric sum (0-12): ours_4b 7.644 vs base_4b 5.692 (**Δ 1.952**).

**B. Litmus — can the best prompt-engineered 4B base match the tune?** **tune still wins**
- instr rank: pbase_4b 5.695 vs ours_4b 4.701 (Δ 0.994); tier-fit Δ -0.036; 6-dim Δ -1.051.

**C. Distance to frontier:** best frontier = gpt (rank 2.529); ours_4b rank 4.701; gap ours_4b−bestfrontier = 2.172 rank positions (ref: ours_v3 32B rank 2.944).

**D. Deterministic gate pass-rates (targets = 100% / 0% fabrication):**
- ours_4b: move-sound 1.000, no-engine-speak 1.000, well-formed 1.000 (gate fallback 0.008, mean attempts 1.194). Post-gate fabrication = 0 by gate design.
- base_4b: move-sound 1.000, no-engine-speak 1.000, well-formed 1.000 (gate fallback 0.000, mean attempts 1.156). Post-gate fabrication = 0 by gate design.
- pbase_4b: move-sound 1.000, no-engine-speak 1.000, well-formed 1.000 (gate fallback 0.003, mean attempts 1.167). Post-gate fabrication = 0 by gate design.

**E. Distinct-moves-per-level on DIFFERENTIATING positions (canonical beginner≠advanced; target ≥95%):**
- ours_4b: 0.260 distinct over 100 differentiating positions (74 B==A collapses); zigzag(B==A≠I) 0.042, flat 0.675.
- base_4b: 0.220 distinct over 100 differentiating positions (78 B==A collapses); zigzag(B==A≠I) 0.042, flat 0.742.
- pbase_4b: 0.460 distinct over 100 differentiating positions (54 B==A collapses); zigzag(B==A≠I) 0.025, flat 0.500.

## Leaderboard (validation field)

| Model | gated | tier-fit↑ | instr rank↓ | 6-dim/12↑ | move-sound↑ | distinct↑ | coh-viol↓ |
|---|:--:|---:|---:|---:|---:|---:|---:|
| GPT-5.5 (frontier) | reuse | 0.425 | 2.529 | 10.39 | 1.000 | 0.220 | 0.442 |
| Claude Opus 4.8 (frontier) | reuse | 0.453 | 2.769 | 9.889 | 1.000 | 0.280 | 0.392 |
| OURS-v3 (32B tuned, reused ungated) | reuse | 0.525 | 2.944 | 9.787 | 1.000 | 0.310 | 0.358 |
| Gemini 3.1 Pro (frontier) | reuse | 0.503 | 3.465 | 9.401 | 1.000 | 0.270 | 0.367 |
| OURS-4B (Qwen3-4B tuned, gated) | yes | 0.386 | 4.701 | 7.644 | 1.000 | 0.260 | 0.342 |
| PROMPT-BASE-4B (Qwen3-4B engineered, gated) | yes | 0.350 | 5.695 | 6.593 | 1.000 | 0.460 | 0.392 |
| BASE-4B (Qwen3-4B untuned, gated) | yes | 0.347 | 5.896 | 5.692 | 1.000 | 0.220 | 0.392 |

## Instructiveness rubric — six dimensions (mean 0/1/2)

| Model | move purpose | transferable principle | board specific reason | how to find | level calibration | grounded concise |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.5 (frontier) | 1.937 | 1.893 | 1.797 | 1.015 | 1.919 | 1.826 |
| Claude Opus 4.8 (frontier) | 1.919 | 1.916 | 1.666 | 1.059 | 1.854 | 1.475 |
| OURS-v3 (32B tuned, reused ungated) | 1.721 | 1.790 | 1.481 | 1.730 | 1.700 | 1.365 |
| Gemini 3.1 Pro (frontier) | 1.831 | 1.863 | 1.508 | 0.834 | 1.784 | 1.581 |
| OURS-4B (Qwen3-4B tuned, gated) | 1.725 | 1.489 | 1.209 | 0.688 | 1.479 | 1.054 |
| PROMPT-BASE-4B (Qwen3-4B engineered, gated) | 1.532 | 1.297 | 0.863 | 1.084 | 1.231 | 0.586 |
| BASE-4B (Qwen3-4B untuned, gated) | 1.460 | 1.176 | 0.777 | 0.461 | 1.209 | 0.609 |

## Gate telemetry (gated contenders)

| Model | mean attempts | fallback rate | no-engine-speak |
|---|---:|---:|---:|
| OURS-4B (Qwen3-4B tuned, gated) | 1.194 | 0.008 | 1.000 |
| PROMPT-BASE-4B (Qwen3-4B engineered, gated) | 1.167 | 0.003 | 1.000 |
| BASE-4B (Qwen3-4B untuned, gated) | 1.156 | 0.000 | 1.000 |

