# HONEST base-vs-tuned eval + the "train by prompting" hard test

First VALIDATION run on a held-out DEV/VAL slice. Every 1.7B/32B contender coaches the SAME positions through the **identical shipped pipeline** — grounding (Stockfish pool + Maia + verified facts) AND the shared faithfulness gate (`src.teacher.coach_gate.run_gate`, the exact code `src/api/server.py` runs) — so base vs tuned differ only in weights, and the prompt-base differs only in its system prompt. Frontier + OURS-v3 rows are REUSED ungated gap803 gens (low-fabrication reference); the core litmus (1.7B) is fully gated on both sides.

- **Validation slice:** 18 positions × 3 tiers; council n_items=54, judges=3, rankings=162.

## Headline

**A. Training as the only variable (1.7B, identical gated pipeline):**
- tier-appropriate move selection: OURS-v2 0.463 vs BASE 0.2963 (**Δ 0.1667**).
- instructiveness council mean rank (lower=better): OURS-v2 5.457 vs BASE 8.451 (**Δ -2.994**).
- instructiveness rubric sum (0-12): OURS-v2 7.038 vs BASE 2.432 (**Δ 4.606**).

**B. Litmus [1p7] — can a well-prompted base match the tune?** **tune still wins**
- instr rank: pbase_1p7 7.58 vs ours_1p7 5.457 (Δ 2.123); tier-fit Δ -0.0741; 6-dim Δ -3.05.

**B. Litmus [32b] — can a well-prompted base match the tune?** **tune still wins** _(caveat: ours_v3 reused UNGATED)_
- instr rank: pbase_32b 5.685 vs ours_v3 2.957 (Δ 2.728); tier-fit Δ -0.0185; 6-dim Δ -3.179.

**C. Distance to frontier:** best frontier = claude (rank 2.821); OURS-v2 rank 5.457, OURS-v3 rank 2.957; gap OURS-v2−bestfrontier = 2.636 rank positions.

**D. Tier-coherence violation rate (deterministic):** OURS-v2 (1.7B tuned, gated) 0.3333, BASE (1.7B untuned, gated) 0.5, PROMPT-BASE (1.7B engineered, gated) 0.6111, PROMPT-BASE-32B (engineered, gated) 0.3333, BASE-32B (Qwen3-32B untuned, gated) 0.3333, OURS-v3 (32B tuned, reused ungated) 0.3889, GPT-5.5 (frontier) 0.3889, Claude Opus 4.8 (frontier) 0.3889, Gemini 3.1 Pro (frontier) 0.4444

## Leaderboard (validation field)

| Model | gated | tier-fit↑ | instr rank↓ | 6-dim/12↑ | move-sound↑ | tier-coh viol↓ |
|---|:--:|---:|---:|---:|---:|---:|
| Claude Opus 4.8 (frontier) | reuse | 0.426 | 2.821 | 10.18 | 1.000 | 0.389 |
| GPT-5.5 (frontier) | reuse | 0.426 | 2.889 | 10.34 | 1.000 | 0.389 |
| OURS-v3 (32B tuned, reused ungated) | reuse | 0.407 | 2.957 | 10.25 | 1.000 | 0.389 |
| Gemini 3.1 Pro (frontier) | reuse | 0.389 | 3.370 | 10.00 | 1.000 | 0.444 |
| OURS-v2 (1.7B tuned, gated) | yes | 0.463 | 5.457 | 7.038 | 1.000 | 0.333 |
| PROMPT-BASE-32B (engineered, gated) | yes | 0.389 | 5.685 | 7.068 | 1.000 | 0.333 |
| BASE-32B (Qwen3-32B untuned, gated) | yes | 0.333 | 5.790 | 7.080 | 1.000 | 0.333 |
| PROMPT-BASE (1.7B engineered, gated) | yes | 0.389 | 7.580 | 3.988 | 1.000 | 0.611 |
| BASE (1.7B untuned, gated) | yes | 0.296 | 8.451 | 2.432 | 1.000 | 0.500 |

## Instructiveness rubric — six dimensions (mean 0/1/2)

| Model | move purpose | transferable principle | board specific reason | how to find | level calibration | grounded concise |
|---|---:|---:|---:|---:|---:|---:|
| Claude Opus 4.8 (frontier) | 1.957 | 1.938 | 1.765 | 1.142 | 1.895 | 1.481 |
| GPT-5.5 (frontier) | 1.920 | 1.889 | 1.741 | 1.062 | 1.938 | 1.790 |
| OURS-v3 (32B tuned, reused ungated) | 1.821 | 1.840 | 1.556 | 1.796 | 1.790 | 1.444 |
| Gemini 3.1 Pro (frontier) | 1.895 | 1.926 | 1.698 | 0.975 | 1.895 | 1.611 |
| OURS-v2 (1.7B tuned, gated) | 1.488 | 1.426 | 0.969 | 1.000 | 1.272 | 0.883 |
| PROMPT-BASE-32B (engineered, gated) | 1.556 | 1.333 | 1.000 | 0.667 | 1.426 | 1.086 |
| BASE-32B (Qwen3-32B untuned, gated) | 1.574 | 1.432 | 1.043 | 0.562 | 1.401 | 1.068 |
| PROMPT-BASE (1.7B engineered, gated) | 1.049 | 0.827 | 0.636 | 0.210 | 0.747 | 0.519 |
| BASE (1.7B untuned, gated) | 0.833 | 0.753 | 0.290 | 0.093 | 0.407 | 0.056 |

## Gate telemetry (gated contenders)

| Model | mean attempts | fallback rate | no-jargon |
|---|---:|---:|---:|
| OURS-v2 (1.7B tuned, gated) | 2.352 | 0.074 | 1.000 |
| PROMPT-BASE-32B (engineered, gated) | 1.167 | 0.000 | 1.000 |
| BASE-32B (Qwen3-32B untuned, gated) | 1.056 | 0.000 | 1.000 |
| PROMPT-BASE (1.7B engineered, gated) | 1.167 | 0.000 | 1.000 |
| BASE (1.7B untuned, gated) | 1.278 | 0.000 | 0.926 |

## Reproduce the FULL eval after v4 lands

```bash
P=~/.venvs/mlx/bin/python
# 1) (once) rebuild the held-out slices if the position set changed:
$P -m scripts.honest_eval seed --dev 8 --val 18
# 2) engineer the best base prompt per size (uses TrueFoundry judge+engineer):
$P -m scripts.honest_eval optimize --size 1p7 --rounds 3
$P -m scripts.honest_eval optimize --size 32b --rounds 2
# 3) gated generation of every contender (1.7B local free; 32B via TFY):
for m in base_1p7 ours_1p7 pbase_1p7 base_32b pbase_32b; do $P -m scripts.honest_eval gen --model $m; done
# 4) reuse existing gap803 frontier + tuned-32B gens for the val positions:
$P -m scripts.honest_eval reuse --models gpt,claude,gemini,ours_v3
# 5) blinded 6-dim cross-family council + report:
$P -m scripts.honest_eval judge --judges gpt,claude,gemini
$P -m scripts.honest_eval report
# For the DEFINITIVE full-scale run: raise --val (e.g. 150) and re-gen ours_1p7
# against the v4 checkpoint (set OURS_1P7 / models/mlx/chess-coach-v4).
```

