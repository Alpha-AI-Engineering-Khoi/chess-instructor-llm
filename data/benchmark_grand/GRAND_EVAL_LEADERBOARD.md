# GRAND EVAL — comprehensive chess-coach leaderboard

One fresh, apples-to-apples comparison of **all 20 models** — our tuned specialists, the untuned baselines, and the full frontier lineup — on the SAME held-out VAL slice, scored with BOTH layers:

- **Deterministic moat metrics** (free; python-chess over pre-computed Stockfish/Maia facts) over **all 120 positions × 3 tiers = 360 scenarios**: tier-fit, distinct-moves-per-level, move-soundness, raw faithfulness (verify-pass on draft 1), tier-coherence, shipped-gate soundness.
- **Blinded cross-family frontier council** (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro via TrueFoundry), 0-10 move + instructiveness with 95% CIs, over **75 of the 120 positions** (675 gradings) — sized to the TFY budget.

Every TFY gateway model was regenerated **FRESH** on these exact positions (never reusing the old frontier gens); our Modal/MLX tuned models are deterministic given their adapter (reused where noted). ours_v5 is the finish-v5 controller's fresh Modal Volume gen.

**New TFY spend:** gen $21.61 + council $32.35 = **$53.96** (under the $60 cap). The council on all 120 positions would cost ~$51.77 (total ~$73.37) at the measured $0.144/scenario — hence the 75-position council + full-field deterministic layer. Modal spend from this run ≈ $0 (v5 reused from the controller's Volume gen; v3/v4/4B reused).

**Frontier reachability:** the 14-model lineup = 3 frontier APIs + 11 open candidates; **12 reachable** (dsr1 via `bedrock-oss-group/deepseek-r1`), **2 blocked**: `llama4-maverick` (400, Meta Llama access denied) and `kimi-k2-thinking` (403, not authorized).

## Metric framing (2026-07-09 honest reframe)

Numbers below are the as-computed grand-eval values (canonical/frozen); the framing terms are aligned to the honest reframe:

- **"tier-fit" = "tier-policy exact match"** — exact agreement with the preregistered `select_tier_move` rule, a PROJECT RULE, not validated pedagogy. Lead with the all-scenario number (v4 0.767 vs best frontier 0.553), not a head-to-head win rate.
- **The "moat" head-to-head is a project-rule metric, not a general win rate.** The per-tuned W/L/T table below is SELECTION-CONDITIONED (only the positions where OURS already gives a distinct, sound, correctly-graded move AND diverges from the frontier). For v4, the honest **unbiased** head-to-head over all 96 diverging positions is **56-28-12** (56-28-36 over all 120); the **51-5-6 over 62** shown below is the conditioned subset.
- **distinct-moves denominator:** the `distinct↑` column is distinct / (positions where the model named both tier moves) as computed at grand-eval time (v4 = 73/93 = 0.785). The honest all-opportunities denominator (every canonical beginner!=advanced position, a no-answer counting as a miss) gives v4 **73/100 = 0.730**; see `data/benchmark_honest/report_v4.json`.

## Leaderboard — ranked by tier-appropriate move selection (the trained behavior)

**Sort key:** ranked by **tier-appropriate move selection** — the deterministic **tier-fit↑** metric (the behavior we trained and the graded axis), ties broken by **distinct-moves-per-level↑** then **move-soundness↑**. Every deterministic axis (tier-fit / distinct / move-sound / coherence and the moat) uses the canonical **STRICT any-legal** move extractor (`coach_gate.pick_recommendation`, accept = any legal move; **no in-pool backfill**) — so an output that names no clearly-legal move is a miss everywhere and the leaderboard method matches the moat method exactly. The per-tuned head-to-head **W/L/T vs the best frontier** is in the moat table below. Instructiveness (the blinded cross-family council) is shown as a **secondary** axis in the `instr 0-10` / `move 0-10` / `rank↓` columns — **OURS-v4 is intentionally weaker on council prose and that is reported here honestly and unchanged.** Only the row order reflects tier-fit; every model's measured numbers are the deterministic + council values.

| # | Model | family | gen | gated | tier-fit↑ | distinct↑ | move-sound↑ | raw-faith↑ | coh-viol↓ | instr 0-10↑ [95% CI] | move 0-10↑ | rank↓ | top1% |
|--:|---|:--:|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | OURS-v4 (Qwen3-32B tuned) | ours | reuse | raw | 0.767 | 0.785 | 0.942 | 0.589 | 0.140 | 4.528 [4.168–4.875] | 7.660 | 12.66 | 9.800 |
| 2 | OURS-v2 (Qwen3-1.7B tuned) | ours | reuse | raw | 0.578 | 0.380 | 1.000 | 0.689 | 0.167 | 4.323 [3.983–4.662] | 8.050 | 13.58 | 8.000 |
| 3 | OURS-v3 (Qwen3-32B tuned) | ours | reuse | raw | 0.558 | 0.585 | 0.950 | 0.942 | 0.229 | 6.428 [6.131–6.738] | 8.540 | 7.764 | 28.90 |
| 4 | Gemini 3.1 Pro | frontier | FRESH | raw | 0.553 | 0.210 | 1.000 | 0.958 | 0.292 | 6.902 [6.721–7.080] | 9.110 | 6.838 | 8.400 |
| 5 | OURS-v5 (Qwen3-32B tuned, v5) | ours | FRESH | raw | 0.536 | 0.726 | 0.828 | 0.575 | 0.260 | 3.863 [3.530–4.188] | 7.250 | 14.37 | 3.100 |
| 6 | Claude Opus 4.8 | frontier | FRESH | raw | 0.508 | 0.200 | 1.000 | 0.944 | 0.308 | 7.062 [6.876–7.249] | 9.160 | 6.011 | 17.80 |
| 7 | GPT-5.5 | frontier | FRESH | raw | 0.494 | 0.280 | 1.000 | 0.986 | 0.342 | 7.984 [7.869–8.098] | 9.380 | 3.024 | 40.40 |
| 8 | DeepSeek-R1 (reasoning) | open | FRESH | raw | 0.436 | 0.370 | 1.000 | 0.978 | 0.300 | 6.117 [5.906–6.319] | 9.060 | 9.524 | 1.300 |
| 9 | GLM-5 | open | FRESH | raw | 0.408 | 0.270 | 1.000 | 0.906 | 0.350 | 6.875 [6.685–7.068] | 9.150 | 6.780 | 9.300 |
| 10 | DeepSeek-V3.2 | open | FRESH | raw | 0.400 | 0.280 | 0.997 | 0.950 | 0.392 | 5.859 [5.612–6.107] | 8.900 | 10.09 | 0.900 |
| 11 | OURS-4B (Qwen3-4B tuned) | ours | reuse | yes | 0.397 | 0.280 | 1.000 | — | 0.325 | 5.828 [5.608–6.040] | 8.930 | 10.37 | 3.600 |
| 12 | PROMPT-BASE-4B (Qwen3-4B engineered) | base | reuse | yes | 0.378 | 0.460 | 1.000 | — | 0.333 | 4.799 [4.592–5.011] | 8.810 | 13.52 | 0.400 |
| 13 | BASE (Qwen3-1.7B untuned) | base | reuse | raw | 0.358 | 0.280 | 0.992 | 0.858 | 0.333 | 1.939 [1.779–2.107] | 6.950 | 18.71 | 0.000 |
| 14 | BASE (Qwen3-32B untuned) | base | FRESH | raw | 0.358 | 0.250 | 1.000 | 0.939 | 0.442 | 5.493 [5.286–5.689] | 8.950 | 11.63 | 0.400 |
| 15 | Llama-3.3-70B | open | FRESH | raw | 0.356 | 0.160 | 1.000 | 0.997 | 0.417 | 6.262 [6.095–6.422] | 9.160 | 9.327 | 1.300 |
| 16 | BASE-4B (Qwen3-4B untuned) | base | reuse | yes | 0.353 | 0.230 | 1.000 | — | 0.375 | 4.723 [4.526–4.927] | 8.750 | 13.94 | 0.000 |
| 17 | Mistral-Large-3 (675B) | open | FRESH | raw | 0.336 | 0.370 | 0.997 | 0.919 | 0.403 | 5.217 [4.982–5.436] | 8.770 | 12.14 | 0.900 |
| 18 | Kimi-K2.5 | open | FRESH | raw | 0.333 | 0.410 | 1.000 | 0.875 | 0.500 | 6.266 [6.058–6.470] | 9.070 | 8.758 | 6.700 |
| 19 | Gemma-3-27B-it | open | FRESH | raw | 0.283 | 0.190 | 1.000 | 0.969 | 0.417 | 5.778 [5.553–5.981] | 9.010 | 10.62 | 0.400 |
| 20 | Qwen3-Next-80B-A3B | open | FRESH | raw | 0.281 | 0.240 | 1.000 | 0.953 | 0.333 | 5.875 [5.654–6.092] | 8.950 | 10.35 | 2.700 |

_gen: FRESH = regenerated this run; reuse = deterministic adapter/MLX gen reused. gated: `yes` = full shipped verify-and-regenerate pipeline (4B trio); `raw` = ungated draft (raw-draft gate axes shown). raw-faith = verify-pass on draft 1 (1 − fabrication). tier-fit / distinct / move-sound / raw-faith / coherence are deterministic (free); instr / move 0-10 + rank are the blinded council._

## The moat — each tuned model vs the best frontier (tier-fit then soundness)

On positions where OURS gives distinct, sound, correctly-graded per-tier moves AND diverges from the best-frontier move, who wins the platform's move-quality moat (the `assemble.derive_wins` definition)? Instructiveness (where the frontier leads) is reported separately above.

| Tuned model | distinct | distinct & diverge | **W** | **L** | **T** |
|---|---:|---:|---:|---:|---:|
| OURS-v2 (Qwen3-1.7B tuned) | 47 | 43 | 21 | 17 | 5 |
| OURS-4B (Qwen3-4B tuned) | 26 | 23 | 5 | 14 | 4 |
| OURS-v3 (Qwen3-32B tuned) | 47 | 44 | 22 | 13 | 9 |
| OURS-v4 (Qwen3-32B tuned) | 67 | 62 | 51 | 5 | 6 |
| OURS-v5 (Qwen3-32B tuned, v5) | 43 | 40 | 26 | 6 | 8 |

## Shipped-gate soundness (tuned models through the SAME verify+fallback gate)

| Tuned model | gated move-sound↑ | gated well-formed↑ | gated no-engine-speak↑ | gate fallback↓ |
|---|---:|---:|---:|---:|
| OURS-v2 (Qwen3-1.7B tuned) | 1.000 | 1.000 | 1.000 | 0.358 |
| OURS-4B (Qwen3-4B tuned) | 1.000 | 1.000 | 1.000 | 0.000 |
| OURS-v3 (Qwen3-32B tuned) | 1.000 | 1.000 | 0.969 | 0.181 |
| OURS-v4 (Qwen3-32B tuned) | 1.000 | 1.000 | 0.983 | 0.444 |
| OURS-v5 (Qwen3-32B tuned, v5) | 1.000 | 1.000 | 0.992 | 0.444 |

_Once gated, tuned soundness/format hit a shared ~100% floor (0 user-visible fabrication by construction) — a fairness floor, not a differentiator; the differentiators are tier-fit / distinct-moves / instructiveness._

## Deterministic gate axes (raw draft for ungated rows; telemetry for gated 4B)

| Model | gated | no-engine-speak↑ | well-formed↑ | move-sound↑ | verify-pass draft1↑ | mean attempts | fallback↓ |
|---|:--:|---:|---:|---:|---:|---:|---:|
| OURS-v4 (Qwen3-32B tuned) | raw | 0.978 | 0.956 | 0.942 | 0.589 | — | — |
| OURS-v2 (Qwen3-1.7B tuned) | raw | 1.000 | 1.000 | 1.000 | 0.689 | — | — |
| OURS-v3 (Qwen3-32B tuned) | raw | 0.964 | 0.958 | 0.950 | 0.942 | — | — |
| Gemini 3.1 Pro | raw | 0.997 | 1.000 | 1.000 | 0.958 | — | — |
| OURS-v5 (Qwen3-32B tuned, v5) | raw | 0.978 | 0.861 | 0.828 | 0.575 | — | — |
| Claude Opus 4.8 | raw | 1.000 | 1.000 | 1.000 | 0.944 | — | — |
| GPT-5.5 | raw | 1.000 | 1.000 | 1.000 | 0.986 | — | — |
| DeepSeek-R1 (reasoning) | raw | 1.000 | 1.000 | 1.000 | 0.978 | — | — |
| GLM-5 | raw | 0.997 | 1.000 | 1.000 | 0.906 | — | — |
| DeepSeek-V3.2 | raw | 1.000 | 1.000 | 0.997 | 0.950 | — | — |
| OURS-4B (Qwen3-4B tuned) | yes | 1.000 | 1.000 | — | — | 1.194 | 0.008 |
| PROMPT-BASE-4B (Qwen3-4B engineered) | yes | 1.000 | 1.000 | — | — | 1.167 | 0.003 |
| BASE (Qwen3-1.7B untuned) | raw | 0.964 | 1.000 | 0.992 | 0.858 | — | — |
| BASE (Qwen3-32B untuned) | raw | 0.992 | 1.000 | 1.000 | 0.939 | — | — |
| Llama-3.3-70B | raw | 1.000 | 1.000 | 1.000 | 0.997 | — | — |
| BASE-4B (Qwen3-4B untuned) | yes | 1.000 | 1.000 | — | — | 1.156 | 0.000 |
| Mistral-Large-3 (675B) | raw | 1.000 | 0.997 | 0.997 | 0.919 | — | — |
| Kimi-K2.5 | raw | 1.000 | 1.000 | 1.000 | 0.875 | — | — |
| Gemma-3-27B-it | raw | 1.000 | 1.000 | 1.000 | 0.969 | — | — |
| Qwen3-Next-80B-A3B | raw | 1.000 | 1.000 | 1.000 | 0.953 | — | — |

## How each row was generated

| Model | fresh/reused | method |
|---|:--:|---|
| OURS-v5 (Qwen3-32B tuned, v5) | FRESH | Modal-adapter FRESH (finish-v5 controller Volume gen) |
| OURS-v4 (Qwen3-32B tuned) | reused | Modal-adapter reuse (honest val, deterministic) |
| OURS-v3 (Qwen3-32B tuned) | reused | Modal-adapter reuse (gap803, deterministic) |
| OURS-v2 (Qwen3-1.7B tuned) | reused | MLX-local reuse (gap803, greedy deterministic) |
| OURS-4B (Qwen3-4B tuned) | reused | Modal reuse (honest val, gated pipeline) |
| BASE (Qwen3-32B untuned) | FRESH | TFY FRESH (aws-bedrock qwen3-32b) |
| BASE (Qwen3-1.7B untuned) | reused | MLX-local reuse (gap803, greedy deterministic) |
| BASE-4B (Qwen3-4B untuned) | reused | Modal reuse (honest val, gated pipeline) |
| PROMPT-BASE-4B (Qwen3-4B engineered) | reused | Modal reuse (honest val, gated pipeline) |
| GPT-5.5 | FRESH | TFY FRESH |
| Claude Opus 4.8 | FRESH | TFY FRESH |
| Gemini 3.1 Pro | FRESH | TFY FRESH |
| Qwen3-Next-80B-A3B | FRESH | TFY FRESH |
| Gemma-3-27B-it | FRESH | TFY FRESH |
| Llama-3.3-70B | FRESH | TFY FRESH |
| DeepSeek-V3.2 | FRESH | TFY FRESH |
| GLM-5 | FRESH | TFY FRESH |
| Mistral-Large-3 (675B) | FRESH | TFY FRESH |
| Kimi-K2.5 | FRESH | TFY FRESH |
| DeepSeek-R1 (reasoning) | FRESH | TFY FRESH |

