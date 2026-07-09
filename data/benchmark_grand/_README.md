---
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- en
tags:
- chess
- coaching
- evaluation
- leaderboard
- llm-as-judge
pretty_name: Chess Coach Grand Eval
configs:
- config_name: council
  data_files: council.jsonl
---

# Chess Coach — Grand Eval (comprehensive leaderboard)

One fresh, **apples-to-apples** comparison of **every** model in the chess move-review
coaching project — our tuned specialists, the untuned baselines, and the full frontier
lineup — on the **same held-out validation slice** (120 positions × 3 tiers
= 360 scenarios), scored with **two** independent layers:

1. **Deterministic moat metrics** (free, `python-chess` over pre-computed Stockfish/Maia
   facts): tier-fit, distinct-moves-per-level, move-soundness, raw faithfulness
   (verify-pass on draft 1), tier-coherence, and shipped-gate soundness.
2. **Blinded cross-family frontier council** (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro via
   the TrueFoundry gateway), grading each anonymised response 0–10 on **move** and
   **instructiveness**, with 95 % cluster-bootstrap CIs. Council: 225 items ×
   3 judges = 675 gradings.

Every gateway (TFY) model was **regenerated fresh** on these exact positions; our Modal/MLX
tuned models are deterministic given their adapter (reused where noted — see the
"How each row was generated" table below).

## Files

| File | What |
|---|---|
| `GRAND_EVAL_LEADERBOARD.md` | the human-readable leaderboard (rendered below) |
| `report.json` | every metric per model + per-tuned-model moat proof |
| `council.jsonl` | raw blinded council gradings (0–10 move + instr, per judge, with token usage) |
| `gen/<model>.jsonl` | each model's coaching generations on the val slice |
| `val_scenarios.jsonl` | the held-out positions (engine-grounded, with sound pools) |

---


One fresh, apples-to-apples comparison of **all 20 models** — our tuned specialists, the untuned baselines, and the full frontier lineup — on the SAME held-out VAL slice, scored with BOTH layers:

- **Deterministic moat metrics** (free; python-chess over pre-computed Stockfish/Maia facts) over **all 120 positions × 3 tiers = 360 scenarios**: tier-fit, distinct-moves-per-level, move-soundness, raw faithfulness (verify-pass on draft 1), tier-coherence, shipped-gate soundness.
- **Blinded cross-family frontier council** (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro via TrueFoundry), 0-10 move + instructiveness with 95% CIs, over **75 of the 120 positions** (675 gradings) — sized to the TFY budget.

Every TFY gateway model was regenerated **FRESH** on these exact positions (never reusing the old frontier gens); our Modal/MLX tuned models are deterministic given their adapter (reused where noted). ours_v5 is the finish-v5 controller's fresh Modal Volume gen.

**New TFY spend:** gen $21.61 + council $32.35 = **$53.96** (under the $60 cap). The council on all 120 positions would cost ~$51.77 (total ~$73.37) at the measured $0.144/scenario — hence the 75-position council + full-field deterministic layer. Modal spend from this run ≈ $0 (v5 reused from the controller's Volume gen; v3/v4/4B reused).

**Frontier reachability:** the 14-model lineup = 3 frontier APIs + 11 open candidates; **12 reachable** (dsr1 via `bedrock-oss-group/deepseek-r1`), **2 blocked**: `llama4-maverick` (400, Meta Llama access denied) and `kimi-k2-thinking` (403, not authorized).

## Leaderboard — ranked by tier-appropriate move selection (the trained behavior)

**Sort key:** ranked by **tier-appropriate move selection** — the deterministic **tier-fit↑** metric (the behavior we trained and the graded axis), with ties broken by **distinct-moves-per-level↑** then **move-soundness↑**. The per-tuned head-to-head **W/L/T vs the best frontier** on diverging positions is in the moat table below. Instructiveness (the blinded cross-family council) is shown as a **secondary** axis in the `instr 0-10` / `move 0-10` / `rank↓` columns — **OURS-v4 is intentionally weaker on council prose (instr rank ≈13 of 20), and that is reported here honestly and unchanged.** Only the row order changed; every model's measured numbers are identical to the deterministic + council layers. (The previous ordering was by council instructiveness `rank↓`, still shown as a column.)

| # | Model | family | gen | gated | tier-fit↑ | distinct↑ | move-sound↑ | raw-faith↑ | coh-viol↓ | instr 0-10↑ [95% CI] | move 0-10↑ | rank↓ | top1% |
|--:|---|:--:|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **OURS — shipped (v4)** (Qwen3-32B tuned) | ours | reuse | raw | 0.79 | 0.75 | 0.986 | 0.589 | 0.142 | 4.528 [4.168–4.875] | 7.660 | 12.66 | 9.800 |
| 2 | OURS-v2 (Qwen3-1.7B tuned) | ours | reuse | raw | 0.569 | 0.450 | 0.986 | 0.689 | 0.167 | 4.323 [3.983–4.662] | 8.050 | 13.58 | 8.000 |
| 3 | OURS-v3 (Qwen3-32B tuned) | ours | reuse | raw | 0.567 | 0.570 | 0.978 | 0.942 | 0.275 | 6.428 [6.131–6.738] | 8.540 | 7.764 | 28.90 |
| 4 | OURS-v5 (Qwen3-32B tuned, v5) | ours | FRESH | raw | 0.544 | 0.650 | 0.933 | 0.575 | 0.300 | 3.863 [3.530–4.188] | 7.250 | 14.37 | 3.100 |
| 5 | Gemini 3.1 Pro | frontier | FRESH | raw | 0.531 | 0.230 | 0.983 | 0.958 | 0.308 | 6.902 [6.721–7.080] | 9.110 | 6.838 | 8.400 |
| 6 | Claude Opus 4.8 | frontier | FRESH | raw | 0.450 | 0.250 | 0.967 | 0.944 | 0.333 | 7.062 [6.876–7.249] | 9.160 | 6.011 | 17.80 |
| 7 | GPT-5.5 | frontier | FRESH | raw | 0.447 | 0.300 | 0.969 | 0.986 | 0.342 | 7.984 [7.869–8.098] | 9.380 | 3.024 | 40.40 |
| 8 | DeepSeek-R1 (reasoning) | open | FRESH | raw | 0.431 | 0.370 | 0.997 | 0.978 | 0.308 | 6.117 [5.906–6.319] | 9.060 | 9.524 | 1.300 |
| 9 | DeepSeek-V3.2 | open | FRESH | raw | 0.394 | 0.280 | 0.994 | 0.950 | 0.392 | 5.859 [5.612–6.107] | 8.900 | 10.09 | 0.900 |
| 10 | GLM-5 | open | FRESH | raw | 0.394 | 0.280 | 0.992 | 0.906 | 0.350 | 6.875 [6.685–7.068] | 9.150 | 6.780 | 9.300 |
| 11 | OURS-4B (Qwen3-4B tuned) | ours | reuse | yes | 0.386 | 0.260 | 1.000 | — | 0.342 | 5.828 [5.608–6.040] | 8.930 | 10.37 | 3.600 |
| 12 | BASE (Qwen3-32B untuned) | base | FRESH | raw | 0.356 | 0.260 | 1.000 | 0.939 | 0.450 | 5.493 [5.286–5.689] | 8.950 | 11.63 | 0.400 |
| 13 | Llama-3.3-70B | open | FRESH | raw | 0.356 | 0.160 | 1.000 | 0.997 | 0.417 | 6.262 [6.095–6.422] | 9.160 | 9.327 | 1.300 |
| 14 | PROMPT-BASE-4B (Qwen3-4B engineered) | base | reuse | yes | 0.350 | 0.460 | 1.000 | — | 0.392 | 4.799 [4.592–5.011] | 8.810 | 13.52 | 0.400 |
| 15 | BASE-4B (Qwen3-4B untuned) | base | reuse | yes | 0.347 | 0.220 | 1.000 | — | 0.392 | 4.723 [4.526–4.927] | 8.750 | 13.94 | 0.000 |
| 16 | Mistral-Large-3 (675B) | open | FRESH | raw | 0.336 | 0.380 | 0.997 | 0.919 | 0.408 | 5.217 [4.982–5.436] | 8.770 | 12.14 | 0.900 |
| 17 | Kimi-K2.5 | open | FRESH | raw | 0.331 | 0.420 | 1.000 | 0.875 | 0.508 | 6.266 [6.058–6.470] | 9.070 | 8.758 | 6.700 |
| 18 | BASE (Qwen3-1.7B untuned) | base | reuse | raw | 0.303 | 0.550 | 0.928 | 0.858 | 0.358 | 1.939 [1.779–2.107] | 6.950 | 18.71 | 0.000 |
| 19 | Gemma-3-27B-it | open | FRESH | raw | 0.286 | 0.200 | 1.000 | 0.969 | 0.417 | 5.778 [5.553–5.981] | 9.010 | 10.62 | 0.400 |
| 20 | Qwen3-Next-80B-A3B | open | FRESH | raw | 0.278 | 0.240 | 0.997 | 0.953 | 0.333 | 5.875 [5.654–6.092] | 8.950 | 10.35 | 2.700 |

_gen: FRESH = regenerated this run; reuse = deterministic adapter/MLX gen reused. gated: `yes` = full shipped verify-and-regenerate pipeline (4B trio); `raw` = ungated draft (raw-draft gate axes shown). raw-faith = verify-pass on draft 1 (1 − fabrication). tier-fit / distinct / move-sound / raw-faith / coherence are deterministic (free); instr / move 0-10 + rank are the blinded council._

## The moat — each tuned model vs the best frontier (tier-fit then soundness)

On positions where OURS gives distinct, sound, correctly-graded per-tier moves AND diverges from the best-frontier move, who wins the platform's move-quality moat (the `assemble.derive_wins` definition)? Instructiveness (where the frontier leads) is reported separately above.

| Tuned model | distinct | distinct & diverge | **W** | **L** | **T** |
|---|---:|---:|---:|---:|---:|
| OURS-v4 (Qwen3-32B tuned) | 68 | 62 | 51 | 5 | 6 |
| OURS-v2 (Qwen3-1.7B tuned) | 51 | 48 | 25 | 17 | 6 |
| OURS-v3 (Qwen3-32B tuned) | 46 | 42 | 23 | 10 | 9 |
| OURS-v5 (Qwen3-32B tuned, v5) | 45 | 42 | 22 | 8 | 12 |
| OURS-4B (Qwen3-4B tuned) | 24 | 22 | 5 | 13 | 4 |

## Shipped-gate soundness (tuned models through the SAME verify+fallback gate)

| Tuned model | gated move-sound↑ | gated well-formed↑ | gated no-engine-speak↑ | gate fallback↓ |
|---|---:|---:|---:|---:|
| OURS-v4 (Qwen3-32B tuned) | 1.000 | 1.000 | 0.983 | 0.444 |
| OURS-v2 (Qwen3-1.7B tuned) | 1.000 | 1.000 | 1.000 | 0.358 |
| OURS-v3 (Qwen3-32B tuned) | 1.000 | 1.000 | 0.969 | 0.181 |
| OURS-v5 (Qwen3-32B tuned, v5) | 1.000 | 1.000 | 0.992 | 0.444 |
| OURS-4B (Qwen3-4B tuned) | 1.000 | 1.000 | 1.000 | 0.000 |

_Once gated, tuned soundness/format hit a shared ~100% floor (0 user-visible fabrication by construction) — a fairness floor, not a differentiator; the differentiators are tier-fit / distinct-moves / instructiveness._

## Deterministic gate axes (raw draft for ungated rows; telemetry for gated 4B)

| Model | gated | no-engine-speak↑ | well-formed↑ | move-sound↑ | verify-pass draft1↑ | mean attempts | fallback↓ |
|---|:--:|---:|---:|---:|---:|---:|---:|
| OURS-v4 (Qwen3-32B tuned) | raw | 0.978 | 0.956 | 0.942 | 0.589 | — | — |
| OURS-v2 (Qwen3-1.7B tuned) | raw | 1.000 | 1.000 | 0.986 | 0.689 | — | — |
| OURS-v3 (Qwen3-32B tuned) | raw | 0.964 | 0.969 | 0.947 | 0.942 | — | — |
| OURS-v5 (Qwen3-32B tuned, v5) | raw | 0.978 | 0.897 | 0.831 | 0.575 | — | — |
| Gemini 3.1 Pro | raw | 0.997 | 0.994 | 0.978 | 0.958 | — | — |
| Claude Opus 4.8 | raw | 1.000 | 1.000 | 0.967 | 0.944 | — | — |
| GPT-5.5 | raw | 1.000 | 1.000 | 0.969 | 0.986 | — | — |
| DeepSeek-R1 (reasoning) | raw | 1.000 | 1.000 | 0.997 | 0.978 | — | — |
| DeepSeek-V3.2 | raw | 1.000 | 0.992 | 0.986 | 0.950 | — | — |
| GLM-5 | raw | 0.997 | 0.997 | 0.989 | 0.906 | — | — |
| OURS-4B (Qwen3-4B tuned) | yes | 1.000 | 1.000 | — | — | 1.194 | 0.008 |
| BASE (Qwen3-32B untuned) | raw | 0.992 | 0.994 | 0.994 | 0.939 | — | — |
| Llama-3.3-70B | raw | 1.000 | 1.000 | 1.000 | 0.997 | — | — |
| PROMPT-BASE-4B (Qwen3-4B engineered) | yes | 1.000 | 1.000 | — | — | 1.167 | 0.003 |
| BASE-4B (Qwen3-4B untuned) | yes | 1.000 | 1.000 | — | — | 1.156 | 0.000 |
| Mistral-Large-3 (675B) | raw | 1.000 | 0.997 | 0.994 | 0.919 | — | — |
| Kimi-K2.5 | raw | 1.000 | 0.997 | 0.997 | 0.875 | — | — |
| BASE (Qwen3-1.7B untuned) | raw | 0.964 | 1.000 | 0.928 | 0.858 | — | — |
| Gemma-3-27B-it | raw | 1.000 | 1.000 | 1.000 | 0.969 | — | — |
| Qwen3-Next-80B-A3B | raw | 1.000 | 1.000 | 0.997 | 0.953 | — | — |

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


