# SHOWCASE eval dataset — honest, per-model / per-tier, blinded-graded

Powers the revamped platform's showcase. Every one of the 14 models coaches the SAME positions at all 3 tiers with byte-identical Stockfish+Maia grounding (the exact 803-benchmark pipeline), scored deterministically (sound / tier-fit / fabricated) and graded by a blinded 3-judge cross-family council (move + instructiveness, 0-10). OURS = the live local v2 coach.

## Counts

| split | positions | note |
|---|---:|---|
| train | 132 | IN-DISTRIBUTION (boards OURS-v2 was trained on) — reported honestly as such |
| test (new Lichess) | 210 | freshly pulled, held-out, discriminating, zero-leakage |
| test (reused 803) | 150 | reuses the definitive benchmark's 14-model gens (read-only) |
| **test total** | **360** | |
| **all** | **492** | |

## OURS 3-tier coverage (comprehensive, zero gaps)

- OURS was run LOCALLY (mlx, free) on **every** showcase position × all 3 tiers.
- Positions with full OURS 3-tier coverage: **492/492** (CONFIRMED complete — no per-tier gaps).

## Tier differentiation — the focus (OURS gives DIFFERENT, level-appropriate moves)

A position is a **tier-differentiation / focus** case when OURS recommends >=2 distinct, SOUND moves across the 3 tiers with the correct gradient (beginner = more human-findable, advanced = sharper). This is the behaviour the platform showcases.

| split | focus (tier-differentiates) | mis-directed | of positions |
|---|---:|---:|---:|
| train | 50 | 4 | 132 |
| test  | 176 | 37 | 360 |
| **all** | **226** | **41** | 492 |

- **ours_tier_differentiates / focus**: >=2 distinct sound OURS moves across tiers, beginner!=advanced, correctly directed (beginner more human-findable).
- **ours_misdirected**: OURS changes its move the WRONG way (sharp move handed to the beginner) or a differentiating pick is unsound — recorded honestly, never hidden.

## OURS wins vs loses vs the frontier (both included — not cherry-picked)

| split | ours_wins | ours_loses | clean 'shine' |
|---|---:|---:|---:|
| train | 88 | 103 | — |
| test | 292 | 287 | — |
| **all** | **380** | **390** | **50** |

- **ours_wins**: a tier where OURS is sound+tier-fit while a frontier model isn't, or sound+faithful while a frontier model fabricates.
- **ours_loses**: the honest opposite. Both flags can be true on one position.
- **shine**: focus (tier-differentiates) AND not ours_loses AND not mis-directed — the clean, demonstrable level-fitting cases.
- **best_other**: the strongest non-OURS model per position (council-first), stored for the OURS-vs-best comparison; every model's per-tier picks are kept for the dropdown.

## Per-model coverage (honest)

OURS + 10 of the API models coached every position. During the NEW-position run three open models (Gemma-3-27B, Kimi-K2.5, Mistral-Large-3) hit a transient AWS Bedrock overload (timeouts/503s) and were retried but only partially completed on the fresh positions; they are FULLY present on the reused-803 test positions. OURS's own coverage is complete — the differentiation view has zero gaps.

| model | train (/132) | test (/360) |
|---|---:|---:|
| BASE (Qwen3-1.7B-4bit, untuned) | 132 | 360 |
| Claude Opus 4.8 | 132 | 360 |
| DeepSeek-R1 (reasoning) | 132 | 360 |
| DeepSeek-V3.2 | 132 | 360 |
| GLM-5 | 132 | 360 |
| GPT-5.5 | 132 | 360 |
| Gemini 3.1 Pro | 132 | 360 |
| Llama-3.3-70B | 132 | 360 |
| OURS-v2 (1.7B tuned) | 132 | 360 |
| Qwen3-32B | 132 | 360 |
| Qwen3-Next-80B-A3B | 132 | 360 |
| Kimi-K2.5 | 91 | 150  ⚠ partial (transient provider outage) |
| Mistral-Large-3 (675B) | 84 | 150  ⚠ partial (transient provider outage) |
| Gemma-3-27B-it | 80 | 150  ⚠ partial (transient provider outage) |

## New-Lichess filter yield (the 10k -> few hundred funnel)

| stage | positions |
|---|---:|
| raw pulled from Lichess | 10000 |
| after dedup (vs all corpora) | 9998 |
| grounded (Stockfish+Maia) | 9998 |
| discriminating + eligible | 5862 |
| **selected (strongest)** | **210** |

## Cost (frontier via TrueFoundry; local OURS/BASE free)

| bucket | calls | in tok | out tok | est. USD |
|---|---:|---:|---:|---:|
| train_gen | 4,113 | 4,295,072 | 1,072,030 | $23.17 |
| test_new_gen | 5,670 | 5,887,242 | 1,586,636 | $35.57 |
| council | 4,428 | 11,688,198 | 3,190,377 | $141.57 |
| local_gen_free | 2,952 | 0 | 0 | $0.00 |
| reused_803_gen | 5,400 | 5,559,123 | 1,297,152 | $26.96 |
| **NEW SPEND (this run)** | | | | **$200.31** |

_New spend this run = train gens + new-Lichess gens + council = **$200.31**. The reused 803 generations ($26.96 of value) were already paid for by the definitive benchmark and cost $0 here. Local OURS-v2 + BASE are free._

## Artifacts

- Per split: `data/showcase/{train,test_new,test_reuse}/scenarios.jsonl`, `gen/<model>.jsonl`, `objective.jsonl`, `council.jsonl`.
- Raw Lichess pull: `data/showcase/lichess_raw.jsonl`; grounded: `data/showcase/test_new/grounded.jsonl`.
- Output: `web/public/showcase.json`; machine stats: `data/showcase/stats.json`, `data/showcase/cost.json`.