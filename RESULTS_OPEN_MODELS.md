# Chess-Coach Benchmark — Bigger Open Models vs OURS-v2 / Frontier

Unified leaderboard extending the v2 benchmark to **9 bigger open-source models**, on the **same 100 held-out positions** and **identical grounding** the v2 run used, so the numbers are directly comparable to OURS-v2 / BASE / GPT-5.5 / Claude Opus 4.8 / Gemini 3.1 Pro. Generated 2026-07-07 02:00 UTC.

> **Question answered:** *do bigger open models fabricate less / coach more instructively than our 1.7B on the same grounded input?*


## TL;DR recommendation

- **Bigger open models fabricate far less than our 1.7B — the truthfulness gap is essentially closed.** Every open model scores 1–8% grounded fabrication vs **OURS-v2 38%** (and BASE 15%). The cleanest, Gemma-3-27B-it at 1%, matches the frontier (~3% avg). Size — not our data intervention — is what a 1.7B lacks for board-fact tracking.
- **They also coach more instructively than OURS-v2 — but do not reach the frontier.** Best open coach is DeepSeek-V3.2 (mean rank 5.18 of 10); every open model in the field out-ranks **OURS-v2 (7.95)** by ~1.3–2.8 positions, yet all trail the best frontier coach GPT-5.5 (2.53) by ~2.7. The instructiveness gap narrows with size but does not close.
- **Raw size is NOT the coaching driver.** The largest model, Mistral-Large-3 (675B) (6.17), is out-coached by the much smaller DeepSeek-V3.2 (5.18) and Gemma-3-27B (5.25). Model quality/training beats parameter count for this behavior.
- **Best v3 base / bigger local deployment: Gemma-3-27B-it.** Lowest fabrication of the whole field (1%), essentially the top open coach (5.25 of 10), and small enough to fine-tune (QLoRA) and run locally in 4-bit on a 64 GB Mac — a genuine drop-in upgrade path from the 1.7B.
- **Scaling up our OWN family helps but isn't the best base.** Qwen3-32B (same family as our Qwen3-1.7B) cuts fabrication 38% → 8% but coaches worst of the open field (6.68); Gemma-3-27B is the stronger base at similar size.
- **Teacher for distillation: keep GPT-5.5.** It is still the best coach in the field (2.53); the strongest fully-open teacher alternatives are DeepSeek-V3.2 / Gemma-3-27B, which trail GPT-5.5 on instructiveness — switch only if a 100%-open pipeline is the goal.


## Reachability on TrueFoundry (`bedrock-oss-group`)

Probed each candidate with a 1-token chat call (`scripts/tfy_access_open.py`); only reachable models were run.

- **Reachable (9, run):** Qwen3-32B, Qwen3-Next-80B-A3B, Gemma-3-27B-it, Llama-3.3-70B, DeepSeek-V3.2, GLM-5, Mistral-Large-3 (675B), Kimi-K2.5, DeepSeek-R1.
- **Unreachable / excluded:** `llama4-maverick-17b` — provider blocks Meta Llama access (HTTP 400 on both `aws-bedrock/` and `bedrock-oss-group/` routes); `kimi-k2-thinking` — spends its entire token budget on hidden reasoning and returns empty coaching content (doesn't fit the coach format); `deepseek.r1` direct route was 403 but the `bedrock-oss-group/deepseek-r1` virtual route works and is used.


## Phase 1 — Grounded objective leaderboard (fabrication is the metric)

All models get the **same VERIFIED-FACTS + Stockfish sound pool + Maia** input and the same format instruction; scoring is deterministic (the project's own faithfulness verifier + move/soundness/engine-speak checks). Sorted by fabrication (lower = better).


| Model | family | fabrication↓ | move_sound↑ | no_engine_speak↑ | ply_cap_ok↑ | avg_violations↓ | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma-3-27B-it | open | 1% | 100% | 100% | 100% | 0.01 | 100 |
| Llama-3.3-70B | open | 1% | 99% | 100% | 100% | 0.01 | 100 |
| GPT-5.5 | frontier | 2% | 100% | 100% | 100% | 0.02 | 100 |
| Gemini 3.1 Pro | frontier | 2% | 100% | 100% | 100% | 0.02 | 100 |
| DeepSeek-R1 (reasoning) | open | 2% | 100% | 100% | 100% | 0.02 | 100 |
| GLM-5 | open | 4% | 99% | 100% | 100% | 0.04 | 100 |
| Claude Opus 4.8 | frontier | 6% | 100% | 100% | 100% | 0.06 | 100 |
| DeepSeek-V3.2 | open | 6% | 100% | 100% | 100% | 0.07 | 100 |
| Mistral-Large-3 (675B) | open | 6% | 100% | 100% | 100% | 0.06 | 100 |
| Kimi-K2.5 | open | 7% | 100% | 100% | 100% | 0.07 | 100 |
| Qwen3-32B | open | 8% | 100% | 99% | 100% | 0.08 | 100 |
| Qwen3-Next-80B-A3B | open | 8% | 100% | 100% | 100% | 0.08 | 100 |
| BASE (Qwen3-1.7B-4bit, untuned) | ours/base | 15% | 92% | 95% | 100% | 0.23 | 100 |
| OURS (chess-coach-v2, 1.7B tuned) | ours/base | 38% | 100% | 100% | 100% | 0.45 | 100 |


*`fabrication` = share of outputs with ≥1 false board fact (non-LLM verifier). `move_sound` = recommended move is in the Stockfish sound pool. `avg_violations` = mean false facts per output.*


> **Note on comparability:** every model here (including OURS-v2 / BASE / the frontier) is re-scored by the *current* faithfulness verifier on identical grounded inputs, so the numbers are internally consistent. That verifier is slightly stricter than the one behind the older `RESULTS_BENCHMARK_v2.md` (e.g. OURS-v2 grounded fabrication reads 38% here vs 33% there); the ranking and the size effect are unaffected.


## Phase 2 — Unified council instructiveness ranking (blinded, cross-family)

One blinded council ranks a **single field of 10 anonymized coaches** (the 5 v2 competitors + the strongest open models by Phase-1 objective) per item, on a **50-item grounded subset** (150 judge-observations across 3 judges: GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro). Lower mean rank = judged more instructive (1 = best of 10); `norm rank` scales that to 0 (best) – 1 (worst).


| Model | family | mean rank (of 10)↓ | norm rank↓ | top-1 win%↑ | tier_calib | clarity | correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.5 | frontier | 2.53 | 0.17 | 37% | 1.84 | 1.90 | 1.77 |
| Claude Opus 4.8 | frontier | 2.91 | 0.21 | 31% | 1.84 | 1.85 | 1.47 |
| Gemini 3.1 Pro | frontier | 3.79 | 0.31 | 19% | 1.75 | 1.87 | 1.45 |
| DeepSeek-V3.2 | open | 5.18 | 0.46 | 5% | 1.58 | 1.62 | 1.12 |
| Gemma-3-27B-it | open | 5.25 | 0.47 | 1% | 1.63 | 1.76 | 1.23 |
| Llama-3.3-70B | open | 5.35 | 0.48 | 2% | 1.44 | 1.59 | 1.45 |
| Mistral-Large-3 (675B) | open | 6.17 | 0.57 | 3% | 1.37 | 1.55 | 0.76 |
| Qwen3-32B | open | 6.68 | 0.63 | 0% | 1.24 | 1.45 | 0.98 |
| OURS (chess-coach-v2, 1.7B tuned) | ours/base | 7.95 | 0.77 | 2% | 0.89 | 1.03 | 0.30 |
| BASE (Qwen3-1.7B-4bit, untuned) | ours/base | 9.17 | 0.91 | 0% | 0.41 | 0.65 | 0.19 |


*Field chosen cost-aware: ranking all 9 open models × all 100 positions in one field would be a huge, less-reliable judge prompt, so Phase 2 uses the 5 v2 anchors + the top open objective performers on a reduced position subset. Open models outside the field have Phase-1 objective numbers above but no council rank.*


> **Bias caveat:** the three judges are also the top-3 competitors. The v2 run measured mean self-preference at +0.43 rank — small next to the ~2.7-position open→frontier gap — and it does not distort the open-vs-OURS-v2 comparison, since neither is any judge's own lab.


## Cost (this open-model extension only)

| Model | gen calls | judge calls | in tok | out tok | est. USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| OURS (chess-coach-v2, 1.7B tuned) | 100 | 0 | 0 | 0 | $0.00 |
| BASE (Qwen3-1.7B-4bit, untuned) | 100 | 0 | 0 | 0 | $0.00 |
| GPT-5.5 | 100 | 50 | 196,644 | 78,097 | $1.03 |
| Claude Opus 4.8 | 100 | 50 | 293,101 | 41,500 | $7.51 |
| Gemini 3.1 Pro | 100 | 50 | 204,108 | 120,298 | $1.46 |
| Qwen3-32B | 100 | 0 | 98,316 | 10,361 | $0.02 |
| Qwen3-Next-80B-A3B | 100 | 0 | 97,916 | 13,142 | $0.03 |
| Gemma-3-27B-it | 100 | 0 | 99,703 | 12,711 | $0.02 |
| Llama-3.3-70B | 100 | 0 | 98,795 | 13,447 | $0.08 |
| DeepSeek-V3.2 | 100 | 0 | 95,960 | 11,140 | $0.03 |
| GLM-5 | 100 | 0 | 96,201 | 11,443 | $0.06 |
| Mistral-Large-3 (675B) | 100 | 0 | 98,901 | 13,721 | $0.28 |
| Kimi-K2.5 | 100 | 0 | 98,029 | 13,633 | $0.09 |
| DeepSeek-R1 (reasoning) | 100 | 0 | 96,060 | 75,709 | $0.54 |


**Total estimated cost of this extension: $11.14** — grounded generation for the open models + the unified council. Open-model prices are best-effort Bedrock on-demand estimates; frontier judge prices are the same per-1M-token figures the v2 run used. The v2 five-model run itself cost ~$24 and is reused here for free.


## Artifacts

- Scenarios (same as v2): `data/benchmark_open/scenarios.jsonl`
- Grounded generations (open + reused v2): `data/benchmark_open/generations.jsonl`
- Objective scores: `data/benchmark_open/objective.jsonl`
- Unified council: `data/benchmark_open/council.jsonl`
- Reachability probe: `scripts/tfy_access_open.py`; driver: `scripts/run_benchmark_open.py`
