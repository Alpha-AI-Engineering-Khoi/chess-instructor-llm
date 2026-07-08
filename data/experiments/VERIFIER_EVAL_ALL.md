# Does the verify-and-regenerate faithfulness gate drive user-visible fabrication to ~0% — for ALL 14 models?

_Generated 2026-07-07T02:48:05.890963+00:00. **Gate** = production `src/api/server.py verify-and-regenerate (imported verbatim)`, N=4 attempts, deterministic checker `verify_text`, engine-derived verified fallback. **Sample** = the SAME 50 held-out positions for every model (fabrication-weighted on OURS-v2 grounded fabrications (33 fab + 17 clean)), grounding reused from `data/benchmark_v2/scenarios.jsonl` — no engine runs live. Locals in-process via `mlx_lm` (server-identical decode); the 12 API models via the TrueFoundry gateway._

## TL;DR

- **Every one of the 14 models lands at **0%** GATED user-visible fabrication** — RAW fabrication ranges from 0% to 40% (OURS-v2 (chess-coach-v2, 1.7B tuned)) before the gate.
- **Answer to "does any model fail to hit 0% after the gate?" — NO.** All 14 models reach 0% user-visible fabrication after the gate. This is by design: the gate only ever serves text that passes `verify_text`, or the engine-derived fallback (true by construction), so the guarantee is model-agnostic.
- **The honest differentiator is the fallback rate** (how often the gate had to replace the model with the verified template). Most self-sufficient: **Gemini 3.1 Pro** (0.0% fallback); most dependent on the safety net: **OURS-v2 (chess-coach-v2, 1.7B tuned)** (10.0%).
- **Total cost: $3.1286** (locals free; API usage on TrueFoundry).

## Method (identical to the prior 2-model run, extended to 14)

- **RAW** — attempt 1 only, gate OFF: exactly what `src/api/server.py` serves with `COACH_FAITHFULNESS_GATE=0` (the reply split into coaching body + `Takeaway:` via `_split_coaching`). Fabrication scored on that **user-visible** text with `verify_text`.
- **GATED** — the real verify-and-regenerate loop: re-sample the whole answer up to **4** times, keep the FIRST reply whose full text passes `verify_text`; if none pass, emit the deterministic engine-derived explanation (`_verified_coaching`), true by construction. Fabrication scored on the FINAL **user-visible** text.
- Attempt 1 of the gated loop **is** the RAW generation, so GATED = "RAW + the gate" on the identical sampling. Every model sees the same grounding (prose VERIFIED-FACTS + ascii board), the same system prompt, and the same 50 positions.

## Per-model: RAW → GATED user-visible fabrication + fallback

| Model | n | RAW fab | GATED fab | Δ | Fallback | Passed within N | Mean attempts |
|---|---|---|---|---|---|---|---|
| OURS-v2 (chess-coach-v2, 1.7B tuned) | 50 | 40% | 0% | -40 pts | 10.0% | 90% | 1.78 |
| BASE (Qwen3-1.7B-4bit, untuned) | 50 | 2% | 0% | -2 pts | 0.0% | 100% | 1.02 |
| GPT-5.5 | 50 | 2% | 0% | -2 pts | 2.0% | 98% | 1.06 |
| Claude Opus 4.8 | 50 | 4% | 0% | -4 pts | 0.0% | 100% | 1.06 |
| Gemini 3.1 Pro | 50 | 0% | 0% | +0 pts | 0.0% | 100% | 1.0 |
| Qwen3-32B | 50 | 6% | 0% | -6 pts | 0.0% | 100% | 1.06 |
| Qwen3-Next-80B-A3B | 50 | 6% | 0% | -6 pts | 0.0% | 100% | 1.06 |
| Gemma-3-27B-it | 50 | 4% | 0% | -4 pts | 0.0% | 100% | 1.04 |
| Llama-3.3-70B | 50 | 0% | 0% | +0 pts | 0.0% | 100% | 1.0 |
| DeepSeek-V3.2 | 50 | 6% | 0% | -6 pts | 0.0% | 100% | 1.06 |
| GLM-5 | 50 | 6% | 0% | -6 pts | 0.0% | 100% | 1.08 |
| Mistral-Large-3 (675B) | 50 | 14% | 0% | -14 pts | 0.0% | 100% | 1.16 |
| Kimi-K2.5 | 50 | 8% | 0% | -8 pts | 0.0% | 100% | 1.08 |
| DeepSeek-R1 (reasoning) | 50 | 2% | 0% | -2 pts | 2.0% | 98% | 1.06 |

> **Read the RAW column as a stress test, not a global fabrication rate.** The 33 fabricated-stratum positions are exactly the ones **OURS-v2** fabricated on in the benchmark, so OURS-v2's RAW is enriched *by construction* (this is its own hard set), and every other model's RAW is *its* fabrication rate on OURS-v2's hardest positions — a deliberately adversarial slice, not that model's unconditional rate. The purpose of the shared, fabrication-weighted set is to exercise the gate hard and keep the **GATED** and **fallback** columns comparable across models; it is not a fair frontier-vs-frontier fabrication leaderboard. (Notably, bigger models write longer, more concrete coaching, giving the checker more surface area — so several of them post higher RAW here than the tiny untuned BASE, which hedges.)

## Does ANY model have a nonzero GATED fabrication rate?

**No.** All 14 models reach **0%** user-visible fabrication after the gate (GATED column above is 0% for every row). The gate's guarantee is structural, not statistical: the only two things that can reach the learner are (a) a model reply that **passed** the deterministic `verify_text` on its full text — and the user-visible slice is a subset of that text, so it passes too — or (b) the **engine-derived fallback**, which is true by construction. Neither depends on which model produced the draft, so a weak open model and a frontier model land at the same 0%; they differ only in how often they need the safety net (the fallback rate).

**Independent audit.** Re-running `verify_text` from scratch on all **700** stored GATED outputs finds **0** fabrications and **0** empty outputs — so the 0% is a fresh-check result, not just the number the gate wrote at run time.

**Coverage gap vs. real leak.** Because GATED fabrication is measured *by the same checker* the gate uses, a nonzero GATED rate would signal an internal inconsistency (a real leak / gate bug); there are none. The residual risk that remains is therefore **not** a model-specific leak but the checker's own **coverage** — a false board claim phrased in a way `verify_text` does not yet recognise would pass the gate *and* be scored clean, so it would be invisible here for **every** model equally. That is a single shared blind spot to harden in the verifier (broader claim/relation coverage), not a reason to trust any one model more than another. The honest per-model differentiator stays the fallback rate below.

## Ranked by fallback rate — the honest self-sufficiency differentiator

Every model reaches ~0% GATED, so fabrication rate no longer separates them. What does is **how often the gate had to throw the model's answer away** and serve the verified template. Lower = more self-sufficient (its own prose reaches the learner more often).

| Rank | Model | Fallback rate | RAW fab | Final output: model prose / verified template | Attempts-to-clean |
|---|---|---|---|---|---|
| 1 | Gemini 3.1 Pro | 0.0% | 0% | 100% / 0% | a1:50 |
| 2 | Llama-3.3-70B | 0.0% | 0% | 100% / 0% | a1:50 |
| 3 | BASE (Qwen3-1.7B-4bit, untuned) | 0.0% | 2% | 100% / 0% | a1:49, a2:1 |
| 4 | Claude Opus 4.8 | 0.0% | 4% | 100% / 0% | a1:48, a2:1, a3:1 |
| 5 | Gemma-3-27B-it | 0.0% | 4% | 100% / 0% | a1:48, a2:2 |
| 6 | Qwen3-32B | 0.0% | 6% | 100% / 0% | a1:47, a2:3 |
| 7 | Qwen3-Next-80B-A3B | 0.0% | 6% | 100% / 0% | a1:47, a2:3 |
| 8 | DeepSeek-V3.2 | 0.0% | 6% | 100% / 0% | a1:47, a2:3 |
| 9 | GLM-5 | 0.0% | 6% | 100% / 0% | a1:47, a2:2, a3:1 |
| 10 | Kimi-K2.5 | 0.0% | 8% | 100% / 0% | a1:46, a2:4 |
| 11 | Mistral-Large-3 (675B) | 0.0% | 14% | 100% / 0% | a1:43, a2:6, a3:1 |
| 12 | GPT-5.5 | 2.0% | 2% | 98% / 2% | a1:49, fallback:1 |
| 13 | DeepSeek-R1 (reasoning) | 2.0% | 2% | 98% / 2% | a1:49, fallback:1 |
| 14 | OURS-v2 (chess-coach-v2, 1.7B tuned) | 10.0% | 40% | 90% / 10% | a1:30, a2:10, a3:1, a4:4, fallback:5 |

_Attempts-to-clean legend: `aK:n` = n positions where the model produced a clean reply on attempt K (K≤4); `fallback:n` = n positions where no attempt passed in budget and the verified engine-derived explanation was served._

## Cost

| Model | Prompt tok | Completion tok | Cost (USD) |
|---|---|---|---|
| OURS-v2 (chess-coach-v2, 1.7B tuned) | — | — | $0.0000 (local) |
| BASE (Qwen3-1.7B-4bit, untuned) | — | — | $0.0000 (local) |
| GPT-5.5 | 51,773 | 13,344 | $0.1982 |
| Claude Opus 4.8 | 76,129 | 10,448 | $1.9255 |
| Gemini 3.1 Pro | 50,373 | 31,270 | $0.3757 |
| Qwen3-32B | 53,298 | 5,682 | $0.0114 |
| Qwen3-Next-80B-A3B | 53,349 | 5,560 | $0.0151 |
| Gemma-3-27B-it | 52,757 | 6,479 | $0.0079 |
| Llama-3.3-70B | 50,466 | 6,060 | $0.0407 |
| DeepSeek-V3.2 | 52,183 | 4,787 | $0.0180 |
| GLM-5 | 53,126 | 6,471 | $0.0316 |
| Mistral-Large-3 (675B) | 58,713 | 8,210 | $0.1667 |
| Kimi-K2.5 | 54,294 | 8,095 | $0.0528 |
| DeepSeek-R1 (reasoning) | 52,085 | 39,749 | $0.2850 |
| **Total** | | | **$3.1286** |

_API prices are the per-model estimates in `src/eval/benchmark/config.py` (frontier priced exactly; open-model Bedrock on-demand best-effort). Locals are free._

## Reproduce

```bash
~/.venvs/mlx/bin/python scripts/run_verifier_eval_all.py
```

Gate harness (imports the real production gate/fallback/verifier, edits nothing): `src/experiments/verifier_gate.py` · driver: `scripts/run_verifier_eval_all.py` · raw per-item rows: `data/experiments/verifier_eval_all_raw.jsonl` · machine summary: `data/experiments/verifier_eval_all_summary.json`. The 2-model precursor lives in `data/experiments/VERIFIER_EVAL.md`.
