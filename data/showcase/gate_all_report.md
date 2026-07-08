# Gate-all + two-layer truthfulness residual — FAIR model comparison

Every displayed coaching cell in `web/public/showcase.json` is now GATED: the raw output is kept only if it passes the widened deterministic checker (`verify_text_ext`); otherwise it is re-sampled from the SAME model with the identical grounded prompt (up to 5 re-samples, first clean kept), and if none verify, replaced by a deterministic engine-derived explanation that is true by construction. The raw output is preserved as `raw_coaching` (a model-capacity metric). This makes the comparison fair: all models are judged on GATED text.

- Positions: **492**; cells with text: **18,135**.
- Raw fabrication (widened checker, pre-gate): **4,472** cells (**24.7%**).
- Post-gate deterministic residual: **0** cells (**0.0%**) — should be ~0.
- Re-generations spent: **7,305**; verified-fallbacks: **708**.
- New spend (this gate run, frontier/open re-gen): **$44.35** (local OURS/BASE re-gen = $0).
- LLM-judge residual sample: **378** gated cells × 3 judges = **1134** calls, **$13.05**.

## Per-model: RAW → GATED (deterministic) → JUDGE residual

| model | kind | cells | RAW fab% (capacity) | GATED fab% (determ.) | JUDGE truthful% [95% CI] (n) | re-gens | fallback% |
|---|---|---:|---:|---:|---:|---:|---:|
| OURS-v2 (1.7B tuned) | OURS | 1,476 | 49.4% | 0.0% | 23.1% [12.7%–38.3%] (n=39) | 1,883 | 7.7% |
| BASE (Qwen3-1.7B-4bit, untuned) | BASE | 1,476 | 35.0% | 0.0% | 0.0% [0.0%–17.6%] (n=18) | 927 | 1.5% |
| Claude Opus 4.8 | frontier | 1,476 | 33.9% | 0.0% | 25.6% [14.6%–41.1%] (n=39) | 1,004 | 2.2% |
| Gemini 3.1 Pro | frontier | 1,476 | 16.9% | 0.0% | 33.3% [20.6%–49.0%] (n=39) | 374 | 0.3% |
| GPT-5.5 | frontier | 1,476 | 16.3% | 0.0% | 79.5% [64.5%–89.2%] (n=39) | 365 | 0.6% |
| Mistral-Large-3 (675B) ⚠ | open | 623 | 36.1% | 0.0% | 38.9% [20.3%–61.4%] (n=18) | 5 | 36.1% |
| Kimi-K2.5 ⚠ | open | 653 | 35.5% | 0.0% | 44.4% [24.6%–66.3%] (n=18) | 20 | 35.5% |
| Qwen3-Next-80B-A3B | open | 1,476 | 25.8% | 0.0% | 5.6% [1.0%–25.8%] (n=18) | 579 | 0.8% |
| DeepSeek-V3.2 | open | 1,476 | 24.9% | 0.0% | 22.2% [9.0%–45.2%] (n=18) | 497 | 0.1% |
| GLM-5 | open | 1,476 | 24.4% | 0.0% | 44.4% [24.6%–66.3%] (n=18) | 541 | 0.4% |
| Qwen3-32B | open | 1,476 | 15.9% | 0.0% | 46.2% [31.6%–61.4%] (n=39) | 275 | 0.0% |
| DeepSeek-R1 (reasoning) | open | 1,476 | 13.6% | 0.0% | 11.1% [3.1%–32.8%] (n=18) | 290 | 0.5% |
| Gemma-3-27B-it ⚠ | open | 623 | 13.5% | 0.0% | 25.6% [14.6%–41.1%] (n=39) | 138 | 0.5% |
| Llama-3.3-70B | open | 1,476 | 10.4% | 0.0% | 72.2% [49.1%–87.5%] (n=18) | 407 | 2.8% |

⚠ = partial model: only a subset of cells exist (the ORIGINAL benchmark hit AWS Bedrock throttling for Gemma-3-27B / Kimi-K2.5 / Mistral-Large-3, so their missing cells stay missing). During THIS gate run Gemma-3-27B re-sampled normally (0.5% fallback); Kimi-K2.5 & Mistral-Large-3 were unreachable (persistent Bedrock 503), so their flagged cells went straight to the verified engine-derived fallback — that is why their fallback% ≈ their raw-fab%. Rates are over the cells each model DOES have.

## Reading the three layers

1. **RAW fab% (capacity)** — how often the model's *own* ungated coaching states a mechanically-false board fact (widened checker). This is the honest capacity metric; it is high for the small local models and non-trivial even for frontier.
2. **GATED fab% (deterministic residual)** — after gating, what the mechanical checker still catches. ~0 by construction (the gate guarantees it).
3. **JUDGE truthful% (residual)** — a cross-family panel (GPT-5.5 + Claude + Gemini, `any`-aggregation, non-circular) fact-checks a stratified sample of the GATED text for the multi-move / evaluative claims the deterministic layer abstains on. This is the honest ceiling on truthfulness the gate cannot itself guarantee.

## Two-layer residual — read it honestly

- **`any`-aggregation is a strict union.** Individual judge flag-rates: CLAUDE 42.3%, GEMINI 31.5%, GPT 59.8%. A cell is 'not truthful' if *any one* of them objects, so the truthful% is a conservative floor. The judges flag concrete FALSE or UNSUPPORTED-beyond-the-1-ply-facts claims, not general plans/principles.
- **High-confidence floor:** all three cross-family judges independently flagged **26.2%** of sampled cells (n=99). Panel was unanimous on 61.6% of cells.
- **The residual lives in model prose, not the fallback.** The engine-derived verified-fallback cells are judged **100.0%** truthful (n=17) — the judges validate engine-truth. The residual is in mechanically-clean *model* text: raw-clean cells 34.5% truthful (n=287), re-sampled cells 24.3% (n=74). Passing the deterministic gate is NOT the same as being truthful.

## Where gating barely moved a model vs did heavy lifting

- **Under the WIDENED checker, frontier was NOT 'mostly clean'.** Raw fabrication was Claude 33.9%, Gemini 16.9%, GPT-5.5 16.3% — so the deterministic gate had real work to do on frontier too (Claude needed the most frontier cleanup by far: 500 flagged cells). The narrow legacy checker had understated this; the widened `verify_text_ext` catches the relational / move-consequence / turn / material lies it missed.
- **OURS/BASE needed the heaviest lifting.** OURS raw-fab 49.4% and BASE 35.0%; together they consumed 2,810 re-samples and fell back to the engine-derived explanation on the cells even re-sampling couldn't fix. Local re-gen is free, so this cost $0 — the point of running them on-device.
- **Deterministically the gate equalises everyone to 0% mechanical fabrication** (that is the fairness guarantee). The honest *quality* gap only appears in the LLM-judge layer: GPT-5.5 79.5% truthful and Llama-3.3-70B 72.2% at the top, OURS 23.1% and BASE 0.0% at the bottom — the 1.7B models say true-sounding but unsupported things that only a semantic judge can catch.

## Artifacts

- `web/public/showcase.json` — gated (default `coaching`) + `raw_coaching`, `raw_fabricated`, `gate_attempts`, `verified_fallback`, post-gate `fabricated`.
- `data/showcase/showcase.pregate.json` — pristine pre-gate backup.
- `data/showcase/gate/gate_stats.json`, `data/showcase/truthfulness.json`.