# Submission — chess-instructor-llm

**Project:** Train Your Own Small Learning Model — a *reliable, level-calibrated chess coach* from
a fine-tuned Qwen3-1.7B, end to end (dataset → model → platform → eval → thesis → demo).

**Win condition (from the brief):** the tuned model beats the base model on **spec-adherence,
level-calibration, and no-engine-speak** — met and exceeded (see the results table below).

> **✅ `v2` is shipped and current.** Every number in this document is the **v2** run
> (`models/mlx/chess-coach-v2`), with the v1→v2 delta shown. The v2 data intervention
> (faithfulness-filtered labels + tier-aware teacher rule + contrastive multi-tier pairs)
> **improved explanation faithfulness** (grounded fabrication 50% → 33%) and **fixed
> tier-differentiated move-selection** (27.5% → 39.2%, direction now correct). It still does not
> out-teach a prompted frontier model on instructiveness — it *narrows* that gap (council rank
> 4.13 → 3.68) — so the headline win stays **cost / privacy / register + now-improved faithful
> grounding**. The HF artifacts are the v2 re-publish. Full detail:
> [`RESULTS_V2.md`](RESULTS_V2.md), [`RESULTS_BENCHMARK_v2.md`](RESULTS_BENCHMARK_v2.md).

---

## Deliverables map

| # | Deliverable | Artifact — path / URL | Status (`v2`) |
|---|---|---|---|
| 1 | **Dataset** (published on HF Hub) | 🤗 [`datasets/khoilamalphaai/chess-coach-benchmark`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-benchmark) — held-out benchmark: 100 positions × 5 models × 2 conditions + objective scores + council rankings + blind-label export | ✅ **Published** (v2) |
| 1b | Training set (the "real deliverable") | Local, 100% synthetic SFT rows: `data/dataset/train_v2.jsonl` (+`valid_v2.jsonl`), **2,586 kept rows incl. 348 contrastive multi-tier sets**, built by the `positions → Stockfish → Maia → GPT-5.5 (tier-aware) → hard filter + faithfulness gate` pipeline | ✅ Built locally *(gitignored; v2 faithfulness gate → **0% false labels**)* |
| 2 | **Fine-tuned model** (published on HF Hub) | 🤗 [`khoilamalphaai/qwen3-1.7b-chess-coach-mlx`](https://huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx) — QLoRA SFT → merged → 4-bit MLX | ✅ **Published** (v2) |
| 2b | **Running inference demo** (local) | **The Analysis Room** — FastAPI (`src/api/server.py`, :8000) + Next.js (`web/`, :3000). One command: **`./run_platform.sh`** → http://localhost:3000 | ✅ **Runs locally** (tuned MLX coach + engine + verifier) |
| 3 | **Eval harness** | `src/eval/evaluate.py` (base-vs-tuned) · `src/eval/benchmark/` (5-model council) · `scripts/frontier_gap*.py` (gap) · `scripts/divergence_*.py` (tier-selection). Protocol + pass bar: [`docs/EVAL_AND_ITERATE.md`](docs/EVAL_AND_ITERATE.md) | ✅ **Complete & re-runnable** |
| 3b | **Base-vs-tuned results table** | [`RESULTS_V2.md`](RESULTS_V2.md) + [`RESULTS_BENCHMARK_v2.md`](RESULTS_BENCHMARK_v2.md) (v2, with v1→v2 deltas) · [`RESULTS.md`](RESULTS.md) (v1 Claude-judged base vs tuned) · [`data/analysis/GAP_REPORT.md`](data/analysis/GAP_REPORT.md) · [`data/analysis/DIVERGENCE_REPORT.md`](data/analysis/DIVERGENCE_REPORT.md) | ✅ **Complete** (v2) |
| 3c | **Results dashboard** (published on HF Hub) | 🤗 [`spaces/khoilamalphaai/chess-coach-benchmark`](https://huggingface.co/spaces/khoilamalphaai/chess-coach-benchmark) — interactive view of the benchmark | ✅ **Published** (v2) |
| 4 | **BrainLift** (behavior thesis + evidence) | [`brainlift/brainlift.md`](brainlift/brainlift.md) — DOK-4 spiky POVs, experts, DOK-2 knowledge tree (~120 sources), all tied to primary sources or the project's own measurement | ✅ **Complete** |
| 5 | **Demo video** (3–5 min) | Script + shot list: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md). Runnable demo provided (`./run_platform.sh`); **recording is the user's step** | 🟡 **Script ready — awaiting user recording** |

---

## Win-condition scorecard (`v2` — current)

Base vs. tuned, cross-family Claude judge, held-out scenarios (from [`RESULTS_V2.md`](RESULTS_V2.md) /
[`RESULTS.md`](RESULTS.md)):

| Behavior (the brief's win condition) | Base | v1 tuned | **v2 tuned** | Verdict |
|---|---:|---:|---:|:---:|
| **Spec-adherence** (judge 0–2) | 0.47 | 0.93 | **0.93** | ✅ WIN |
| **Level-calibration** (judge 0–2) | 0.60 | 1.13 | **1.13** | ✅ WIN |
| **No-engine-speak** (objective %) | 33% | 100% | **100%** | ✅ WIN |
| **No-engine-speak** (judge 0–2) | 0.87 | 1.87 | **1.73** | ✅ WIN |
| Move soundness (objective %) | 87% | 100% | **100%** | ✅ WIN |
| Ply-cap adherence (objective %) | 67% | 100% | **100%** | ✅ WIN |
| Truthfulness (judge 0–2) | 0.13 | 0.13 | **0.20** | 🟡 **IMPROVING** — was flat; grounded fabrication 50% → 33% |
| Tier-differentiation (% + direction) | — | 27.5% (mis-directed) | **39.2% (correct)** | 🟡 **DIRECTION FIXED** by v2 |

**Bottom line:** v2 still clears the stated win condition on all three target behaviors, and the two
honest v1 gaps both moved: **truthfulness is no longer flat** (0.13 → 0.20 on the rubric; grounded
fabrication 50% → 33%) and **tier-differentiation is fixed in direction** (27.5% → 39.2%, beginners
now steered to the human-findable move). What data-shaping still can't buy is out-teaching a much
larger model on raw instructiveness — v2 *narrows* that gap (council rank 4.13 → 3.68) but doesn't
close it. Dependable truth at deployment stays carried by grounding + a non-LLM verifier — precisely
the spiky claim of the BrainLift.

---

## The honest gaps — what v2 moved, what remains

Reported plainly so the submission is not oversold (details in [`RESULTS_V2.md`](RESULTS_V2.md),
[`GAP_REPORT.md`](data/analysis/GAP_REPORT.md), [`DIVERGENCE_REPORT.md`](data/analysis/DIVERGENCE_REPORT.md)):

- **Truthfulness — improved, not solved; and *ungrounded*, v2 is more brittle.** Grounded (the
  deployment mode) v2 cut fabrication **50% → 33%**, and the faithfulness gate produced **0% false
  labels**. But **ungrounded**, v2 fabricates *more* than v1 (**87% → 99%**) — it teaches more
  concretely (an explicit "how to find it" that cites squares/captures), so a 1.7B invents more
  without the engine facts. The product always runs grounded; the verify-and-regenerate gate remains
  the production defense. Truth is carried by grounding + the verifier, not the weights.
- **Tier-differentiation — fixed in direction, now partial (not universal).** v2 raised it **27.5%
  → 39.2%** and *reversed the direction*: beginners now get the more human-findable move, advanced the
  sharpest ("beginner move == the human/Maia move" 39% → 62%), after taking contrastive multi-tier
  pairs from **0% → 348 FENs × 3 tiers**. The move still varies by tier on ~39% of positions, not all.
- **Instructiveness still trails the frontier.** Even grounded, the blinded council ranks the big
  models above the 1.7B; v2 *narrows* the gap (rank 4.13 → 3.68, gap to best frontier +2.22 → +1.60)
  but does not erase it. The small model's defensible win is **form factor + register consistency +
  now-improved faithful grounding** (~$0, local, private, low-variance voice), not raw coaching
  quality.

---

## Reproduce (one command per instrument)

```bash
cd chess-instructor-llm
set -a && source .env && set +a                                   # keys, never printed
~/.venvs/mlx/bin/python scripts/run_benchmark_v2.py all --n 100   # → RESULTS_BENCHMARK_v2.md
~/.venvs/mlx/bin/python -m scripts.divergence_compare_v2 \
  --v1 data/analysis/divergence_v1_matched.jsonl \
  --v2 data/analysis/divergence_v2.jsonl \
  --out data/analysis/divergence_compare_v2.json                  # → RESULTS_V2.md deltas
~/.venvs/mlx/bin/python -m scripts.frontier_gap --num 50 && \
  ~/.venvs/mlx/bin/python -m scripts.frontier_gap_report          # → data/analysis/GAP_REPORT.md
./run_platform.sh                                                 # → http://localhost:3000 (serves chess-coach-v2)
```

All eval FENs are verified held-out (absent from `train.jsonl`/`valid.jsonl` by board + side-to-move
key); grounding is identical across every model; local decoding is greedy. See
[`docs/EVAL_AND_ITERATE.md`](docs/EVAL_AND_ITERATE.md) for the full protocol and the v2 pass bar.
