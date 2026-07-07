# chess-instructor-llm

**A level-calibrated, engine-grounded chess coach — a fine-tuned 1.7B open model that turns
engine truth into intuitive, tier-appropriate teaching, and does it *consistently* where a
prompted frontier model drifts.**

You set a position, mark the move you were unsure about, pick your rating tier
(Beginner / Intermediate / Advanced), and the coach returns **one** sound teaching move with a
plain-language explanation — no centipawns, no engine jargon, no GM-only lines. Every
recommendation is grounded in Stockfish (a pool of *sound* moves + short lines) and Maia (how
likely a human at your level is to find each move), and each board claim is checked against the
real position before it reaches you.

**Live artifacts (`v2` — current shipped model):**
- 🤗 Model — [`khoilamalphaai/qwen3-1.7b-chess-coach-mlx`](https://huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx)
- 🤗 Benchmark dataset — [`khoilamalphaai/chess-coach-benchmark`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-benchmark)
- 🤗 Results dashboard (Space) — [`khoilamalphaai/chess-coach-benchmark`](https://huggingface.co/spaces/khoilamalphaai/chess-coach-benchmark)
- 💻 Local platform — **The Analysis Room** (FastAPI + Next.js), one command: `./run_platform.sh` (serves `models/mlx/chess-coach-v2`)

> **✅ `v2` is shipped and is the current model** (`models/mlx/chess-coach-v2`); every number
> below is v2, with the v1→v2 delta shown. The v2 data intervention (faithfulness-filtered labels +
> tier-aware teacher rule + contrastive multi-tier pairs) did what it set out to: it **improved
> explanation faithfulness** (grounded fabrication 50% → 33%) and **fixed tier-differentiated
> move-selection** (27.5% → 39.2%, and the *direction* is now correct). It still does **not** beat a
> prompted frontier model on raw instructiveness — it *narrows* that gap (council rank 4.13 → 3.68).
> The honest win is unchanged and now stronger: **reliable, local, ~$0, private, no-engine-speak
> coaching with a non-LLM verifier in the loop** — the fine-tune is still the last-mile compressor,
> not the source of truth. The engine + grounding + verifier remain required.

---

## The gap (why this is worth building)

The pitch is not "a 1.7B model plays better chess than GPT-5.5." It never will. The bet is
narrower and measurable:

> **One specific behavior — leveled, human-findable "teaching-move" coaching — is *not*
> reliably delivered by a prompted frontier model, and *can* be trained into a small model to run
> reliably, cheaply, and locally.**

We proved the gap exists **before** claiming to fill it. On 50 held-out positions with grounding
byte-identical to the app, the frontier models (GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro) are
strong players with fluent prose but **weak at the narrow behavior**:

| Frontier behavior (avg of 3 models) | Rate | What it means |
|---|---:|---|
| Tier-differentiation | **22.7%** | Usually recommends the *same* move regardless of the stated Elo |
| Engine-mirroring at **every** tier | **68.7%** | Mostly just returns Stockfish's #1, blind to level |
| Beginner steered to the *findable* move (opportunity subset) | **20.5%** | Rarely gives a 1200 the move a 1200 would actually find |

**The canonical failure:** serving a 1200-rated beginner the 3000-Elo engine-best move wrapped in
a GM-level line. It's sound, but it's not *findable* and not *instructive* for that student.

### The honest counter-finding

The gap above is about **move selection**. On **truthfulness** the frontier is *strong*: it
fabricates a board fact in only ~3.3% of coaching outputs. So the frontier is weak at leveling
the move but good at not lying about the board — and any credible small-model win has to clear
**both** bars.

### The thesis (where dependability actually comes from)

Dependability in a coach like this is **not carried by the model weights**. It is carried by
three parts that sit *outside* the language model:

1. **A strong engine (Stockfish)** certifies which moves are sound.
2. **Grounding + detectors** expose the concrete features of the position as *verified facts*.
3. **A non-LLM verifier** checks every explanation claim against the real board **before** it
   reaches the student.

The fine-tuned 1.7B model is the **last-mile compressor**: it renders that already-grounded,
already-verified behavior locally, cheaply, privately, and in a steady, low-variance,
no-engine-speak voice. Its honest wins are **form factor** and **register consistency** — not
dependability in general. (Full argument, evidence, and a menu of testable stances:
[`brainlift/brainlift.md`](brainlift/brainlift.md).)

---

## Results at a glance (`v2` — current)

> Every table below is **v2** (`models/mlx/chess-coach-v2`), with the v1 number shown for the delta.
> Full detail: [`RESULTS_V2.md`](RESULTS_V2.md) and [`RESULTS_BENCHMARK_v2.md`](RESULTS_BENCHMARK_v2.md).

### 0. What v2 moved (v1 → v2, grounded — the mode the product deploys in)

The v2 data intervention targeted exactly the two measured v1 gaps, and hit both while narrowing
the instructiveness gap to the frontier:

| Metric (grounded, 100 held-out) | v1 | v2 | Direction |
|---|---:|---:|---|
| Fabrication rate (≥1 false board fact) | 50% | **33%** | ↓ better (avg false facts/answer 0.62 → 0.46) |
| Council rank (1 = best of 5) | 4.13 | **3.68** | ↓ better; gap to best frontier **+2.22 → +1.60** |
| Top-1 instructiveness win-rate | 2% | **8%** | ↑ better |
| Tier-differentiation (move varies by tier) | 27.5% | **39.2%** | ↑ better **and now correctly directed** |
| Move-soundness | 97% | **98%** | ↑ |
| No-engine-speak (no jargon) | 100% | **100%** | held |

"Correctly directed" = beginners now get the more human-findable move and advanced the sharpest
(mean pool-rank beginner 0.43 → **0.78**, advanced 0.62 → **0.45**; "beginner move == the human/Maia
move" 39% → **62%**). In v1 this was *mis-directed* — beginners got the sharper engine move.

### 1. Base vs. fine-tuned — the trained behavior (Claude Opus judge, cross-family, held-out)

The fine-tune wins **decisively on everything it can control by shaping the training
distribution** — exactly the behaviors that resisted prompting on the base model.

| Objective check (deterministic, %) | Base | Tuned (v2) | Δ |
|---|---:|---:|---:|
| move_sound | 87% | **100%** | **+13** |
| no_engine_speak | 33% | **100%** | **+67** |
| ply_cap_ok | 67% | **100%** | **+33** |

| LLM-judge (mean 0–2) | Base | v1 tuned | **v2 tuned** | Δ (base→v2) |
|---|---:|---:|---:|---:|
| spec_adherence | 0.47 | 0.93 | **0.93** | +0.46 |
| level_calibration | 0.60 | 1.13 | **1.13** | +0.53 |
| no_engine_speak | 0.87 | 1.87 | **1.73** | +0.86 |
| **truthfulness** | 0.13 | 0.13 | **0.20** | **+0.07 ← no longer flat** |
| task_quality | 0.13 | 0.27 | **0.33** | +0.20 |

**Truthfulness was the flat axis in v1 (0.13 → 0.13); v2 finally moves it.** On this 15-scenario
rubric it lifts to **0.20**, and the more robust deterministic signal — the 100-position fabrication
rate — falls **50% → 33%** in the grounded (deployment) mode. The v2 faithfulness gate rejected
every fabricated *label* (**0% false labels**, down from 6.3% in the v1 candidate pool), and
grounded teacher generation + a verify-and-retry loop pushed source fabrication to 1.6%. This is
still the project's hardest axis — a 1.7B/4-bit model cannot reliably track 32 pieces from a FEN, so
truth is carried by **grounding + the verifier, not the weights** — but it is now *improving*, not
flat. See [`RESULTS_V2.md`](RESULTS_V2.md) / [`RESULTS.md`](RESULTS.md).

### 2. The 5-model benchmark — grounded vs. ungrounded, ours vs. frontier (100 held-out positions)

A blinded, cross-family council (GPT-5.5 + Claude + Gemini) ranks all 5 anonymized outputs by
*instructiveness for the tier*, alongside deterministic objective metrics. `OURS` = **chess-coach-v2**.
See [`RESULTS_BENCHMARK_v2.md`](RESULTS_BENCHMARK_v2.md).

| Metric | OURS v2 (ungr → grnd) | Frontier avg (grnd) | Reading |
|---|---:|---:|---|
| move_sound | 41% → **98%** | 99% | Grounding hands everyone the sound pool; choosing from it is easy |
| no_engine_speak | 100% → **100%** | 100% | The fine-tune owns the style gate |
| fabrication_rate | 99% → **33%** | ~3% | Grounded fabrication is down from v1's 50%; still trails the frontier on truth |
| council rank (1 = best of 5) | 4.11 → **3.68** | ~2.21 (best 2.09) | Narrowed from v1's 4.13; the coaching gap shrinks but does **not** close |
| top-1 instructiveness win% | 1% → **8%** | — | Up from v1's 2% grounded |

- **Self-preference check:** mean signed self-preference is **+0.43 rank** — small relative to the
  gaps, so the council isn't just lab loyalty.
- **Cost:** running the whole benchmark cost **~$24** total; the local MLX models (OURS + BASE)
  are **$0.00** — the standing form-factor advantage.

**Takeaway:** the fine-tune is a **reliable last-mile behavior compressor**, and v2 shows the two
gaps that resisted v1 *can* be moved by data-shaping: grounded fabrication fell 50% → 33% and
tier-differentiation rose to 39.2% with the direction corrected. What data-shaping still can't do is
out-teach a much larger model on raw instructiveness — v2 *narrows* that gap (4.13 → 3.68) but does
not erase it, and truthfulness at deployment is still carried by **grounding + the non-LLM verifier**,
not the weights. That division of labor is the spiky, defensible claim of the BrainLift.

---

## Architecture

### Data pipeline (offline, produces the training set)

```
Lichess positions  →  Stockfish (sound pool + mistake magnitude)  →  Maia (human likelihood by tier)
   →  GPT-5.5 teacher (max reasoning, grounded + tier-aware move rule: pick the teaching move,
      the why, AND how to find it + leveled coaching)
   →  hard filter (soundness · no-engine-speak · ply-cap · faithfulness gate [v2, shipped])
   →  data/dataset/train_v2.jsonl  →  QLoRA (Qwen3-1.7B)  →  base-vs-tuned eval
```

Locked design decisions:

- **Engine as guardrail, not dictator.** Stockfish supplies the sound-move pool (within ~150cp of
  best, never a blunder ≥250cp) + mistake magnitude; it does **not** pick the lesson.
- **Teaching move ≠ engine's #1.** From all *sound* moves, pick the one with the most extractable
  lesson for the tier — sometimes #1, sometimes #5.
- **Maia (human-at-rating)** ranks candidate moves by "would a human at this tier even play this?"
  — filtering superhuman-only moves. Used as a **descriptive** level signal, not a teaching target.
- **Teacher = GPT-5.5 (max reasoning), grounded in engine analysis** (explains, never invents).
  Judged by a **different** model family (Claude) — no grading your own homework.
- **YouTube transcripts (Naroditsky, GothamChess) = pedagogy reference**, distilled **once** into
  principles + few-shots baked into the teacher prompt. Internal use only; the **dataset stays 100%
  synthetic**.
- **Task:** move review. **Tiers:** Beginner 1000–1200 / Intermediate 1300–1600 / Advanced 1700–2000.
- **Fix disappointing models in DATA, not hyperparameters.**

### Live platform — "The Analysis Room" (online, serves the coach)

A thin FastAPI backend wires the repo's existing pieces to a calm, board-centric Next.js front end.
It re-implements no chess logic:

- **Stockfish** → the sound-move pool + how bad the student's move was.
- **Maia** → which sound moves a player at the chosen tier would actually consider (best-effort;
  the API degrades gracefully if lc0 / the weights are missing).
- **`config/schema.py`** → assembles those facts into the exact `TeacherInput` prompt text the
  model was trained on (`render_user_prompt`).
- **`src/engine/position_facts.py`** → prepends a **VERIFIED FACTS** block (the exact pieces on the
  board, which are loose, what each candidate move concretely does) so the model explains *from
  truth* instead of guessing off the ASCII board.
- **The MLX model** → reads `prompts/coach_system.md` + that prompt and produces the coaching.
- **`src/engine/faithfulness.py` (the verifier)** → a **verify-and-regenerate gate**: after the
  model writes a reply, every board claim is checked against the real position; if any is false the
  **whole answer is re-sampled** (never sentence-stripped) up to a small budget, keeping the first
  reply that verifies clean. If none verify, the API emits a deterministic, engine-derived
  explanation of a sound move that is **truthful by construction**. This is the inference-time half
  of the thesis' remedy, running in production today.

The tuned-model swap is a single env var (`COACH_MODEL_PATH` / `COACH_ADAPTER_PATH`) — nothing
else changes.

---

## Quickstart

```bash
cd chess-instructor-llm
./run_platform.sh
```

This starts the FastAPI backend (the tuned MLX coach) on **:8000** and the Next.js front end on
**:3000**, then waits (Ctrl-C stops both). Open **http://localhost:3000** — the page auto-runs the
coach on the classic `1.e4 e5 2.Qh5?` example so you land straight in a coaching reveal.

**Prerequisites:**

- The MLX venv Python with `mlx_lm`, `python-chess`, `fastapi`, `uvicorn`
  (default `~/.venvs/mlx/bin/python`; override with `PY=...`).
- **Stockfish** (`/opt/homebrew/bin/stockfish`) — required.
- **lc0 + Maia nets** in `models/maia/` — optional; without them the coach still runs and the
  human-likelihood panel shows "unavailable."
- **Node 18.18+** (built/tested on Node 26) and `npm install` in `web/` (first run only).

**Overrides** (all optional): `COACH_MODEL_PATH` (tuned model dir/repo), `COACH_ADAPTER_PATH` (MLX
LoRA adapter), `API_PORT`, `WEB_PORT`, `PY`. Secrets (if any) live only in `./.env` and are read at
call time — never printed.

---

## Repo layout

```
config/     tiers, engine tolerances, Maia mapping, the BEHAVIOR_SPEC (the one gate), schema/rendering
data/       positions / transcripts / generated / dataset / analysis / benchmark / eval  (gitignored)
prompts/    coach_system.md (the spec), principles.md + fewshots.json (distilled style), tier_guides, rubric
src/engine  Stockfish + Maia wrappers, position_facts (grounding), faithfulness (the verifier)
src/ingest  Lichess sampler, YouTube transcript harvester
src/teacher GPT-5.5 generation (v1 + v2) + principle distillation + tier selection
src/filter  soundness + spec checks + LLM judge + faithfulness gate (v2, shipped)
src/train   split_data + Modal QLoRA trainers
src/eval    base-vs-tuned harness (evaluate.py) + the 5-model benchmark (benchmark/)
src/api     FastAPI backend (server.py) — the platform's thin HTTP layer
web/        Next.js 16 + Tailwind v4 + HeroUI v3 + react-chessboard front end
docs/       DATASET_PLAN · EXTERNAL_DATASETS · EVAL_AND_ITERATE · DEMO_SCRIPT
run_platform.sh   one command to run the whole platform locally
```

---

## Evaluation & reproducing the numbers

The eval is a **referee, not a marketing tool**: we don't get to claim a win by assertion, we prove
it against a fixed, re-runnable yardstick — and where v2 still trails the frontier, we report that.
The full protocol, pass bar, and diagnosis tree are in
[`docs/EVAL_AND_ITERATE.md`](docs/EVAL_AND_ITERATE.md). All instruments are held-out and grounded
identically to the live app, and v2 is scored against a matched v1 baseline:

```bash
set -a && source .env && set +a          # loads keys (never printed)

# A) 5-model instructiveness benchmark (v2)     → RESULTS_BENCHMARK_v2.md
~/.venvs/mlx/bin/python scripts/run_benchmark_v2.py all --n 100

# B) v1↔v2 tier-differentiation + fabrication   → RESULTS_V2.md
~/.venvs/mlx/bin/python -m scripts.divergence_compare_v2 \
  --v1 data/analysis/divergence_v1_matched.jsonl \
  --v2 data/analysis/divergence_v2.jsonl \
  --out data/analysis/divergence_compare_v2.json

# C) frontier move-selection gap                → data/analysis/GAP_REPORT.md
~/.venvs/mlx/bin/python -m scripts.frontier_gap --num 50
~/.venvs/mlx/bin/python -m scripts.frontier_gap_report

# D) base-vs-tuned objective + Claude judge      → RESULTS_V2.md / RESULTS.md
~/.venvs/mlx/bin/python -m src.eval.evaluate --model tuned --tuned-path models/mlx/chess-coach-v2 \
  --num-scenarios 18 --positions data/positions/positions.jsonl \
  --compare-to data/eval/results_base_claude.json --out data/eval/results_tuned_v2_claude.json
```

Held-out & anti-leak invariants are non-negotiable: every eval FEN is verified absent from
`train.jsonl`/`valid.jsonl` by board + side-to-move key; grounding is identical across all models;
local decoding is greedy so tier differences are genuine conditioning, not sampling noise.

---

## Honest limitations (`v2` — current)

We report the weak spots as plainly as the wins:

1. **Explanation truthfulness improved but is still the hardest axis — and *ungrounded*, v2 is more
   brittle.** In the grounded (deployment) mode v2 cuts the fabrication rate **50% → 33%** (and the
   divergence-harness raw rate **46.1% → 31.7%**): the faithfulness gate removed fabricated labels
   (**0% false labels**) and grounded generation + a verify-and-retry loop cut source fabrication to
   1.6%. But **ungrounded**, v2 fabricates *more* than v1 (**87% → 99%** on the benchmark) — it now
   teaches more concretely (the explicit "how to find it" cites squares/captures), so without the
   engine facts a 1.7B invents more. The product always runs grounded, so this is the expected trade,
   but it's real and reported. Truth is still carried by **grounding + the non-LLM verifier**, not the
   weights; the verify-and-regenerate gate remains required in production.
2. **Tier-differentiated move-selection is now correctly directed — but still partial.** v2 raised
   tier-differentiation **27.5% → 39.2%** and *fixed the direction* (beginners get the more
   human-findable move, advanced the sharpest; "beginner move == the human/Maia move" 39% → 62%),
   after taking contrastive multi-tier pairs from **0% → 348 FENs × 3 tiers**. It's fixed in
   *direction*, not yet universal — the move still varies by tier on ~39% of positions, not all.
   See [`data/analysis/DIVERGENCE_REPORT.md`](data/analysis/DIVERGENCE_REPORT.md) / [`RESULTS_V2.md`](RESULTS_V2.md).
3. **The frontier still explains more instructively.** Even grounded, the blinded council ranks the
   big models above the 1.7B on instructiveness. v2 **narrows** the gap (council rank 4.13 → 3.68;
   gap to best frontier +2.22 → +1.60) but does not erase it. A bigger model still teaches better.
4. **The small model's real, defensible edge is form factor + register + now-improved faithfulness**,
   not raw coaching quality: ~$0 marginal cost, local, private, offline, a steady no-engine-speak
   voice with low variance, and — with the v2 gate — meaningfully fewer fabricated board facts when
   grounded.

---

## What v2 shipped

v2 is a **data intervention** (same prompt/format, so the v1→v2 comparison stays clean) that
targeted exactly the gaps above. All four levers landed:

1. **Faithfulness-filtered dataset** — a deterministic `python-chess` gate (`verify_text` in
   `src/filter/filter.py`) rejects any teacher candidate that references a piece/square/capture/tactic
   absent from the FEN, on top of grounded generation + a 1-retry loop. Source fabrication 6.3% →
   1.6% residual, then rejected → **0% false labels** (only 42 of 2,628 candidates dropped).
2. **Strongly tier-aware teacher move rule** (`src/teacher/tier_select.py`) — beginner → highest
   **Maia-1100** (most findable) sound move; intermediate → eval×policy blend; advanced → engine best.
   Always inside the sound pool; **37%** of picks are *not* the engine's #1.
3. **Contrastive multi-tier pairs** — from **0** to **348 FENs × 3 tiers** (same position taught per
   tier), directly supervising the model to vary the move by tier.
4. **Method clause** — the v2 target teaches the move, the *why*, **and how to find it** at that Elo
   (an explicit "How to find it: …" field).

Dataset grew 1,448 → **2,586** kept rows (train / valid **2,457 / 129**). Teacher generation cost
**$50.36** (GPT-5.5, 2,797 calls, 0 failures); training was ~$1 (Modal A10G, ~20 min). Against the
pass bar in [`docs/EVAL_AND_ITERATE.md`](docs/EVAL_AND_ITERATE.md), **v2 cleared soundness,
correctly-directed differentiation, and a real faithfulness *improvement*, but did not clear
"fabrication ≤ frontier" or "instructiveness ≥ frontier"** — so, as promised, the honest headline
stays the **form-factor + truthful-grounding win**, now with measurably better faithfulness and
correct tier-direction than v1.

---

## Compute

Data-gen and eval run locally (Mac, `~/.venvs/mlx`). Fine-tuning runs on a CUDA GPU (Modal / RunPod
/ TrueFoundry) via QLoRA — see `requirements-train.txt`. Inference (the live coach) runs locally in
4-bit MLX at ~$0 marginal cost.

## Data sourcing & licensing

Positions come from the CC0 Lichess Open Database (via the sampler / HF mirrors). Teacher-style
transcripts and any external commentary are **distilled to paraphrase and used internally only** —
the SFT dataset stays 100% synthetic. External datasets are always **re-grounded** through our own
Stockfish + Maia; external evals/solutions are context, never labels. See
[`docs/DATASET_PLAN.md`](docs/DATASET_PLAN.md) and [`docs/EXTERNAL_DATASETS.md`](docs/EXTERNAL_DATASETS.md).
