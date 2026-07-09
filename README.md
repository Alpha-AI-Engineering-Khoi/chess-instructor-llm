# chess-instructor-llm

A level-calibrated, engine-grounded chess coach. The shipped model is `chess-coach-32b-v4`, a
QLoRA fine-tune of Qwen3-32B (base `unsloth/Qwen3-32B-unsloth-bnb-4bit`) trained to do one thing
reliably: given a position and the student's rating tier (Beginner / Intermediate / Advanced),
select the tier-appropriate instructive move and tag it with a short principle (for example,
"Nf3, develop toward the center").

That single move choice is the trained, graded behavior. The four-part English explanation is a
secondary, optional display layer: it can be rendered by the engine's own templates, a
detector-driven writer, or a prompted frontier model, and its faithfulness is enforced by a
separate non-LLM verifier before anything reaches a student. The point is one dependable behavior
from data, not out-teaching a frontier model and not winning a prose contest.

## Canonical artifacts (v4)

- Model: [`khoilamalphaai/chess-coach-32b-v4-qlora`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v4-qlora) (Qwen3-32B QLoRA adapter)
- Dataset: [`khoilamalphaai/chess-coach-move-review`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review) (default config = v4)
- Grand eval: [`khoilamalphaai/chess-coach-grand-eval`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-grand-eval) (in-repo: [`RESULTS_HONEST_EVAL_V4.md`](RESULTS_HONEST_EVAL_V4.md), `data/benchmark_honest/report_v4.json`, `src/eval/`, `scripts/grand_eval.py`)
- Live demo (Space): [`khoilamalphaai/chess-coach-studio`](https://huggingface.co/spaces/khoilamalphaai/chess-coach-studio) (live: https://khoilamalphaai-chess-coach-studio.static.hf.space), backed by the Modal endpoint `chess-coach-v4-4bit-maia` (Maia-enabled, scale-to-zero, ~2.5-3 min cold start)
- BrainLift (behavior thesis + evidence): [`BRAINLIFT.md`](BRAINLIFT.md)
- Local platform: The Analysis Room (FastAPI + Next.js), one command: `./run_platform.sh`

## Headline result (strict held-out eval)

The trained behavior is graded deterministically against the engine and a human-move model, with
no LLM judge in the loop. On 120 held-out positions x 3 tiers, fine-tuning is the whole difference
on the one graded axis:

| Deterministic axis (held-out, no judge) | Qwen3-32B base | OURS-v4 (tuned) |
|---|---:|---:|
| Tier-fit (picks the tier-appropriate move) up | 0.347 | 0.767 |
| Distinct move per level up | 0.290 | 0.785 |
| Move soundness (raw draft) up | 1.000 | 0.942 |

- v4 has the top tier-fit of all 20 models measured. The tuned checkpoints take 4 of the top 5;
  the best frontier model (Gemini 3.1 Pro, tier-fit 0.553) is #4.
- On the 62 held-out positions where v4 diverges from the best frontier's move, v4 wins the
  tier-appropriate move 51-5 (6 ties) on the platform's move-quality moat (tier-fit then soundness).
- Move soundness is a shared fairness floor, not a differentiator: the shipped
  verify-and-regenerate gate lifts every model, including v4, to ~100% move-sound with zero
  user-visible fabrication.

Honest by design: v4 is deliberately weaker on prose. On the blinded, cross-family instructiveness
council it lands around 15th of 20 (grade about 4.5), below the smaller 4B tune and the prior 32B
v3. Prose is the optional, gated display layer, not the trained behavior, so this trade is
on-thesis. See [`RESULTS_HONEST_EVAL_V4.md`](RESULTS_HONEST_EVAL_V4.md).

Eval validated honest. Two independent audits back the headline: the human-move model (Maia) is
present and symmetric across all 20 models (it feeds both the ground-truth tier move and every
model's grounding equally), and there is zero train/test leakage (board-key intersection 0 of 120
between the validation slice and v4's training data).

---

## The gap (why this is worth building)

The pitch is not "a small open model plays better chess than GPT-5.5." It never will. The bet is
narrower and measurable:

- One specific behavior, tier-appropriate move selection, is not reliably delivered by a prompted
  frontier model, and can be trained into an open model to run reliably.

We proved the gap before claiming to fill it. With grounding held byte-identical to the app, the
frontier models are strong players with fluent prose but weak at the narrow behavior: they hand the
engine's single best move to every level, repeating one move across the three tiers about 77% of
the time regardless of the stated rating. The canonical failure is serving a 1200-rated beginner
the 3000-Elo engine-best move wrapped in a GM-level line: sound, but not findable and not
instructive for that student.

The un-promptable part is the point. Holding the same weights and only swapping the system prompt,
a carefully engineered prompt on the base does not reach the tuned model's tier-fit at 1.7B, 4B, or
32B, and at 1.7B it actually hurt the behavior. Move selection has to be added by data, not
prompting. The full controlled experiment at three model sizes is in [`BRAINLIFT.md`](BRAINLIFT.md).

### Where dependability actually comes from

Dependability in a coach like this is not carried by the model weights writing English. It is
carried by parts that sit outside the language model:

1. A strong engine (Stockfish) certifies which moves are sound.
2. A human-move model (Maia) says which sound move a player at a given rating would actually find.
3. A tier rule turns those two signals into the single canonical move per level.
4. A non-LLM verifier checks every prose claim against the real board before it reaches the student.

The fine-tuned model's job is to emit that tier-appropriate move reliably and locally, which a
prompt on the same weights does not do. Prose, if the product wants it, is rendered and separately
verified on top.

---

## Iteration history / training journey (v2 to v3 to v4 to v5)

The shipped v4 was reached through a documented sequence. The honesty is the point, including the
wrong turns.

- v2 (Qwen3-1.7B QLoRA): the original data intervention, faithfulness-filtered labels + a
  tier-aware teacher rule + contrastive multi-tier pairs. At 1.7B it fixed the direction of
  tier-differentiated move selection and improved explanation faithfulness. In the 20-model grand
  eval it still posts tier-fit 0.578, second among the tuned checkpoints. This established that the
  behavior is trainable into a small model.
- v3 (Qwen3-32B QLoRA): the all-rounder. It kept a strong balance of move and prose, landing about
  5th of 20 on the blinded prose council (instructiveness grade about 6.35) at tier-fit 0.558.
- v4 (Qwen3-32B QLoRA): the shipped model. Trained to own the moat, it leads the field on
  tier-appropriate move selection (tier-fit 0.767, distinct-moves 0.785, move-soundness 0.942) and
  wins the head-to-head 51-5 (6 ties) over the 62 diverging positions, while deliberately trading
  prose down to about 15th of 20 (grade about 4.5).
- v5 (Qwen3-32B QLoRA): the attempt to keep the moat while fixing v4's prose and raw faithfulness
  with a cleaner, filtered dataset. It backfired: tier-fit fell to 0.536, move-soundness to 0.828,
  prose to about 3.9 (near the bottom of the field), and faithfulness stayed flat around 0.58. The
  lesson is concrete: the moat's signal is the density of contrastive multi-tier examples, and
  aggressively cleaning the data thinned that signal. v5 is not shipped.

v3, v4, and v5 use the same low-rank QLoRA recipe on the same base
(`unsloth/Qwen3-32B-unsloth-bnb-4bit`), differing essentially only in the data. Going to 32B was a
deliberate quality push toward a near-frontier coach on the one trained axis, not a retreat from
the small-model thesis: the on-spec, defensible form factor is still a small (~4B) local model,
which is the honest floor of the claim, while the 32B v4 is the strongest instance of the behavior.

---

## Architecture

### Data pipeline (offline, produces the training set)

```
Lichess positions -> Stockfish (sound pool + mistake magnitude) -> Maia (human likelihood by tier)
   -> GPT-5.5 teacher (max reasoning, grounded + tier-aware move rule: pick the teaching move,
      the why, AND how to find it + leveled coaching)
   -> hard filter (soundness . no-engine-speak . ply-cap . faithfulness gate)
   -> contrastive multi-tier SFT set -> QLoRA (Qwen3-32B) -> deterministic base-vs-tuned eval
```

Locked design decisions:

- Engine as guardrail, not dictator. Stockfish supplies the sound-move pool (within ~150cp of
  best, never a blunder >=250cp) plus mistake magnitude; it does not pick the lesson.
- Teaching move is not the engine's #1. From all sound moves, pick the one with the most
  extractable lesson for the tier: sometimes #1, sometimes #5.
- Maia (human-at-rating) ranks candidate moves by "would a human at this tier even play this?",
  filtering superhuman-only moves. Used to define the canonical tier move, not as a training target
  the model sees directly.
- Teacher = GPT-5.5 (max reasoning), grounded in engine analysis (explains, never invents). Prose
  is judged by a different model family (Claude): no grading your own homework.
- YouTube transcripts (Naroditsky, GothamChess) = pedagogy reference, distilled once into
  principles + few-shots baked into the teacher prompt. Internal use only; the dataset stays 100%
  synthetic.
- Task: move review. Tiers: Beginner 1000-1200 / Intermediate 1300-1600 / Advanced 1700-2000.
- Fix disappointing models in DATA, not hyperparameters.

### Serving the coach

The shipped live demo is the Hugging Face Space `chess-coach-studio` (a static Next.js export)
talking to a Modal endpoint, `chess-coach-v4-4bit-maia`, that serves the v4 adapter on the 4-bit
base with Maia enabled and greedy-first decoding. The endpoint is scale-to-zero, so the first
request after idle has a ~2.5-3 min cold start.

The same behavior runs locally as "The Analysis Room": a thin FastAPI backend wires the repo's
existing pieces to a calm, board-centric Next.js front end. It re-implements no chess logic:

- Stockfish supplies the sound-move pool and how bad the student's move was.
- Maia supplies which sound moves a player at the chosen tier would actually consider (best-effort;
  the API degrades gracefully if lc0 / the weights are missing).
- `config/schema.py` assembles those facts into the exact `TeacherInput` prompt text the model was
  trained on (`render_user_prompt`).
- `src/engine/position_facts.py` prepends a VERIFIED FACTS block (the exact pieces on the board,
  which are loose, what each candidate move concretely does) so the model explains from truth.
- `src/engine/faithfulness.py` (the verifier) is the verify-and-regenerate gate: after the model
  writes a reply, every board claim is checked against the real position; if any is false the whole
  answer is re-sampled (never sentence-stripped) up to a small budget. If none verify, the API
  emits a deterministic, engine-derived explanation that is truthful by construction. This is the
  inference-time defense for the optional prose layer, running in production today.

Two-surface honesty. The curated showcase is the canonical, deterministic proof of the moat: it is
generated locally with the full Maia grounding, so it differentiates cleanly by tier. The live tool
is the interactive differentiator that demonstrates the behavior end to end, but it is not
guaranteed to be move-for-move identical to the curated showcase. (Maia was initially missing on
the serving container, which collapsed the tiers to one move; adding Maia + greedy-first decoding
restored per-tier differentiation on the live coach.)

---

## Quickstart (local platform)

```bash
cd chess-instructor-llm
./run_platform.sh
```

This starts the FastAPI backend and the Next.js front end, then waits (Ctrl-C stops both). Open
http://localhost:3000. For the hosted v4 coach with no local setup, use the live Space instead:
https://khoilamalphaai-chess-coach-studio.static.hf.space.

Prerequisites:

- A Python env with `mlx_lm` (Apple Silicon) or the CUDA path, plus `python-chess`, `fastapi`,
  `uvicorn`.
- Stockfish (`/opt/homebrew/bin/stockfish` by default; override with `STOCKFISH_PATH`): required.
- lc0 + Maia nets in `models/maia/`: optional; without them the coach still runs and the
  human-likelihood panel shows "unavailable" (and tiers may not differentiate).
- Node 18.18+ and `npm install` in `web/` (first run only).

Overrides (all optional): `COACH_MODEL_PATH`, `COACH_ADAPTER_PATH`, `API_PORT`, `WEB_PORT`, `PY`.
Secrets live only in `./.env` and are read at call time, never printed.

---

## Repo layout

```
config/     tiers, engine tolerances, Maia mapping, the BEHAVIOR_SPEC (the one gate), schema/rendering
data/       positions / transcripts / generated / dataset / analysis / benchmark / eval (gitignored)
prompts/    coach_system.md (the spec), principles.md + fewshots.json (distilled style), tier_guides, rubric
src/engine  Stockfish + Maia wrappers, position_facts (grounding), faithfulness (the verifier)
src/ingest  Lichess sampler, YouTube transcript harvester
src/teacher GPT-5.5 generation + principle distillation + tier selection + the coach gate
src/train   split_data + Modal QLoRA trainers
src/eval    base-vs-tuned harness (evaluate.py), the blinded council (benchmark/), honest gated eval (honest/)
scripts/    grand_eval.py (20-model leaderboard), honest_v4.py (v4 regression + moat proof)
src/api     FastAPI backend (server.py): the platform's thin HTTP layer
web/        Next.js 16 + Tailwind v4 + HeroUI v3 + react-chessboard front end
run_platform.sh  one command to run the whole platform locally
```

---

## Evaluation & reproducing the numbers

The eval is a referee, not a marketing tool. The core behavior is scored deterministically against
the engine and Maia, with no model judge in the loop, because the deliverable is a move and a move
has a checkable right answer per tier. Instructiveness of the optional prose layer is a separate,
held-out, cross-family council the model never trains against.

```bash
# strict v4 regression verdict + vs-frontier moat proof -> RESULTS_HONEST_EVAL_V4.md + data/benchmark_honest/report_v4.json
python -m scripts.honest_v4 report

# full 20-model leaderboard (deterministic moat + blinded council) -> data/benchmark_grand/GRAND_EVAL_LEADERBOARD.md
python -m scripts.grand_eval report
```

Held-out and anti-leak invariants are non-negotiable: every eval FEN is verified absent from the
training set by board + side-to-move key (0 of 120 leakage), grounding is identical across all
models, Maia is symmetric across the field, and local decoding is greedy so tier differences are
genuine conditioning, not sampling noise. Re-scoring the published generations reproduces tier-fit
0.767 and distinct-moves 0.785 exactly.

---

## Honest limitations (v4)

Reported as plainly as the wins:

1. Prose is weaker by design. v4 is the strongest model on the trained move axis and about 15th of
   20 on the blinded instructiveness council (grade about 4.5), below the 4B tune and v3. This is
   on-thesis: prose is the optional display layer, not the graded behavior, and a product that wants
   rich prose renders it on top of the tuned move (engine templates or a frontier model) and
   verifies it separately.
2. Truth is carried by grounding + the verifier, not the weights. About 40% of v4's raw drafts trip
   the prose faithfulness check before the gate; the shipped verify-and-regenerate gate drives
   user-visible fabrication to zero. The core move claim cannot fabricate a board fact, which is why
   narrowing the graded behavior to the move makes faithfulness free for the core deliverable.
3. Live vs curated showcase. The curated showcase is the canonical deterministic proof; the live
   tool differentiates by tier but is not guaranteed to be move-for-move identical to the showcase.
4. Size vs form factor. The on-spec, defensible form factor is a small (~4B) local model, and that
   stays the honest floor of the claim. The 32B v4 is a deliberate quality push to the strongest
   instance of the behavior, not a move to the local form factor.

---

## Compute

Data-gen and eval run locally (Mac, `~/.venvs/mlx`) plus the TrueFoundry gateway for the frontier
council. Fine-tuning runs on a CUDA GPU (Modal) via QLoRA on the 4-bit Qwen3-32B base. Live
inference is served on Modal in 4-bit (`chess-coach-v4-4bit-maia`, scale-to-zero); the same coach
runs locally in MLX.

## Data sourcing & licensing

Positions come from the CC0 Lichess Open Database (via the sampler / HF mirrors). Teacher-style
transcripts and any external commentary are distilled to paraphrase and used internally only; the
SFT dataset stays 100% synthetic. External datasets are always re-grounded through our own
Stockfish + Maia; external evals/solutions are context, never labels. See
[`docs/DATASET_PLAN.md`](docs/DATASET_PLAN.md) and [`docs/EXTERNAL_DATASETS.md`](docs/EXTERNAL_DATASETS.md).
