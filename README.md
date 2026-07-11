# chess-instructor-llm

A level-calibrated, engine-grounded chess coach. The SFT base is `chess-coach-32b-v4`, a
QLoRA fine-tune of Qwen3-32B (base `unsloth/Qwen3-32B-unsloth-bnb-4bit`) and the model behind the
full reproducible evaluation below; the live demo now serves `chess-coach-32b-v6-dpo2` (0.892 on the
corrected 120 TEST, a strict no-regression improvement over v4's 0.861), the best-DPO refinement of
v4. The genuinely small, on-spec model is the 1.7B tune. It is trained to do one thing: given a position and the student's
rating tier (Beginner / Intermediate / Advanced), emit the move that a fixed rule, `select_tier_move`,
designates as the canonical tier move, tagged with a short principle (for example, "Nf3, develop
toward the center").

That single move choice is the trained, evaluated behavior. The four-part English explanation is
secondary to the evaluation claim and is not separately optimized; note it is still in the SFT loss,
so it is not "not trained." Its faithfulness is enforced by a separate non-LLM verifier before
anything reaches a student.

## What is and is not validated (read this first)

1. LEARNABILITY (validated). The `select_tier_move` policy is distillable into weights: with identical
   Maia grounding on both sides, a fine-tune reproduces it where a prompt on the same base cannot. The
   small on-spec model leads: 1.7B tune, strict tier-policy match 0.358 -> 0.578, #2 of 20 and above
   every frontier (Gemini 0.553). 1.7B (0.578) beats 4B (0.397), so the behavior is from the data, not
   capacity. 32B v4 (0.767) is the strongest mid-size instance, not a small model.
2. DEPLOYMENT-NECESSITY (false as built). The same Stockfish + Maia grounding that feeds the model's
   prompt also feeds the ~20-line `select_tier_move` rule, which computes the canonical move directly
   at ~1.0 by construction (it IS the target). So the model approximates a policy the product already
   produces; the deterministic rule is the true ceiling. The model would be load-bearing only in a
   grounding-free, fully-local deployment, which we did not build or measure.
3. PEDAGOGY / VALUE (unvalidated). We optimized agreement with our own heuristic, not evidence coaches
   or students prefer these moves or improve. Behavior validated / feature demonstrated: NOT a product,
   NOT a moat, NOT value validated.

Metric: "tier-policy exact match" (agreement with the preregistered `select_tier_move` policy),
shortened to "tier-policy match." The canonical move is a project rule, not validated pedagogy.

## Canonical artifacts (v4)

- Model: [`khoilamalphaai/chess-coach-32b-v4-qlora`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v4-qlora) (Qwen3-32B QLoRA adapter)
- Dataset: [`khoilamalphaai/chess-coach-move-review`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review) (default config = v4)
- Grand eval: [`khoilamalphaai/chess-coach-grand-eval`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-grand-eval) (in-repo: [`RESULTS_HONEST_EVAL_V4.md`](RESULTS_HONEST_EVAL_V4.md), `data/benchmark_honest/report_v4.json`, `src/eval/`, `scripts/grand_eval.py`)
- Live demo (Space): [`khoilamalphaai/chess-coach-studio`](https://huggingface.co/spaces/khoilamalphaai/chess-coach-studio) (live: https://khoilamalphaai-chess-coach-studio.static.hf.space), now backed by the Modal endpoint `chess-coach-v6dpo2-4bit-maia` serving v6-dpo2 (a verbatim clone of the v4 serving app with only the LoRA adapter swapped; Maia-enabled, scale-to-zero, ~2.5-3 min cold start); the v4 endpoint `chess-coach-v4-4bit-maia` stays deployed as the fallback
- BrainLift (behavior thesis + evidence): [`BRAINLIFT.md`](BRAINLIFT.md)
- Local platform: The Analysis Room (FastAPI + Next.js), one command: `./run_platform.sh`

Stretch and refinement adapters (see the corrected-benchmark results below; v6-dpo2 is now the live-served model):

- Preference-tuned, best DPO (now live-served): [`khoilamalphaai/chess-coach-32b-v6-dpo2`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v6-dpo2) (v4 + stronger tier-targeted DPO, checkpoint step 200; overall tier-policy 0.892 on the corrected 120 TEST, supersedes v6-dpo and is now the live-served model on the demo, the best-DPO refinement of v4)
- Preference-tuned, earlier DPO: [`khoilamalphaai/chess-coach-32b-v6-dpo`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v6-dpo) (v4 + DPO on tier-move pairs)
- Engine-distilled (no-grounding): [`khoilamalphaai/chess-coach-32b-v6-distill`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v6-distill) (tier rule distilled into the weights)
- Deep-verified v6 labels (dataset): [`khoilamalphaai/chess-coach-v6`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-v6) (corrected canonical/sound labels, deeper Stockfish-17 + Syzygy, behind the v6 stretch results)
- Stage-4 corrected-benchmark results: [`RESULTS_STAGE4_CORRECTED.md`](RESULTS_STAGE4_CORRECTED.md); full-field corrected re-score: [`RESULTS_FULL_EVAL_803.md`](RESULTS_FULL_EVAL_803.md)

## Headline result (strict held-out eval)

The trained behavior is graded deterministically against the engine and a human-move model, with no
LLM judge in the loop. Metric = tier-policy exact match (agreement with `select_tier_move`). On 120
held-out positions x 3 tiers, with grounding held identical, fine-tuning reproduces the policy where
the base does not:

| Contender (identical grounding) | tier-policy match up | note |
|---|---:|---|
| `select_tier_move` deterministic rule | ~1.000 | true ceiling: it IS the target, computed from the same grounding without any model |
| 1.7B base -> 1.7B tuned (v2, on-spec) | 0.358 -> 0.578 | learnability lead: #2 of 20, above every frontier |
| 4B base / prompt-base / tuned | 0.353 / 0.378 / 0.397 | same-backend prompt control: tune > prompt > base |
| 32B base -> 32B tuned (v4, shipped) | 0.347 -> 0.767 | strongest instance; mid-size, not small |
| best frontier (Gemini 3.1 Pro) | 0.553 | #4 overall |

(v4 distinct-moves-per-level 0.730 = 73/100 canonical beginner!=advanced opportunities, vs base 0.290;
raw move soundness 0.942.)

- All-scenario lead (unbiased): across all 120 x 3 scenarios, v4 tier-policy match 0.767 vs the best
  frontier 0.553 (Gemini, #4). The tuned checkpoints take 4 of the top 5. This is the primary unbiased
  comparison.
- Unbiased head-to-head: over ALL positions where v4 diverges from the best frontier (not conditioned
  on v4 succeeding), v4 goes 56-24-12 over the 92 diverging positions (56-24-40 over all 120) on the
  moat (tier-policy match then soundness).
- Selection-conditioned subset (NOT a general win rate): within the v4-success/divergence subset (the
  62 of 120 positions where v4 already gives a distinct, sound, correctly-graded move AND diverges from
  the frontier), v4 wins 51-5 (6 ties). Conditioned on v4 succeeding, so it overstates a win rate; use
  the unbiased 56-24-12 above.
- The served move is the evaluated move: the prose gate changes only the explanation, never the move (it
  keeps the model's own greedy sound move on a prose failure and rewrites only the prose), so v4's served-move
  tier-policy match (0.789 replayed over the val drafts) matches its evaluated greedy 0.767.
- Deployment-necessity is false as built: the deterministic rule already computes the canonical move
  at ~1.0 from the same grounding, so the model approximates a policy the product produces without it.
- Move soundness equalizes to a shared ~100% floor once every model passes the shipped gate; it is a
  fairness floor, and soundness itself is fidelity to a shallow sound pool, not certified best-move
  truth.

Honest by design: v4 is deliberately weaker on prose. On the blinded, cross-family instructiveness
council it lands around 15th of 20 (grade about 4.5), below the smaller 4B tune and the prior 32B v3.
Prose is secondary to the evaluation claim and is not separately optimized (but it is in the SFT
loss). See [`RESULTS_HONEST_EVAL_V4.md`](RESULTS_HONEST_EVAL_V4.md).

Corrected-benchmark stretch results (Stage-4): the benchmark labels were rebuilt under deeper
Stockfish-17 search plus Syzygy (the 120 held-out test FENs are unchanged, only the canonical and
engine-best targets moved), and every tuned model was re-scored in one controlled run. On the corrected
grounded held-out benchmark, tier-policy match is base 0.428, v4 0.861, the preference-tuned v6-dpo 0.881,
and the stronger tier-targeted v6-dpo2 0.892 (checkpoint step 200), the best DPO result. v6-dpo2 is now the
live-served model on the demo (the best-DPO refinement of v4), but it is honestly a stronger v6-dpo, not a beginner or advanced breakthrough:
the whole gain is the intermediate tier (0.842 vs v4 0.750, vs v6-dpo 0.808), while beginner (0.858) and
advanced (0.975) are byte-identical to v4 and v6-dpo because both already sit at their ceiling under
grounding; soundness (0.983) and distinct-moves (0.987) are unchanged, and format (0.925) is marginally
under v4 (0.939), a token-cap prose-length artifact. Stripped of grounding, the untuned base collapses to
tier-policy 0.022 (names-a-move 0.250) while the engine-distilled adapter recovers it to 0.325 (names-a-move
0.983), a behavior-in-weights result with an honest advanced-tier limit (0.217). v4 is the SFT base and
the model behind the full reproducible evaluation; v6-dpo2 (the best-DPO refinement of v4) is now the
live-served model on the demo, and v6-dpo and v6-distill remain stretch-ladder results. A free cached re-score of the full field
against the corrected labels ([`RESULTS_FULL_EVAL_803.md`](RESULTS_FULL_EVAL_803.md)) keeps OURS on top of
the moat (OURS-v2 #1, +0.042 over the best frontier; tuned-over-base +0.151 / +0.162) and preserves the
cross-family order (OURS then frontier then open), while the frontier reshuffles internally so Claude Opus
4.8 now edges Gemini 3.1 Pro as the strongest single frontier coach. Scope caveat: these are
v4-era-grounding cached generations judged by the sharper v6 targets, so absolutes are lower than a
fresh-grounding eval, valid for the relative and ranking read, not each model's ceiling. Full detail:
[`RESULTS_STAGE4_CORRECTED.md`](RESULTS_STAGE4_CORRECTED.md).

Eval audited honest. Two independent audits back the base-vs-tuned comparison: the human-move model
(Maia) is present and symmetric across all 20 models (it feeds both the ground-truth tier move and
every model's grounding equally), and there is zero train/test leakage (board-key intersection 0 of
120 between the validation slice and v4's training data). The gate's zero-fabrication figure is zero
verifier-detectable mechanical violations, not certified truthfulness (see limitations).

---

## The gap (why this is worth building)

The pitch is not "a small open model plays better chess than GPT-5.5." It never will. The bet is
narrower and measurable:

- One specific behavior, tier-appropriate move selection under a fixed policy, is not reliably
  delivered by a prompted frontier model, and can be distilled into an open model's weights.

We proved the gap before claiming to fill it. With grounding held byte-identical to the app, the
frontier models are strong players with fluent prose but weak at the narrow behavior: they hand the
engine's single best move to every level, repeating one move across the three tiers about 77% of
the time regardless of the stated rating. The canonical failure is serving a 1200-rated beginner
the 3000-Elo engine-best move wrapped in a GM-level line: sound, but not findable and not
instructive for that student.

Learnability is the point, validated at 1.7B and 4B. Holding the same weights and only swapping the
system prompt, a fine-tune reaches the policy where an engineered prompt on the base does not, at
1.7B and 4B; at 1.7B the prompt actually hurt the behavior. Move selection under the rule has to be
added by data, not prompting. Honesty correction: the matched same-backend 32B prompt control was
never run, so 32B un-promptability is a falsifiable HYPOTHESIS (future work), not a result. The full
controlled experiment is in [`BRAINLIFT.md`](BRAINLIFT.md).

### Where the canonical move comes from (and why the model is not load-bearing as built)

Dependability in a coach like this is not carried by the model weights writing English. It is
carried by parts that sit outside the language model:

1. A strong engine (Stockfish) certifies which moves are sound (within a shallow pool; see limitations).
2. A human-move model (Maia) says which sound move a player at a given rating would actually find.
3. A tier rule (`select_tier_move`) turns those two signals into the single canonical move per level.
4. A non-LLM verifier checks detector-covered prose claims against the real board before it reaches
   the student.

Because that rule is a ~20-line pure function of (tier, sound pool, Maia policy), the product already
computes the canonical move directly, at ~1.0 by construction. The fine-tuned model is trained to
reproduce that same move: a genuine learnability result, but it means the model APPROXIMATES a target
the product already has. The model only becomes load-bearing in a grounding-free, fully-local
deployment (no engine or Maia at inference), which we did not build or measure. And the tune does not
internalize the rule: at inference it still needs Maia's per-tier grounding in the prompt, and without
it the three tiers collapse to one move. The honest claim is reliable grounded EXECUTION, not that the
behavior lives in the weights.

---

## Iteration history / training journey (v2 to v3 to v4 to v5)

The shipped v4 was reached through a documented sequence. The honesty is the point, including the
wrong turns.

- v2 (Qwen3-1.7B QLoRA): the original data intervention, faithfulness-filtered labels + a
  tier-aware teacher rule + contrastive multi-tier pairs. At 1.7B it fixed the direction of
  tier-differentiated move selection. In the 20-model grand eval it posts tier-policy match 0.578,
  #2 of 20 and above every frontier. This small, local, on-spec model is the learnability result.
- v3 (Qwen3-32B QLoRA): the all-rounder. It kept a strong balance of move and prose, landing about
  5th of 20 on the blinded prose council (instructiveness grade about 6.35) at tier-policy match 0.558.
- v4 (Qwen3-32B QLoRA): the shipped model. Trained to maximize the policy match, it leads the field
  (tier-policy match 0.767, distinct-moves 0.730, raw move-soundness 0.942); the unbiased head-to-head
  over the 92 diverging positions is 56-24-12 (the 51-5-6 over 62 is the v4-success-conditioned subset,
  not a win rate), while it trades prose down to about 15th of 20 (grade about 4.5). 32B is a mid-size
  extension, not a small model.
- v5 (Qwen3-32B QLoRA): an attempt to fix v4's prose and raw faithfulness with a cleaner, filtered
  dataset. It regressed (tier-policy match 0.536, move-soundness 0.828, prose about 3.9, faithfulness
  flat around 0.58) but the run was CONFOUNDED and does not prove filtering alone caused it: about 27%
  less optimization / token exposure, contrastive triads broken by row-wise filtering, about 42%
  boilerplate-principle pollution, retrained from base rather than from v4, and no checkpoint
  selection. So "cleaner data killed the tier-policy lead" is not an established lesson. v5 is not
  shipped.

v3, v4, and v5 use the same low-rank QLoRA recipe on the same base
(`unsloth/Qwen3-32B-unsloth-bnb-4bit`), differing essentially only in the data. Going to 32B was a
deliberate quality push, not a retreat from the small-model thesis: the on-spec, defensible form
factor is the small 1.7B/4B local model (the honest floor of the claim), while the 32B v4 is the
strongest mid-size instance of the behavior.

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
talking to a Modal endpoint, `chess-coach-v6dpo2-4bit-maia`, that serves the v6-dpo2 adapter (the
best-DPO refinement of v4) on the same 4-bit base with Maia enabled and greedy-first decoding. It is
a verbatim clone of the v4 serving app with only the LoRA adapter swapped; the v4 endpoint
`chess-coach-v4-4bit-maia` stays deployed as the fallback. The full reproducible evaluation below was
run on v4 (the SFT base). The endpoint is scale-to-zero, so the first request after idle has a
~2.5-3 min cold start.

The same behavior runs locally as "The Analysis Room": a thin FastAPI backend wires the repo's
existing pieces to a calm, board-centric Next.js front end. It re-implements no chess logic:

- Stockfish supplies the sound-move pool and how bad the student's move was.
- Maia supplies which sound moves a player at the chosen tier would actually consider. Maia is NOT
  optional for the behavior: without it the per-tier signal disappears and the three tiers collapse to
  one move (there is no graceful degradation here). If lc0 / the weights are missing the API still
  returns an answer, but tier differentiation is lost.
- `config/schema.py` assembles those facts into the exact `TeacherInput` prompt text the model was
  trained on (`render_user_prompt`). The tuned model needs this per-tier Maia grounding at inference;
  it does not internalize the rule.
- `src/engine/position_facts.py` prepends a VERIFIED FACTS block (the exact pieces on the board,
  which are loose, what each candidate move concretely does) so the model explains from truth.
- `src/engine/faithfulness.py` (the verifier) is the verify-and-regenerate gate: after the model
  writes a reply, every detector-covered board claim is checked against the real position; if any is
  false the whole answer is re-sampled (never sentence-stripped) up to a small budget. If none verify,
  the API emits a deterministic, engine-derived explanation that is truthful by construction. This is
  the inference-time defense for the optional prose layer, running in production today. It is
  high-precision but low-recall: semantic falsehoods the detectors do not cover (relational pawn-SAN
  claims, forks, threats, negations, eval claims) can still pass, so it is not certified truthfulness.

Two-surface honesty. The curated showcase is the canonical, deterministic proof of the tier-policy
behavior: it is generated locally with the full Maia grounding, so it differentiates cleanly by tier.
The live tool is the interactive differentiator that demonstrates the behavior end to end, but it is
not guaranteed to be move-for-move identical to the curated showcase. (Maia was initially missing on
the serving container, which collapsed the tiers to one move; adding Maia + greedy-first decoding
restored per-tier differentiation on the live coach. That collapse is exactly why "the behavior lives
in the weights" is false: the model needs Maia's grounding at inference.)

---

## Quickstart (local platform)

```bash
cd chess-instructor-llm
./run_platform.sh
```

This starts the FastAPI backend and the Next.js front end, then waits (Ctrl-C stops both). Open
http://localhost:3000. For the hosted coach with no local setup (the live Space now serves v6-dpo2),
use it instead: https://khoilamalphaai-chess-coach-studio.static.hf.space.

Prerequisites:

- A Python env with `mlx_lm` (Apple Silicon) or the CUDA path, plus `python-chess`, `fastapi`,
  `uvicorn`.
- Stockfish (`/opt/homebrew/bin/stockfish` by default; override with `STOCKFISH_PATH`): required.
- lc0 + Maia nets in `models/maia/`: without them the coach still returns an answer, but the
  human-likelihood panel shows "unavailable" and the three tiers collapse to a single move (the
  trained behavior is lost). Maia is required for tier differentiation, not optional.
- Node 18.18+ and `npm install` in `web/` (first run only).

Overrides (all optional): `COACH_MODEL_PATH`, `COACH_ADAPTER_PATH`, `API_PORT`, `WEB_PORT`, `PY`.
Secrets live only in `./.env` and are read at call time, never printed.

---

## Repo layout

```
config/     tiers, engine tolerances, Maia mapping, an older prose-full BEHAVIOR_SPEC (pending update; the canonical one-behavior spec is in BRAINLIFT.md), schema/rendering
data/       raw inputs + derived training sets (positions/transcripts/generated/dataset/bank/curate) are gitignored;
            the benchmark + eval artifacts (benchmark_*/, eval/, showcase/, analysis/ — 250+ tracked files) ARE
            committed, including the v4 headline inputs (ours_v4 gens, gap803 scenarios, val_ids, report_v4.json)
prompts/    coach_system.md (the spec), principles.md + fewshots.json (distilled style), tier_guides, rubric
src/engine  Stockfish + Maia wrappers, position_facts (grounding), faithfulness (the verifier)
src/ingest  Lichess sampler, YouTube transcript harvester
src/teacher GPT-5.5 generation + principle distillation + tier selection + the coach gate
src/train   split_data + Modal QLoRA trainers
src/eval    base-vs-tuned harness (evaluate.py), the blinded council (benchmark/), honest gated eval (honest/)
scripts/    reproduce_v4.py (clean-clone headline re-score, no GPU/network), grand_eval.py (20-model leaderboard), honest_v4.py (v4 regression + selection-conditioned head-to-head)
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

### Reproduce the headline (no GPU, no network)

The v4 headline re-scores from committed files alone — one command, only `python-chess` required:

```bash
python -m scripts.reproduce_v4
```

This is **Level 1 (artifact re-score)**: it re-extracts v4's recommended move from the published
generations (`data/benchmark_honest/gen/ours_v4.jsonl`) with the same strict any-legal extractor the
report uses (`src/eval/evaluate.py::extract_recommended_move` ->
`src/teacher/coach_gate.py::pick_recommendation`), scores them against the committed ground truth
(`data/benchmark_gap803/scenarios.jsonl` + the 120 position ids in
`data/benchmark_honest/val_ids.txt`), and ASSERTS the numbers equal
`data/benchmark_honest/report_v4.json`: tier-policy match **0.7667**, distinct-moves **0.7300**
(73/100 canonical beginner!=advanced opportunities), raw move-soundness **0.9417**. It re-scores
published generations against committed ground truth; it does
NOT re-derive the ground truth (engine / Maia / `select_tier_move`) and does NOT re-run inference.

Three levels of reproduction, labeled by what each actually re-runs:

| Level | What it re-runs | How | Status |
|---|---|---|:--|
| **1 — artifact re-score** | re-scores the published v4 generations against committed ground truth; asserts 0.7667 / 0.7300 / 0.9417 | `python -m scripts.reproduce_v4` | **supported today** — no GPU, no network, no engine |
| **2 — retrain from final dataset** | re-trains v4 from the shipped dataset (new checkpoint; decoding + hardware variance) | `src/train` QLoRA on `chess-coach-move-review` (v4 config) | **approximate** — needs a CUDA GPU |
| **3 — full teacher / data regen** | regenerates ground truth + dataset from scratch (Stockfish + Maia + GPT-5.5 teacher + filter), then Level 2 | the offline data pipeline | **not bit-exact** — engine / Maia / teacher nondeterminism; needs GPU + API keys + engines |

### Deeper re-scores (require the full generation field present locally)

```bash
# strict v4 regression verdict + selection-conditioned head-to-head -> RESULTS_HONEST_EVAL_V4.md + data/benchmark_honest/report_v4.json
python -m scripts.honest_v4 report

# full 20-model leaderboard (deterministic tier-policy match + selection-conditioned head-to-head + blinded council) -> data/benchmark_grand/GRAND_EVAL_LEADERBOARD.md
python -m scripts.grand_eval report
```

These two re-score the whole field and need every model's generations present (including the 4B trio
and the frontier council); `scripts.honest_v4` will error on a clean clone that lacks the 4B gens. For
a bare clean clone, use `scripts.reproduce_v4` above — it needs only the committed v4 artifacts.

Held-out and anti-leak invariants are non-negotiable: every eval FEN is verified absent from the
training set by board + side-to-move key (0 of 120 leakage), grounding is identical across all
models, Maia is symmetric across the field, and local decoding is greedy so tier differences are
genuine conditioning, not sampling noise. Re-scoring the published generations reproduces tier-policy
match 0.767 and distinct-moves 0.730 exactly.

---

## Honest limitations (v4)

Reported as plainly as the wins:

1. Deployment-necessity is false as built. `select_tier_move` computes the canonical move directly
   from the same grounding at ~1.0 by construction, so the tuned model approximates a policy the
   product already produces. The model is load-bearing only grounding-free and fully local, which we
   did not build or measure. The deterministic rule is the true ceiling baseline.
2. The tune does not internalize the rule. At inference it needs Maia's per-tier grounding in the
   prompt; without it the tiers collapse to one move. Honest claim: reliable grounded EXECUTION, not
   "the behavior lives in the weights." No graceful degradation; it degrades to collapse.
3. Faithfulness is not guaranteed truthfulness. About 40% of v4's raw drafts trip the prose check;
   the gate drives verifier-DETECTABLE mechanical violations to zero. But the checker is
   high-precision / low-recall: relational pawn-SAN claims, forks, threats, negations, and eval claims
   can still be wrong and reach a user, and a cross-family LLM-judge residual exists. We do not claim
   guaranteed truthfulness.
4. Tier-policy match is fidelity to a heuristic, not certified best teaching. "Advanced = engine best"
   is contradicted in the data (advanced maximizes stored cp and diverges from the persisted
   engine_best on ~43/803), and the sound pool is a shallow 300ms / MultiPV-8 / 150cp search that can
   include position-worsening moves. v6 roadmap: deeper pool + tablebases, Maia-as-constraint,
   titled-coach validation.
5. Pedagogy is unvalidated. We measured agreement with our own move rule, not whether coaches or
   students prefer these moves or learn faster. Behavior validated, value not validated.
6. Prose is weaker and secondary, not "not trained." v4 lands about 15th of 20 on the blinded council
   (grade about 4.5). Prose is secondary to the evaluation claim and is not separately optimized, but
   it IS in the SFT loss. A product that wants rich prose renders it on top of the chosen move and
   verifies it separately.
7. v5 did not prove clean data kills the lead. The v5 regression was confounded (see iteration
   history); it does not isolate filtering as the cause.
8. Live vs curated showcase. The curated showcase is the canonical deterministic proof; the live tool
   differentiates by tier but is not guaranteed move-for-move identical to the showcase.

---

## Compute

Data-gen and eval run locally (Mac, `~/.venvs/mlx`) plus the TrueFoundry gateway for the frontier
council. Fine-tuning runs on a CUDA GPU (Modal) via QLoRA on the 4-bit Qwen3-32B base. Live
inference is served on Modal in 4-bit (`chess-coach-v6dpo2-4bit-maia`, scale-to-zero, with the v4
endpoint `chess-coach-v4-4bit-maia` kept as the fallback); the same coach runs locally in MLX.

## Data sourcing & licensing

Positions come from the CC0 Lichess Open Database (via the sampler / HF mirrors). Teacher-style
transcripts and any external commentary are distilled to paraphrase and used internally only; the
SFT dataset stays 100% synthetic. External datasets are always re-grounded through our own
Stockfish + Maia; external evals/solutions are context, never labels. See
[`docs/DATASET_PLAN.md`](docs/DATASET_PLAN.md) and [`docs/EXTERNAL_DATASETS.md`](docs/EXTERNAL_DATASETS.md).
