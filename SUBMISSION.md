# Submission: chess-instructor-llm

Project: Train Your Own Small Learning Model. A level-calibrated chess coach fine-tuned to reliably
select the tier-appropriate instructive move, end to end (dataset -> model -> platform -> eval ->
thesis -> demo). The shipped model is v4, a QLoRA fine-tune of Qwen3-32B (base
`unsloth/Qwen3-32B-unsloth-bnb-4bit`).

The one trained behavior: given a position and the student's rating tier (Beginner / Intermediate /
Advanced), select the tier-appropriate, sound, instructive move and tag it with a short principle.
The English explanation is a secondary, gated display layer, not the graded claim.

Win condition (from the brief): the tuned model beats the base model on the trained behavior,
graded deterministically. Met: tier-appropriate move selection (tier-fit) lifts from 0.347 on the
Qwen3-32B base to 0.767 on v4 on the strict held-out eval, the top tier-fit of all 20 models
measured.

---

## Canonical deliverables map (v4)

| # | Deliverable | Artifact: path / URL |
|---|---|---|
| 1 | Dataset (published on HF Hub) | [`datasets/khoilamalphaai/chess-coach-move-review`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review), default config = v4: the engine-grounded, contrastive multi-tier SFT set built by `positions -> Stockfish -> Maia -> GPT-5.5 (tier-aware) -> hard filter + faithfulness gate` |
| 2 | Fine-tuned model (published on HF Hub) | [`khoilamalphaai/chess-coach-32b-v4-qlora`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v4-qlora): QLoRA adapter on the 4-bit Qwen3-32B base |
| 2b | Running demo | Live Space: [`spaces/khoilamalphaai/chess-coach-studio`](https://huggingface.co/spaces/khoilamalphaai/chess-coach-studio) (https://khoilamalphaai-chess-coach-studio.static.hf.space), backed by the Modal endpoint `chess-coach-v4-4bit-maia` (Maia-enabled, scale-to-zero, ~2.5-3 min cold start). Also local: The Analysis Room, `./run_platform.sh` |
| 3 | Eval harness | `src/eval/` (base-vs-tuned `evaluate.py` · blinded council `benchmark/` · honest gated `honest/`) · `scripts/honest_v4.py` (v4 regression + moat proof) · `scripts/grand_eval.py` (20-model leaderboard). Protocol + pass bar: [`docs/EVAL_AND_ITERATE.md`](docs/EVAL_AND_ITERATE.md) |
| 3b | Base-vs-tuned results | [`RESULTS_HONEST_EVAL_V4.md`](RESULTS_HONEST_EVAL_V4.md) + `data/benchmark_honest/report_v4.json` (strict, deterministic) · [`data/benchmark_grand/GRAND_EVAL_LEADERBOARD.md`](data/benchmark_grand/GRAND_EVAL_LEADERBOARD.md) (20-model field) |
| 3c | Grand eval (published on HF Hub) | [`datasets/khoilamalphaai/chess-coach-grand-eval`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-grand-eval): all 20 models on the same held-out slice, deterministic moat + blinded council with 95% CIs |
| 4 | BrainLift (behavior thesis + evidence) | [`BRAINLIFT.md`](BRAINLIFT.md): the one-behavior thesis, the 32B training story (v2 -> v3 -> v4 -> v5), DOK-4 spiky POVs, all tied to primary sources or the project's own measurement |
| 5 | Demo video (3-5 min) | Script + shot list: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md). Runnable demo provided (live Space + `./run_platform.sh`); recording is the user's step |

---

## Win-condition scorecard (v4, strict held-out eval)

Deterministic, no LLM judge in the loop (120 held-out positions x 3 tiers). From
[`RESULTS_HONEST_EVAL_V4.md`](RESULTS_HONEST_EVAL_V4.md):

| Behavior (brief's win condition, v4 framing) | Qwen3-32B base | OURS-v4 (tuned) | Verdict |
|---|---:|---:|:---:|
| Level-calibration = tier-appropriate move (tier-fit up) | 0.347 | 0.767 | WIN (top of all 20 models) |
| Distinct move per level (up) | 0.290 | 0.785 | WIN |
| Move soundness (up) | 1.000 | 0.942 raw / ~1.0 gated | Shared gate floor (fairness, not differentiator) |
| No-engine-speak (up) | 0.992 | 0.978 raw / ~1.0 gated | Near-ceiling on the 32B base at this size |

- v4 has the top tier-fit of the 20-model field; the tuned checkpoints take 4 of the top 5, and the
  best frontier model (Gemini 3.1 Pro, 0.553) is #4.
- Moat: on the 62 held-out positions where v4 diverges from the best frontier's move, v4 wins the
  tier-appropriate move 51-5 (6 ties).
- Prompting cannot buy it: on the same weights, an engineered prompt on the base does not reach the
  tuned tier-fit at 1.7B, 4B, or 32B (full controlled table in [`BRAINLIFT.md`](BRAINLIFT.md)).

At the small 1.7B form factor (v2), no-engine-speak was the win the fine-tune had to earn (base 0.33
-> tuned 1.00). At 32B the base already writes clean, well-formed, no-jargon prose, so v4's entire
value-add is the tier-appropriate move: the fine-tune's differentiator at this size is the move, and
soundness / no-engine-speak equalize to a shared ~100% floor once through the shipped gate.

---

## The honest gaps (so the submission is not oversold)

- Prose is weaker by design. v4 lands about 15th of 20 on the blinded instructiveness council (grade
  about 4.5), below the 4B tune and the prior 32B v3. Prose is the optional, gated display layer,
  not the trained behavior; a product that wants rich prose renders it on top of the tuned move and
  verifies it separately.
- Truth is carried by grounding + a non-LLM verifier, not the weights. About 40% of v4's raw drafts
  trip the prose faithfulness check; the shipped verify-and-regenerate gate drives user-visible
  fabrication to zero. The core move claim cannot fabricate a board fact.
- Live vs curated showcase. The curated showcase is the canonical deterministic proof of the moat;
  the live tool differentiates by tier but is not guaranteed to be move-for-move identical to the
  showcase.
- Size vs form factor. The on-spec, defensible form factor is a small (~4B) local model, the honest
  floor of the claim. The 32B v4 is a deliberate quality push to the strongest instance of the
  behavior.

---

## Eval integrity

Two independent audits back the headline: Maia (the human-move model) is present and symmetric
across all 20 models, feeding both the ground-truth tier move and every model's grounding equally;
and there is zero train/test leakage (board-key intersection 0 of 120 between the val slice and v4's
training data).

---

## Reproduce

```bash
cd chess-instructor-llm
python -m scripts.honest_v4 report     # -> RESULTS_HONEST_EVAL_V4.md + data/benchmark_honest/report_v4.json
python -m scripts.grand_eval report    # -> data/benchmark_grand/GRAND_EVAL_LEADERBOARD.md
./run_platform.sh                      # local Analysis Room, or use the live Space
```

All eval FENs are verified held-out (absent from the training set by board + side-to-move key, 0 of
120); grounding is identical across every model; Maia is symmetric; local decoding is greedy.
Re-scoring the published generations reproduces tier-fit 0.767 and distinct-moves 0.785 exactly. See
[`docs/EVAL_AND_ITERATE.md`](docs/EVAL_AND_ITERATE.md) for the full protocol and pass bar.
