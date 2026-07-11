# Submission: chess-instructor-llm

Project: Train Your Own Small Learning Model. A level-calibrated chess coach fine-tuned to reproduce
one preregistered policy, tier-appropriate move selection, end to end (dataset -> model -> platform
-> eval -> thesis -> demo). The SFT base is v4, a QLoRA fine-tune of Qwen3-32B (base
`unsloth/Qwen3-32B-unsloth-bnb-4bit`) and the model behind the full reproducible evaluation; the live
demo now serves v6-dpo2 (0.892 on the corrected 120 TEST, a strict no-regression improvement over
v4's 0.861), the best-DPO refinement of v4. The genuinely small, on-spec model is the 1.7B tune.

The one trained behavior: given a position and the student's rating tier (Beginner / Intermediate /
Advanced), emit the move that a fixed rule, `select_tier_move`, designates as the canonical tier
move, tagged with a short principle. The English explanation is secondary to that evaluation claim
and is not separately optimized; note it is still in the SFT loss, so it is not "not trained" (see
the Behavior Spec chronology in [`BRAINLIFT.md`](BRAINLIFT.md)).

## What is and is not validated (read this first)

This submission separates three claims that are easy to conflate. Keeping them separate is what makes
it rigorous.

1. LEARNABILITY (validated, the assignment's real win). The tier-selection policy is DISTILLABLE INTO
   WEIGHTS: with identical Maia grounding on both sides, a fine-tune reproduces the policy where a
   prompt on the same base cannot. Led by the small, on-spec model: the 1.7B tune lifts strict
   tier-policy match from 0.358 (base) to 0.578, #2 of 20 models and above every frontier (best
   frontier Gemini 3.1 Pro 0.553), running locally on the leakage-checked held-out slice. The 1.7B
   tune (0.578) beats the 4B tune (0.397): the behavior comes from the data and the contrastive
   signal, not from capacity. The 32B v4 (0.767) is the strongest mid-size extension, not a small
   model.
2. DEPLOYMENT-NECESSITY (false as built, stated plainly). The same Stockfish sound pool and Maia
   policy that feed the model's prompt also feed a roughly 20-line deterministic rule
   (`select_tier_move`) that computes the canonical tier move directly, at about 1.0 by construction,
   because that move IS the target the model is graded against. As built, the model APPROXIMATES a
   policy the product already produces without it. The deterministic rule is the true ceiling and the
   honest baseline. The model would be load-bearing only in a grounding-free, fully-local deployment
   (no engine or Maia at inference), which we did not build or measure.
3. PEDAGOGY / VALUE (unvalidated). We optimized AGREEMENT with our own heuristic, not evidence that
   coaches or students prefer these moves or that students improve. This is behavior validated and a
   feature demonstrated: NOT a product, NOT a moat, NOT value validated.

Metric name: "tier-policy exact match" (exact agreement with the preregistered `select_tier_move`
policy), shortened to "tier-policy match" below. The canonical move is a PROJECT RULE, not validated
pedagogy.

---

## Canonical deliverables map (v4)

| # | Deliverable | Artifact: path / URL |
|---|---|---|
| 1 | Dataset (published on HF Hub) | [`datasets/khoilamalphaai/chess-coach-move-review`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review), default config = v4: the engine-grounded, contrastive multi-tier SFT set built by `positions -> Stockfish -> Maia -> GPT-5.5 (tier-aware) -> hard filter + faithfulness gate` |
| 2 | Fine-tuned model (published on HF Hub) | [`khoilamalphaai/chess-coach-32b-v4-qlora`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v4-qlora): QLoRA adapter on the 4-bit Qwen3-32B base |
| 2b | Running demo | Live Space: [`spaces/khoilamalphaai/chess-coach-studio`](https://huggingface.co/spaces/khoilamalphaai/chess-coach-studio) (https://khoilamalphaai-chess-coach-studio.static.hf.space), now backed by the Modal endpoint `chess-coach-v6dpo2-4bit-maia` serving v6-dpo2 (a verbatim clone of the v4 serving app with only the LoRA adapter swapped; Maia-enabled, scale-to-zero, ~2.5-3 min cold start); the v4 endpoint `chess-coach-v4-4bit-maia` stays deployed as the fallback. Also local: The Analysis Room, `./run_platform.sh` |
| 3 | Eval harness | `src/eval/` (base-vs-tuned `evaluate.py` · blinded council `benchmark/` · honest gated `honest/`) · `scripts/honest_v4.py` (v4 regression + selection-conditioned head-to-head) · `scripts/grand_eval.py` (20-model leaderboard). Protocol + pass bar: [`docs/EVAL_AND_ITERATE.md`](docs/EVAL_AND_ITERATE.md) |
| 3b | Base-vs-tuned results | [`RESULTS_HONEST_EVAL_V4.md`](RESULTS_HONEST_EVAL_V4.md) + `data/benchmark_honest/report_v4.json` (strict, deterministic) · [`data/benchmark_grand/GRAND_EVAL_LEADERBOARD.md`](data/benchmark_grand/GRAND_EVAL_LEADERBOARD.md) (20-model field) |
| 3c | Grand eval (published on HF Hub) | [`datasets/khoilamalphaai/chess-coach-grand-eval`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-grand-eval): all 20 models on the same held-out slice, deterministic tier-policy match + selection-conditioned head-to-head + blinded council with 95% CIs |
| 4 | BrainLift (behavior thesis + evidence) | [`BRAINLIFT.md`](BRAINLIFT.md): the one-behavior thesis, the 32B training story (v2 -> v3 -> v4 -> v5), DOK-4 spiky POVs, all tied to primary sources or the project's own measurement |
| 5 | Demo video (3-5 min) | Script + shot list: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md). Runnable demo provided (live Space + `./run_platform.sh`); recording is the user's step |
| 6 | Stretch: preference-tuned adapter (best DPO, now live-served) | [`khoilamalphaai/chess-coach-32b-v6-dpo2`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v6-dpo2): v4 + stronger tier-targeted DPO (checkpoint step 200); overall tier-policy 0.892 on the corrected 120 TEST, supersedes v6-dpo and is now the live-served model on the demo, the best-DPO refinement of v4 (all the gain is the intermediate tier; beginner/advanced ceilinged) |
| 6b | Stretch: preference-tuned adapter (earlier DPO) | [`khoilamalphaai/chess-coach-32b-v6-dpo`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v6-dpo): v4 + DPO on tier-move pairs (sharpens the moat, no regression) |
| 7 | Stretch: engine-distilled adapter | [`khoilamalphaai/chess-coach-32b-v6-distill`](https://huggingface.co/khoilamalphaai/chess-coach-32b-v6-distill): the tier rule distilled into the weights, scored no-grounding |
| 7b | Stretch results (corrected benchmark) | [`RESULTS_STAGE4_CORRECTED.md`](RESULTS_STAGE4_CORRECTED.md): v4 / base / v6-dpo / v6-dpo2 / v6-distill on the deep-verified v6 labels, 120 held-out TEST. Full-field corrected re-score: [`RESULTS_FULL_EVAL_803.md`](RESULTS_FULL_EVAL_803.md) |
| 7c | Stretch: deep-verified v6 labels (published on HF Hub) | [`datasets/khoilamalphaai/chess-coach-v6`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-v6): the corrected canonical/sound labels (deeper Stockfish-17 + Syzygy) behind the v6 stretch results above |

---

## Scorecard: base vs tuned, and the deterministic ceiling (strict held-out eval)

Deterministic, no LLM judge in the loop (120 held-out positions x 3 tiers, strict any-legal
extractor). Metric = tier-policy exact match (agreement with `select_tier_move`). From
[`RESULTS_HONEST_EVAL_V4.md`](RESULTS_HONEST_EVAL_V4.md) and the 20-model grand eval:

| Contender (identical grounding) | tier-policy match up | note |
|---|---:|---|
| `select_tier_move` deterministic rule | ~1.000 | the true ceiling: it IS the target, computed from the same grounding without any model |
| 1.7B base -> 1.7B tuned (v2, on-spec) | 0.358 -> 0.578 | learnability lead: tuned is #2 of 20, above every frontier |
| 4B base / prompt-base / tuned | 0.353 / 0.378 / 0.397 | same-backend prompt control: tune > prompt > base |
| 32B base -> 32B tuned (v4, shipped) | 0.347 -> 0.767 | strongest instance; mid-size, not small |
| best frontier (Gemini 3.1 Pro) | 0.553 | #4 overall, all-scenario |

- Learnability (validated): the fine-tune reproduces the policy where the base cannot, at 1.7B, 4B,
  and 32B, on identical grounding. At 1.7B and 4B a same-backend engineered-prompt control was run and
  the tune beats it; at 1.7B the prompt actually hurt cross-tier coherence.
- Un-promptability at 32B is a FALSIFIABLE HYPOTHESIS, not a result: the matched same-backend 32B
  prompt control was never run. It is listed as future work.
- All-scenario lead (the primary unbiased number): across all 120 x 3 scenarios, v4 tier-policy match
  0.767 vs the best frontier 0.553 (Gemini, #4). The tuned checkpoints take 4 of the top 5. (v4
  distinct-moves-per-level 0.730 = 73/100 canonical beginner!=advanced opportunities.)
- Unbiased head-to-head: over ALL positions where v4 diverges from the best frontier (not conditioned
  on v4 succeeding), v4 goes 56-24-12 over the 92 diverging positions (56-24-40 over all 120) on the
  moat (tier-policy match then soundness).
- v4-success-conditioned subset (NOT a general win rate): within the subset of 62 of 120 positions
  where v4 already gives a distinct, sound, correctly-graded move AND diverges from the frontier, v4
  wins 51-5 (6 ties). Conditioned on v4 succeeding, so it overstates a win rate; use the unbiased
  56-24-12 above.
- The served move is the evaluated move: the prose gate changes only the explanation, never the move, so
  v4's served-move tier-policy match (0.789 replayed over the val drafts) equals its evaluated greedy
  0.767 (pre-fix it could collapse to 0.589; fixed in `d4afd73`).
- Deployment-necessity (false as built): the ~1.0 deterministic-rule row is the ceiling; the model
  approximates a move the product already computes from the same grounding.

Soundness and no-engine-speak equalize to a shared ~100% floor once every model passes the shipped
gate, so they are a fairness floor, not a differentiator. Move soundness itself is fidelity to a
shallow sound pool, not certified best-move truth (see honest gaps).

### Stretch results on the corrected benchmark (Stage-4)

The benchmark labels were rebuilt under deeper Stockfish-17 search plus Syzygy (the 120 held-out TEST
FENs are unchanged, only the canonical and engine-best targets moved). Re-scored in one controlled run
with the same strict extractor, GROUNDED for base/v4/v6-dpo/v6-dpo2 and NO-GROUNDING for base/v6-distill
([`RESULTS_STAGE4_CORRECTED.md`](RESULTS_STAGE4_CORRECTED.md)):

| Model | condition | tier-policy match | move-sound | distinct | names-a-move |
|---|---|---:|---:|---:|---:|
| BASE (Qwen3-32B untuned) | grounded | 0.428 | 0.969 | 0.303 | 0.975 |
| OURS-v4 (shipped) | grounded | 0.861 | 0.983 | 0.987 | 0.983 |
| OURS-v6-dpo | grounded | 0.881 | 0.983 | 0.987 | 0.983 |
| OURS-v6-dpo2 (best DPO) | grounded | 0.892 | 0.983 | 0.987 | 0.986 |
| BASE (Qwen3-32B untuned) | no-grounding | 0.022 | 0.081 | 0.040 | 0.250 |
| OURS-v6-distill | no-grounding | 0.325 | 0.653 | 0.461 | 0.983 |

- Preference tuning sharpens the moat with no regression. The stronger tier-targeted v6-dpo2 (checkpoint
  step 200) is the best DPO result: overall tier-policy 0.892 (+0.031 vs v4, +0.011 vs v6-dpo), with the
  entire gain at the intermediate tier (0.842 vs v4 0.750, vs v6-dpo 0.808, out of distribution). Beginner
  (0.858) and advanced (0.975) are byte-identical to v4 and v6-dpo, already ceilinged under grounding, so it
  is a stronger v6-dpo, not a beginner/advanced breakthrough; soundness (0.983) and distinct-moves (0.987)
  are unchanged, names-a-move is nominally higher (0.986), and format (0.925) is marginally under v4 (0.939),
  a token-cap prose-length artifact. v4 is the SFT base and the model behind the full reproducible
  evaluation; v6-dpo2 supersedes v6-dpo and is now the live-served model on the demo (the best-DPO
  refinement of v4).
- Distillation puts the tier rule in the weights: stripped of grounding, the base collapses (0.022,
  names-a-move 0.250); the distilled adapter recovers it to 0.325 (names-a-move 0.983), with an honest
  advanced-tier limit (0.217, the sharpest move genuinely needs grounding).
- Base-vs-tuned is preserved under the correction: grounded tuned-minus-base is +0.433 (v4) and +0.453
  (v6-dpo) tier-policy, matching the pre-correction gap.
- Corrected full-field re-score (free, cached, no model re-run;
  [`RESULTS_FULL_EVAL_803.md`](RESULTS_FULL_EVAL_803.md)): OURS still tops the moat (OURS-v2 #1, +0.042 over
  the best frontier; tuned-over-base +0.151 / +0.162) and the cross-family order holds (OURS then frontier
  then open), while the frontier reshuffles internally so Claude Opus 4.8 now edges Gemini 3.1 Pro as the
  strongest single frontier coach. Scope caveat: these are v4-era-grounding cached generations judged by the
  sharper v6 targets, so absolutes are lower than a fresh-grounding eval, valid for the relative and ranking
  read, not each model's ceiling.

---

## The honest gaps (so the submission is not oversold)

- Deployment-necessity is false as built. `select_tier_move` computes the canonical move directly
  from the same Stockfish + Maia grounding, at about 1.0 by construction. The tuned model approximates
  a policy the product already produces, so the model is not load-bearing in the grounded deployment
  we shipped. It would be load-bearing only grounding-free and fully local, which we did not build or
  measure.
- The tuned model does not internalize the rule. At inference it still needs Maia's per-tier grounding
  in the prompt; without Maia the three tiers collapse to a single move (exactly what happened when
  Maia was missing on the serving container). The honest claim is reliable grounded EXECUTION, not
  "the behavior lives in the weights." There is no graceful degradation: it degrades to collapse.
- Faithfulness is not guaranteed truthfulness. The retracted claim is "zero user-visible fabrication."
  The accurate claim is zero verifier-DETECTABLE mechanical violations after gating. The deterministic
  checker is high-precision but low-recall: relational pawn-SAN claims, forks, threats, negations, and
  eval claims can still be wrong and reach a user, and a cross-family LLM-judge residual exists. We do
  not claim the prose is guaranteed true.
- Tier-policy match is fidelity to a heuristic, not certified best teaching. The rule's "advanced =
  engine best" is contradicted in the data: advanced maximizes the stored cp in a shallow pool and
  diverges from the persisted engine_best on about 43 of 803 positions, and the sound pool comes from
  a shallow 300ms / MultiPV-8 / 150cp search that can include position-worsening moves. v6 roadmap:
  deeper pool + tablebases, Maia-as-constraint, titled-coach validation.
- Pedagogy is unvalidated. We measured agreement with our own move rule, not whether coaches or
  students prefer these moves or learn faster. Behavior validated, value not validated.
- Prose is weaker and secondary, not "not trained." v4 lands about 15th of 20 on the blinded
  instructiveness council (grade about 4.5). Prose is secondary to the evaluation claim and is not
  separately optimized, but it IS in the SFT loss, so it is not "not a training objective." A product
  that wants rich prose renders it on top of the chosen move and verifies it separately.
- v5 did not prove that clean data kills the tier-policy lead. v5 regressed (tier-policy match 0.536),
  but the run was confounded: about 27% less optimization / token exposure, contrastive triads broken by row-wise
  filtering, about 42% boilerplate-principle pollution, retrained from base rather than from v4, and no
  checkpoint selection. It does not isolate filtering as the cause.
- Live vs curated showcase. The curated showcase is the canonical deterministic proof; the live tool
  differentiates by tier but is not guaranteed to be move-for-move identical to the showcase.

---

## Eval integrity

Two independent audits back the base-vs-tuned comparison: Maia (the human-move model) is present and
symmetric across all 20 models, feeding both the ground-truth tier move and every model's grounding
equally; and there is zero train/test leakage (board-key intersection 0 of 120 between the val slice
and v4's training data). Local decoding is greedy, and re-scoring the published generations reproduces
tier-policy match 0.767 and distinct-moves 0.730 exactly.

Caveat on the faithfulness audit: the gate's zero-fabrication figure means zero verifier-detectable
mechanical violations, not certified truthfulness (see honest gaps).

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
Re-scoring the published generations reproduces tier-policy match 0.767 and distinct-moves 0.730 exactly. See
[`docs/EVAL_AND_ITERATE.md`](docs/EVAL_AND_ITERATE.md) for the full protocol and pass bar.
