# What a Fine-Tune Actually Buys a Chess Coach: A Tier-Selection Policy That Distills Into Weights, a Named Deterministic Ceiling, and an Honest Map of What Is and Is Not Validated

**Owner:** Khoi Lam

**Date:** July 8, 2026 (reframed July 9, 2026 after a converged multi-agent audit)

## Purpose

This BrainLift separates three claims that are easy to run together. Keeping them separate is what makes the submission rigorous: a clean distillation result, a named ceiling baseline, and explicit open questions are what the brief rewards.

**Claim 1: LEARNABILITY (validated, the assignment's real win).** The tier-selection policy is DISTILLABLE INTO WEIGHTS. With Maia grounding held identical on both sides, a fine-tune reproduces the policy where a prompt on the same base cannot. The genuinely small, on-spec model carries this result: the 1.7B tune lifts strict tier-policy exact match from 0.358 (base) to 0.578, #2 of the 20-model field and above every frontier model (best frontier Gemini 3.1 Pro at 0.553), running locally on the leakage-checked held-out slice. The 1.7B tune (0.578) beats the 4B tune (0.397): the behavior comes from the data and the contrastive signal, not from capacity. The 32B v4 (0.767) is the strongest MID-SIZE extension of the same result, not a small model.

**Claim 2: DEPLOYMENT-NECESSITY (false as built, stated plainly).** The same Stockfish sound pool and Maia policy that feed the model's prompt also feed a roughly 20-line deterministic rule, `select_tier_move` (a pure function of tier, sound pool, and Maia policy), which computes the canonical tier move directly. That rule scores about 1.0 by construction, because the move it returns IS the target the model is graded against. So as built, the fine-tuned model APPROXIMATES a policy the product already produces without it. The deterministic rule is the true ceiling and the honest baseline; the model is not load-bearing in the grounded deployment we shipped. It would be load-bearing only in a grounding-free, fully-local deployment (no engine or Maia at inference), which we did not build or measure.

**Claim 3: PEDAGOGY / VALUE (unvalidated).** Every number here measures AGREEMENT with our own heuristic, not evidence that coaches or students prefer these moves or that students improve. The honest status is behavior validated and a feature demonstrated: NOT a product, NOT a moat, and NOT value validated.

**Metric name.** Throughout, "tier-policy exact match" (shortened to "tier-policy match") means exact agreement with the preregistered `select_tier_move` policy. The canonical move is a PROJECT RULE, not validated pedagogy, and this document never treats agreement with it as evidence of teaching quality.

**Behavior Spec (one spec, and its chronology).** Given a chess position and the student's rating tier (beginner about 1000 to 1200, intermediate about 1300 to 1600, advanced about 1700 to 2000), the model's one trained job is to emit the move `select_tier_move` designates as canonical, rendered as the move plus a short principle tag (for example, "Nf3 - develop toward the center"). That single choice is graded pass or fail on three deterministic clauses a stranger can check with no opinion in the loop: (1) sound, meaning not a blunder (engine cp-loss below 250, within the shallow sound pool described below); (2) tier-policy match, meaning it equals `select_tier_move`'s canonical tier move; (3) distinct across levels, meaning a beginner and an advanced player are not handed the same move on a differentiating position. Chronology, disclosed for honesty: this spec was NARROWED to the move only after the evidence separated the move from the prose (an earlier prose-full spec graded the four-part explanation too). The prose is therefore secondary to the evaluation claim and is not separately optimized; it is NOT "not a training objective," because the prose is still in the SFT loss. (Maintainer note: `config/settings.py` still holds the older prose-full `BEHAVIOR_SPEC`; this document is the canonical one-behavior spec, and the code copy is flagged for the code worker to update.)

**The learnability evidence (base versus prompt versus tune, identical grounding).** The test holds the shipped grounding identical, so the only thing that changes is the weights (untuned base versus fine-tuned) or the system prompt (fine-tuned versus an engineered prompt on the base). Same held-out positions, same grounding (a strong engine's sound-move pool, a human-move model, and the identified tier move), and the move is scored deterministically against `select_tier_move` with no model judge anywhere. On the 20-model grand eval (all 120 held-out positions x 3 tiers, strict any-legal extractor):

| contender (identical grounding) | tier-policy match (up better) | note |
|---|---:|---|
| `select_tier_move` deterministic rule | ~1.000 | the ceiling: it IS the target, computed from the same grounding without any model |
| 1.7B base -> 1.7B tuned (v2, on-spec) | 0.358 -> 0.578 | learnability lead: #2 of 20, above every frontier |
| 4B base / prompt-base / tuned | 0.353 / 0.378 / 0.397 | clean same-slice prompt control: tune > prompt > base |
| 32B base -> 32B tuned (v4, shipped) | 0.347 -> 0.767 | strongest mid-size instance |
| best frontier (Gemini 3.1 Pro) | 0.553 | #4 of 20 |

Three things follow, and they are the point of this document. First, the fine-tune reproduces the policy where its base does not, and the deterministic `select_tier_move` rule sits above all of them at about 1.0 by construction, because the model is trained to match exactly what that rule already outputs from the same grounding. That is claim 1 and claim 2 together: the policy is learnable into weights, and the model is not necessary in the grounded product. Second, the prompt control was run at the small sizes and the tune wins: at 4B on this grand slice the tune (0.397) beats the engineered prompt-base (0.378), which beats the base (0.353); at 1.7B on the earlier byte-identical litmus slice the tune (0.463) beat the engineered prompt (0.389), and the prompt actually HURT cross-tier coherence (violation 0.500 base to 0.611 prompt) while the tune lowered it to 0.333, because prompting made the base vary its move without making it level-aware. Third, a converged honesty correction: the matched same-backend 32B prompt control was never run, so the claim that a well-prompted 32B base cannot reproduce the policy is a FALSIFIABLE HYPOTHESIS, listed as future work, not a result. The real, run prompt-versus-tune evidence is at 1.7B and 4B.

**The all-scenario lead, and the demoted head-to-head.** The unbiased comparison is the all-scenario number: across all 120 x 3 scenarios v4 tier-policy match is 0.767 versus the best frontier's 0.553 (Gemini, #4), and the tuned checkpoints take four of the top five. The honest unbiased head-to-head, over all 96 positions where v4 diverges from the best frontier with no v4-success gate, is 56-28-12 (56-28-36 over all 120). The often-quoted 51-5-6 head-to-head is SELECTION-CONDITIONED and is not a general win rate: it is computed only within the v4-success/divergence subset, the 62 of 120 positions where v4 already gives a distinct, sound, correctly-graded move AND diverges from the frontier. Because it conditions on v4 already succeeding, it overstates a raw win rate and is reported here as a subset figure only, never as a win rate over all positions; use the unbiased 56-28-12.

**The spiky bonus, honestly bounded: on the all-scenario metric the small tune leads the field, but that is not "beats the frontier at coaching."** On tier-policy match v4 reaches 0.767 against about 0.49 to 0.55 for the frontier models, and gives distinct moves across the tiers about 73 percent of the time against roughly 21 to 28 percent for the frontier, because the frontier hands the engine's single best move to every level, repeating one move across the three tiers about 77 percent of the time. This is a lead on AGREEMENT WITH OUR OWN RULE, not evidence the frontier coaches worse: the frontier could be right and our rule wrong, since the rule's pedagogy is unvalidated (claim 3). The 51-5-6 head-to-head is the selection-conditioned subset figure above, not a win rate. The defensible reading is narrow and true: the tuned small model reproduces our tier-selection policy and a prompted frontier does not follow it.

**Grounded execution, not weights.** The tuned model does not internalize the rule. At inference it still needs Maia's per-tier grounding in the prompt; without Maia the three tiers collapse to a single move, which is exactly what happened when Maia was missing on the serving container. The honest claim is reliable grounded EXECUTION, not that the behavior lives in the weights, and there is no graceful degradation here: it degrades to collapse.

**Tier-policy match is fidelity to a heuristic, not certified best teaching.** Two honest limits sit under the deterministic grade. First, the "advanced = engine best" leg of the rule is contradicted in the data: advanced maximizes the stored cp in the sound pool and diverges from the separately persisted engine_best on about 43 of 803 positions. Second, that sound pool comes from a shallow 300ms / MultiPV-8 / 150cp search that can include position-worsening moves. So a passing tier-policy match is fidelity to this heuristic, not certified best teaching. The v6 roadmap addresses this directly: a deeper pool plus tablebases, Maia used as a constraint rather than only a ranker, and validation against titled coaches.

**Faithfulness is not guaranteed truthfulness.** The retracted claim is "zero user-visible fabrication." The accurate claim is zero verifier-DETECTABLE mechanical violations after gating. The deterministic checker is high-precision but low-recall: relational pawn-SAN claims, forks, threats, negations, and eval claims can still be wrong and reach a user, and a cross-family LLM-judge residual exists. This document does not claim the prose is guaranteed true; it claims the gate removes the violations it can detect and falls back to a truthful engine-derived explanation when it cannot. Critically, the gate changes only the explanation, never the served MOVE: on a prose failure it keeps the model's own greedy sound move and rewrites just the prose (post-fix `d4afd73`), so the served move equals the evaluated greedy move — replaying the shipped gate over the 120 val drafts measures served tier-policy match 0.789, at or above the raw 0.767; pre-fix the gate could substitute the engine-best move on a prose failure, collapsing the served match to 0.589 (now fixed).

**The v5 regression was confounded.** v5 retrained on a cleaner, faithfulness-filtered dataset and regressed (tier-policy match 0.536), but the run does NOT isolate filtering as the cause: it saw about 27 percent less optimization / token exposure, its contrastive triads were broken by row-wise filtering, about 42 percent of its principles were boilerplate pollution, it was retrained from the base rather than from v4, and there was no checkpoint selection. So "cleaner data killed the tier-policy lead" is not an established lesson; it is a confounded experiment, and the confounds are the honest takeaway.

**Why narrowing to the move still sharpens the learnability claim (without overclaiming).** Two things get cleaner once the graded behavior is the move. The litmus is cleaner, because a good prompt can already write fluent chess prose, so prose was never the place to prove a fine-tune earns its keep, whereas the tier-policy move is a discrete target with a deterministic answer. And the primary grade is judge-free, because the move is scored against `select_tier_move` by the engine and the human-move model with no council in the core claim. What narrowing does NOT buy is deployment-necessity (claim 2) or validated pedagogy (claim 3); it makes the learnability claim clean, and it must not be read as more. The shipped model is the 32B v4 (the strongest mid-size instance); the on-spec, defensible floor is the small 1.7B/4B local model, which is where the learnability result actually lives.

This document answers one honest question for anyone building an AI chess coach, or any tool that turns a verified answer into a level-appropriate recommendation. Can a small open model, fine-tuned on engine-grounded distilled data, reproduce a tier-selection policy that a prompt on the same base cannot, and what does that actually buy? The measured answer, in three parts: yes, the policy distills into weights (claim 1); no, that does not make the model necessary in the grounded product, because the deterministic `select_tier_move` rule already computes the move from the same grounding (claim 2); and no, none of it validates pedagogy, because everything here scores agreement with our own rule (claim 3). Dependability in this system is carried by the parts outside the language model: a strong engine that certifies soundness (within a shallow pool), a human-move model that says which sound move a rating would find, and the tier rule that combines them. The fine-tune's honest contribution is reproducing that rule's move locally, which a prompt on the same weights does not, while it still needs the grounding at inference.

Several measurement rules follow. The core behavior is graded deterministically against `select_tier_move`, with no model judge in the loop, because the deliverable is a move with a checkable answer against the rule. Faithfulness of the prose, where it exists, is gated by a non-LLM claim checker before anything reaches a student, but that gate is high-precision and low-recall, so it removes detectable mechanical violations rather than certifying truth, and any prose quality score comes from a different model family, corrected for a measured self-preference of about 1.44 rank positions. Read against those rules, the honest answers to the three claims are: learnability yes (validated at 1.7B and 4B, hypothesized at 32B pending the matched prompt control), deployment-necessity no as built, pedagogy unvalidated. The stances below are a depth-gated menu around that answer, led by the four strongest, and deliberately not a single verdict.

**In scope:** whether a small fine-tuned engine-grounded model reproduces the tier-selection policy more reliably than the same base under a prompt, graded deterministically; the deterministic rule as the ceiling and the honest question of when the model is load-bearing; the move-versus-prose split and why the prose is secondary but still in the training loss; where dependability comes from (the engine, the human-move model, and the tier rule versus the weights); the honest size and location of the fine-tune's contribution; the faithfulness gate and its recall limits; and the reward design that keeps the primary signal deterministic.

**Out of scope (flagged, not claimed):** validated pedagogy or learning outcomes; deployment-necessity in a grounding-free local product; a matched same-backend 32B prompt control; certified best-move truth beyond the shallow pool; making a language model itself play strong chess; the raw cost and hardware absolutes, kept low-confidence; non-chess domains except as an explicit generalization test; and any claim that could not be tied to a primary source or the project's own measurement.

## The 32B Training Story: What Each Attempt Actually Did (v2 -> v3 -> v4 -> v5)

This section records how the shipped result was reached, including the wrong turns, because the honesty is the point. It changes none of the three claims above.

**Why 32B at all, when the on-spec answer is small.** The defensible, on-spec deliverable is a small local move-chooser (the 1.7B and 4B tunes), and that stays the honest floor of the learnability claim, because the behavior is reproducing the tier-selection policy, which the 1.7B tune already does at 0.578 (#2 of 20). Going to 32B was a deliberate quality push to see how far a fine-tuned open coach could push the policy match, not a retreat from the small-model thesis and not a claim that 32B is small. The 32B v4 is reported as the strongest MID-SIZE instance of the behavior.

**The v2, v3, v4, v5 arc, and what happened to each.** v2 (Qwen3-1.7B) is the learnability result: on the 20-model grand eval it posts tier-policy match 0.578, #2 of 20 and above every frontier, small and local. v3, v4, and v5 (Qwen3-32B) were trained with the same low-rank recipe on the same base, unsloth's four-bit Qwen3-32B, differing essentially only in the data. v3 was the all-rounder: a strong balance of move and prose, about fifth of twenty on the blinded prose council at an instructiveness grade of about 6.4, tier-policy match about 0.56. v4 is the shipped model: trained to maximize the policy match, it leads the field at 0.767 and distinct-moves 0.730 (73/100 canonical beginner!=advanced opportunities; the honest unbiased head-to-head over the 96 diverging positions is 56-28-12, and the selection-conditioned 51-5-6 over 62 is a subset, not a win rate), while trading prose down to about fifteenth of twenty at about 4.5, which is on-thesis because prose is secondary to the evaluation claim. v5 was an attempt to fix v4's prose and raw faithfulness with a cleaner, faithfulness-filtered dataset, and it regressed: tier-policy match fell to 0.536, move-soundness to 0.828, the prose council to about 3.9, and raw faithfulness stayed flat around 0.58. But the v5 run is CONFOUNDED and does not isolate filtering as the cause: it saw about 27 percent less optimization and token exposure, its contrastive triads were broken by row-wise filtering (so the beginner / intermediate / advanced contrast within a position was split apart), about 42 percent of its principles were boilerplate pollution, it was retrained from the base rather than continued from v4, and there was no checkpoint selection. The earlier framing, "aggressively cleaning the data thinned the contrastive signal and killed the tier-policy lead," is therefore a HYPOTHESIS at best; the honest statement is that v5 regressed under several simultaneous confounds and does not prove filtering alone caused it. v5 is not shipped.

**Two honesty corrections found while measuring v4.** First, an early move extractor sometimes read an avoid-move (a move named as one to avoid before the recommended move), which understated the tuned model's policy match; making the extractor avoid-framing-aware corrected the reading, to distinct-moves about 0.75 and policy match about 0.79. Second, the leaderboard and the head-to-head were using two different parsers, a lenient in-pool one and a strict any-legal one; making the strict any-legal parser canonical everywhere, so that an output naming no clearly legal move is a miss on every axis, moved the headline tier-policy match from about 0.79 to 0.767 and made the report reproducible, because re-scoring the published generations then reproduces tier-policy match 0.767 and distinct-moves 0.730 exactly. The number went down and the honesty went up.

**The Maia serving gap, and why the behavior is grounded execution, not weights.** Maia supplies the human-findability signal that, with the engine and the tier rule, defines the canonical tier move, and it grounds every model's prompt in the eval. When Maia was not installed on the live serving container, the interactive coach lost the per-tier signal and COLLAPSED all three tiers to a single move, differentiating on only about zero to one of seven showcase positions, even though the eval and the curated showcase differentiate cleanly. This was an environment defect, but it is also the direct proof that the tuned model does NOT internalize the rule: the four-bit and the full-precision endpoints produced the same collapse, and the curated showcase differentiates only because it was generated locally with Maia. Adding Maia and greedy-first decoding restored differentiation (four of the seven showcase forks). The honest framing is two-surface: the curated showcase is the deterministic proof of the tier-policy behavior, generated with the full grounding, while the live tool demonstrates the behavior end to end but is not guaranteed move-for-move identical. And the honest claim about the weights is reliable grounded EXECUTION: the tune needs Maia's per-tier grounding in the prompt at inference, and without it the behavior degrades to collapse, not gracefully.

**Two audits that back the base-vs-tuned comparison.** Because the comparison rests on the evaluation being fair, two independent checks were run, and both came back clean. The first traced the human-move model end to end and confirmed it is present and symmetric across all twenty models, feeding both the eval's ground-truth tier move and every model's grounding equally, so no model, this project's included, gets a human-move advantage the others lack. The second checked for train-test leakage and found none: the 120-position validation set is genuinely held out from the shipped model's fine-tuning data, with a board-key intersection of zero out of 120. Together these make the learnability comparison defensible, because the tier-policy lead is measured on a clean, held-out, symmetrically grounded set that reproduces from the published generations. They do NOT validate pedagogy or deployment-necessity, which are separate claims addressed above.

## DOK 4: Spiky Points of View

These are a depth-gated menu of candidate-valid stances, positions worth testing rather than settled truths, and deliberately not one chosen winner. Each is labeled by how strong its backing is now: Validated (established by the project's own measurement and still off-consensus), Strong (candidate-valid, awaiting a further test), and Weak (a softened caution). Read the whole set under the three-claim lens. The four strongest, and the ones to read first, are: (a) the tier-selection policy is LEARNABLE into weights where a prompt on the same base is not, validated at 1.7B and 4B and a hypothesis at 32B (POV 1); (b) a discrete deterministic choice is the right thing to train and grade, and its prose is not, because a verified move does not buy a verified explanation (POV 2 and POV 3); (c) the dependability, and the canonical move itself, live in the engine-plus-Maia-plus-rule SYSTEM, not the weights, so as built the model is not load-bearing (POV 9); and (d) any model-judged prose score is untrustworthy, so the primary grade is judge-free (POV 4). The rest are retained as tested sub-stances, but note two demotions the audit forced: the "beats the frontier on a moat" framing (POV 5) is a selection-conditioned subset result plus an all-scenario agreement lead, not a coaching-quality win, and "zero user-visible fabrication" (POV 15) means zero verifier-DETECTABLE violations, not certified truth.

**Spiky POV 1 (Validated at 1.7B and 4B; hypothesis at 32B): The tier-selection policy is DISTILLABLE INTO WEIGHTS. Hold the grounding identical and a fine-tune reproduces `select_tier_move` where the same base under an engineered prompt does not, at 1.7B and 4B. The matched same-backend 32B prompt control was never run, so 32B un-promptability is a falsifiable hypothesis, not a result.**

**Elaboration:** The common expectation is that for a narrow behavior a good prompt matches or beats a fine-tune. This project ran the clean version of that test on the tier-selection policy, graded deterministically against `select_tier_move` with no model judge, freezing the grounding so the only variable was the weights or the system prompt. The fine-tune reproduced the policy where the base did not: on the grand slice the 1.7B tune reaches tier-policy match 0.578 (base 0.358), #2 of 20 and above every frontier, and the 4B trio is base 0.353, prompt-base 0.378, tune 0.397, so the tune beats the prompt beats the base on the same slice; at 1.7B on the earlier byte-identical litmus the engineered prompt actually HURT cross-tier coherence (0.500 base to 0.611 prompt) while the tune lowered it to 0.333, because prompting made the base vary its move without making it level-aware. The 32B tune reaches 0.767, but the matched same-backend 32B prompt control was never run, so the 32B leg is a hypothesis. Two honest boundaries carried from the Purpose: this is learnability, not deployment-necessity, because `select_tier_move` already computes the same move at about 1.0 from the same grounding (the ceiling); and it is agreement with our rule, not validated pedagogy.

**Prediction or Disconfirmer:** With grounding held identical, the fine-tuned model reproduces `select_tier_move` at a materially higher rate than both its untuned base and the best engineered prompt on the same weights, at 1.7B and 4B. If a matched same-backend engineered prompt on the base reaches the fine-tune's tier-policy match, the claim is wrong; running that control at 32B is the open test.

**How to resolve it:** Keep the grounding identical and vary only the weights and the prompt, then score base, best-prompted base, and fine-tune on the same held-out set with the deterministic `select_tier_move` check. This has been done with a prompt arm at 1.7B and 4B and the tune wins; the matched 32B prompt arm is the missing control and is the confirming or disconfirming work.

**Testing note:** Cold raters treated this as genuinely disputed, because the field's working belief is that prompting usually suffices for a narrow behavior, so a flat "a well-prompted base cannot match the tune on this policy" earns its spikiness. An adversary could not make it retreat at 1.7B and 4B once the grounding is frozen and the metric is the deterministic policy match, and it reaches to any leveled recommender whose target is checkable without a model, such as difficulty-tiered hints in a math or programming tutor. It holds as established at 1.7B and 4B, where the prompt arm was run; it is a hypothesis at 32B, where the matched prompt arm was not. The honest caveats are kept in view: the arms are validation slices rather than the full definitive set, the tuned rows were scored on their raw draft for the deterministic axes, this is the project's own controlled experiment rather than an outside replication, and the whole result is learnability, not deployment-necessity (the deterministic rule is the ceiling) and not validated pedagogy.

**Spiky POV 2 (Validated): Because the trained deliverable is a move and not prose, faithfulness is free and the evaluation is fully deterministic. A move cannot hallucinate a board fact, so the core claim needs no verifier and no model judge; it is graded purely as tier-policy match against the canonical tier move.**

**Elaboration:** The sharpest consequence of making the move the behavior is that the two hardest measurement problems in the old prose-centric framing simply disappear from the graded claim. A prose explanation can invent a fork, a pin, or a mate that is not there, which is why the old framing needed a claim-level verifier and a cross-family council. A move cannot do any of that: it is a single legal action on the board, and whether it is sound comes from the engine and whether it is the level-right choice comes from the engine plus a human-move model. So the entire faithfulness apparatus, the verifier and the judge, is unnecessary for the thing being graded, and the score is a deterministic comparison to a fixed canonical move. This is not a softening of the claim, it is a hardening: the project's own measurements score move selection on the engine and the human-move model alone, and the 803-position leaderboard, the three-size litmus, and the v4 head-to-head are all judge-free on this axis. The rule "anchor on deterministic gates, not judges" is usually a measurement aspiration; narrowing the trained behavior to the move makes it literally true for the whole graded claim.

**Prediction or Disconfirmer:** The core behavior can be scored to full agreement by two independent deterministic checkers (engine soundness plus canonical-tier-move match) with no model judge, and repeated scoring returns identical results. If grading the move requires a model judge to resolve disagreement, or if two deterministic scorers cannot agree on tier-policy match, the claim is wrong.

**How to resolve it:** Re-score the same generations twice with the deterministic tier-move checker and confirm identical tier-policy match, and confirm that no board-fact fabrication metric applies to a bare move because there is no board claim to check. This is how every move-axis number in the project is already produced.

**Testing note:** Cold raters found it non-obvious that narrowing the deliverable removes the hallucination problem by construction rather than by better verification, which is where it earns its edge. An adversary could not make it retreat, because a move genuinely has no free-text claim to falsify, and it reaches to any recommender that outputs a discrete verifiable choice rather than a rationale, such as a triage system that outputs a category checkable against a rule base. It holds as established, because it follows from the definition of the deliverable and is instantiated by every judge-free move measurement in the project.

**Spiky POV 3 (Validated): Grounding the move does not ground the prose, which is exactly why prose must not be the trained behavior. Even when the engine has proven the move is sound, a model narrating the reasons invents tactics, so the right design is to make the un-fabricable move the deliverable and treat prose as an optional, separately-verified layer.**

**Elaboration:** It is tempting to assume that once the engine has picked and verified the move, the surrounding explanation inherits that correctness. It does not. Choosing the move and narrating the reasons are different tasks. The engine certifies the choice, but the words about why, such as "this knight is trapped" or "this threatens mate in two," are generated by the language model and are only as reliable as that model. The project's base run makes the split concrete, with move soundness at 1.00 and prose truthfulness at zero on the same outputs at the same time, and broader chess evidence agrees, with a strong frontier model making factually incorrect chess claims about 22 percent of the time and smaller open models more than 50 percent regardless of whether the move was right. In the old framing this drove a whole verifier-and-council apparatus to rescue the prose. The sharpened framing draws the opposite, cleaner lesson: if the prose cannot be trusted from weights, do not make the prose the trained behavior. Train the move, which cannot fabricate, and render the prose separately with a non-LLM claim checker in front of it. The project's own gate drives shipped prose to zero verifier-DETECTABLE violations for every model, including the 32B v4 whose raw prose fabricates about 40 percent of the time; that is not certified truth, because the checker is low-recall, and it now protects an optional display layer rather than standing between the fine-tune and a passing grade.

**Prediction or Disconfirmer:** On positions where the move is engine-verified as sound, a model with no claim-level verifier will still make at least one false tactical claim in a large share of its free-text explanations, so prose faithfulness cannot be assumed from a verified move. A model whose unverified prose is faithful on its own at high rates would weaken the need to demote prose, though it would not touch the move claim.

**How to resolve it:** Hold move grounding constant and count fabricated claims per free-text explanation with and without a claim-level verifier, confirming the move being sound does not make the prose sound.

**Testing note:** Cold raters found the split, that a verified move does not buy a verified explanation, genuinely non-obvious once the why is separated from the what. An adversary could not make it retreat, because the move-versus-reason split is concrete, and it reaches to radiology, where a report can verify the nodule and still invent a comorbidity in the impression. The core holds as established, from the 1.00 versus zero base run and the measured chess-claim error rates, and under the sharpened thesis it is the direct argument for why the move, not the prose, is the trained behavior.

**Spiky POV 4 (Validated): An unaided explanation judge, and especially one from the same model family, systematically passes false chess prose as truthful, which is a second reason the graded claim cannot rest on prose. The move axis needs no judge; the optional prose layer, if scored at all, must be gated by a non-LLM check first and judged cross-family.**

**Elaboration:** A language model asked to judge a chess explanation does not run an engine check; it reacts to fluency and confidence. In a controlled chess-commentary evaluation a vanilla judge rated a hallucinated commentary about 4.9 out of 5 while two of its three factual claims were false, and the project's own base run showed a strong frontier judge returning a truthfulness score of zero on outputs it still rated as readable. The project's own blinded cross-family council later measured same-family inflation directly, at about 1.44 rank positions, with every judge favoring its own lab's model. In the old framing this was a warning about how to score the prose. Under the sharpened thesis it is a second, independent reason the graded claim is the move and not the prose: the move needs no judge at all because it has a deterministic right answer, so the sycophancy and same-family problems that plague explanation scoring never touch the core claim. Where the optional prose layer is scored, the rule stands, gate faithfulness with a non-LLM engine-and-detector check first and draw any quality score from a different family, corrected for self-preference.

**Prediction or Disconfirmer:** On held-out positions, an unaided or same-family model judge passes as truthful a much larger share of explanations than a non-LLM gate accepts, so a judge cannot certify prose truth; meanwhile the move axis is unaffected because it uses no judge. If the model judge and the non-LLM gate agree within noise, the prose half is wrong.

**How to resolve it:** Score one batch of prose twice, once with the engine-and-detector gate and once with an unaided model judge, compare a same-family judge against a different-family judge, and confirm the move axis is graded without either.

**Testing note:** Cold raters split on whether a model judge is fine for chess, which marks it off-consensus. An adversary confirmed the crux holds as long as the unaided judge is fixed in advance, and it reaches to any setting with a cheap external checker, such as a legal assistant whose sibling-model grader blesses fabricated citations. It holds as established from the chess-commentary judge result and the project's own zero-truthfulness reading, and under the sharpened thesis it reinforces that the graded claim is deliberately judge-free.

**Spiky POV 5 (Strong, honestly bounded): On the all-scenario policy-match metric the tuned model leads the field, including the frontier, because the frontier follows the engine's best move regardless of level. This is a lead on AGREEMENT WITH OUR RULE, not evidence the frontier coaches worse, and the 51-5-6 head-to-head is a selection-conditioned subset, not a win rate.**

**Elaboration:** The field keeps assuming a stronger-playing frontier must give better coaching moves, when the opposite holds for leveling: an engine-aligned model defaults to the engine's single best move regardless of who is asking, repeating one move across the three tiers about 77 percent of the time. On the twenty-model grand slice the 32B v4 tune leads at tier-policy match 0.767 against about 0.49 to 0.55 for the frontier, with distinct moves about 73 percent of the time against roughly 21 to 28 percent. The unbiased comparison is that all-scenario number, 0.767 versus the best frontier's 0.553; the honest unbiased head-to-head over the 96 diverging positions is 56-28-12 (56-28-36 over all 120). The 51-5-6 head-to-head is a SUBSET figure, computed only on the 62 of 120 positions where v4 already gives a distinct, sound, correctly-graded move AND diverges from the frontier, so it conditions on v4 succeeding and is not a win rate. Crucially, all of this is agreement with `select_tier_move`, our own rule, so it is not evidence the frontier coaches worse: the frontier could be following a defensible different policy. The defensible reading is narrow: the tuned model reproduces our policy and a prompted frontier does not follow it.

**Prediction or Disconfirmer:** With identical grounding, the tuned model reaches a higher deterministic tier-policy match than a well-prompted frontier. If a well-prompted frontier matches or beats the tuned model on tier-policy match, the lead is not there. Note this measures agreement with our rule, not coaching quality; a pedagogy study (do coaches or students prefer these moves) is the separate, unrun test.

**How to resolve it:** Score tier-policy match deterministically on a large held-out set for the tuned model and for a well-prompted frontier, and compare tier-policy match, distinct-moves-per-level, and the selection-conditioned head-to-head. The twenty-model grand leaderboard and the earlier 803 leaderboard are computed and the tuned model leads on policy match; whether that policy is the right teaching policy is a separate, unvalidated question.

**Testing note:** This is off-consensus because most people assume the frontier's stronger play makes its move choice better for teaching, when in fact it defaults to the engine line regardless of level. The crux is falsifiable and rests on the measured frontier behavior, the gap density, and the grand and 803 numbers where the tune leads on policy match. It reaches to any leveled recommender where a bigger general model over-optimizes the single best answer instead of the level-appropriate one. It is Strong, not Validated, and explicitly bounded: it is a lead on agreement with our own rule, the head-to-head is selection-conditioned, and it says nothing about whether our rule teaches better, which is claim 3 and unvalidated.

**Spiky POV 6 (Strong): The coaching prose is an optional display layer, not the product, so a model's prose quality, including the 32B v4 prose regression, is irrelevant to the graded claim. Render prose with the engine's templates or a frontier model on top of the tuned move.**

**Elaboration:** Because the trained behavior is the move, the prose is free to be produced by whatever writes English best, and the choice of writer is a product decision, not a training one. The v4 evidence makes the separation vivid. The 32B v4 is the strongest model on the move, leading the field on tier-policy match and distinct-moves, and simultaneously the weakest recent checkpoint on prose, with a blinded instructiveness grade of 4.67 below the 4B tune at 5.32 and the prior v3 at 6.35, and about 40 percent of its raw drafts failing the prose faithfulness check before the gate. Under a prose-centric thesis that regression would sink v4. Under the sharpened thesis it is a non-issue: v4 is kept for the move it chooses, and the prose beside that move is rendered by the engine's templates or by a prompted frontier model, then run through the same non-LLM gate that drives shipped verifier-detectable violations to zero for every model (not certified truth). The honest boundary is that a product wanting rich prose should render it with a strong writer, which the frontier is, precisely because prose is not where the small model needs to win.

**Prediction or Disconfirmer:** Swapping the prose writer, from the tuned model's own text to engine templates or a frontier renderer, leaves the graded tier-policy match unchanged, because the move is chosen before the prose is written. If changing the prose writer changes the graded move axis, the layers are not actually separable and the claim is wrong.

**How to resolve it:** Hold the tuned model's chosen move fixed, render its prose three ways (self, templates, frontier), and confirm tier-policy match is identical while only the prose quality varies.

**Testing note:** Cold raters found the demotion of prose to an optional layer off-consensus, since most coaching work treats the explanation as the deliverable. An adversary found the crux resistant because move choice provably precedes prose rendering, and it reaches to any system that separates a verifiable decision from its natural-language justification, such as a recommender whose explanation is a swappable template. It is Strong on the project's own v4 split, where the strongest-move model is the weakest-prose model, which is only coherent once prose is the optional layer.

**Spiky POV 7 (Strong): When a small local coach beats a frontier model on anything other than the move, what it actually wins is form factor. On prose it trails; its clean edges are cost, latency, privacy, offline use, and the leveled move choice.**

**Elaboration:** The honest framing separates three axes. One is the leveled move choice, which Spiky POV 1 and Spiky POV 5 show the fine-tune genuinely adds and now leads the frontier on. Another is prose, meaning explaining the move correctly, in plain language, at the right depth. A third is form factor, meaning running cheaply, privately, offline, and fast. On the prose axis, with grounding held constant, a blinded cross-family council still judges the frontier models more instructive than the tuned small model, and the frontier fabricates a board fact only about 3 percent of the time against the small model's roughly a third before the verifier. So with grounding equalized the small model does not win the prose axis, and it does not need to, because prose is the optional layer. Its clean standalone wins are the deployment envelope and the leveled move choice, and a product that wants frontier-grade prose renders it on top of the tuned move.

**Prediction or Disconfirmer:** With identical grounding, a prompted frontier model matches or beats the tuned small model on prose instructiveness under repeated sampling, while the tuned model leads on the deterministic move axis. If the tuned small model clearly beats the equally grounded frontier on prose instructiveness, this specific claim is wrong; note it is about prose, not the move, where the tune leads.

**How to resolve it:** Score prose instructiveness separately from cost, latency, and the deterministic move axis, under repeated sampling at deployment temperature with grounding held constant.

**Testing note:** Cold raters put the core near the mainstream, since few dispute that grounding helps both, so it earns its edge from the sharp reframing that the field mislabels an economics win as a capability win and from cleanly separating the move from the prose. An adversary found the crux resistant once grounding is equalized, and it reaches to on-device speech-to-intent, where the win is latency and privacy rather than accuracy. The single-sample run supports it; the repeated-sampling version is pending, so it stays Strong.

**Spiky POV 8 (Strong): On the optional prose layer, the small tuned model's honest capability win is register consistency, not truth. A narrowly tuned small model is the lowest-variance renderer of plain, no-engine-speak voice, and it should beat a prompted frontier on staying in that register under sampling.**

**Elaboration:** Within the optional prose layer there are two separable qualities. One is faithfulness, which the non-LLM verifier carries for whichever model writes. The other is register, meaning never leaking centipawns or deep engine lines and holding a steady voice for a given rating. Register is exactly the narrow stylistic behavior fine-tuning compresses well into a small model. The base run hints at the size of the prize, with no-engine-speak at 0.11 for the untuned small model, so there is real headroom to own. The newest measurements complicate this honestly: when the frontier models are handed the same grounded prompt they also stay jargon-free almost all of the time on a single greedy pass, so on a one-shot pass the register axis is near a tie. The bet relocates onto variance, that under repeated sampling at deployment temperature the tuned model holds the register with a lower failure rate, and that comparison has not been run. None of this touches the graded move claim; it is a property of the optional layer.

**Prediction or Disconfirmer:** Under repeated sampling at deployment temperature, the tuned small model beats an equally grounded prompted frontier on no-engine-speak and register-consistency pass rates. If the grounded frontier matches or beats the tuned model on register consistency under sampling, the claim is wrong.

**How to resolve it:** Sample both systems many times at deployment temperature with grounding held constant and compare the register pass rates and their variance, kept separate from both truth and the move axis.

**Testing note:** Cold raters treated this as a real empirical bet rather than a truism. An adversary found it resistant as long as the register metrics are frozen in advance, and it reaches to on-device command-grammar parsing. On a single pass the grounded frontier now matches the tuned model, so the claim rests on the repeated-sampling variance measurement, which has not been run; it is Strong and explicitly awaiting that number, and it lives entirely in the optional layer.

**Spiky POV 9 (Strong, and the deployment-necessity claim): The dependability asset, and the canonical move itself, live in the engine-plus-Maia-plus-`select_tier_move` SYSTEM, not the weights. Because that ~20-line rule computes the canonical move directly at about 1.0 from the same grounding, the fine-tuned model APPROXIMATES a policy the product already produces, so as built it is not load-bearing. It would be load-bearing only grounding-free and fully local, which was not built or measured.**

**Elaboration:** It is natural to treat the distilled dataset and the fine-tuned checkpoint as the product. The evidence splits the roles, more sharply than the earlier framing admitted. The canonical move's correctness comes from the engine plus the human-move model plus the tier rule, all non-weight parts, and `select_tier_move` is a pure function of (tier, sound pool, Maia policy) that returns exactly the move the model is trained and graded to match. So in the shipped, grounded product the rule already produces the move at about 1.0 by construction, and the model reproduces it at 0.578 (1.7B) to 0.767 (32B): a genuine learnability result, but not a necessity result. The model becomes the load-bearing chooser only where the grounding is absent at inference, a grounding-free fully-local deployment we did not build or measure. Prose truth, separately, comes from a non-LLM verifier and from capacity, and is bounded by the verifier's recall. So the honest division is: the engine-and-Maia-and-rule system carries the move, the fine-tune reproduces it (useful mainly if you later remove the grounding), and the verifier plus capacity bounds prose fabrication.

**Prediction or Disconfirmer:** In the grounded product, the deterministic rule matches or exceeds the fine-tuned model's tier-policy match at near-1.0, so removing the model does not lower the canonical-move quality. If the fine-tuned model produced materially better canonical moves than `select_tier_move` on the same grounding, the deployment-necessity claim would flip; that has not been observed, because the model is trained to match the rule.

**How to resolve it:** In the grounded product, compare the canonical move from `select_tier_move` against the fine-tuned model's move on the same held-out grounding; then, separately, build a grounding-free local variant and measure whether the model recovers the policy without the engine and Maia, which is the only setting where it is load-bearing.

**Testing note:** Cold raters found the claim off-consensus about what the product actually is, and the audit sharpened it into the deployment-necessity result: the system, not the weights, produces the move, and the rule is the ceiling. An adversary could not rescue necessity in the grounded product, because the model is trained to match a rule that already runs. It reaches to any system where a deterministic policy over verified signals is the real asset and the model is a convenience. It is Strong, and it is the honest counterweight to the learnability win: the policy is learnable (POV 1), but as built the model is not required (this POV).

**Spiky POV 10 (Strong): This recipe of training a small model to emit a verified, level-appropriate choice generalizes only where the choice has a cheap deterministic checker. Chess is safe for the move because the engine and a human-move model fix the right answer per tier; it is the prose, lacking a complete checker, that is the trap.**

**Elaboration:** The recipe looks general: ground a small model in a solver, distill a teacher, fine-tune, ship a cheap local chooser. Whether it is safe depends on whether the trained output has a cheap deterministic checker. The move does: soundness from the engine, human-findability from a human-move model, and a tier rule combine into a single canonical move, so the trained behavior is verifiable and safe to grade without a judge. The prose does not have a complete checker, which is exactly why the sharpened design does not train the prose. Domains where the trained choice is deterministically checkable, such as tier-appropriate hints in a math tutor graded against a solver, are safe; domains where the trained output is open-ended rationale with no checker are traps. Chess is instructive because it contains both a safe target (the move) and a trap (open-ended prose), and the right move is to train the safe one and render the trap separately behind a verifier.

**Prediction or Disconfirmer:** Across several domains, a small fine-tune reliably adds the trained behavior only where that behavior has a cheap deterministic checker; where the trained output is unverifiable rationale, the fine-tune cannot be graded cleanly and inherits the teacher's errors. A domain whose only trainable target is unverifiable prose that nonetheless grades cleanly would complicate it.

**How to resolve it:** A pre-registered multi-domain study that sorts candidate trained behaviors by whether a cheap deterministic checker exists, then measures whether the fine-tune adds the behavior cleanly in each. This is the one menu item that needs evidence beyond the chess project.

**Testing note:** Cold raters rated the verifiable-choice-versus-unverifiable-rationale distinction durable and non-obvious. An adversary found it resistant if the domains are pre-registered, and it reaches to a code hinter, safe because tests verify, versus a finance rationale, unsafe because nothing verifies. It is Strong rather than Validated only because it awaits that cross-domain study.

**Spiky POV 11 (Strong): There is one output but two independent axes, and the sharpened thesis picks the right one to train. The move is deterministically checkable and weight-learnable, so the fine-tune owns it; prose faithfulness is not reliably weight-learnable, so it is demoted to a verified optional layer. Scoring a coach with one blended quality number hides this and is a category error.**

**Elaboration:** Because the move and the prose are carried by different mechanisms, they move independently, and a single combined score lets a gain on one hide a failure on the other. The v4 checkpoint is the cleanest possible demonstration: it is the best model on the move and the worst recent model on prose at the same time, which is only coherent once the two are separated. Fine-tuning reliably teaches the move, a discrete choice with a right answer per tier, and the litmus shows it does so where a prompt cannot. Fine-tuning does not reliably teach prose truth, because the training transcripts themselves can contain confident wrong claims, and imitating them teaches confident wrongness, which is why the small model's raw prose fabricates more than the untuned base even after the data was filtered. The practical rule is to train the move, report the move deterministically, and treat prose as a separate, verifier-gated number that no fine-tune is expected to carry.

**Prediction or Disconfirmer:** From base to tuned, the deterministic tier-policy match rises steeply while raw prose truthfulness stays low unless a verifier is added. Any fine-tune-only lift of raw prose truthfulness by a large margin, with no verifier, would complicate the split, though it would not touch the move claim.

**How to resolve it:** Measure the base-to-tuned change on deterministic tier-policy match and on raw prose truthfulness separately, with no verifier in the loop, and confirm the move moves while raw prose truth does not.

**Testing note:** Cold raters found the two-axes claim off-consensus once the absolute wording was dropped, and the v4 split makes it concrete. An adversary noted it can leak if "learnable" is stretched, so the crux is pinned to a no-verifier fine-tune, and it reaches to a support bot that nails a discrete routing decision while inventing refund policy in prose. It is Strong, backed by the base-to-tuned split and the v4 move-versus-prose divergence from the project's own runs.

**Spiky POV 12 (Strong): Fine-tuning a small model on raw, unfiltered frontier prose risks being worse than base-plus-prompt on prose faithfulness, because it imitates the teacher's confident-assertion style. This is a reason to keep prose out of the trained objective, or to filter it hard, not a reason to distrust the trained move.**

**Elaboration:** Distillation copies the teacher's manner along with its content, and a frontier teacher narrating chess makes confident claims that are sometimes wrong. A small student trained on those transcripts learns to sound just as sure, including when wrong, and known distillation failure modes push this the wrong way. The project's own numbers show the mechanism: filtering the data beat not filtering it, cutting grounded prose fabrication from about half to about a third, but even the filtered fine-tune fabricates more than the untouched base, because the fine-tune also taught a more assertive, concrete voice with more surface to be wrong about. Under the sharpened thesis this is precisely why prose is not the trained objective: the move is trained because it cannot be fabricated, and prose is either kept out of the objective or filtered with a verifier and rendered separately. The trained move is untouched by this failure mode because a move has no assertion to inherit.

**Prediction or Disconfirmer:** A small model tuned on raw prose transcripts has raw prose truthfulness no better than base-plus-prompt, while a model tuned only on verifier-passed prose beats raw on fabrication, and neither affects the deterministic tier-policy match. If raw-transcript prose tuning matches verifier-filtered prose tuning, the prose half is wrong.

**How to resolve it:** Train two fine-tunes differing only in whether the prose data was verifier-filtered, compare raw prose truthfulness against base-plus-prompt, and confirm the move axis is unchanged by either.

**Testing note:** Cold raters found the amplification claim off-consensus once softened from always net-negative. An adversary noted "value" can be redefined, so the crux is fixed to raw prose truthfulness, and it reaches to distilling a clinician's off-hand wrong guesses into a junior model. It is Strong, and under the sharpened thesis it is a design argument for training the move and gating the prose rather than a threat to the graded claim.

**Spiky POV 13 (Strong): Because no complete verifier for chess prose exists, the optional prose layer should be coverage-bounded: assert only what the detectors can verify and abstain otherwise. Saying less, truthfully, beats saying more, fluently. This governs the display layer, not the trained move.**

**Elaboration:** Motif and threat detectors cover many but not all of the things a coach might say, so there will always be prose claims the system cannot check. For the optional layer the safe default is to speak only inside the verifiable set and stay silent elsewhere, trading richness for truth. The project's own gate is a working instance: when a model cannot produce a fully checkable explanation within a few tries, the system falls back to a short, verified, engine-derived explanation that deliberately says less, at a fallback rate of about one in ten for the small model, guaranteeing zero fabrication on the hardest cases. This is entirely a property of the display layer; the trained move is already truthful by construction because it is a move.

**Prediction or Disconfirmer:** Constraining prose to detector-verified claims raises truthfulness toward the coverage rate while richness drops, and beats say-more variants on truthfulness. If a say-more variant matches the bounded prose on truthfulness while keeping higher richness, the claim is wrong.

**How to resolve it:** Compare a coverage-bounded prose configuration against richer variants on the same positions, measuring truthfulness and a richness proxy together, with the move held fixed.

**Testing note:** Cold raters found the say-less-truthfully rule off-consensus once "only" was softened. An adversary found the crux resolvable by the truthfulness-versus-richness trade, and it reaches to a clinical bot that refuses anything outside the retrieved guideline. The fallback path in the project's own gate already shows the safe direction working, and it stays Strong pending the fuller measurement, scoped to the optional layer.

**Spiky POV 14 (Strong, riskiest): Dependability should be defined as the worst-case, all-constraints-at-once pass rate under repeated deployment sampling, not mean quality. The trained move is the anchor of that stack, and the small tuned model's plausible edge lives on the tail.**

**Elaboration:** A shipped coaching output is only good if several things hold at once: the move is sound and tier-appropriate, it is distinct across levels where it should be, and the rendered prose stays truthful and in register. Averaging quality hides how often the whole stack fails together. The right measurement samples each system many times at deployment temperature and asks how often every constraint passes at once. The move constraints are now deterministic and lead the field, which is the sturdy anchor of the stack; the open question is the repeated-sampling tail across the full stack, which the current single-sample evaluations do not capture. This is the riskiest item because its affirmative half, that the small tuned model wins on that tail once prose is included, has no supporting evidence yet, though the move anchor is measured and strong.

**Prediction or Disconfirmer:** At many samples and deployment temperature, the tuned small model pass-all-constraints rate exceeds an equally grounded prompted frontier's, even if its mean prose is lower, because it anchors the stack on a move axis it leads. If the grounded frontier's pass-all rate is at least the tuned model's, the affirmative is falsified, though the metric still stands.

**How to resolve it:** Repeated sampling at deployment temperature with grounding held constant, scoring the fraction of outputs that pass every constraint at once, comparing worst cases rather than averages.

**Testing note:** Cold raters accepted the worst-case framing as a real, non-obvious measurement choice. An adversary found the crux resistant once the metric is fixed, and it reaches to an aviation-checklist phraser graded on zero omissions across eight of eight runs. The single-sample benchmarks do not touch its affirmative half, so it is Strong on the definition and flagged riskiest, with the move axis as the one part of the stack already won.

**Spiky POV 15 (Strong): Prose faithfulness is table-stakes bought by a verifier and by capacity, so it can never be a moat, and it does not touch the graded move claim. A verify-and-regenerate gate drives verifier-DETECTABLE violations to zero for every model (not certified truth: the checker is high-precision, low-recall), and a larger open base fabricates only a few percent for free.**

**Elaboration:** It is tempting to treat prose faithfulness as the differentiator, since small models fabricate so much more than the frontier. The gate result says otherwise: running each explanation through a checker that vetoes false board claims, re-samples a few times, and otherwise substitutes a verified engine-derived explanation drives verifier-DETECTABLE violations to zero for the small model, from about 40 percent, and to zero for a strong frontier model, from about 7 percent, and across the fifteen-model field it drove every model to zero on the detectable set. This is zero verifier-detectable mechanical violations, not certified truth: semantic falsehoods the detectors miss (relational pawn-SAN claims, forks, threats, negations, eval claims) can still pass, so the between-model collapse is on detectable fabrication only. Independently, the small model's raw deficit is capacity-bound: a 27-billion-parameter open model reaches about 1 percent grounded fabrication for free while the project's own data rebuild only reached about a third. So prose faithfulness is a shared floor any serious system installs, not a place to build advantage. Under the sharpened thesis this matters twice: prose faithfulness is table-stakes for the optional layer, and it is entirely outside the graded claim, because the trained deliverable is a move that cannot fabricate.

**Prediction or Disconfirmer:** With the verify-and-regenerate gate in front of them, models of very different sizes and families all reach near-zero verifier-DETECTABLE user-visible violations, so the between-model spread collapses; and grounded raw fabrication falls steadily as base size rises. If some models keep meaningfully higher fabrication behind the gate, or a much larger base fabricates as much as the small one on identical input, the stance is wrong.

**How to resolve it:** Put the same gate in front of a wide field and score verifier-detectable violations with and without it, and score raw grounded fabrication across a ladder of open model sizes. Both have been run; the gate zeroed the detectable violations across a fifteen-model field (not semantic truth) and the open-model spread holds.

**Testing note:** This is off-consensus because the field treats small-model hallucination as a hard capability limit rather than a solved deployment detail and a capacity artifact. The crux is falsifiable and holds so far. It reaches to any generation task with a cheap external claim checker and a safe fallback. It is Strong because the evidence is the project's own gate and open-model measurements, and under the sharpened thesis it is doubly demoted, both a commodity and off the graded axis.

**Spiky POV 16 (Strong): The reward that trains and grades this coach is now cleanly split, with the sharpened thesis making the primary signal fully deterministic. The move is trained and graded against un-gameable engine-and-tier gates with no judge; any prose council is a held-out, cross-family check on the optional layer that the model never trains against.**

**Elaboration:** Every axis of the sharpened Behavior Spec is deterministic: move soundness from the engine, tier-appropriateness and distinct-moves-per-level from the engine plus a human-move model and the tier rule. None can be flattered, so the primary reward is the deterministic tier-move check itself, which is exactly what the litmus and the 803 leaderboard use. The one thing that would need a model judge, prose instructiveness, is not the trained behavior and is scored only as a held-out, blinded, cross-family council on the optional layer, corrected for the measured self-preference of about 1.44 rank positions and never used as a training target. This is the strongest possible version of the anti-Goodhart rule: because the trained behavior is a move, the primary reward cannot be gamed by a judge at all, and the judge is quarantined to the layer the fine-tune is not graded on.

**Prediction or Disconfirmer:** A coach trained toward a same-family learned judge would climb that judge's score while its deterministic tier-policy match stagnates, whereas training toward the deterministic tier-move gate lifts tier-policy match directly. If training toward a single learned judge improves the deterministic gate as much as training toward the gate does, the trap is not real.

**How to resolve it:** Run two training loops differing only in the reward, one toward a same-family judge and one toward the deterministic tier-move gate, and compare deterministic tier-policy match for each.

**Testing note:** Cold raters found the split sharper than the usual advice to just use a good judge, because it makes the primary reward judge-free by construction. An adversary found the crux resistant once the judge is held out and cross-family, and it reaches to any preference-trained system where a learned reward can be gamed. It is Strong, backed by the measured self-preference and the deterministic move grading the project already uses.

**Spiky POV 17 (Weak, supporting): Human-move modeling is a descriptive learner signal that feeds the deterministic tier rule, not the pedagogical objective. It tells you what a player of a given rating would probably play, which is an input to the canonical tier move, not the thing to teach on its own.**

**Elaboration:** The strength of a human-move predictor like Maia is describing behavior: it predicts the move a rated human would probably make about half the time, which is genuinely useful for meeting a student where they are. But most likely is not most instructive; a likely move can be a misconception or a bad habit. So the human-move signal belongs as a descriptive input to the tier rule that fixes the canonical move, clearly labeled as such, rather than as the selector itself. Under the sharpened thesis this is exactly its role: it is one of the two deterministic signals, alongside the engine, that define the tier-appropriate move the fine-tune is trained to emit. The strong form, that the signal is useless or harmful, does not survive scrutiny, which is why it is a supporting caution.

**Prediction or Disconfirmer:** Two tier rules identical except for how the human-move signal is used, as a raw selector versus as an input to a pedagogical tier rule, will differ, with the tier rule winning on instructiveness. If the raw human-likely selector ties or wins, the caution is wrong.

**How to resolve it:** A dedicated study comparing the two selectors on learning outcomes, separate from the deterministic move evaluation.

**Testing note:** Cold raters flagged the strong version as overstated, and primary sources confirm the human-move signal is useful but not sufficient rather than harmful, so it is softened to a supporting caution. Its reach is narrower and its resolution needs a separate learning study, so it is Weak and kept as support that explains one of the two deterministic inputs to the tier move.

The thread that ties these together is the three-claim map. Claim 1, learnability: what the fine-tune reproduces that the same weights under a prompt do not is the tier-selection policy, validated at 1.7B and 4B (the 1.7B tune leads the field at 0.578 and above every frontier) and a hypothesis at 32B, where the matched prompt control was not run. Claim 2, deployment-necessity: because `select_tier_move` computes the canonical move directly at about 1.0 from the same grounding, the model APPROXIMATES a policy the product already produces, so as built it is not load-bearing and the rule is the ceiling; the model needs the grounding at inference and collapses without it. Claim 3, pedagogy: every number is agreement with our own rule, not evidence coaches or students prefer these moves, so this is behavior validated, not value validated. The supporting structure holds for the optional prose layer: a verified move never buys a verified explanation, a non-LLM gate drives shipped verifier-detectable violations to zero (not certified truth), and capacity buys back most of the small model's prose deficit for free, so prose is table-stakes, not a moat. The genuine, defensible differentiator is that the tuned model reproduces the tier-selection policy the frontier does not follow on the all-scenario metric, which is a distillation result, not a claim to out-teach anyone.

## Experts

These are the voices worth following, including the ones who disagree with each other. The disagreement is the point.

**Asbjorn Steinskog and Anant Dole**

- Who: builders of the Take Take Take and Play Magnus chess coach.
- Focus: shipping a production coach where the engine is the source of truth and the language model only translates.
- Why follow: they argue from production that a language model cannot calculate and should be confined to translating engine and detector output into English, and their shipped coach uses a prompted frontier model plus grounding rather than a fine-tune, which both supports the system-not-the-weights view of prose truth and, under the sharpened thesis, illustrates the frontier-as-prose-renderer role while leaving the leveled move choice unaddressed.
- Where: "Building a Chess Coach," AI Engineer, 2026 - [ai.engineer](https://ai.engineer)

**Zhenwei Tang and the CSSLab C1 team**

- Who: authors of C1, a 4B chess model trained on engine-grounded reasoning distilled from a frontier teacher.
- Focus: a small grounded model that reasons about chess and beats its teacher.
- Why follow: C1 reaches about 48.1 percent puzzle accuracy and surpasses its distillation teacher with far fewer tokens, which shows grounded small models can go further than expected at exactly the 4B size the production coach targets, and is the strongest opposing signal to any translate-only stance.
- Where: [arxiv.org/abs/2603.20510](https://arxiv.org/abs/2603.20510)

**Reid McIlroy-Young and Ashton Anderson**

- Who: creators of Maia, the human-move prediction models, at CSSLab.
- Focus: rating-conditioned modeling of what a human of a given strength would actually play.
- Why follow: Maia predicts human moves about half the time and peaks near its training rating, which makes it a strong descriptive level signal, and it is one of the two deterministic inputs (with the engine) that define the canonical tier move the fine-tune is trained to emit.
- Where: [maiachess.com](https://maiachess.com)

**Nathan Lambert**

- Who: researcher and writer on open models and post-training at Interconnects.
- Focus: the gap between benchmark scores and real deployment robustness.
- Why follow: he warns that open models are very jagged, easy to overfit on benchmarks, and often not specialized enough, and that closed models tend to be more robust where users keep presenting new challenges, which is exactly why the move litmus was run with grounding held constant and graded deterministically.
- Where: [interconnects.ai](https://interconnects.ai)

**Kevin Lu and Thinking Machines Lab**

- Who: authors of the on-policy distillation work.
- Focus: making small models strong in a trained domain while watching what training costs them elsewhere.
- Why follow: they show small models with strong domain training can outperform larger generalists, and they document that fine-tuning small models on new knowledge causes catastrophic forgetting of instruction-following, which is the mechanism behind treating the fine-tune as a narrow move-chooser and rendering prose elsewhere.
- Where: [thinkingmachines.ai](https://thinkingmachines.ai)

**Mathieu Acher**

- Who: professor and strong chess player who benchmarks LLM chess play empirically.
- Focus: how well general and reasoning language models actually play legal, sound chess.
- Why follow: he shows one older model plays around 1750 Elo yet produces an illegal move in about 16 percent of games, and that reasoning models are illegal most of the time, which guts the assumption that a frontier model is a strong chess reasoner out of the box and underlines why the move must be grounded, not generated.
- Where: [blog.mathieuacher.com](https://blog.mathieuacher.com)

**Adam Karvonen**

- Who: researcher on the empirical chess ability of language models.
- Focus: measuring legal-move rates and playing strength across model families.
- Why follow: his work on the one model that plays strong chess, and the finding that chat and instruction tuning degrade a well-defined task, is a caution that fine-tuning can move behavior in the wrong direction if the objective is not held straight, which is why the trained objective is the deterministic tier move.
- Where: [adamkarvonen.github.io](https://adamkarvonen.github.io)

**Simon Willison**

- Who: widely read practitioner writer on applied language models.
- Focus: what actually works when building with models.
- Why follow: he found prompt-engineering results on chess more convincing than fine-tuning, and argues that tools combined with reasoning are the most powerful current technique, which is the mainstream position the byte-identical move litmus is spiky against and now has direct deterministic evidence to challenge for move selection specifically.
- Where: [simonwillison.net](https://simonwillison.net)

**Tim Dettmers**

- Who: author of QLoRA.
- Focus: cheap, low-memory fine-tuning of small and mid-size models.
- Why follow: QLoRA makes fine-tuning a small model nearly free in cost and hardware, which is what makes the last-mile move-chooser role practical, while his own caution that chatbot benchmarks are untrustworthy reinforces grading the move deterministically rather than with a judge.
- Where: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

**Mrinank Sharma and Ethan Perez**

- Who: authors of the sycophancy study in language models.
- Focus: why models, including model judges, prefer convincing answers over truthful ones.
- Why follow: they show a preference model chose a convincing sycophantic answer over a truthful one a large majority of the time, which is the mechanism behind keeping the graded move claim judge-free and gating any optional prose before a preference score.
- Where: [arxiv.org/abs/2310.13548](https://arxiv.org/abs/2310.13548)

**Lianmin Zheng and colleagues**

- Who: authors of the LLM-as-a-judge evaluation.
- Focus: how well a strong model judge agrees with humans, and where it is biased.
- Why follow: they establish that judges reach high human agreement but carry position, verbosity, and self-enhancement biases, which is why the move is graded by an engine and the optional prose, if scored, is judged cross-family and corrected for self-preference.
- Where: [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

**John Sweller**

- Who: originator of Cognitive Load Theory.
- Focus: minimizing extraneous load so limited working memory can build schemas.
- Why follow: the requirement that the optional prose never leak engine internals is a direct application of reducing extraneous load, which gives the no-engine-speak register a real learning-science justification, and the tier rule's shift toward a human-findable move for weaker players is the same principle applied to the move itself.
- Where: [link.springer.com/article/10.1007/s10648-019-09465-5](https://link.springer.com/article/10.1007/s10648-019-09465-5)

## DOK 3: Insights

These are the conclusions that fell out of connecting the sources. Each drew on facts that no single source stated together.

### On what the fine-tune actually adds

**Insight 1: Holding the grounding identical and flipping only the weights isolates what the data adds, which is reproduction of the tier-selection policy a prompt on the same base does not reproduce, validated at 1.7B and 4B.** When grounding is frozen so the only variable is the weights or the system prompt, the fine-tune reproduces the policy where the base and an engineered prompt do not, at 1.7B and 4B; at 1.7B the prompt regressed on cross-tier coherence, and at 4B it produced more varied but mis-directed moves. The matched same-backend 32B prompt control was not run, so the 32B leg is a hypothesis. This connects the base run where move selection under the tier rule was the open axis, the prompt-versus-fine-tune literature that says prompting usually suffices, and the project's own controlled comparison that contradicts it at 1.7B and 4B. It is a learnability result, not a deployment-necessity result: the deterministic rule already computes the move at about 1.0 from the same grounding.

**Insight 2: Narrowing the trained behavior to the move makes the litmus cleaner and the primary grade judge-free, though it does not buy deployment-necessity or validated pedagogy.** Prose was never the clean place to prove a fine-tune earns its keep, because a prompt can already write fluent chess prose, whereas the tier-selection policy is what a prompt on the same base did not reproduce at 1.7B and 4B. And because the deliverable is a move, it cannot hallucinate a board fact, so the core claim needs no verifier, and it has a right answer against the rule, so it needs no judge. This connects the base run split (sound move, unfaithful prose), the prompt-controlled litmus at 1.7B and 4B, and the project's judge-free deterministic scoring of the move against `select_tier_move`. The clean litmus proves learnability, not that the model is necessary (the rule is the ceiling) and not that the rule teaches well.

### On where dependability comes from

**Insight 3: A small model can win only if the system turns coaching into a deterministic, level-appropriate move choice, not open-ended chess reasoning or prose.** The engine supplies which moves are sound, the human-move model supplies which sound move a rating would find, the tier rule turns those into the single canonical move, and the fine-tune learns to emit it locally. Prose, if wanted, is rendered on top and separately verified. Without this narrowing the task is under-constrained, which is why the raw model fabricates in prose. This connects the production coaches that confine the model to translation, the C1 result that grounded small reasoning is possible, the chess-commentary evidence that fluent prose is often wrong, and the base run where move selection under the tier rule was the real open axis.

**Insight 4: The fine-tune is the last-mile move-chooser, and its contribution is the leveled move, which a prompt on the same weights demonstrably does not add, while prose truth is carried by a verifier and by capacity, not the weights.** Move truth comes from the engine and the tier rule; prose truth comes from a non-LLM verifier and from capacity; the fine-tune's own contribution is emitting the tier-appropriate move reliably and locally. This rests on the finding that a smaller aligned model was preferred over a much larger one, on prompt-optimization beating fine-tuning on structured reliability for prose-like axes, on the distillation failure modes that make raw prose fine-tuning risky, and on the project's own litmus, verifier ablation, and the v4 move-versus-prose divergence.

### On how to measure it

**Insight 5: The primary reward and grade are now fully deterministic, which is the strongest form of the anti-Goodhart rule.** Because the trained behavior is a move, tier-policy match and distinct-moves-per-level are computed on the engine plus the human-move model with no judge, so the primary signal cannot be flattered. The one axis that would need a model, prose instructiveness, is not the trained behavior and is quarantined to a held-out cross-family council on the optional layer, corrected for a measured self-preference of about 1.44 rank positions and never trained toward. This connects the chess-commentary error rates, the sycophancy and judge-bias findings, and the project's own deterministic move grading and council design.

**Insight 6: The optional prose layer must still gate faithfulness with non-LLM checks before any quality score, because fluent falsehood contaminates holistic scores, but this now protects a display layer rather than the graded claim.** Every claimed motif, threat, and plan is cross-checked against engine lines and detector output before any holistic score, and any prose quality score comes from a different model family. Chess is unusually gate-able because the engine and detectors form a non-LLM source of truth. This connects the chess-commentary judge that rated false commentary highly, the sycophancy and judge-bias findings, the base run where readable prose still scored zero on truthfulness, and the production gate that drives shipped verifier-detectable prose violations to zero for every model (not certified truth).

### On what to teach and where the advantage lives

**Insight 7: The human-move signal is a descriptive input to the deterministic tier rule, not a prescription for what to teach on its own.** Human-likely is not the same as pedagogically useful, so the signal is one of the two deterministic inputs (with the engine) that define the canonical tier move, clearly labeled descriptive, rather than the selector itself. This connects the measured human-move accuracy and its volatility across adjacent ratings, the industry move toward picking the most human among strong moves, feedback theory, and expertise reversal.

**Insight 8: The defensible differentiator is reproducing the tier-selection policy the frontier does not follow, measured as an all-scenario agreement lead, not a validated coaching moat.** The frontier changes its move across tiers only about a fifth to a third of the time and mirrors the engine's best move to every level about 77 percent of the time, yet about two-thirds of real held-out positions are discriminating, so the policy divergence is common. Because soundness comes from the engine and human-findability from a human-move model, `select_tier_move` fixes the canonical move per tier without any judge, so it can be trained against a mechanical reward and graded cleanly, and the 32B v4 tune leads the twenty-model grand field at tier-policy match 0.767 (the 1.7B tune is #2 at 0.578), while the earlier 803-position field showed the tuned models leading at about 53 percent. The honest unbiased head-to-head over the 96 diverging positions is 56-28-12 (56-28-36 over all 120); the 51-5-6 head-to-head is a selection-conditioned subset, not a win rate, and all of this is agreement with our rule, not evidence of better teaching (claim 3). This connects the gap-density measurement, the frontier-mirror measurement, the prompt-controlled litmus at 1.7B and 4B, and the grand eval.

## DOK 2: Knowledge Tree

This is the verified evidence behind the stances above. Each entry lists its objective facts, a short plain-language summary, and a link. About 120 sources were reviewed across the full effort, and the highest-leverage ones are collected here, grouped by topic. Every load-bearing and off-consensus fact was checked against a primary source, with no fabricated or hallucinated citations.

### A. Distillation and small-model specialization

**Knowledge distillation and step-by-step distillation (Hinton, Vinyals, Dean 2015; Hsieh et al., ACL Findings 2023)**

- Fact: knowledge distillation transfers a large model's soft-target dark knowledge to a smaller student.
- Fact: distilling step-by-step let a 770M model beat a few-shot 540B model while using about 80 percent of the data.
- Summary: distillation can move a specific capability into a much smaller model, which is the mechanism the project is betting on for the move-choosing behavior.
- Link to source: [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

**Small-model parity and specialization (Qwen3 Technical Report 2025; NVIDIA SLM position, Belcak et al. 2025; Finetuner's Fallacy 2026)**

- Fact: a 1.7B base model reached parity with a 2.5B to 3B base model, though this is a pretraining-parity result, distinct from distillation.
- Fact: a position paper argues small models are sufficient and economical for specialized agentic tasks, and a separate result shows a 1B specialized model beating a 3B standard model on under-represented domains through specialized pretraining.
- Summary: small models can match larger ones on narrow targets, but the strongest results lean on specialized pretraining rather than fine-tuning alone.
- Link to source: [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

### B. Prompting versus fine-tuning for reliability

**Alignment and constraint adherence (Ouyang et al. 2022; structured-output reliability 2026)**

- Fact: a 1.3B aligned model's outputs were preferred over a 175B model's, so bigger is not automatically better at following intent.
- Fact: naive prompting reached high task accuracy but zero valid structured output in one study, while prompt-optimization, not fine-tuning, brought a frontier model to about 95 percent valid output.
- Summary: reliability and format adherence are often won by alignment and prompt design rather than by scale, which is the mainstream expectation the project's own byte-identical move litmus was built to test for the leveled move-selection behavior specifically.
- Link to source: [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

### C. Distillation failure modes

**Model collapse and small-student limits (Shumailov et al., Nature 2024; Small Model Learnability Gap, ACL Findings 2025; distillation traps 2026)**

- Fact: training on recursively generated data erases the tails of the distribution, and preserving those tails needs real human data.
- Fact: small models, at or below about 3B, learn better from shorter and simpler reasoning chains, and tail noise plus a teacher-student gap can drive overconfident hallucination.
- Summary: distilling a frontier teacher's confident chess prose into a small student risks copying confident wrongness, which is why prose is kept out of the trained objective and the move, which cannot be fabricated, is trained instead.
- Link to source: [nature.com/articles/s41586-024-07566-y](https://www.nature.com/articles/s41586-024-07566-y)

**Adapter forgetting (LoRA intruder dimensions, NeurIPS 2025)**

- Fact: low-rank adapters introduce intruder dimensions and forget more of pretraining than full fine-tuning, and still trail full fine-tuning on some measures.
- Summary: cheap fine-tuning has a real cost in retained general ability, which reinforces keeping the fine-tune narrow, as a move-chooser, and late.
- Link to source: [arxiv.org/abs/2410.21228](https://arxiv.org/abs/2410.21228)

### D. Chess engines and human-move modeling

**Human-move prediction (Maia, McIlroy-Young et al., KDD 2020; Maia-2, NeurIPS 2024)**

- Fact: Maia predicts human moves about 46 to 52 percent of the time, against roughly 33 to 41 percent for engine-style predictors, with accuracy peaking near the training rating, and personalization can reach up to about 65 percent.
- Fact: the authors note that per-level models can be volatile and incoherent across adjacent ratings and are limited as teaching tools, and that the human-move ceiling is well below 100 percent.
- Summary: human-move modeling is a strong descriptive level signal and one of the two deterministic inputs that define the canonical tier move, which is why it is treated as descriptive rather than as the selector.
- Link to source: [maiachess.com](https://maiachess.com)

**Compact rating-conditioned prediction (Maia-3 / Chessformer, ICLR 2026)**

- Fact: a 79M rating-conditioned model reached about 57.1 percent human-move accuracy at under a quarter of the previous state-of-the-art parameter count.
- Summary: human-move prediction is improving and getting cheaper, which strengthens the deterministic tier rule, but it still describes behavior rather than prescribing what to teach.
- Link to source: [arxiv.org/abs/2605.19091](https://arxiv.org/abs/2605.19091)

### E. LLMs playing and explaining chess

**Empirical LLM chess ability (Acher 2024; Karvonen 2024; reasoning-LLM chess 2025)**

- Fact: one older model plays around 1750 Elo with under 0.1 percent illegal moves at the move level but an illegal move in about 16 percent of full games, while reasoning models are illegal in the large majority of cases.
- Fact: chat and instruction tuning were found to degrade performance on the well-defined task of chess.
- Summary: a frontier model is not a dependable chess reasoner by default, which is why the move is grounded in the engine and the tier rule rather than generated, and why the trained objective is held to the deterministic move.
- Link to source: [arxiv.org/abs/2512.01992](https://arxiv.org/abs/2512.01992)

**Grounded small chess reasoning (C1, CSSLab 2026; faithful reasoning training 2026)**

- Fact: a 4B model trained on engine-grounded reasoning distilled from a frontier teacher, then reinforced, reached about 48.1 percent puzzle accuracy, surpassing its teacher at roughly 40.8 percent, with about 100 times fewer tokens, improving from 42.3 percent after supervised training to 48.3 percent after reinforcement.
- Fact: separate work found best-move supervised training strong but reasoning sometimes unfaithful, while multi-move trajectory training was more faithful.
- Summary: grounded small models can reason well at 4B, which is the size the production coach targets, and the finding that best-move training is strong while free-text reasoning is sometimes unfaithful is direct support for training the move and demoting the prose.
- Link to source: [arxiv.org/abs/2603.20510](https://arxiv.org/abs/2603.20510)

**Commentary hallucination and its evaluation (ACT-Eval 2026; CCC and GCC-Eval, Kim et al., NAACL 2025)**

- Fact: a strong frontier model without tools produced factually incorrect chess claims about 22 percent of the time and smaller open models more than 50 percent, and standard reference-based model-as-a-judge scoring could not reliably detect these hallucinations, rating a false commentary highly.
- Fact: concept-guided generation that integrates an expert model with the language model produces more accurate commentary, and evaluation is more reliable when expert-model knowledge is folded into the judge.
- Summary: fluent chess prose is frequently false and an unaided judge misses it, which is why prose is the optional, verifier-gated layer and the graded claim is the un-fabricable move.
- Link to source: [openreview.net/forum?id=nne0ti66KT](https://openreview.net/forum?id=nne0ti66KT)

### F. Grounded coaching products and shipped small tutors

**Engine-as-truth production systems (Play Magnus and Take Take Take; DecodeChess; Chess.com Game Review 2026)**

- Fact: production coaches use the engine as ground truth and detectors for structured concepts, with the language model confined to translating into English, a choice made because independent chess reasoning by a language model hallucinates.
- Fact: one major platform's game review picks the most human among strong moves so the feedback feels like a real coach.
- Summary: production coaches already confine the model to prose translation over an engine-chosen move, which both supports the optional-prose-layer design and leaves the tier-appropriate move choice, the trained behavior here, as the open axis.
- Link to source: [decodechess.com](https://decodechess.com)

**Shipped small fine-tuned tutors (community LoRA tutors 2026)**

- Fact: a LoRA fine-tune of a 4B model on distilled explanations reported high completeness and near-zero hallucination on a small 50-puzzle test set, and a 270M model was fine-tuned for offline move classification and rating prediction.
- Summary: small fine-tuned chess models exist at the 4B size the production coach targets, but the strongest reports measure prose completeness rather than deterministic tier-appropriate move selection.
- Link to source: [huggingface.co](https://huggingface.co)

### G. Learning science

**Tutoring effectiveness (VanLehn 2011)**

- Fact: intelligent tutoring systems reached an effect size of about 0.76 against no tutoring, close to human tutoring at about 0.79, so structured computer tutoring can approach human tutoring.
- Summary: a well-designed tutor can be nearly as effective as a human, which sets a real bar and motivates getting the level-appropriate move right, since the move is the load-bearing pedagogical choice.
- Link to source: [doi.org/10.1080/00461520.2011.611369](https://doi.org/10.1080/00461520.2011.611369)

**Cognitive load and expertise reversal (Sweller et al. 2019; expertise-reversal literature)**

- Fact: novel information passes through a limited working memory, so instruction should minimize extraneous load, and guidance that helps novices can harm more advanced learners and must fade with proficiency.
- Fact: deliberate practice explains only about 21 to 26 percent of performance variance, less than once claimed.
- Summary: showing a weaker player a more human-findable move rather than the engine's sharpest line is expertise-reversal applied to the move itself, and no-engine-speak in the optional prose is load reduction.
- Link to source: [link.springer.com/article/10.1007/s10648-019-09465-5](https://link.springer.com/article/10.1007/s10648-019-09465-5)

### H. Evaluation: LLM-as-judge and sycophancy

**Judge validity and sycophancy (Zheng et al., NeurIPS 2023; Sharma et al., ICLR 2024)**

- Fact: strong model judges reach over 80 percent agreement with humans but carry position, verbosity, and self-enhancement biases.
- Fact: a preference model preferred a convincing sycophantic answer over a truthful one the large majority of the time, and sampling many candidates only partly reduced this.
- Summary: model judges are useful for style but unreliable for truth and biased toward their own family, which is why the graded move claim uses no judge and any prose score is cross-family and corrected.
- Link to source: [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

### I. The project's own measurements

These are the project's own internal measurements, not outside primary sources. For a claim about this specific system, such as whether this fine-tune beats this prompt on the move, a controlled experiment is the appropriate primary source, because no outside literature can settle it. The move-selection axis is graded deterministically on the engine and the human-move model with no judge, which is what makes these measurements clean.

**The move litmus, base versus tuned versus best-prompted base, with a prompt control at 1.7B and 4B (project measurement)**

- Fact: holding the shipped grounding identical and grading the move deterministically against `select_tier_move`, the fine-tune beat both its untuned base and the best engineered prompt on that base at the sizes where a prompt arm was run. On the earlier litmus slice: at 1.7B tier-policy match 0.296 (base) and 0.389 (prompt) versus 0.463 (tune), coherence violation 0.500 and 0.611 versus 0.333; at 4B 0.347 and 0.350 versus 0.386. On the 20-model grand slice: the 4B trio is base 0.353, prompt-base 0.378, tune 0.397, and the 1.7B tune is 0.578 (base 0.358), #2 of 20.
- Fact: at 32B there is a base-versus-tune result (0.347 versus 0.767) but NO matched same-backend prompt control was run, so the claim that a prompt cannot reproduce the policy at 32B is a hypothesis, not a measured result.
- Fact: prompting failed on the graded axis at the small sizes rather than merely trailing: at 1.7B the engineered prompt pushed cross-tier coherence violation to 0.611, worse than the untuned base, and at 4B it produced more varied but mis-directed moves and still lost on the policy match.
- Summary: with grounding frozen so only the weights or the prompt change, fine-tuning reproduces the policy where a prompt on the same base does not, validated at 1.7B and 4B and hypothesized at 32B; and because `select_tier_move` already computes the same move at about 1.0 from that grounding, this is learnability, not deployment-necessity.
- Link to source: the project's own 1.7B and 4B prompt-controlled move litmus plus the 20-model grand eval (internal measurement)

**The 32B v4 all-scenario lead and selection-conditioned head-to-head, and the prose trade (project measurement)**

- Fact: on 120 held-out validation positions across three tiers, the 32B v4 fine-tune reached tier-policy match 0.767 versus about 0.49 to 0.55 for the frontier (best frontier Gemini 0.553), distinct-moves-per-level 0.730 (73/100 canonical beginner!=advanced opportunities) versus roughly 0.21 to 0.28, and move-soundness 0.942. The honest unbiased head-to-head over all 96 diverging positions is 56-28-12 (56-28-36 over all 120). The 51-5-6 head-to-head is SELECTION-CONDITIONED: it is computed only on the 62 of 120 positions where v4 already gives a distinct, sound, correctly-graded move AND diverges from the frontier, so it is a subset figure, not a win rate over all positions.
- Fact: the same v4 checkpoint is the weakest recent model on prose, blinded instructiveness grade 4.67 below the 4B tune at 5.32 and the prior 32B v3 at 6.35, with about 40 percent of raw drafts failing the prose check before the gate; prose is secondary to the evaluation claim (still in the SFT loss), and the gate drives shipped verifier-detectable violations to zero, which is not certified truth.
- Summary: the strongest-move checkpoint being the weakest-prose checkpoint follows from separating the trained move from the prose; v4 leads on agreement with our tier-selection rule, not on validated coaching, and the head-to-head is a subset figure, not a win rate.
- Link to source: the project's own v4-centered honest eval and 4B eval (internal measurement)

**The definitive twenty-model grand evaluation, v4-centered (project measurement)**

- Fact: on the shipped held-out validation slice of 120 positions across three tiers, scored across all twenty models with identical grounding and the single strict any-legal move extractor, the 32B v4 tune leads the field on tier-policy match at 0.767, the 1.7B tune is #2 at 0.578 (above every frontier), the tuned checkpoints take four of the top five, and the only frontier in the top five is Gemini at fourth (0.553). The deterministic `select_tier_move` rule scores about 1.0 by construction on the same grounding and is the ceiling.
- Fact: the same v4 checkpoint is intentionally weaker on the blinded cross-family prose council, about fifteenth of twenty at an instructiveness grade of 4.53, on-thesis because prose is secondary; the prior v3 all-rounder sits about fifth at 6.43, and the faithfulness-filtered v5 retrain regressed to tier-policy match 0.536 and about nineteenth on prose without improving raw faithfulness (near 0.58), BUT the v5 run was confounded (about 27 percent less optimization/token exposure, contrastive triads broken by row-wise filtering, about 42 percent boilerplate-principle pollution, retrained from base not from v4, no checkpoint selection), so it does not isolate filtering as the cause. The whole grand evaluation cost about 54 dollars.
- Fact: the grand evaluation was audited for fairness on two axes, both clean: the human-move model is symmetric across all twenty models, feeding the ground-truth tier move and every model's grounding equally, and the 120-position validation set has zero train-test leakage against the shipped model's fine-tuning data, a board-key intersection of zero out of 120; re-scoring the published generations reproduces tier-policy match 0.767 and distinct-moves 0.730 exactly.
- Summary: the definitive twenty-model evaluation confirms the tuned models reproduce the tier-selection policy better than the field on a clean, held-out, symmetrically grounded set, with the deterministic rule as the ceiling; it does not validate pedagogy or deployment-necessity.
- Link to source: the project's own twenty-model grand evaluation (internal measurement)

**The earlier 803-position gap leaderboard across the model field (project measurement)**

- Fact: on a curated, zero-leakage set of 803 held-out positions, each discriminating so that the tier-appropriate move differs from the engine's first choice for at least one tier, scored across fifteen models with identical grounding, the tuned models reached the highest tier-policy match in the field at about 53 percent, above every frontier model at about 43 to 48 percent, with the widest lead at the beginner and intermediate tiers, and the frontier mirrored the engine's best move at every tier a high fraction of the time.
- Fact: tier-policy match is weak across the whole field, most models between about a third and a half, precisely because it is a trained behavior rather than an emergent one; faithfulness after the verifier is a fairness floor at zero verifier-detectable violations (not certified truth) for all fifteen models and is deliberately not a scoring axis; the blinded prose council's measured self-preference was about 1.44 rank, corrected in the reported ranking; the whole evaluation cost about 112 dollars.
- Summary: the large evaluation confirms tier-policy match is the one axis where the small trained models lead the field while the field stays weak because the behavior is trained rather than emergent, all graded deterministically against the rule with no judge; leading on policy match is not the same as validated teaching.
- Link to source: the project's own 803-position gap evaluation, earlier and larger-position (internal measurement)

**The verify-and-regenerate faithfulness gate for the optional prose layer (project measurement)**

- Fact: the production gate, which re-samples an explanation up to four times and otherwise substitutes a verified engine-derived explanation, drove verifier-detectable prose violations from about 40 percent to zero for the small model and from about 7 percent to zero for a frontier model, and across the fifteen-model field drove every model to zero on the detectable set. This is zero verifier-detectable mechanical violations, not certified truth: semantic falsehoods the detectors miss (relational pawn-SAN claims, forks, threats, negations, eval claims) can still pass.
- Fact: the small model fell back to the verified explanation about one time in ten and the frontier about one time in fourteen, and no raw model reached zero on its own, because even the frontier repeated the same false claim across retries.
- Summary: a claim-level non-LLM gate with a deterministic fallback guarantees zero verifier-detectable prose violations for any model (not certified truth), which makes mechanical faithfulness table-stakes for the optional layer and, under the three-claim framing, outside the graded move claim.
- Link to source: the project's own verifier evaluation (internal measurement)

**Bigger open models on the same input (project measurement)**

- Fact: on the identical grounded input, larger open models fabricated between about 1 and 8 percent in prose, with a 27 billion parameter open model at about 1 percent matching the frontier, while the tuned small model sat near a third and the untuned base near an eighth.
- Fact: every open model was judged more instructive in prose than the small model but none reached the frontier, and the very largest model did not coach best, so training quality and size both matter more than raw parameter count for the prose voice, while the best locally runnable base was a model of about 27 to 32 billion parameters.
- Summary: the small model's prose deficit is closed for free by capacity, not by the data intervention, and a mid-size open base is the natural stronger starting point for a local coach, none of which touches the graded move claim.
- Link to source: the project's own open-model benchmark (internal measurement)

**Base evaluation and the move-versus-prose split (project measurement)**

- Fact: an untuned 4-bit small base model, scored by a strong frontier judge, reached move soundness 1.00 while prose truthfulness was zero and no-engine-speak was 0.11 on the same outputs at the same time.
- Fact: rebuilding the training data to be faithfulness-filtered, tier-aware, and more concrete cut grounded prose fabrication from about half to about a third and lifted prose instructiveness, but even the filtered fine-tune fabricates more than the untuned base, so prose truth is not reliably weight-learnable at small size while the move under the tier rule is.
- Summary: on the same outputs the move was sound while the prose was unfaithful, which is the earliest evidence that the move and the prose are separate axes and that the move is the one to train.
- Link to source: the project's own base and retrain evaluations (internal measurement)

**Move-selection gap, density, and richer input at inference (project measurement)**

- Fact: across the frontier models on held-out positions, tier-differentiation averaged about a fifth to a third, the frontier repeated the engine's best move across the tiers about 77 percent of the time, and on a large curated set about two-thirds of the decidable positions were discriminating, so the leveled-move gap is common in ordinary play rather than a niche.
- Fact: replacing the trained prose grounding with a fuller structured board state at inference raised the small model's prose fabrication from about 40 percent to about 56 percent, while the same change barely moved a frontier model, which is format-agnostic.
- Summary: the frontier is weak at leveled move selection but strong at not lying about the board in prose, the leveled-move behavior is exercised in most normal positions, and a small fine-tune is coupled to its trained input format, so any prose faithfulness is fixed by the verifier and capacity while the trained move is what leads the field.
- Link to source: the project's own gap-density and rich-grounding analyses (internal measurement)

**Reward design and the deterministic primary signal (project measurement)**

- Fact: the training and evaluation loop uses a fully deterministic primary reward (tier-appropriate move, distinct-moves-per-level, move soundness, well-formed, no-engine-speak) and, only for the optional prose layer, a held-out blinded cross-family instructiveness council of three frontier judges that the model never trains against, self-preference-corrected by leaving out each judge's own family.
- Summary: because the trained behavior is a move, the primary reward is judge-free and un-gameable, and the learned prose judge is quarantined to the optional layer, which is the strongest realization of the anti-Goodhart rule.
- Link to source: the project's own training and evaluation harness (internal measurement)

### J. Economics and local deployment (secondary, low-confidence)

**Cheap fine-tuning and local inference (QLoRA and on-device runtimes)**

- Fact: quantized low-rank fine-tuning of a small model is inexpensive in cost and hardware, and on-device runtimes keep data local for privacy, though the specific cost and speed figures come from vendors and practitioners and were not independently verified.
- Summary: the form-factor advantages of a small local move-chooser are real in kind, so they are used only as the honest deployment win and never as a load-bearing number.
- Link to source: [unsloth.ai](https://unsloth.ai)
