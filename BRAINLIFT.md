# What a Fine-Tune Actually Buys a Chess Coach: A Tier-Selection Policy That Distills Into Weights, a Named Deterministic Ceiling, and an Honest Map of What Is and Is Not Validated

**Owner:** Khoi Lam

## Purpose

This BrainLift answers one honest question for anyone building an AI chess coach, or any tool that has to turn a verified answer into a level-appropriate recommendation. Can a small open model, fine-tuned on engine-grounded distilled data, reproduce a tier-selection policy that a prompt on the same base cannot, and what does that actually buy? The measured answer comes in three parts that are easy to run together and worth keeping apart: the policy does distill into weights; that does not make the model necessary in the product we shipped, because a short deterministic rule already computes the same move from the same grounding; and none of it validates teaching quality, because every number here measures agreement with our own rule rather than what helps students. Keeping those three apart is what makes the work rigorous instead of a scoreboard.

The trained job is a single behavior. Given a chess position and the student's rating tier (beginner about 1000 to 1200, intermediate about 1300 to 1600, advanced about 1700 to 2000), the model emits the move the project's canonical policy designates for that tier, rendered as the move plus a short principle tag, for example "Nf3, develop toward the center." That one choice is graded pass or fail on three deterministic clauses a stranger can check with no opinion in the loop: it is sound, meaning not a blunder (engine centipawn loss below 250, within the sound pool described below); it matches the canonical tier move; and it is distinct across levels, so a beginner and an advanced player are not handed the same move on a position that should separate them. The coaching prose that can sit beside the move (the plain-English reason) is a secondary, optional display layer, not the trained target.

The name of the headline metric matters. What earlier drafts called "tier-fit" is here called tier-policy exact match, shortened to tier-policy match: exact agreement with the preregistered `select_tier_move` policy. That canonical move is a project rule, not validated pedagogy, and this document never treats agreement with it as proof of teaching quality.

The first claim, learnability, is validated and is the assignment's real win. With Maia grounding held identical on both sides, a fine-tune reproduces the policy where a prompt on the same base cannot. The genuinely small, on-spec model carries the result: the Qwen3-1.7B tune lifts tier-policy match from 0.358 at its base to 0.578, second of a twenty-model field and above every frontier model (best frontier Gemini 3.1 Pro at 0.553), running locally on a leakage-checked held-out slice. The 1.7B tune (0.578) beats the 4B tune (0.397), so the behavior comes from the data and the contrastive signal, not from raw capacity. The shipped Qwen3-32B v4 (base 0.347 to 0.767 raw, 0.789 as served) is the strongest mid-size instance of the same result, and it is not a small model.

The second claim, deployment-necessity, is false as built and is stated plainly rather than hidden. The same Stockfish sound pool and Maia policy that feed the model's prompt also feed a roughly twenty-line deterministic rule, `select_tier_move`, which computes the canonical tier move directly. That rule scores about 1.0 by construction, because the move it returns is the target the model is graded against. So as built, the fine-tuned model approximates a policy the product already produces without it. The deterministic rule is the true ceiling and the honest baseline. The model would become load-bearing only in a grounding-free, fully-local deployment with no engine or Maia at inference, which was not built or measured.

The third claim, pedagogy and value, is unvalidated. Every number here measures agreement with our own heuristic, not evidence that coaches or students prefer these moves or that students improve. The honest status is behavior validated and a feature demonstrated: not a product, not a moat, and not value validated.

### In Scope

Whether a small fine-tuned engine-grounded model reproduces the tier-selection policy more reliably than the same base under a prompt, graded deterministically; the deterministic rule as the named ceiling and the honest question of when the model is load-bearing; the split between the move (the trained, checkable behavior) and the prose (a secondary layer that is still in the training loss but not separately optimized); where dependability comes from, meaning the engine, the human-move model, and the tier rule versus the weights; the honest size and location of the fine-tune's contribution; the faithfulness gate for the optional prose and its recall limits; and the reward design that keeps the primary signal deterministic.

### Out of Scope

Validated pedagogy or learning outcomes, which no number here establishes; deployment-necessity in a grounding-free local product, which was not built; a matched same-backend 32B prompt control, so the claim that a well-prompted 32B base cannot reproduce the policy stays a falsifiable hypothesis, not a result; certified best-move truth beyond the shallow sound pool; making a language model itself play strong chess; the raw cost and hardware absolutes, kept low-confidence; a clean lesson from the v5 retrain, which regressed under several simultaneous confounds and does not isolate any single cause; non-chess domains except as an explicit generalization test; and any claim that could not be tied to a primary source or the project's own measurement.

## DOK 4: Spiky Points of View

These are a depth-gated menu of candidate-valid stances, positions worth testing rather than settled truths, and deliberately not one chosen winner. Most can only be resolved by an experiment the reader would run, so the useful deliverable is the honest set of testable stances with the backing to choose among them. Each is labeled by how strong its backing is now: Validated (established by the project's own measurement or primary sources and still off-consensus), Strong (candidate-valid, awaiting a further test), and Weak (a softened caution). Read the whole set under the three-claim lens. The first four carry the honest thesis and are the ones to read first: grounding carries the selection and the fine-tune is its local executor (POV 1); a verified move is not a verified explanation (POV 2); unaided model judges pass fluent chess falsehoods (POV 3); and behavior from data is the real win while beating the frontier is a bounded bonus (POV 4).

**Spiky POV 1 (Strong): Grounding carries the selection. The fine-tune is the reliable local executor of a policy the pipeline already computes, not weights that have internalized the rule, and it collapses to one move across tiers when the human-move grounding is missing at inference.**

Elaboration: It is natural to treat the distilled dataset and the fine-tuned checkpoint as the product. The evidence splits the roles. The canonical move's correctness comes from the engine, the human-move model, and the tier rule, all parts that are not the weights. `select_tier_move` is a pure function of the tier, the sound pool, and the Maia policy, and it returns exactly the move the model is trained and graded to match, so in the shipped grounded product it already produces that move at about 1.0 by construction while the model reproduces it at 0.578 (1.7B) to 0.767 (32B). That is a genuine learnability result, not a necessity result. The tuned model also does not internalize the rule: at inference it still needs Maia's per-tier grounding in the prompt, and when Maia was missing on the serving container the three tiers collapsed to a single move, on both the quantized and the full-precision endpoints. The honest claim about the weights is reliable grounded execution, and the degradation without grounding is collapse, not graceful decline.

Prediction (or Disconfirmer): In the grounded product, the deterministic rule matches or exceeds the fine-tuned model's tier-policy match at near 1.0, so removing the model does not lower canonical-move quality. If the fine-tuned model produced materially better canonical moves than `select_tier_move` on the same grounding, or recovered the policy with no engine and no Maia at inference, the necessity picture would change.

How to resolve it: In the grounded product, compare the rule's move against the model's move on the same held-out grounding; then, separately, build a grounding-free local variant and measure whether the model recovers the policy without the engine and Maia, which is the only setting where it is load-bearing.

Validation: Cold raters found it off-consensus about what the product actually is. The crux (the rule computes the move at about 1.0 from the same grounding) is resistant, because the model is trained to match a rule that already runs, and it is backed by the rule being the target by construction and the observed grounding-absent collapse. It reaches to any system where a deterministic policy over verified signals is the real asset and the model is a convenience.

**Spiky POV 2 (Validated): A verified move is not a verified explanation. Choosing the sound move and narrating why are separate tasks, so the prose is the harder, still-open part and must not be the trained and graded behavior.**

Elaboration: It is tempting to assume that once the engine has picked and verified the move, the explanation inherits that correctness. It does not. The engine certifies the choice, but the words about why, such as "this knight is trapped" or "this threatens mate in two," are generated by the language model and are only as reliable as that model. The project's base run makes the split concrete, with move soundness at 1.00 and prose truthfulness at zero on the same outputs at the same time, and the broader evidence agrees, with a strong frontier model making factually incorrect chess claims about 22 percent of the time and smaller open models more than half the time regardless of whether the move was right. The clean lesson is to train the move, which cannot fabricate a board fact, and render the prose separately behind a non-LLM claim checker.

Prediction (or Disconfirmer): On positions where the move is engine-verified as sound, a model with no claim-level verifier still makes at least one false tactical claim in a large share of its free-text explanations, so prose faithfulness cannot be assumed from a verified move. A model whose unverified prose is faithful on its own at high rates would weaken the need to demote prose, though it would not touch the move claim.

How to resolve it: Hold move grounding constant and count fabricated claims per free-text explanation with and without a claim-level verifier, confirming that the move being sound does not make the prose sound.

Validation: Cold raters found the split genuinely non-obvious once the why is separated from the what. An adversary could not make it retreat, because the move-versus-reason split is concrete, and against primary sources it holds as established, from the 1.00 versus zero base run and the measured chess-claim error rates. It reaches to radiology, where a report can verify the nodule and still invent a comorbidity in the impression.

**Spiky POV 3 (Validated): Unaided model judges, and especially same-family ones, pass fluent chess falsehoods as truthful. That is why the move is graded deterministically and any prose score uses a non-LLM verifier first and a cross-family council.**

Elaboration: A language model asked to judge a chess explanation does not run an engine check; it reacts to fluency and confidence. In a controlled chess-commentary evaluation a vanilla judge rated a hallucinated commentary about 4.9 out of 5 while two of its three factual claims were false, and the project's own base run showed a strong frontier judge returning a truthfulness score of zero on outputs it still rated as readable. The project's blinded cross-family council later measured same-family inflation directly, at about 1.44 rank positions, with every judge favoring its own lab's model. So the move needs no judge at all because it has a deterministic right answer, and the sycophancy and same-family problems that plague explanation scoring never touch the core claim. Where the optional prose layer is scored, faithfulness is gated with a non-LLM engine-and-detector check first and any quality score is drawn from a different family, corrected for self-preference.

Prediction (or Disconfirmer): On held-out positions, an unaided or same-family model judge passes as truthful a much larger share of explanations than a non-LLM gate accepts, so a judge cannot certify prose truth, while the move axis is unaffected because it uses no judge. If the model judge and the non-LLM gate agree within noise, the prose half is wrong.

How to resolve it: Score one batch of prose twice, once with the engine-and-detector gate and once with an unaided model judge, compare a same-family judge against a different-family judge, and confirm the move axis is graded without either.

Validation: Cold raters split on whether a model judge is fine for chess, which marks it off-consensus. The crux holds against primary sources, settled by the chess-commentary judge result and the project's own zero-truthfulness reading, and reaches to any setting with a cheap external checker, such as a legal assistant whose sibling-model grader blesses fabricated citations.

**Spiky POV 4 (Validated at 1.7B and 4B; hypothesis at 32B): Behavior from data is the real, on-spec win. A fine-tune reproduces the tier-selection policy where a prompt on the same base cannot, and it tracks the data, not model size. Leading the frontier on this metric is a bounded bonus, not the thesis.**

Elaboration: The field's working belief is that for a narrow behavior a good prompt matches or beats a fine-tune. This project ran the clean version of that test, graded deterministically against `select_tier_move` with no model judge, freezing the grounding so the only variable was the weights or the system prompt. The fine-tune reproduced the policy where the base did not: the 1.7B tune reaches 0.578 (base 0.358), second of twenty and above every frontier, and the 4B trio is base 0.353, prompt-base 0.378, tune 0.397, so the tune beats the prompt beats the base on the same slice. Because the 1.7B tune (0.578) beats the 4B tune (0.397), the lift is from the data, not capacity. The 32B tune reaches 0.767, but the matched same-backend 32B prompt control was never run, so the 32B leg is a hypothesis. The bonus, honestly bounded: on the all-scenario metric the tuned model leads the field, reaching 0.767 against about 0.49 to 0.55 for the frontier and giving distinct moves across tiers about 73 percent of the time (distinct-moves-per-level 0.730, 73 of 100) against roughly 21 to 28 percent, because the frontier hands the engine's single best move to every level and repeats one move across the three tiers about 77 percent of the time. This is a lead on agreement with our own rule, not evidence the frontier coaches worse.

Prediction (or Disconfirmer): With grounding held identical, the fine-tuned model reproduces `select_tier_move` at a materially higher rate than both its untuned base and the best engineered prompt on the same weights, at 1.7B and 4B. If a matched same-backend engineered prompt on the base reaches the fine-tune's tier-policy match, the claim is wrong; running that control at 32B is the open test.

How to resolve it: Keep the grounding identical and vary only the weights and the prompt, then score base, best-prompted base, and fine-tune on the same held-out set with the deterministic check. This has been done at 1.7B and 4B and the tune wins; the matched 32B prompt arm is the missing confirming or disconfirming work. For the frontier comparison, the honest unbiased number is the all-scenario tier-policy match, 0.767 versus 0.553. In the position-by-position head-to-head, the selection-conditioned showcase subset (the 62 of 120 positions where v4 already gives a distinct, sound, correctly-graded move and diverges from the best frontier) is 51 wins, 5 losses, and 6 ties, which overstates a raw win rate and is a subset figure only. Over all the positions where v4's move diverges from the best frontier's, with no success gate, it is about 53 wins, 18 losses, and 12 ties: several times more losses than the conditioned subset, still a clear net win, and the direction holds.

Validation: Cold raters treated a flat "a well-prompted base cannot match the tune on this policy" as genuinely disputed, since the field assumes prompting suffices for a narrow behavior. An adversary could not make it retreat at 1.7B and 4B once the grounding is frozen and the metric is the deterministic policy match, and against primary sources the crux holds as established there (it is the project's own controlled experiment, the appropriate primary source for a claim about this system) and stays a hypothesis at 32B. It reaches to any leveled recommender whose target is checkable without a model, such as difficulty-tiered hints in a math or programming tutor.

The remaining candidate-valid stances complete the menu, ordered by strength.

**Spiky POV 5 (Validated): Because the trained deliverable is a move and not prose, the primary claim is fully deterministic. A move cannot hallucinate a board fact, so the core claim needs no verifier and no model judge; it is graded purely as tier-policy match against the canonical tier move.**

Elaboration: The sharpest consequence of making the move the behavior is that the two hardest measurement problems in the old prose-centric framing disappear from the graded claim. A prose explanation can invent a fork, a pin, or a mate that is not there, which is why the old framing needed a claim-level verifier and a cross-family council. A move cannot do any of that: it is a single legal action, whose soundness comes from the engine and whose level-fit comes from the engine plus a human-move model. So the entire faithfulness apparatus is unnecessary for the thing being graded, and the score is a deterministic comparison to a fixed canonical move. This is a hardening, not a softening: the project's move-axis numbers are all judge-free, and re-scoring the published generations reproduces them exactly.

Prediction (or Disconfirmer): The core behavior can be scored to full agreement by two independent deterministic checkers (engine soundness plus canonical-tier-move match) with no model judge, and repeated scoring returns identical results. If grading the move requires a model judge to resolve disagreement, the claim is wrong.

How to resolve it: Re-score the same generations twice with the deterministic tier-move checker and confirm identical tier-policy match, and confirm that no board-fact fabrication metric applies to a bare move because there is no board claim to check.

Validation: Cold raters found it non-obvious that narrowing the deliverable removes the hallucination problem by construction rather than by better verification. An adversary could not make it retreat, because a move has no free-text claim to falsify, and it holds as established because it follows from the definition of the deliverable and is instantiated by every judge-free move measurement in the project. It reaches to any recommender that outputs a discrete verifiable choice rather than a rationale.

**Spiky POV 6 (Strong): The coaching prose is an optional display layer, so a model's prose quality, including the 32B v4 prose regression, is irrelevant to the graded claim. Render prose with the engine's templates or a frontier model on top of the tuned move.**

Elaboration: Because the trained behavior is the move, the prose is free to be produced by whatever writes English best, and the choice of writer is a product decision, not a training one. The v4 evidence makes the separation vivid. The 32B v4 is the strongest model on the move, leading the field on tier-policy match and distinct-moves, and at the same time the weakest recent checkpoint on prose, with a blinded instructiveness grade of 4.67 below the 4B tune at 5.32 and the prior v3 at 6.35, and about 40 percent of its raw drafts failing the prose faithfulness check before the gate. Under a prose-centric thesis that regression would sink v4. Here it is a non-issue: v4 is kept for the move it chooses, and the prose beside that move is rendered by templates or a prompted frontier model, then run through the same non-LLM gate.

Prediction (or Disconfirmer): Swapping the prose writer, from the tuned model's own text to engine templates or a frontier renderer, leaves the graded tier-policy match unchanged, because the move is chosen before the prose is written. If changing the prose writer changes the graded move axis, the layers are not separable and the claim is wrong.

How to resolve it: Hold the tuned model's chosen move fixed, render its prose three ways (self, templates, frontier), and confirm tier-policy match is identical while only the prose quality varies.

Validation: Cold raters found the demotion of prose to an optional layer off-consensus, since most coaching work treats the explanation as the deliverable. An adversary found the crux resistant because move choice provably precedes prose rendering, and it is Strong on the project's own v4 split, where the strongest-move model is the weakest-prose model, which is only coherent once prose is the optional layer.

**Spiky POV 7 (Strong): When a small local coach beats a frontier model on anything other than the move, what it actually wins is form factor. On prose it trails; its clean edges are cost, latency, privacy, offline use, and the leveled move choice.**

Elaboration: The honest framing separates three axes. One is the leveled move choice, which the fine-tune genuinely adds and now leads the frontier on. Another is prose, meaning explaining the move correctly at the right depth. A third is form factor, meaning running cheaply, privately, offline, and fast. On the prose axis, with grounding held constant, a blinded cross-family council still judges the frontier models more instructive than the tuned small model, and the frontier fabricates a board fact only about 3 percent of the time against the small model's roughly a third before the verifier. So with grounding equalized the small model does not win prose, and it does not need to, because prose is the optional layer. Its clean standalone wins are the deployment envelope and the leveled move choice.

Prediction (or Disconfirmer): With identical grounding, a prompted frontier model matches or beats the tuned small model on prose instructiveness under repeated sampling, while the tuned model leads on the deterministic move axis. If the tuned small model clearly beats the equally grounded frontier on prose instructiveness, this specific claim is wrong; note it is about prose, not the move.

How to resolve it: Score prose instructiveness separately from cost, latency, and the deterministic move axis, under repeated sampling at deployment temperature with grounding held constant.

Validation: Cold raters put the core near the mainstream, so it earns its edge from the sharp reframing that the field mislabels an economics win as a capability win. An adversary found the crux resistant once grounding is equalized. The single-sample run supports it and the repeated-sampling version is pending, so it stays Strong. It reaches to on-device speech-to-intent, where the win is latency and privacy rather than accuracy.

**Spiky POV 8 (Strong): On the optional prose layer, the small tuned model's honest capability win is register consistency, not truth. A narrowly tuned small model should be the lowest-variance renderer of a plain, no-engine-speak voice under sampling.**

Elaboration: Within the optional prose layer there are two separable qualities. One is faithfulness, which the non-LLM verifier carries for whichever model writes. The other is register, meaning never leaking centipawns or deep engine lines and holding a steady voice for a given rating. Register is exactly the narrow stylistic behavior fine-tuning compresses well into a small model. The base run hints at the size of the prize, with no-engine-speak at 0.11 for the untuned small model. The newest measurements complicate this honestly: when the frontier models are handed the same grounded prompt they also stay jargon-free almost all of the time on a single greedy pass, so on a one-shot pass the register axis is near a tie. The bet relocates onto variance, that under repeated sampling at deployment temperature the tuned model holds the register with a lower failure rate, and that comparison has not been run.

Prediction (or Disconfirmer): Under repeated sampling at deployment temperature, the tuned small model beats an equally grounded prompted frontier on no-engine-speak and register-consistency pass rates. If the grounded frontier matches or beats the tuned model on register consistency under sampling, the claim is wrong.

How to resolve it: Sample both systems many times at deployment temperature with grounding held constant and compare the register pass rates and their variance, kept separate from both truth and the move axis.

Validation: Cold raters treated this as a real empirical bet rather than a truism. An adversary found it resistant as long as the register metrics are frozen in advance. On a single pass the grounded frontier now matches the tuned model, so the claim rests on the unrun repeated-sampling measurement; it is Strong and explicitly awaiting that number. It reaches to on-device command-grammar parsing.

**Spiky POV 9 (Strong): This recipe of training a small model to emit a verified, level-appropriate choice generalizes only where the choice has a cheap deterministic checker. Chess is safe for the move because the engine and a human-move model fix the right answer per tier; the prose, lacking a complete checker, is the trap.**

Elaboration: The recipe looks general: ground a small model in a solver, distill a teacher, fine-tune, ship a cheap local chooser. Whether it is safe depends on whether the trained output has a cheap deterministic checker. The move does: soundness from the engine, human-findability from a human-move model, and a tier rule combine into a single canonical move, so the trained behavior is verifiable and safe to grade without a judge. The prose does not have a complete checker, which is exactly why the sharpened design does not train the prose. Domains where the trained choice is deterministically checkable, such as tier-appropriate hints in a math tutor graded against a solver, are safe; domains where the trained output is open-ended rationale with no checker are traps.

Prediction (or Disconfirmer): Across several domains, a small fine-tune reliably adds the trained behavior only where that behavior has a cheap deterministic checker; where the trained output is unverifiable rationale, the fine-tune cannot be graded cleanly and inherits the teacher's errors. A domain whose only trainable target is unverifiable prose that nonetheless grades cleanly would complicate it.

How to resolve it: A pre-registered multi-domain study that sorts candidate trained behaviors by whether a cheap deterministic checker exists, then measures whether the fine-tune adds the behavior cleanly in each.

Validation: Cold raters rated the verifiable-choice-versus-unverifiable-rationale distinction durable and non-obvious. An adversary found it resistant if the domains are pre-registered. It is Strong rather than Validated only because it awaits that cross-domain study, the one menu item that needs evidence beyond the chess project. It reaches to a code hinter, safe because tests verify, versus a finance rationale, unsafe because nothing verifies.

**Spiky POV 10 (Strong): There is one output but two independent axes, and the design trains the right one. The move is deterministically checkable and weight-learnable, so the fine-tune owns it; prose faithfulness is not reliably weight-learnable, so it is a verified optional layer. Scoring a coach with one blended number hides this and is a category error.**

Elaboration: Because the move and the prose are carried by different mechanisms, they move independently, and a single combined score lets a gain on one hide a failure on the other. The v4 checkpoint is the cleanest demonstration: it is the best model on the move and the worst recent model on prose at the same time, which is only coherent once the two are separated. Fine-tuning reliably teaches the move, a discrete choice with a right answer per tier, and the controlled test shows it does so where a prompt cannot. Fine-tuning does not reliably teach prose truth, because the training transcripts can contain confident wrong claims, and imitating them teaches confident wrongness, which is why the small model's raw prose fabricates more than the untuned base even after the data was filtered. The rule is to train the move, report it deterministically, and treat prose as a separate, verifier-gated number no fine-tune is expected to carry.

Prediction (or Disconfirmer): From base to tuned, the deterministic tier-policy match rises steeply while raw prose truthfulness stays low unless a verifier is added. Any fine-tune-only lift of raw prose truthfulness by a large margin, with no verifier, would complicate the split, though it would not touch the move claim.

How to resolve it: Measure the base-to-tuned change on deterministic tier-policy match and on raw prose truthfulness separately, with no verifier in the loop, and confirm the move moves while raw prose truth does not.

Validation: Cold raters found the two-axes claim off-consensus once the absolute wording was dropped, and the v4 split makes it concrete. An adversary noted it can leak if "learnable" is stretched, so the crux is pinned to a no-verifier fine-tune. It is Strong, backed by the base-to-tuned split and the v4 move-versus-prose divergence. It reaches to a support bot that nails a discrete routing decision while inventing refund policy in prose.

**Spiky POV 11 (Strong): Fine-tuning a small model on raw, unfiltered frontier prose risks being worse than base-plus-prompt on prose faithfulness, because it imitates the teacher's confident-assertion style. This is a reason to keep prose out of the trained objective, or filter it hard, not a reason to distrust the trained move.**

Elaboration: Distillation copies the teacher's manner along with its content, and a frontier teacher narrating chess makes confident claims that are sometimes wrong. A small student trained on those transcripts learns to sound just as sure, including when wrong. The project's own numbers show the mechanism: filtering the data beat not filtering it, cutting grounded prose fabrication from about half to about a third, but even the filtered fine-tune fabricates more than the untouched base, because the fine-tune also taught a more assertive, concrete voice with more surface to be wrong about. This is precisely why prose is not the trained objective: the move is trained because it cannot be fabricated, and prose is filtered with a verifier and rendered separately. The trained move is untouched by this failure mode because a move has no assertion to inherit.

Prediction (or Disconfirmer): A small model tuned on raw prose transcripts has raw prose truthfulness no better than base-plus-prompt, while a model tuned only on verifier-passed prose beats raw on fabrication, and neither affects the deterministic tier-policy match. If raw-transcript prose tuning matches verifier-filtered prose tuning, the prose half is wrong.

How to resolve it: Train two fine-tunes differing only in whether the prose data was verifier-filtered, compare raw prose truthfulness against base-plus-prompt, and confirm the move axis is unchanged by either.

Validation: Cold raters found the amplification claim off-consensus once softened from always net-negative. An adversary noted "value" can be redefined, so the crux is fixed to raw prose truthfulness. It is Strong, and it is a design argument for training the move and gating the prose. It reaches to distilling a clinician's off-hand wrong guesses into a junior model.

**Spiky POV 12 (Strong): Because no complete verifier for chess prose exists, the optional prose layer should be coverage-bounded: assert only what the detectors can verify and abstain otherwise. Saying less, truthfully, beats saying more, fluently. This governs the display layer, not the trained move.**

Elaboration: Motif and threat detectors cover many but not all of the things a coach might say, so there will always be prose claims the system cannot check. For the optional layer the safe default is to speak only inside the verifiable set and stay silent elsewhere, trading richness for truth. The project's own gate is a working instance: when a model cannot produce a fully checkable explanation within a few tries, the system falls back to a short, verified, engine-derived explanation that deliberately says less, at a fallback rate of about one in ten for the small model. This is entirely a property of the display layer; the trained move is already truthful by construction because it is a move.

Prediction (or Disconfirmer): Constraining prose to detector-verified claims raises truthfulness toward the coverage rate while richness drops, and beats say-more variants on truthfulness. If a say-more variant matches the bounded prose on truthfulness while keeping higher richness, the claim is wrong.

How to resolve it: Compare a coverage-bounded prose configuration against richer variants on the same positions, measuring truthfulness and a richness proxy together, with the move held fixed.

Validation: Cold raters found the say-less-truthfully rule off-consensus once "only" was softened. An adversary found the crux resolvable by the truthfulness-versus-richness trade. The fallback path in the project's own gate already shows the safe direction working, and it stays Strong pending the fuller measurement. It reaches to a clinical bot that refuses anything outside the retrieved guideline.

**Spiky POV 13 (Strong, riskiest): Dependability should be defined as the worst-case, all-constraints-at-once pass rate under repeated deployment sampling, not mean quality. The trained move is the anchor of that stack, and the small tuned model's plausible edge lives on the tail.**

Elaboration: A shipped coaching output is only good if several things hold at once: the move is sound and tier-appropriate, it is distinct across levels where it should be, and the rendered prose stays truthful and in register. Averaging quality hides how often the whole stack fails together. The right measurement samples each system many times at deployment temperature and asks how often every constraint passes at once. The move constraints are now deterministic and lead the field, which is the sturdy anchor of the stack; the open question is the repeated-sampling tail across the full stack, which the current single-sample evaluations do not capture. This is the riskiest item because its affirmative half, that the small tuned model wins on that tail once prose is included, has no supporting evidence yet.

Prediction (or Disconfirmer): At many samples and deployment temperature, the tuned small model pass-all-constraints rate exceeds an equally grounded prompted frontier's, even if its mean prose is lower, because it anchors the stack on a move axis it leads. If the grounded frontier's pass-all rate is at least the tuned model's, the affirmative is falsified, though the metric still stands.

How to resolve it: Repeated sampling at deployment temperature with grounding held constant, scoring the fraction of outputs that pass every constraint at once, comparing worst cases rather than averages.

Validation: Cold raters accepted the worst-case framing as a real, non-obvious measurement choice. An adversary found the crux resistant once the metric is fixed. The single-sample benchmarks do not touch its affirmative half, so it is Strong on the definition and flagged riskiest, with the move axis as the one part of the stack already won. It reaches to an aviation-checklist phraser graded on zero omissions across eight of eight runs.

**Spiky POV 14 (Strong): Prose faithfulness is table-stakes bought by a verifier and by capacity, so it can never be a moat, and it does not touch the graded move claim. A verify-and-regenerate gate drives verifier-detectable violations to zero for every model, and a larger open base fabricates only a few percent for free.**

Elaboration: It is tempting to treat prose faithfulness as the differentiator, since small models fabricate so much more than the frontier. The gate result says otherwise: running each explanation through a checker that vetoes false board claims, re-samples a few times, and otherwise substitutes a verified engine-derived explanation drove verifier-detectable violations to zero for the small model, from about 40 percent, and to zero for a strong frontier model, from about 7 percent, and across a fifteen-model field it drove every model to zero on the detectable set. This is zero verifier-detectable mechanical violations, not certified truth: the checker is high-precision but low-recall, so semantic falsehoods it misses (relational pawn-square claims, forks, threats, negations, eval claims) can still pass, and a cross-family LLM-judge residual remains. Independently, the small model's raw deficit is capacity-bound: a 27-billion-parameter open model reaches about 1 percent grounded fabrication for free while the project's own data rebuild only reached about a third. So prose faithfulness is a shared floor any serious system installs, not a place to build advantage.

Prediction (or Disconfirmer): With the gate in front of them, models of very different sizes and families all reach near-zero verifier-detectable violations, so the between-model spread collapses; and grounded raw fabrication falls steadily as base size rises. If some models keep meaningfully higher detectable fabrication behind the gate, or a much larger base fabricates as much as the small one on identical input, the stance is wrong.

How to resolve it: Put the same gate in front of a wide field and score verifier-detectable violations with and without it, and score raw grounded fabrication across a ladder of open model sizes. Both have been run; the gate zeroed the detectable violations across the field (not semantic truth) and the open-model spread holds.

Validation: This is off-consensus because the field treats small-model hallucination as a hard capability limit rather than a solved deployment detail and a capacity artifact. The crux is falsifiable and holds so far. It is Strong because the evidence is the project's own gate and open-model measurements, and it is doubly demoted, both a commodity and off the graded axis. It reaches to any generation task with a cheap external claim checker and a safe fallback.

**Spiky POV 15 (Strong): The reward that trains and grades this coach is cleanly split, with the primary signal fully deterministic. The move is trained and graded against un-gameable engine-and-tier gates with no judge; any prose council is a held-out, cross-family check on the optional layer that the model never trains against.**

Elaboration: Every axis of the move behavior is deterministic: soundness from the engine, tier-appropriateness and distinct-moves-per-level from the engine plus a human-move model and the tier rule. None can be flattered, so the primary reward is the deterministic tier-move check itself. The one thing that would need a model judge, prose instructiveness, is not the trained behavior and is scored only as a held-out, blinded, cross-family council on the optional layer, corrected for the measured self-preference of about 1.44 rank positions and never used as a training target. This is the strongest version of the anti-Goodhart rule: because the trained behavior is a move, the primary reward cannot be gamed by a judge at all, and the judge is quarantined to the layer the fine-tune is not graded on.

Prediction (or Disconfirmer): A coach trained toward a same-family learned judge would climb that judge's score while its deterministic tier-policy match stagnates, whereas training toward the deterministic gate lifts tier-policy match directly. If training toward a single learned judge improves the deterministic gate as much as training toward the gate does, the trap is not real.

How to resolve it: Run two training loops differing only in the reward, one toward a same-family judge and one toward the deterministic tier-move gate, and compare deterministic tier-policy match for each.

Validation: Cold raters found the split sharper than the usual advice to just use a good judge, because it makes the primary reward judge-free by construction. An adversary found the crux resistant once the judge is held out and cross-family. It is Strong, backed by the measured self-preference and the deterministic move grading the project already uses. It reaches to any preference-trained system where a learned reward can be gamed.

**Spiky POV 16 (Weak, supporting): Human-move modeling is a descriptive learner signal that feeds the deterministic tier rule, not the pedagogical objective. It tells you what a player of a given rating would probably play, which is an input to the canonical tier move, not the thing to teach on its own.**

Elaboration: The strength of a human-move predictor like Maia is describing behavior: it predicts the move a rated human would probably make about half the time, which is useful for meeting a student where they are. But most likely is not most instructive; a likely move can be a misconception or a bad habit. So the human-move signal belongs as a descriptive input to the tier rule that fixes the canonical move, clearly labeled as such, rather than as the selector itself. The strong form, that the signal is useless or harmful, does not survive scrutiny, which is why it is a supporting caution.

Prediction (or Disconfirmer): Two tier rules identical except for how the human-move signal is used, as a raw selector versus as an input to a pedagogical tier rule, will differ, with the tier rule winning on instructiveness. If the raw human-likely selector ties or wins, the caution is wrong.

How to resolve it: A dedicated study comparing the two selectors on learning outcomes, separate from the deterministic move evaluation.

Validation: Cold raters flagged the strong version as overstated, and primary sources confirm the human-move signal is useful but not sufficient rather than harmful, so it is softened to a supporting caution. Its reach is narrower and its resolution needs a separate learning study, so it is Weak and kept as support that explains one of the two deterministic inputs to the tier move.

## Experts

These are the voices worth following, including the ones who disagree with each other. The disagreement is the point.

**Asbjorn Steinskog and Anant Dole**

- **Who:** builders of the Take Take Take and Play Magnus chess coach.
- **Focus:** shipping a production coach where the engine is the source of truth and the language model only translates.
- **Why Follow:** they argue from production that a language model cannot calculate and should be confined to translating engine and detector output into English, and their shipped coach uses a prompted frontier model plus grounding rather than a fine-tune, which supports the system-not-the-weights view of prose truth and illustrates the frontier-as-prose-renderer role while leaving the leveled move choice unaddressed.
- **Where:** [ai.engineer](https://ai.engineer)

**Zhenwei Tang and the CSSLab C1 team**

- **Who:** authors of C1, a 4B chess model trained on engine-grounded reasoning distilled from a frontier teacher.
- **Focus:** a small grounded model that reasons about chess and beats its teacher.
- **Why Follow:** C1 reaches about 48.1 percent puzzle accuracy and surpasses its distillation teacher with far fewer tokens, which shows grounded small models can go further than expected at exactly the 4B size the production coach targets, and is the strongest opposing signal to any translate-only stance.
- **Where:** [arxiv.org/abs/2603.20510](https://arxiv.org/abs/2603.20510)

**Reid McIlroy-Young and Ashton Anderson**

- **Who:** creators of Maia, the human-move prediction models, at CSSLab.
- **Focus:** rating-conditioned modeling of what a human of a given strength would actually play.
- **Why Follow:** Maia predicts human moves about half the time and peaks near its training rating, which makes it a strong descriptive level signal, and it is one of the two deterministic inputs (with the engine) that define the canonical tier move the fine-tune is trained to emit.
- **Where:** [maiachess.com](https://maiachess.com)

**Nathan Lambert**

- **Who:** researcher and writer on open models and post-training at Interconnects.
- **Focus:** the gap between benchmark scores and real deployment robustness.
- **Why Follow:** he warns that open models are very jagged, easy to overfit on benchmarks, and often not specialized enough, and that closed models tend to be more robust where users keep presenting new challenges, which is exactly why the move test was run with grounding held constant and graded deterministically.
- **Where:** [interconnects.ai](https://interconnects.ai)

**Kevin Lu and Thinking Machines Lab**

- **Who:** authors of the on-policy distillation work.
- **Focus:** making small models strong in a trained domain while watching what training costs them elsewhere.
- **Why Follow:** they show small models with strong domain training can outperform larger generalists, and they document that fine-tuning small models on new knowledge causes catastrophic forgetting of instruction-following, which is the mechanism behind treating the fine-tune as a narrow move-chooser and rendering prose elsewhere.
- **Where:** [thinkingmachines.ai](https://thinkingmachines.ai)

**Mathieu Acher**

- **Who:** professor and strong chess player who benchmarks LLM chess play empirically.
- **Focus:** how well general and reasoning language models actually play legal, sound chess.
- **Why Follow:** he shows one older model plays around 1750 Elo yet produces an illegal move in about 16 percent of games, and that reasoning models are illegal most of the time, which guts the assumption that a frontier model is a strong chess reasoner out of the box and underlines why the move must be grounded, not generated.
- **Where:** [blog.mathieuacher.com](https://blog.mathieuacher.com)

**Adam Karvonen**

- **Who:** researcher on the empirical chess ability of language models.
- **Focus:** measuring legal-move rates and playing strength across model families.
- **Why Follow:** his work on the one model that plays strong chess, and the finding that chat and instruction tuning degrade a well-defined task, is a caution that fine-tuning can move behavior in the wrong direction if the objective is not held straight, which is why the trained objective is the deterministic tier move.
- **Where:** [adamkarvonen.github.io](https://adamkarvonen.github.io)

**Simon Willison**

- **Who:** widely read practitioner writer on applied language models.
- **Focus:** what actually works when building with models.
- **Why Follow:** he found prompt-engineering results on chess more convincing than fine-tuning, and argues that tools combined with reasoning are the most powerful current technique, which is the mainstream position the move test is spiky against and now has direct deterministic evidence to challenge for move selection specifically.
- **Where:** [simonwillison.net](https://simonwillison.net)

**Tim Dettmers**

- **Who:** author of QLoRA.
- **Focus:** cheap, low-memory fine-tuning of small and mid-size models.
- **Why Follow:** QLoRA makes fine-tuning a small model nearly free in cost and hardware, which is what makes the last-mile move-chooser role practical, while his own caution that chatbot benchmarks are untrustworthy reinforces grading the move deterministically rather than with a judge.
- **Where:** [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

**Mrinank Sharma and Ethan Perez**

- **Who:** authors of the sycophancy study in language models.
- **Focus:** why models, including model judges, prefer convincing answers over truthful ones.
- **Why Follow:** they show a preference model chose a convincing sycophantic answer over a truthful one a large majority of the time, which is the mechanism behind keeping the graded move claim judge-free and gating any optional prose before a preference score.
- **Where:** [arxiv.org/abs/2310.13548](https://arxiv.org/abs/2310.13548)

**Lianmin Zheng and colleagues**

- **Who:** authors of the LLM-as-a-judge evaluation.
- **Focus:** how well a strong model judge agrees with humans, and where it is biased.
- **Why Follow:** they establish that judges reach high human agreement but carry position, verbosity, and self-enhancement biases, which is why the move is graded by an engine and the optional prose, if scored, is judged cross-family and corrected for self-preference.
- **Where:** [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

**John Sweller**

- **Who:** originator of Cognitive Load Theory.
- **Focus:** minimizing extraneous load so limited working memory can build schemas.
- **Why Follow:** the requirement that the optional prose never leak engine internals is a direct application of reducing extraneous load, which gives the no-engine-speak register a real learning-science justification, and the tier rule's shift toward a human-findable move for weaker players is the same principle applied to the move itself.
- **Where:** [link.springer.com/article/10.1007/s10648-019-09465-5](https://link.springer.com/article/10.1007/s10648-019-09465-5)

## DOK 3: Insights

These are the conclusions that fell out of connecting the sources. Each drew on facts that no single source stated together.

### On what the fine-tune actually adds

**Insight 1: Holding the grounding identical and flipping only the weights isolates what the data adds, which is reproduction of the tier-selection policy a prompt on the same base does not reproduce, validated at 1.7B and 4B.** When grounding is frozen so the only variable is the weights or the system prompt, the fine-tune reproduces the policy where the base and an engineered prompt do not, at 1.7B and 4B; at 1.7B the prompt regressed on cross-tier coherence, and at 4B it produced more varied but mis-directed moves. The matched same-backend 32B prompt control was not run, so the 32B leg is a hypothesis. This connects the base run where move selection under the tier rule was the open axis, the prompt-versus-fine-tune literature that says prompting usually suffices, and the project's own controlled comparison that contradicts it at 1.7B and 4B. It is a learnability result, not a deployment-necessity result: the deterministic rule already computes the move at about 1.0 from the same grounding.

**Insight 2: Narrowing the trained behavior to the move makes the test cleaner and the primary grade judge-free, though it does not buy deployment-necessity or validated pedagogy.** Prose was never the clean place to prove a fine-tune earns its keep, because a prompt can already write fluent chess prose, whereas the tier-selection policy is what a prompt on the same base did not reproduce at 1.7B and 4B. And because the deliverable is a move, it cannot hallucinate a board fact, so the core claim needs no verifier, and it has a right answer against the rule, so it needs no judge. This connects the base run split (sound move, unfaithful prose), the prompt-controlled test at 1.7B and 4B, and the project's judge-free deterministic scoring of the move against the canonical rule. The clean test proves learnability, not that the model is necessary and not that the rule teaches well.

### On where dependability comes from

**Insight 3: A small model can win only if the system turns coaching into a deterministic, level-appropriate move choice, not open-ended chess reasoning or prose.** The engine supplies which moves are sound, the human-move model supplies which sound move a rating would find, the tier rule turns those into the single canonical move, and the fine-tune learns to emit it locally. Prose, if wanted, is rendered on top and separately verified. Without this narrowing the task is under-constrained, which is why the raw model fabricates in prose. This connects the production coaches that confine the model to translation, the C1 result that grounded small reasoning is possible, the chess-commentary evidence that fluent prose is often wrong, and the base run where move selection under the tier rule was the real open axis.

**Insight 4: The fine-tune is the last-mile move-chooser, and its contribution is the leveled move, which a prompt on the same weights demonstrably does not add, while prose truth is carried by a verifier and by capacity, not the weights.** Move truth comes from the engine and the tier rule; prose truth comes from a non-LLM verifier and from capacity; the fine-tune's own contribution is emitting the tier-appropriate move reliably and locally. This rests on the finding that a smaller aligned model was preferred over a much larger one, on prompt-optimization beating fine-tuning on structured reliability for prose-like axes, on the distillation failure modes that make raw prose fine-tuning risky, and on the project's own controlled test, verifier ablation, and the v4 move-versus-prose divergence.

### On how to measure it

**Insight 5: The primary reward and grade are fully deterministic, which is the strongest form of the anti-Goodhart rule.** Because the trained behavior is a move, tier-policy match and distinct-moves-per-level are computed on the engine plus the human-move model with no judge, so the primary signal cannot be flattered. The one axis that would need a model, prose instructiveness, is not the trained behavior and is quarantined to a held-out cross-family council on the optional layer, corrected for a measured self-preference of about 1.44 rank positions and never trained toward. This connects the chess-commentary error rates, the sycophancy and judge-bias findings, and the project's own deterministic move grading and council design.

**Insight 6: The optional prose layer must still gate faithfulness with non-LLM checks before any quality score, because fluent falsehood contaminates holistic scores, but this now protects a display layer rather than the graded claim.** Every claimed motif, threat, and plan is cross-checked against engine lines and detector output before any holistic score, and any prose quality score comes from a different model family. Chess is unusually gate-able because the engine and detectors form a non-LLM source of truth. This connects the chess-commentary judge that rated false commentary highly, the sycophancy and judge-bias findings, the base run where readable prose still scored zero on truthfulness, and the production gate that drives shipped verifier-detectable prose violations to zero for every model, which is not certified truth.

### On what to teach and where the advantage lives

**Insight 7: The human-move signal is a descriptive input to the deterministic tier rule, not a prescription for what to teach on its own.** Human-likely is not the same as pedagogically useful, so the signal is one of the two deterministic inputs (with the engine) that define the canonical tier move, clearly labeled descriptive, rather than the selector itself. This connects the measured human-move accuracy and its volatility across adjacent ratings, the industry move toward picking the most human among strong moves, feedback theory, and expertise reversal.

**Insight 8: The defensible differentiator is reproducing the tier-selection policy the frontier does not follow, measured as an all-scenario agreement lead, not a validated coaching moat.** The frontier changes its move across tiers only about a fifth to a third of the time and mirrors the engine's best move to every level about 77 percent of the time, yet about two-thirds of real held-out positions are discriminating, so the policy divergence is common. Because soundness comes from the engine and human-findability from a human-move model, the tier rule fixes the canonical move per tier without any judge, so it can be trained against a mechanical reward and graded cleanly, and the 32B v4 tune leads the twenty-model grand field at tier-policy match 0.767 (the 1.7B tune is second at 0.578), while the earlier 803-position field showed the tuned models leading at about 53 percent. The all-scenario lead is 0.767 versus the best frontier's 0.553; the position-by-position head-to-head over all diverging positions is about 53 wins, 18 losses, and 12 ties, while the 51-5-6 figure is a selection-conditioned subset, not a win rate, and all of it is agreement with our rule, not evidence of better teaching. This connects the gap-density measurement, the frontier-mirror measurement, the prompt-controlled test at 1.7B and 4B, and the grand eval.

## DOK 2: Knowledge Tree

This is the verified evidence behind the stances above. Each entry lists its objective facts, a short plain-language summary, and a link. About 120 sources were reviewed across the full effort, and the highest-leverage ones are collected here, grouped by topic. Every load-bearing and off-consensus fact was checked against a primary source, with no fabricated or hallucinated citations.

### A. Distillation and small-model specialization

**Knowledge distillation and step-by-step distillation (Hinton, Vinyals, Dean 2015; Hsieh et al., ACL Findings 2023)**

- DOK 1 - Fact: knowledge distillation transfers a large model's soft-target dark knowledge to a smaller student.
- DOK 1 - Fact: distilling step-by-step let a 770M model beat a few-shot 540B model while using about 80 percent of the data.
- DOK 2 - Summary: distillation can move a specific capability into a much smaller model, which is the mechanism the project bets on for the move-choosing behavior.
- Link to source: [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

**Small-model parity and specialization (Qwen3 Technical Report 2025; NVIDIA SLM position, Belcak et al. 2025; Finetuner's Fallacy 2026)**

- DOK 1 - Fact: a 1.7B base model reached parity with a 2.5B to 3B base model, though this is a pretraining-parity result, distinct from distillation.
- DOK 1 - Fact: a position paper argues small models are sufficient and economical for specialized agentic tasks, and a separate result shows a 1B specialized model beating a 3B standard model on under-represented domains through specialized pretraining.
- DOK 2 - Summary: small models can match larger ones on narrow targets, but the strongest results lean on specialized pretraining rather than fine-tuning alone.
- Link to source: [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

### B. Prompting versus fine-tuning for reliability

**Alignment and constraint adherence (Ouyang et al. 2022; structured-output reliability 2026)**

- DOK 1 - Fact: a 1.3B aligned model's outputs were preferred over a 175B model's, so bigger is not automatically better at following intent.
- DOK 1 - Fact: naive prompting reached high task accuracy but zero valid structured output in one study, while prompt-optimization, not fine-tuning, brought a frontier model to about 95 percent valid output.
- DOK 2 - Summary: reliability and format adherence are often won by alignment and prompt design rather than by scale, which is the mainstream expectation the project's own move test was built to challenge for the leveled move-selection behavior specifically.
- Link to source: [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

### C. Distillation failure modes

**Model collapse and small-student limits (Shumailov et al., Nature 2024; Small Model Learnability Gap, ACL Findings 2025; distillation traps 2026)**

- DOK 1 - Fact: training on recursively generated data erases the tails of the distribution, and preserving those tails needs real human data.
- DOK 1 - Fact: small models, at or below about 3B, learn better from shorter and simpler reasoning chains, and tail noise plus a teacher-student gap can drive overconfident hallucination.
- DOK 2 - Summary: distilling a frontier teacher's confident chess prose into a small student risks copying confident wrongness, which is why prose is kept out of the trained objective and the move, which cannot be fabricated, is trained instead.
- Link to source: [nature.com/articles/s41586-024-07566-y](https://www.nature.com/articles/s41586-024-07566-y)

**Adapter forgetting (LoRA intruder dimensions, NeurIPS 2025)**

- DOK 1 - Fact: low-rank adapters introduce intruder dimensions and forget more of pretraining than full fine-tuning, and still trail full fine-tuning on some measures.
- DOK 2 - Summary: cheap fine-tuning has a real cost in retained general ability, which reinforces keeping the fine-tune narrow, as a move-chooser, and late.
- Link to source: [arxiv.org/abs/2410.21228](https://arxiv.org/abs/2410.21228)

### D. Chess engines and human-move modeling

**Human-move prediction (Maia, McIlroy-Young et al., KDD 2020; Maia-2, NeurIPS 2024)**

- DOK 1 - Fact: Maia predicts human moves about 46 to 52 percent of the time, against roughly 33 to 41 percent for engine-style predictors, with accuracy peaking near the training rating, and personalization can reach up to about 65 percent.
- DOK 1 - Fact: the authors note that per-level models can be volatile and incoherent across adjacent ratings and are limited as teaching tools, and that the human-move ceiling is well below 100 percent.
- DOK 2 - Summary: human-move modeling is a strong descriptive level signal and one of the two deterministic inputs that define the canonical tier move, which is why it is treated as descriptive rather than as the selector.
- Link to source: [maiachess.com](https://maiachess.com)

**Compact rating-conditioned prediction (Maia-3 / Chessformer, ICLR 2026)**

- DOK 1 - Fact: a 79M rating-conditioned model reached about 57.1 percent human-move accuracy at under a quarter of the previous state-of-the-art parameter count.
- DOK 2 - Summary: human-move prediction is improving and getting cheaper, which strengthens the deterministic tier rule, but it still describes behavior rather than prescribing what to teach.
- Link to source: [arxiv.org/abs/2605.19091](https://arxiv.org/abs/2605.19091)

### E. LLMs playing and explaining chess

**Empirical LLM chess ability (Acher 2024; Karvonen 2024; reasoning-LLM chess 2025)**

- DOK 1 - Fact: one older model plays around 1750 Elo with under 0.1 percent illegal moves at the move level but an illegal move in about 16 percent of full games, while reasoning models are illegal in the large majority of cases.
- DOK 1 - Fact: chat and instruction tuning were found to degrade performance on the well-defined task of chess.
- DOK 2 - Summary: a frontier model is not a dependable chess reasoner by default, which is why the move is grounded in the engine and the tier rule rather than generated, and why the trained objective is held to the deterministic move.
- Link to source: [arxiv.org/abs/2512.01992](https://arxiv.org/abs/2512.01992)

**Grounded small chess reasoning (C1, CSSLab 2026; faithful reasoning training 2026)**

- DOK 1 - Fact: a 4B model trained on engine-grounded reasoning distilled from a frontier teacher, then reinforced, reached about 48.1 percent puzzle accuracy, surpassing its teacher at roughly 40.8 percent, with about 100 times fewer tokens, improving from 42.3 percent after supervised training to 48.3 percent after reinforcement.
- DOK 1 - Fact: separate work found best-move supervised training strong but reasoning sometimes unfaithful, while multi-move trajectory training was more faithful.
- DOK 2 - Summary: grounded small models can reason well at 4B, which is the size the production coach targets, and the finding that best-move training is strong while free-text reasoning is sometimes unfaithful is direct support for training the move and demoting the prose.
- Link to source: [arxiv.org/abs/2603.20510](https://arxiv.org/abs/2603.20510)

**Commentary hallucination and its evaluation (ACT-Eval 2026; CCC and GCC-Eval, Kim et al., NAACL 2025)**

- DOK 1 - Fact: a strong frontier model without tools produced factually incorrect chess claims about 22 percent of the time and smaller open models more than 50 percent, and standard reference-based model-as-a-judge scoring could not reliably detect these hallucinations, rating a false commentary highly.
- DOK 1 - Fact: concept-guided generation that integrates an expert model with the language model produces more accurate commentary, and evaluation is more reliable when expert-model knowledge is folded into the judge.
- DOK 2 - Summary: fluent chess prose is frequently false and an unaided judge misses it, which is why prose is the optional, verifier-gated layer and the graded claim is the un-fabricable move.
- Link to source: [openreview.net/forum?id=nne0ti66KT](https://openreview.net/forum?id=nne0ti66KT)

### F. Grounded coaching products and shipped small tutors

**Engine-as-truth production systems (Play Magnus and Take Take Take; DecodeChess; Chess.com Game Review 2026)**

- DOK 1 - Fact: production coaches use the engine as ground truth and detectors for structured concepts, with the language model confined to translating into English, a choice made because independent chess reasoning by a language model hallucinates.
- DOK 1 - Fact: one major platform's game review picks the most human among strong moves so the feedback feels like a real coach.
- DOK 2 - Summary: production coaches already confine the model to prose translation over an engine-chosen move, which supports the optional-prose-layer design and leaves the tier-appropriate move choice, the trained behavior here, as the open axis.
- Link to source: [decodechess.com](https://decodechess.com)

**Shipped small fine-tuned tutors (community LoRA tutors 2026)**

- DOK 1 - Fact: a LoRA fine-tune of a 4B model on distilled explanations reported high completeness and near-zero hallucination on a small 50-puzzle test set, and a 270M model was fine-tuned for offline move classification and rating prediction.
- DOK 2 - Summary: small fine-tuned chess models exist at the 4B size the production coach targets, but the strongest reports measure prose completeness rather than deterministic tier-appropriate move selection.
- Link to source: [huggingface.co](https://huggingface.co)

### G. Learning science

**Tutoring effectiveness (VanLehn 2011)**

- DOK 1 - Fact: intelligent tutoring systems reached an effect size of about 0.76 against no tutoring, close to human tutoring at about 0.79, so structured computer tutoring can approach human tutoring.
- DOK 2 - Summary: a well-designed tutor can be nearly as effective as a human, which sets a real bar and motivates getting the level-appropriate move right, since the move is the load-bearing pedagogical choice.
- Link to source: [doi.org/10.1080/00461520.2011.611369](https://doi.org/10.1080/00461520.2011.611369)

**Cognitive load and expertise reversal (Sweller et al. 2019; expertise-reversal literature)**

- DOK 1 - Fact: novel information passes through a limited working memory, so instruction should minimize extraneous load, and guidance that helps novices can harm more advanced learners and must fade with proficiency.
- DOK 1 - Fact: deliberate practice explains only about 21 to 26 percent of performance variance, less than once claimed.
- DOK 2 - Summary: showing a weaker player a more human-findable move rather than the engine's sharpest line is expertise-reversal applied to the move itself, and no-engine-speak in the optional prose is load reduction.
- Link to source: [link.springer.com/article/10.1007/s10648-019-09465-5](https://link.springer.com/article/10.1007/s10648-019-09465-5)

### H. Evaluation: LLM-as-judge and sycophancy

**Judge validity and sycophancy (Zheng et al., NeurIPS 2023; Sharma et al., ICLR 2024)**

- DOK 1 - Fact: strong model judges reach over 80 percent agreement with humans but carry position, verbosity, and self-enhancement biases.
- DOK 1 - Fact: a preference model preferred a convincing sycophantic answer over a truthful one the large majority of the time, and sampling many candidates only partly reduced this.
- DOK 2 - Summary: model judges are useful for style but unreliable for truth and biased toward their own family, which is why the graded move claim uses no judge and any prose score is cross-family and corrected.
- Link to source: [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

### I. The project's own measurements

These are the project's own internal measurements, not outside primary sources. For a claim about this specific system, such as whether this fine-tune beats this prompt on the move, a controlled experiment is the appropriate primary source, because no outside literature can settle it. The move-selection axis is graded deterministically on the engine and the human-move model with no judge, which is what makes these measurements clean.

**The move test, base versus tuned versus best-prompted base, with a prompt control at 1.7B and 4B**

- DOK 1 - Fact: holding the shipped grounding identical and grading the move deterministically against the canonical rule, the fine-tune beat both its untuned base and the best engineered prompt on that base at the sizes where a prompt arm was run. On the earlier slice: at 1.7B tier-policy match 0.296 (base) and 0.389 (prompt) versus 0.463 (tune), coherence violation 0.500 and 0.611 versus 0.333; at 4B 0.347 and 0.350 versus 0.386. On the 20-model grand slice: the 4B trio is base 0.353, prompt-base 0.378, tune 0.397, and the 1.7B tune is 0.578 (base 0.358), second of 20.
- DOK 1 - Fact: at 32B there is a base-versus-tune result (0.347 versus 0.767) but no matched same-backend prompt control was run, so the claim that a prompt cannot reproduce the policy at 32B is a hypothesis, not a measured result.
- DOK 1 - Fact: prompting failed on the graded axis at the small sizes rather than merely trailing: at 1.7B the engineered prompt pushed cross-tier coherence violation to 0.611, worse than the untuned base, and at 4B it produced more varied but mis-directed moves and still lost on the policy match.
- DOK 2 - Summary: with grounding frozen so only the weights or the prompt change, fine-tuning reproduces the policy where a prompt on the same base does not, validated at 1.7B and 4B and hypothesized at 32B; and because the canonical rule already computes the same move at about 1.0 from that grounding, this is learnability, not deployment-necessity.
- Link to source: the project's own 1.7B and 4B prompt-controlled move test plus the 20-model grand eval (internal measurement)

**The 32B v4 all-scenario lead and selection-conditioned head-to-head, and the prose trade**

- DOK 1 - Fact: on 120 held-out validation positions across three tiers, the 32B v4 fine-tune reached tier-policy match 0.767 raw (0.789 as served through the shipped gate) versus about 0.49 to 0.55 for the frontier (best frontier Gemini 0.553), distinct-moves-per-level 0.730 (73 of 100 canonical beginner-not-equal-advanced opportunities) versus roughly 0.21 to 0.28, and raw move-soundness 0.942. Over all the positions where v4's move diverges from the best frontier's, with no success gate, the head-to-head is about 53 wins, 18 losses, and 12 ties. The 51 wins, 5 losses, 6 ties figure is selection-conditioned: it is computed only on the 62 of 120 positions where v4 already gives a distinct, sound, correctly-graded move and diverges from the frontier, so it is a subset figure, not a win rate over all positions.
- DOK 1 - Fact: the same v4 checkpoint is the weakest recent model on prose, blinded instructiveness grade 4.67 below the 4B tune at 5.32 and the prior 32B v3 at 6.35, with about 40 percent of raw drafts failing the prose check before the gate; prose is secondary to the evaluation claim (still in the training loss), and the gate drives shipped verifier-detectable violations to zero, which is not certified truth.
- DOK 2 - Summary: the strongest-move checkpoint being the weakest-prose checkpoint follows from separating the trained move from the prose; v4 leads on agreement with our tier-selection rule, not on validated coaching, and the head-to-head win rate is the unbiased all-diverging figure, not the conditioned subset.
- Link to source: the project's own v4-centered honest eval and 4B eval (internal measurement)

**The definitive twenty-model grand evaluation, v4-centered**

- DOK 1 - Fact: on the shipped held-out validation slice of 120 positions across three tiers, scored across all twenty models with identical grounding and the single strict any-legal move extractor, the 32B v4 tune leads the field on tier-policy match at 0.767, the 1.7B tune is second at 0.578 (above every frontier), the tuned checkpoints take four of the top five, and the only frontier in the top five is Gemini at fourth (0.553). The canonical rule scores about 1.0 by construction on the same grounding and is the named ceiling.
- DOK 1 - Fact: the same v4 checkpoint is intentionally weaker on the blinded cross-family prose council, on-thesis because prose is secondary; the prior v3 all-rounder sits near the middle of the field, and the faithfulness-filtered v5 retrain regressed to tier-policy match 0.536 without improving raw faithfulness (near 0.58), but the v5 run was confounded (about 27 percent less optimization and token exposure, contrastive triads broken by row-wise filtering, about 42 percent boilerplate-principle pollution, retrained from base not from v4, no checkpoint selection), so it does not isolate filtering as the cause. The whole grand evaluation cost about 54 dollars.
- DOK 1 - Fact: the grand evaluation was audited for fairness on two axes, both clean: the human-move model is symmetric across all twenty models, feeding the ground-truth tier move and every model's grounding equally, and the 120-position validation set has zero train-test leakage against the shipped model's fine-tuning data, an exact-board-key intersection of zero out of 120; re-scoring the published generations reproduces tier-policy match 0.767 and distinct-moves 0.730 exactly.
- DOK 2 - Summary: the definitive twenty-model evaluation confirms the tuned models reproduce the tier-selection policy better than the field on a clean, held-out, symmetrically grounded set, with the deterministic rule as the ceiling; it does not validate pedagogy or deployment-necessity.
- Link to source: the project's own twenty-model grand evaluation (internal measurement)

**The earlier 803-position gap leaderboard across the model field**

- DOK 1 - Fact: on a curated, zero-leakage set of 803 held-out positions, each discriminating so that the tier-appropriate move differs from the engine's first choice for at least one tier, scored across fifteen models with identical grounding, the tuned models reached the highest tier-policy match in the field at about 53 percent, above every frontier model at about 43 to 48 percent, with the widest lead at the beginner and intermediate tiers, and the frontier mirrored the engine's best move at every tier a high fraction of the time.
- DOK 1 - Fact: tier-policy match is weak across the whole field, most models between about a third and a half, precisely because it is a trained behavior rather than an emergent one; faithfulness after the verifier is a fairness floor at zero verifier-detectable violations (not certified truth) for all fifteen models and is deliberately not a scoring axis; the blinded prose council's measured self-preference was about 1.44 rank, corrected in the reported ranking; the whole evaluation cost about 112 dollars.
- DOK 2 - Summary: the large evaluation confirms tier-policy match is the one axis where the small trained models lead the field while the field stays weak because the behavior is trained rather than emergent, all graded deterministically against the rule with no judge; leading on policy match is not the same as validated teaching.
- Link to source: the project's own 803-position gap evaluation (internal measurement)

**The verify-and-regenerate faithfulness gate for the optional prose layer**

- DOK 1 - Fact: the production gate, which re-samples an explanation up to four times and otherwise substitutes a verified engine-derived explanation, drove verifier-detectable prose violations from about 40 percent to zero for the small model and from about 7 percent to zero for a frontier model, and across the fifteen-model field drove every model to zero on the detectable set. This is zero verifier-detectable mechanical violations, not certified truth: semantic falsehoods the detectors miss (relational pawn-square claims, forks, threats, negations, eval claims) can still pass, and a cross-family LLM-judge residual remains.
- DOK 1 - Fact: on a prose failure the post-fix gate changes only the explanation and never the move, keeping the model's own greedy sound move and rewriting the prose with a verified engine-derived explanation of that same move, so the served move equals the evaluated greedy move; the small model fell back to the verified explanation about one time in ten.
- DOK 2 - Summary: a claim-level non-LLM gate with a deterministic fallback guarantees zero verifier-detectable prose violations for any model (not certified truth) and never alters the served move, which makes mechanical faithfulness table-stakes for the optional layer and, under the three-claim framing, outside the graded move claim.
- Link to source: the project's own verifier evaluation (internal measurement)

**Bigger open models on the same input**

- DOK 1 - Fact: on the identical grounded input, larger open models fabricated between about 1 and 8 percent in prose, with a 27-billion-parameter open model at about 1 percent matching the frontier, while the tuned small model sat near a third and the untuned base near an eighth.
- DOK 1 - Fact: every open model was judged more instructive in prose than the small model but none reached the frontier, and the very largest model did not coach best, so training quality and size both matter more than raw parameter count for the prose voice, while the best locally runnable base was a model of about 27 to 32 billion parameters.
- DOK 2 - Summary: the small model's prose deficit is closed for free by capacity, not by the data intervention, and a mid-size open base is the natural stronger starting point for a local coach, none of which touches the graded move claim.
- Link to source: the project's own open-model benchmark (internal measurement)

**Base evaluation and the move-versus-prose split**

- DOK 1 - Fact: an untuned 4-bit small base model, scored by a strong frontier judge, reached move soundness 1.00 while prose truthfulness was zero and no-engine-speak was 0.11 on the same outputs at the same time.
- DOK 1 - Fact: rebuilding the training data to be faithfulness-filtered, tier-aware, and more concrete cut grounded prose fabrication from about half to about a third and lifted prose instructiveness, but even the filtered fine-tune fabricates more than the untuned base, so prose truth is not reliably weight-learnable at small size while the move under the tier rule is.
- DOK 2 - Summary: on the same outputs the move was sound while the prose was unfaithful, which is the earliest evidence that the move and the prose are separate axes and that the move is the one to train.
- Link to source: the project's own base and retrain evaluations (internal measurement)

**Move-selection gap, density, and richer input at inference**

- DOK 1 - Fact: across the frontier models on held-out positions, tier-differentiation averaged about a fifth to a third, the frontier repeated the engine's best move across the tiers about 77 percent of the time, and on a large curated set about two-thirds of the decidable positions were discriminating, so the leveled-move gap is common in ordinary play rather than a niche.
- DOK 1 - Fact: replacing the trained prose grounding with a fuller structured board state at inference raised the small model's prose fabrication from about 40 percent to about 56 percent, while the same change barely moved a frontier model, which is format-agnostic.
- DOK 2 - Summary: the frontier is weak at leveled move selection but strong at not lying about the board in prose, the leveled-move behavior is exercised in most normal positions, and a small fine-tune is coupled to its trained input format, so any prose faithfulness is fixed by the verifier and capacity while the trained move is what leads the field.
- Link to source: the project's own gap-density and rich-grounding analyses (internal measurement)

**Reward design and the deterministic primary signal**

- DOK 1 - Fact: the training and evaluation loop uses a fully deterministic primary reward (tier-appropriate move, distinct-moves-per-level, move soundness, well-formed, no-engine-speak) and, only for the optional prose layer, a held-out blinded cross-family instructiveness council of three frontier judges that the model never trains against, self-preference-corrected by leaving out each judge's own family.
- DOK 2 - Summary: because the trained behavior is a move, the primary reward is judge-free and un-gameable, and the learned prose judge is quarantined to the optional layer, which is the strongest realization of the anti-Goodhart rule.
- Link to source: the project's own training and evaluation harness (internal measurement)

### J. Economics and local deployment (secondary, low-confidence)

**Cheap fine-tuning and local inference (QLoRA and on-device runtimes)**

- DOK 1 - Fact: quantized low-rank fine-tuning of a small model is inexpensive in cost and hardware, and on-device runtimes keep data local for privacy, though the specific cost and speed figures come from vendors and practitioners and were not independently verified.
- DOK 2 - Summary: the form-factor advantages of a small local move-chooser are real in kind, so they are used only as the honest deployment win and never as a load-bearing number.
- Link to source: [unsloth.ai](https://unsloth.ai)
