# Teaching a Small AI to Coach Chess at the Player's Level

**Owner:** Khoi Lam

## Purpose

This BrainLift answers one honest question for anyone building an AI chess coach, or any tool that must turn a verified answer into advice that fits the user's level. The question is this. Can a small, openly available AI model, trained a little further on carefully prepared examples, learn to pick the right move for a given skill level when a plain instruction to the same starting model cannot? And if it can, what does that actually buy you?

A quick note on two terms. "Fine-tuning" means taking a model that is already trained and training it a bit more on specific examples so it learns one narrow behavior. A "small open model" is a model with relatively few internal settings, whose files are public, so it can run on an ordinary computer.

The measured answer has three parts. They are easy to run together, so it matters to keep them apart. First, the behavior does move into the model's weights. (A model's "weights" are the numbers it learned. Putting a behavior "into the weights" means the model does it on its own, without extra help in the prompt.) Second, that does not make the model necessary in the product we shipped, because a short, fixed rule already computes the same move from the same inputs. Third, none of this proves the coaching is good for students, because every number here measures agreement with our own rule, not whether students learn more. Keeping these three apart is what makes the work honest instead of a scoreboard.

The trained job is a single behavior. Given a chess position and the student's rating tier (beginner is about 1000 to 1200, intermediate about 1300 to 1600, and advanced about 1700 to 2000), the model outputs the move our project's official rule assigns to that tier. It writes the move plus a short principle tag, for example "Nf3, develop toward the center." That one choice is graded pass or fail on three clauses a stranger can check with no opinion involved. The move must be sound, meaning not a blunder. (We measure this with a chess engine. The move must lose less than 250 centipawns, where a centipawn is one hundredth of a pawn's worth of advantage.) The move must match the official tier move. And the moves must differ across levels, so a beginner and an advanced player are not handed the same move on a position that should separate them. The written reason that can sit beside the move is a secondary, optional display layer. It is not the trained target.

The name of the main measurement matters. Earlier drafts called it "tier-fit." Here it is called tier-policy exact match, shortened to tier-policy match. It means exact agreement with our preregistered rule, called `select_tier_move`. That rule is a project decision, not proven teaching, and this document never treats agreement with it as proof that the coaching is good.

The first claim is about learnability, and it is proven. It is the real win of the assignment. With the same background facts given to both sides, a fine-tuned model reproduces the rule where a plain instruction to the same starting model cannot. The genuinely small, on-target model carries the result. The Qwen3-1.7B fine-tune lifts tier-policy match from 0.358 at its starting point to 0.578. That is second best in a field of twenty models, and above every large frontier model (the best frontier model, Gemini 3.1 Pro, scored 0.553). It runs locally, on a held-out set of positions checked for leakage. (A "held-out" set is test data the model never saw in training. "Leakage" is when test data sneaks into training and inflates the score.) The 1.7B fine-tune (0.578) beats the 4B fine-tune (0.397), so the result comes from the data and the training signal, not from raw model size. The shipped Qwen3-32B model, version 4 (v4), is the strongest mid-size example of the same result. It goes from 0.347 at its starting point to 0.767 raw, or 0.789 as actually served. It is not a small model.

The second claim is about whether the product needs the model at all, and the honest answer is no as built. We state this plainly rather than hide it. The same engine-approved list of sound moves and the same human-move estimates that feed the model's prompt (together, the "grounding") also feed a fixed rule of about twenty lines, `select_tier_move`, which computes the official tier move directly. That rule scores about 1.0 by construction, because the move it returns is the exact target the model is graded against. So as built, the fine-tuned model approximates a move the product already produces without it. The fixed rule is the true ceiling and the honest baseline. The model would only become necessary in a version with no engine and no human-move model at the moment of use, running fully offline. We did not build or measure that version.

The third claim is about teaching value, and it is unproven. Every number here measures agreement with our own rule. None of it shows that coaches or students prefer these moves, or that students improve. The honest status is this. The behavior is proven and a feature is demonstrated. It is not a finished product, not a durable advantage, and not proven value.

### In Scope

Whether a small, engine-grounded fine-tuned model reproduces the tier-selection rule more reliably than the same starting model under a plain instruction, graded automatically. The fixed rule as the named ceiling, and the honest question of when the model actually carries weight. The split between the move (the trained, checkable behavior) and the written reason (a secondary layer that is still present in training but not separately optimized). Where the reliability comes from, meaning the engine, the human-move model, and the tier rule, versus the model's weights. The honest size and location of the fine-tune's contribution. The faithfulness check for the optional written reason and its limits. And the reward design that keeps the main signal fully automatic.

### Out of Scope

Proven teaching or learning outcomes, which no number here establishes. Whether the model is needed in an offline, grounding-free product, which we did not build. A matched, same-model 32B plain-instruction comparison, so the claim that a well-prompted 32B model cannot reproduce the rule stays a testable hypothesis, not a result. Certified best-move truth beyond the shallow list of sound moves. Making a language model itself play strong chess. The exact cost and hardware numbers, which we keep low-confidence. A clean lesson from the version 5 (v5) retrain, which got worse under several tangled changes at once and does not isolate a single cause. Non-chess uses, except as an explicit test of how far the recipe travels. And any claim we could not tie to a primary source or our own measurement.

## Main Claims (Spiky Points of View)

In the BrainLift method these are called "spiky points of view." A spiky point of view is a claim that goes against what most people in the field believe, stated clearly enough that someone could test it. The list below is a menu of such claims worth testing, not one chosen winner. Most can only be settled by an experiment the reader would run, so the useful thing to deliver is the honest set of testable claims with the evidence to choose among them.

Each claim is labeled by how strong its backing is right now. "Validated" means our own measurement or a primary source establishes it, and it still goes against the common view. "Strong" means it is a good candidate that still awaits a further test. "Weak" means it has been softened to a careful caution.

Read the whole set with the three-part frame in mind. The first four claims carry the honest thesis and should be read first. The background facts carry the move choice, and the fine-tune is only their local executor (Claim 1). A verified move is not a verified explanation (Claim 2). AI graders left on their own approve fluent chess falsehoods (Claim 3). And getting a behavior from data is the real win, while beating the frontier is a bounded bonus (Claim 4).

**Claim 1 (Strong): The background facts carry the move choice. The fine-tune reliably runs, on its own machine, a rule the pipeline already computes. It has not truly absorbed the rule, and it collapses to a single move for all levels when the human-move estimates are missing at the moment of use.**

What it means: It is natural to treat the prepared dataset and the fine-tuned model as the product. The evidence splits the roles. The move's correctness comes from three things that are not the weights: the engine, the human-move model, and the tier rule. `select_tier_move` is a plain function of the tier, the list of sound moves, and Maia's estimate. (Maia is a model that predicts the move a human of a given rating would likely play.) It returns exactly the move the model is trained and graded to match. So in the shipped, grounded product, the rule already produces that move at about 1.0 by construction, while the model reproduces it at 0.578 (the 1.7B fine-tune) to 0.767 (the 32B fine-tune). That is a real learnability result, not a proof that the model is needed. The fine-tuned model also does not truly absorb the rule. At the moment of use it still needs Maia's per-level estimates in the prompt. When Maia was missing on the serving machine, the three levels collapsed to a single move, on both the compressed and the full-precision versions. The honest claim about the weights is reliable execution when grounded. Without grounding the model does not degrade gently. It collapses.

Prediction, and what would disprove it: In the grounded product, the fixed rule matches or beats the fine-tuned model's tier-policy match at close to 1.0, so removing the model does not lower the quality of the official move. If the fine-tuned model instead produced clearly better official moves than `select_tier_move` on the same grounding, or recovered the rule with no engine and no Maia at the moment of use, the picture would change.

How to settle it: In the grounded product, compare the rule's move against the model's move on the same held-out grounding. Then, separately, build an offline version with no grounding and measure whether the model can still recover the rule. That offline version is the only setting where the model truly carries weight.

How it held up: Fresh reviewers found it goes against the common view of what the product really is. The core point, that the rule computes the move at about 1.0 from the same grounding, is hard to argue with, because the model is trained to match a rule that already runs, and because we observed the collapse when grounding was absent. It applies to any system where a fixed rule over verified signals is the real asset and the model is a convenience.

**Claim 2 (Validated): A verified move is not a verified explanation. Choosing the sound move and explaining why are separate jobs. The explanation is the harder, still-open part, so it must not be the trained and graded behavior.**

What it means: It is tempting to assume that once the engine has picked and verified the move, the explanation is correct too. It is not. The engine certifies the choice, but the words about why, such as "this knight is trapped" or "this threatens mate in two," are written by the language model and are only as reliable as that model. Our base run makes the split concrete. On the very same outputs at the same time, move soundness was 1.00 while the truthfulness of the written reasons was zero. The wider evidence agrees. A strong frontier model makes factually incorrect chess claims about 22 percent of the time, and smaller open models more than half the time, whether or not the move was right. The clean lesson is to train the move, which cannot invent a board fact, and to write the explanation separately behind a checker that is not itself a language model.

Prediction, and what would disprove it: On positions where the move is engine-verified as sound, a model with no claim-level checker still makes at least one false tactical claim in a large share of its free-text explanations, so you cannot assume the explanation is faithful just because the move is verified. A model whose unchecked explanations are faithful on their own at high rates would weaken the case for demoting the explanation, though it would not touch the move claim.

How to settle it: Hold the move grounding constant and count invented claims per free-text explanation, with and without a claim-level checker. Confirm that a sound move does not make the explanation sound.

How it held up: Fresh reviewers found the split genuinely non-obvious once the "why" is separated from the "what." A challenger could not make it retreat, because the move-versus-reason split is concrete, and it holds against primary sources, from the 1.00-versus-zero base run and the measured chess-claim error rates. It carries over to radiology, where a report can verify the nodule and still invent a second condition in the summary.

**Claim 3 (Validated): AI graders left on their own, and especially ones from the same model family, pass fluent chess falsehoods as truthful. That is why the move is graded automatically, and why any explanation score uses a non-language-model checker first and a panel of graders from different families.**

What it means: A language model asked to grade a chess explanation does not run an engine check. It reacts to how fluent and confident the text sounds. In a controlled chess-commentary test, a plain grader rated a made-up commentary about 4.9 out of 5 while two of its three factual claims were false. Our own base run showed a strong frontier grader returning a truthfulness score of zero on outputs it still rated as readable. Our blinded, cross-family panel later measured same-family favoritism directly, at about 1.44 rank positions, with every grader favoring its own lab's model. So the move needs no grader at all, because it has an automatic right answer, and the flattery and same-family problems that plague explanation scoring never touch the core claim. Where the optional explanation is scored, faithfulness is first checked by an engine-and-detector system that is not a language model, and any quality score comes from a different family, corrected for self-preference.

Prediction, and what would disprove it: On held-out positions, a lone or same-family AI grader passes as truthful a much larger share of explanations than the non-language-model checker accepts, so a grader cannot certify that an explanation is true. The move axis is unaffected, because it uses no grader. If the AI grader and the non-language-model checker agree within noise, the explanation half is wrong.

How to settle it: Score one batch of explanations twice, once with the engine-and-detector checker and once with a lone AI grader. Compare a same-family grader against a different-family grader. Confirm the move axis is graded without either.

How it held up: Fresh reviewers split on whether an AI grader is fine for chess, which marks it as against consensus. The core point holds against primary sources, settled by the chess-commentary grader result and our own zero-truthfulness reading. It reaches to any setting with a cheap outside checker, such as a legal assistant whose sibling-model grader blesses invented citations.

**Claim 4 (Validated at 1.7B and 4B; hypothesis at 32B): Getting a behavior from data is the real, on-target win. A fine-tune reproduces the tier-selection rule where a plain instruction to the same starting model cannot, and it tracks the data, not the model size. Leading the frontier on this measurement is a bounded bonus, not the thesis.**

What it means: The field's working belief is that for a narrow behavior, a good prompt matches or beats a fine-tune. This project ran the clean version of that test. It graded the move automatically against `select_tier_move` with no AI grader, and it froze the grounding so the only thing that changed was the weights or the instruction. The fine-tune reproduced the rule where the base did not. The 1.7B fine-tune reaches 0.578 (from a base of 0.358), second of twenty and above every frontier model. The 4B trio is base 0.353, prompted base 0.378, and fine-tune 0.397, so the fine-tune beats the prompt, which beats the base, on the same set. Because the 1.7B fine-tune (0.578) beats the 4B fine-tune (0.397), the gain comes from the data, not from size. The 32B fine-tune reaches 0.767, but the matched, same-model 32B plain-instruction comparison was never run, so the 32B leg is a hypothesis. The bonus, stated honestly. On the all-scenario measurement the fine-tuned model leads the field, reaching 0.767 against about 0.49 to 0.55 for the frontier, and it gives different moves across levels about 73 percent of the time (73 of 100 chances, a rate of 0.730) against roughly 21 to 28 percent for the frontier. The frontier hands the engine's single best move to every level and repeats one move across the three tiers about 77 percent of the time. This is a lead on agreement with our own rule, not evidence that the frontier coaches worse.

Prediction, and what would disprove it: With grounding held the same, the fine-tuned model reproduces `select_tier_move` at a clearly higher rate than both its untuned base and the best engineered prompt on the same weights, at 1.7B and 4B. If a matched, same-model engineered prompt on the base reaches the fine-tune's tier-policy match, the claim is wrong. Running that comparison at 32B is the open test.

How to settle it: Keep the grounding the same and change only the weights and the prompt. Then score the base, the best-prompted base, and the fine-tune on the same held-out set with the automatic check. This has been done at 1.7B and 4B, and the fine-tune wins. The matched 32B prompt arm is the missing confirming or disconfirming work. For the frontier comparison, the honest unbiased number is the all-scenario tier-policy match, 0.767 versus 0.553. In the position-by-position head-to-head, the selection-conditioned showcase subset (the 62 of 120 positions where v4 already gives a distinct, sound, correctly graded move and differs from the best frontier) is 51 wins, 5 losses, and 6 ties. That figure overstates a raw win rate and is a subset figure only. Across all positions where v4's move differs from the best frontier's, with no success filter, it is 56-24-12 over the 92 diverging positions (56-24-40 over all 120). That is several times more losses than the conditioned subset, still a clear net win, and the direction holds.

How it held up: Fresh reviewers treated a flat claim that "a well-prompted base cannot match the fine-tune on this rule" as genuinely disputed, since the field assumes prompting is enough for a narrow behavior. A challenger could not make it retreat at 1.7B and 4B once the grounding is frozen and the measurement is the automatic rule match. Against primary sources the core point holds as established there (it is our own controlled experiment, the right primary source for a claim about this system) and stays a hypothesis at 32B. It reaches to any leveled recommender whose target is checkable without a model, such as difficulty-tiered hints in a math or programming tutor.

The remaining claims complete the menu, ordered by strength.

**Claim 5 (Validated): Because the trained deliverable is a move and not prose, the main claim is fully automatic. A move cannot invent a board fact, so the core claim needs no checker and no AI grader. It is graded purely as tier-policy match against the official tier move.**

What it means: The sharpest result of making the move the behavior is that the two hardest measurement problems in the old, explanation-centered framing disappear from the graded claim. A written explanation can invent a fork, a pin, or a mate that is not there, which is why the old framing needed a claim-level checker and a cross-family panel. A move cannot do any of that. It is a single legal action, whose soundness comes from the engine and whose level-fit comes from the engine plus a human-move model. So the whole faithfulness apparatus is unnecessary for the thing being graded, and the score is an automatic comparison to a fixed official move. This is a hardening, not a softening. The project's move-axis numbers are all grader-free, and re-scoring the published outputs reproduces them exactly.

Prediction, and what would disprove it: The core behavior can be scored to full agreement by two independent automatic checkers (engine soundness plus official-tier-move match) with no AI grader, and repeated scoring returns identical results. If grading the move requires an AI grader to break a tie, the claim is wrong.

How to settle it: Re-score the same outputs twice with the automatic tier-move checker and confirm identical tier-policy match. Confirm that no board-fact invention metric applies to a bare move, because there is no board claim to check.

How it held up: Fresh reviewers found it non-obvious that narrowing the deliverable removes the invention problem by design rather than by better checking. A challenger could not make it retreat, because a move has no free-text claim to falsify. It holds as established because it follows from the definition of the deliverable and appears in every grader-free move measurement in the project. It reaches to any recommender that outputs a discrete, checkable choice rather than a written rationale.

**Claim 6 (Strong): The coaching explanation is an optional display layer, so a model's writing quality, including the 32B v4 writing drop, does not affect the graded claim. Write the explanation with the engine's templates or a frontier model on top of the fine-tuned move.**

What it means: Because the trained behavior is the move, the explanation is free to be produced by whatever writes English best, and the choice of writer is a product decision, not a training one. The v4 evidence makes the separation vivid. The 32B v4 is the strongest model on the move, leading the field on tier-policy match and on distinct moves, and at the same time it is the weakest recent version on writing. Its blinded instructiveness grade is 4.67, below the 4B fine-tune at 5.32 and the earlier v3 at 6.35, and about 40 percent of its raw drafts fail the explanation-faithfulness check before the gate. Under an explanation-centered thesis that drop would sink v4. Here it is a non-issue. v4 is kept for the move it chooses, and the explanation beside that move is written by templates or a prompted frontier model, then run through the same non-language-model gate.

Prediction, and what would disprove it: Swapping the explanation writer, from the fine-tuned model's own text to engine templates or a frontier writer, leaves the graded tier-policy match unchanged, because the move is chosen before the explanation is written. If changing the explanation writer changes the graded move axis, the layers are not separable and the claim is wrong.

How to settle it: Hold the fine-tuned model's chosen move fixed, write its explanation three ways (its own, templates, and a frontier model), and confirm the tier-policy match is identical while only the writing quality varies.

How it held up: Fresh reviewers found the demotion of the explanation to an optional layer against consensus, since most coaching work treats the explanation as the deliverable. A challenger found the core point hard to move, because move choice provably comes before explanation writing. It is Strong on the project's own v4 split, where the strongest-move model is the weakest-writing model, which only makes sense once the explanation is the optional layer.

**Claim 7 (Strong): When a small local coach beats a frontier model on anything other than the move, what it actually wins is form factor. On writing it trails. Its clean edges are cost, speed, privacy, offline use, and the leveled move choice.**

What it means: The honest framing separates three things. One is the leveled move choice, which the fine-tune genuinely adds and now leads the frontier on. Another is writing, meaning explaining the move correctly at the right depth. A third is form factor, meaning running cheaply, privately, offline, and fast. On the writing axis, with grounding held constant, a blinded cross-family panel still judges the frontier models more instructive than the fine-tuned small model, and the frontier invents a board fact only about 3 percent of the time, against the small model's roughly a third before the checker. So with grounding equalized the small model does not win on writing, and it does not need to, because writing is the optional layer. Its clean standalone wins are the deployment envelope and the leveled move choice.

Prediction, and what would disprove it: With identical grounding, a prompted frontier model matches or beats the fine-tuned small model on writing instructiveness under repeated sampling, while the fine-tuned model leads on the automatic move axis. If the fine-tuned small model clearly beats the equally grounded frontier on writing instructiveness, this specific claim is wrong. Note it is about the writing, not the move.

How to settle it: Score writing instructiveness separately from cost, speed, and the automatic move axis, under repeated sampling at deployment temperature with grounding held constant.

How it held up: Fresh reviewers put the core near the mainstream, so it earns its edge from the sharp reframing that the field mislabels an economics win as a capability win. A challenger found the core point hard to move once grounding is equalized. The single-sample run supports it and the repeated-sampling version is pending, so it stays Strong. It reaches to on-device speech-to-intent, where the win is speed and privacy rather than accuracy.

**Claim 8 (Strong): On the optional writing layer, the small fine-tuned model's honest capability win is a consistent voice, not truth. A narrowly tuned small model should be the steadiest writer of a plain, no-engine-jargon voice under sampling.**

What it means: Within the optional writing layer there are two separate qualities. One is faithfulness, which the non-language-model checker carries for whichever model writes. The other is voice, meaning never leaking centipawns or deep engine lines and holding a steady tone for a given rating. Voice is exactly the narrow stylistic behavior that fine-tuning packs well into a small model. The base run hints at the size of the prize, with the no-engine-jargon rate at 0.11 for the untuned small model. The newest measurements complicate this honestly. When the frontier models are handed the same grounded prompt, they also stay jargon-free almost all of the time on a single best-guess pass, so on a one-shot pass the voice axis is near a tie. The bet moves onto variability, that under repeated sampling at deployment temperature the fine-tuned model holds the voice with a lower failure rate. That comparison has not been run.

Prediction, and what would disprove it: Under repeated sampling at deployment temperature, the fine-tuned small model beats an equally grounded prompted frontier on the no-engine-jargon rate and on voice consistency. If the grounded frontier matches or beats the fine-tuned model on voice consistency under sampling, the claim is wrong.

How to settle it: Sample both systems many times at deployment temperature with grounding held constant, and compare the voice pass rates and their variability, kept separate from both truth and the move axis.

How it held up: Fresh reviewers treated this as a real empirical bet rather than an obvious truth. A challenger found it hard to move as long as the voice measurements are fixed in advance. On a single pass the grounded frontier now matches the fine-tuned model, so the claim rests on the unrun repeated-sampling measurement. It is Strong and openly awaiting that number. It reaches to on-device command parsing.

**Claim 9 (Strong): This recipe, training a small model to output a verified, level-appropriate choice, travels only where the choice has a cheap automatic checker. Chess is safe for the move, because the engine and a human-move model fix the right answer per level. The explanation, which has no complete checker, is the trap.**

What it means: The recipe looks general. Ground a small model in a solver, learn from a teacher, fine-tune, and ship a cheap local chooser. Whether it is safe depends on whether the trained output has a cheap automatic checker. The move does. Soundness from the engine, human-findability from a human-move model, and a tier rule combine into a single official move, so the trained behavior is checkable and safe to grade with no grader. The explanation does not have a complete checker, which is exactly why the sharpened design does not train the explanation. Areas where the trained choice is automatically checkable, such as level-appropriate hints in a math tutor graded against a solver, are safe. Areas where the trained output is open-ended rationale with no checker are traps.

Prediction, and what would disprove it: Across several areas, a small fine-tune reliably adds the trained behavior only where that behavior has a cheap automatic checker. Where the trained output is uncheckable rationale, the fine-tune cannot be graded cleanly and inherits the teacher's errors. An area whose only trainable target is uncheckable writing that still grades cleanly would complicate it.

How to settle it: Run a preregistered multi-area study that sorts candidate trained behaviors by whether a cheap automatic checker exists, then measures whether the fine-tune adds the behavior cleanly in each.

How it held up: Fresh reviewers rated the checkable-choice-versus-uncheckable-rationale distinction durable and non-obvious. A challenger found it hard to move if the areas are chosen in advance. It is Strong rather than Validated only because it awaits that cross-area study, the one menu item that needs evidence beyond the chess project. It reaches to a code hinter, safe because tests verify, versus a finance rationale, unsafe because nothing verifies.

**Claim 10 (Strong): There is one output but two independent parts, and the design trains the right one. The move is automatically checkable and learnable in the weights, so the fine-tune owns it. Explanation faithfulness is not reliably learnable in the weights, so it is a checked, optional layer. Scoring a coach with one blended number hides this and is a category error.**

What it means: Because the move and the explanation are carried by different mechanisms, they move independently, and a single combined score lets a gain on one hide a failure on the other. The v4 version is the cleanest demonstration. It is the best model on the move and the worst recent model on writing at the same time, which only makes sense once the two are separated. Fine-tuning reliably teaches the move, a discrete choice with a right answer per level, and the controlled test shows it does so where a prompt cannot. Fine-tuning does not reliably teach explanation truth, because the training transcripts can contain confident wrong claims, and copying them teaches confident wrongness. That is why the small model's raw writing invents more than the untuned base even after the data was filtered. The rule is to train the move, report it automatically, and treat the explanation as a separate, checker-gated number that no fine-tune is expected to carry.

Prediction, and what would disprove it: From base to fine-tuned, the automatic tier-policy match rises steeply while raw explanation truthfulness stays low unless a checker is added. Any fine-tune-only jump in raw explanation truthfulness by a large margin, with no checker, would complicate the split, though it would not touch the move claim.

How to settle it: Measure the base-to-fine-tuned change on automatic tier-policy match and on raw explanation truthfulness separately, with no checker in the loop, and confirm the move moves while raw explanation truth does not.

How it held up: Fresh reviewers found the two-parts claim against consensus once the absolute wording was dropped, and the v4 split makes it concrete. A challenger noted it can leak if "learnable" is stretched, so the core point is pinned to a no-checker fine-tune. It is Strong, backed by the base-to-fine-tuned split and the v4 move-versus-writing divergence. It reaches to a support bot that nails a discrete routing decision while inventing refund policy in its writing.

**Claim 11 (Strong): Fine-tuning a small model on raw, unfiltered frontier writing risks being worse than the base plus a prompt on explanation faithfulness, because it copies the teacher's confident-assertion style. This is a reason to keep writing out of the trained objective, or to filter it hard, not a reason to distrust the trained move.**

What it means: Distillation copies the teacher's manner along with its content. (Distillation means training a smaller model to copy a larger model's answers, where the larger model is the teacher.) A frontier teacher narrating chess makes confident claims that are sometimes wrong. A small student trained on those transcripts learns to sound just as sure, including when it is wrong. The project's own numbers show the mechanism. Filtering the data beat not filtering it, cutting grounded writing invention from about half to about a third, but even the filtered fine-tune invents more than the untouched base, because the fine-tune also taught a more assertive, concrete voice with more surface to be wrong about. This is precisely why writing is not the trained objective. The move is trained because it cannot be invented, and the writing is filtered with a checker and produced separately. The trained move is untouched by this failure mode, because a move has no assertion to inherit.

Prediction, and what would disprove it: A small model tuned on raw writing transcripts has raw explanation truthfulness no better than the base plus a prompt, while a model tuned only on checker-passed writing beats raw on invention, and neither affects the automatic tier-policy match. If raw-transcript writing tuning matches checker-filtered writing tuning, the writing half is wrong.

How to settle it: Train two fine-tunes that differ only in whether the writing data was checker-filtered, compare raw explanation truthfulness against the base plus a prompt, and confirm the move axis is unchanged by either.

How it held up: Fresh reviewers found the amplification claim against consensus once softened from always net-negative. A challenger noted "value" can be redefined, so the core point is fixed to raw explanation truthfulness. It is Strong, and it is a design argument for training the move and gating the writing. It reaches to distilling a clinician's offhand wrong guesses into a junior model.

**Claim 12 (Strong): Because no complete checker for chess writing exists, the optional writing layer should stay within what it can verify. Assert only what the detectors can confirm, and stay silent otherwise. Saying less, truthfully, beats saying more, fluently. This governs the display layer, not the trained move.**

What it means: Detectors for motifs and threats cover many but not all of the things a coach might say, so there will always be written claims the system cannot check. For the optional layer the safe default is to speak only inside the checkable set and stay silent elsewhere, trading richness for truth. The project's own gate is a working example. When a model cannot produce a fully checkable explanation within a few tries, the system falls back to a short, verified, engine-derived explanation that deliberately says less, at a fallback rate of about one in ten for the small model. This is entirely a property of the display layer. The trained move is already truthful by design, because it is a move.

Prediction, and what would disprove it: Limiting writing to detector-verified claims raises truthfulness toward the coverage rate while richness drops, and beats say-more versions on truthfulness. If a say-more version matches the limited writing on truthfulness while keeping higher richness, the claim is wrong.

How to settle it: Compare a coverage-limited writing setting against richer versions on the same positions, measuring truthfulness and a richness proxy together, with the move held fixed.

How it held up: Fresh reviewers found the say-less-truthfully rule against consensus once "only" was softened. A challenger found the core point resolvable by the truthfulness-versus-richness trade. The fallback path in the project's own gate already shows the safe direction working, and it stays Strong pending the fuller measurement. It reaches to a clinical bot that refuses anything outside the retrieved guideline.

**Claim 13 (Strong, riskiest): Reliability should be defined as the worst-case pass rate with every constraint holding at once under repeated deployment sampling, not average quality. The trained move is the anchor of that stack, and the small fine-tuned model's plausible edge lives on the tail.**

What it means: A shipped coaching output is only good if several things hold at once. The move is sound and level-appropriate, it differs across levels where it should, and the written explanation stays truthful and in voice. Averaging quality hides how often the whole stack fails together. The right measurement samples each system many times at deployment temperature and asks how often every constraint passes at once. The move constraints are now automatic and lead the field, which is the sturdy anchor of the stack. The open question is the repeated-sampling tail across the full stack, which the current single-sample tests do not capture. This is the riskiest item, because its positive half, that the small fine-tuned model wins on that tail once the writing is included, has no supporting evidence yet.

Prediction, and what would disprove it: At many samples and deployment temperature, the fine-tuned small model's pass-all-constraints rate is higher than an equally grounded prompted frontier's, even if its average writing is lower, because it anchors the stack on a move axis it leads. If the grounded frontier's pass-all rate is at least the fine-tuned model's, the positive half is disproven, though the measurement still stands.

How to settle it: Sample repeatedly at deployment temperature with grounding held constant, scoring the fraction of outputs that pass every constraint at once, comparing worst cases rather than averages.

How it held up: Fresh reviewers accepted the worst-case framing as a real, non-obvious measurement choice. A challenger found the core point hard to move once the measurement is fixed. The single-sample benchmarks do not touch its positive half, so it is Strong on the definition and flagged riskiest, with the move axis as the one part of the stack already won. It reaches to an aviation-checklist writer graded on zero omissions across eight of eight runs.

**Claim 14 (Strong): Explanation faithfulness is table stakes, bought by a checker and by model size, so it can never be a durable advantage, and it does not touch the graded move claim. A check-and-rewrite gate drives checkable violations to zero for every model, and a larger open model invents only a few percent for free.**

What it means: It is tempting to treat explanation faithfulness as the differentiator, since small models invent so much more than the frontier. The gate result says otherwise. Running each explanation through a checker that vetoes false board claims, re-samples a few times, and otherwise substitutes a verified engine-derived explanation drove checkable violations to zero for the small model, from about 40 percent, and to zero for a strong frontier model, from about 7 percent. Across a fifteen-model field it drove every model to zero on the checkable set. This is zero checkable mechanical violations, not certified truth. The checker is high-precision but low-recall, so meaning-level falsehoods it misses (relational pawn-square claims, forks, threats, negations, and evaluation claims) can still pass, and a cross-family AI-grader residual remains. Separately, the small model's raw deficit is a size effect. A 27-billion-parameter open model reaches about 1 percent grounded invention for free, while our own data rebuild only reached about a third. So explanation faithfulness is a shared floor any serious system installs, not a place to build advantage.

Prediction, and what would disprove it: With the gate in front of them, models of very different sizes and families all reach near-zero checkable violations, so the spread between models collapses. And grounded raw invention falls steadily as base size rises. If some models keep clearly higher checkable invention behind the gate, or a much larger base invents as much as the small one on identical input, the claim is wrong.

How to settle it: Put the same gate in front of a wide field and score checkable violations with and without it, and score raw grounded invention across a ladder of open model sizes. Both have been run. The gate zeroed the checkable violations across the field (not meaning-level truth), and the open-model spread holds.

How it held up: This is against consensus, because the field treats small-model invention as a hard capability limit rather than a solved deployment detail and a size effect. The core point is falsifiable and holds so far. It is Strong because the evidence is the project's own gate and open-model measurements, and it is doubly demoted, both a commodity and off the graded axis. It reaches to any writing task with a cheap outside claim checker and a safe fallback.

**Claim 15 (Strong): The reward that trains and grades this coach is cleanly split, with the main signal fully automatic. The move is trained and graded against un-gameable engine-and-tier checks with no grader. Any writing panel is a held-out, cross-family check on the optional layer that the model never trains against.**

What it means: Every part of the move behavior is automatic. Soundness comes from the engine. Level-appropriateness and distinct-moves-across-levels come from the engine plus a human-move model and the tier rule. None of these can be flattered, so the main reward is the automatic tier-move check itself. The one thing that would need an AI grader, writing instructiveness, is not the trained behavior. It is scored only as a held-out, blinded, cross-family panel on the optional layer, corrected for the measured self-preference of about 1.44 rank positions, and never used as a training target. This is the strongest version of the anti-gaming rule. Because the trained behavior is a move, the main reward cannot be gamed by a grader at all, and the grader is quarantined to the layer the fine-tune is not graded on.

Prediction, and what would disprove it: A coach trained toward a same-family learned grader would climb that grader's score while its automatic tier-policy match stays flat, whereas training toward the automatic check lifts tier-policy match directly. If training toward a single learned grader improves the automatic check as much as training toward the check does, the trap is not real.

How to settle it: Run two training loops that differ only in the reward, one toward a same-family grader and one toward the automatic tier-move check, and compare the automatic tier-policy match for each.

How it held up: Fresh reviewers found the split sharper than the usual advice to just use a good grader, because it makes the main reward grader-free by design. A challenger found the core point hard to move once the grader is held out and cross-family. It is Strong, backed by the measured self-preference and the automatic move grading the project already uses. It reaches to any preference-trained system where a learned reward can be gamed.

**Claim 16 (Weak, supporting): Human-move modeling is a descriptive input that feeds the automatic tier rule, not the teaching goal itself. It tells you what a player of a given rating would probably play, which is an input to the official tier move, not the thing to teach on its own.**

What it means: The strength of a human-move predictor like Maia is describing behavior. It predicts the move a rated human would probably make about half the time, which is useful for meeting a student where they are. But most likely is not most instructive. A likely move can be a misconception or a bad habit. So the human-move signal belongs as a descriptive input to the tier rule that fixes the official move, clearly labeled as such, rather than as the selector itself. The strong form, that the signal is useless or harmful, does not survive scrutiny, which is why this is a supporting caution.

Prediction, and what would disprove it: Two tier rules that are identical except for how the human-move signal is used, as a raw selector versus as an input to a teaching tier rule, will differ, with the tier rule winning on instructiveness. If the raw human-likely selector ties or wins, the caution is wrong.

How to settle it: A dedicated study comparing the two selectors on learning outcomes, separate from the automatic move evaluation.

How it held up: Fresh reviewers flagged the strong version as overstated, and primary sources confirm the human-move signal is useful but not sufficient, rather than harmful, so it is softened to a supporting caution. Its reach is narrower and its resolution needs a separate learning study, so it is Weak and kept as support that explains one of the two automatic inputs to the tier move.

## Experts We Followed

These are the voices worth following, including the ones who disagree with each other. The disagreement is the point.

**Asbjorn Steinskog and Anant Dole**

- **Who:** builders of the Take Take Take and Play Magnus chess coach.
- **Focus:** shipping a production coach where the engine is the source of truth and the language model only translates.
- **Why they matter:** they argue from production that a language model cannot calculate and should be limited to translating engine and detector output into English. Their shipped coach uses a prompted frontier model plus grounding rather than a fine-tune, which supports the system-not-the-weights view of explanation truth and shows the frontier-as-writer role, while leaving the leveled move choice unaddressed.
- **Where:** [ai.engineer](https://ai.engineer)

**Zhenwei Tang and the CSSLab C1 team**

- **Who:** authors of C1, a 4B chess model trained on engine-grounded reasoning learned from a frontier teacher.
- **Focus:** a small grounded model that reasons about chess and beats its teacher.
- **Why they matter:** C1 reaches about 48.1 percent puzzle accuracy and surpasses its teacher with far fewer tokens, which shows grounded small models can go further than expected at exactly the 4B size the production coach targets. It is the strongest opposing signal to any translate-only stance.
- **Where:** [arxiv.org/abs/2603.20510](https://arxiv.org/abs/2603.20510)

**Reid McIlroy-Young and Ashton Anderson**

- **Who:** creators of Maia, the human-move prediction models, at CSSLab.
- **Focus:** rating-conditioned modeling of what a human of a given strength would actually play.
- **Why they matter:** Maia predicts human moves about half the time and peaks near its training rating, which makes it a strong descriptive level signal. It is one of the two automatic inputs (with the engine) that define the official tier move the fine-tune is trained to output.
- **Where:** [maiachess.com](https://maiachess.com)

**Nathan Lambert**

- **Who:** researcher and writer on open models and post-training at Interconnects.
- **Focus:** the gap between benchmark scores and real deployment robustness.
- **Why they matter:** he warns that open models are very uneven, easy to overfit on benchmarks, and often not specialized enough, and that closed models tend to be more robust where users keep bringing new challenges. That is exactly why the move test was run with grounding held constant and graded automatically.
- **Where:** [interconnects.ai](https://interconnects.ai)

**Kevin Lu and Thinking Machines Lab**

- **Who:** authors of the on-policy distillation work.
- **Focus:** making small models strong in a trained area while watching what training costs them elsewhere.
- **Why they matter:** they show small models with strong domain training can outperform larger generalists, and they document that fine-tuning small models on new knowledge causes catastrophic forgetting of instruction-following. (Catastrophic forgetting is when training on new material makes a model lose skills it used to have.) That mechanism is the reason to treat the fine-tune as a narrow move-chooser and write the explanation elsewhere.
- **Where:** [thinkingmachines.ai](https://thinkingmachines.ai)

**Mathieu Acher**

- **Who:** professor and strong chess player who benchmarks how well language models play chess.
- **Focus:** how well general and reasoning language models actually play legal, sound chess.
- **Why they matter:** he shows one older model plays around 1750 Elo (Elo is a chess rating number) yet produces an illegal move in about 16 percent of games, and that reasoning models are illegal most of the time. That guts the assumption that a frontier model is a strong chess reasoner out of the box, and underlines why the move must be grounded, not generated.
- **Where:** [blog.mathieuacher.com](https://blog.mathieuacher.com)

**Adam Karvonen**

- **Who:** researcher on the real chess ability of language models.
- **Focus:** measuring legal-move rates and playing strength across model families.
- **Why they matter:** his work on the one model that plays strong chess, and the finding that chat and instruction tuning degrade a well-defined task, is a caution that fine-tuning can move behavior in the wrong direction if the objective is not held straight. That is why the trained objective is the automatic tier move.
- **Where:** [adamkarvonen.github.io](https://adamkarvonen.github.io)

**Simon Willison**

- **Who:** widely read practitioner writer on applied language models.
- **Focus:** what actually works when building with models.
- **Why they matter:** he found prompt-engineering results on chess more convincing than fine-tuning, and argues that tools combined with reasoning are the most powerful current technique. That is the mainstream position the move test is spiky against, and it now has direct automatic evidence to challenge for move selection specifically.
- **Where:** [simonwillison.net](https://simonwillison.net)

**Tim Dettmers**

- **Who:** author of QLoRA. (QLoRA is a cheap, memory-saving way to fine-tune a model.)
- **Focus:** cheap, low-memory fine-tuning of small and mid-size models.
- **Why they matter:** QLoRA makes fine-tuning a small model nearly free in cost and hardware, which is what makes the last-mile move-chooser role practical. His own caution that chatbot benchmarks are untrustworthy reinforces grading the move automatically rather than with a grader.
- **Where:** [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

**Mrinank Sharma and Ethan Perez**

- **Who:** authors of the sycophancy study in language models. (Sycophancy is a model's tendency to prefer answers that sound convincing over ones that are true.)
- **Focus:** why models, including AI graders, prefer convincing answers over truthful ones.
- **Why they matter:** they show a preference model chose a convincing sycophantic answer over a truthful one a large majority of the time, which is the mechanism behind keeping the graded move claim grader-free and checking any optional writing before a preference score.
- **Where:** [arxiv.org/abs/2310.13548](https://arxiv.org/abs/2310.13548)

**Lianmin Zheng and colleagues**

- **Who:** authors of the LLM-as-a-judge evaluation. (LLM-as-a-judge means using one AI model to grade another's answer.)
- **Focus:** how well a strong AI grader agrees with humans, and where it is biased.
- **Why they matter:** they establish that graders reach high human agreement but carry position, length, and self-enhancement biases, which is why the move is graded by an engine and the optional writing, if scored, is judged cross-family and corrected for self-preference.
- **Where:** [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

**John Sweller**

- **Who:** originator of Cognitive Load Theory.
- **Focus:** minimizing extra mental load so limited working memory can build understanding.
- **Why they matter:** the requirement that the optional writing never leak engine internals is a direct application of reducing extra load, which gives the no-engine-jargon voice a real learning-science justification. The tier rule's shift toward a human-findable move for weaker players is the same principle applied to the move itself.
- **Where:** [link.springer.com/article/10.1007/s10648-019-09465-5](https://link.springer.com/article/10.1007/s10648-019-09465-5)

## Insights (What We Concluded by Connecting Sources)

These are the conclusions that fell out of connecting the sources. Each drew on facts that no single source stated together.

### On what the fine-tune actually adds

**Insight 1: Holding the grounding identical and flipping only the weights isolates what the data adds, which is reproduction of the tier-selection rule that a plain instruction to the same base does not reproduce, proven at 1.7B and 4B.** When grounding is frozen so the only thing that changes is the weights or the instruction, the fine-tune reproduces the rule where the base and an engineered prompt do not, at 1.7B and 4B. At 1.7B the prompt got worse on cross-level coherence. At 4B it produced more varied but mis-directed moves. The matched, same-model 32B prompt comparison was not run, so the 32B leg is a hypothesis. This connects the base run, where move selection under the tier rule was the open axis, the prompt-versus-fine-tune literature that says prompting usually suffices, and the project's own controlled comparison that contradicts it at 1.7B and 4B. It is a learnability result, not a proof that the model is needed. The fixed rule already computes the move at about 1.0 from the same grounding.

**Insight 2: Narrowing the trained behavior to the move makes the test cleaner and the main grade grader-free, though it does not buy necessity or proven teaching.** Writing was never the clean place to prove a fine-tune earns its keep, because a prompt can already write fluent chess prose, whereas the tier-selection rule is what a prompt on the same base did not reproduce at 1.7B and 4B. And because the deliverable is a move, it cannot invent a board fact, so the core claim needs no checker, and it has a right answer against the rule, so it needs no grader. This connects the base run split (sound move, unfaithful writing), the prompt-controlled test at 1.7B and 4B, and the project's grader-free automatic scoring of the move against the official rule. The clean test proves learnability, not that the model is necessary and not that the rule teaches well.

### On where the reliability comes from

**Insight 3: A small model can win only if the system turns coaching into an automatic, level-appropriate move choice, not open-ended chess reasoning or writing.** The engine supplies which moves are sound, the human-move model supplies which sound move a rating would find, the tier rule turns those into the single official move, and the fine-tune learns to output it locally. The explanation, if wanted, is written on top and separately checked. Without this narrowing the task is under-constrained, which is why the raw model invents in its writing. This connects the production coaches that limit the model to translation, the C1 result that grounded small reasoning is possible, the chess-commentary evidence that fluent prose is often wrong, and the base run where move selection under the tier rule was the real open axis.

**Insight 4: The fine-tune is the last-mile move-chooser, and its contribution is the leveled move, which a prompt on the same weights clearly does not add, while explanation truth is carried by a checker and by model size, not the weights.** Move truth comes from the engine and the tier rule. Explanation truth comes from a non-language-model checker and from model size. The fine-tune's own contribution is outputting the level-appropriate move reliably and locally. This rests on the finding that a smaller aligned model was preferred over a much larger one, on prompt-optimization beating fine-tuning on structured reliability for writing-like tasks, on the distillation failure modes that make raw writing fine-tuning risky, and on the project's own controlled test, checker ablation, and the v4 move-versus-writing divergence.

### On how to measure it

**Insight 5: The main reward and grade are fully automatic, which is the strongest form of the anti-gaming rule.** Because the trained behavior is a move, tier-policy match and distinct-moves-across-levels are computed on the engine plus the human-move model with no grader, so the main signal cannot be flattered. The one axis that would need a model, writing instructiveness, is not the trained behavior and is quarantined to a held-out cross-family panel on the optional layer, corrected for a measured self-preference of about 1.44 rank positions and never trained toward. This connects the chess-commentary error rates, the sycophancy and grader-bias findings, and the project's own automatic move grading and panel design.

**Insight 6: The optional writing layer must still check faithfulness with non-language-model checks before any quality score, because fluent falsehood contaminates overall scores, but this now protects a display layer rather than the graded claim.** Every claimed motif, threat, and plan is cross-checked against engine lines and detector output before any overall score, and any writing quality score comes from a different model family. Chess is unusually checkable, because the engine and detectors form a source of truth that is not a language model. This connects the chess-commentary grader that rated false commentary highly, the sycophancy and grader-bias findings, the base run where readable writing still scored zero on truthfulness, and the production gate that drives shipped checkable writing violations to zero for every model, which is not certified truth.

### On what to teach and where the advantage lives

**Insight 7: The human-move signal is a descriptive input to the automatic tier rule, not a prescription for what to teach on its own.** Human-likely is not the same as pedagogically useful, so the signal is one of the two automatic inputs (with the engine) that define the official tier move, clearly labeled descriptive, rather than the selector itself. This connects the measured human-move accuracy and its swings across nearby ratings, the industry move toward picking the most human among strong moves, feedback theory, and expertise reversal.

**Insight 8: The defensible differentiator is reproducing the tier-selection rule the frontier does not follow, measured as an all-scenario agreement lead, not a proven coaching advantage.** The frontier changes its move across levels only about a fifth to a third of the time and mirrors the engine's best move to every level about 77 percent of the time, yet about two-thirds of real held-out positions are discriminating, so the rule divergence is common. Because soundness comes from the engine and human-findability from a human-move model, the tier rule fixes the official move per level with no grader, so it can be trained against a mechanical reward and graded cleanly. The 32B v4 fine-tune leads the twenty-model grand field at tier-policy match 0.767 (the 1.7B fine-tune is second at 0.578), while the earlier 803-position field showed the fine-tuned models leading at about 53 percent. The all-scenario lead is 0.767 versus the best frontier's 0.553. The position-by-position head-to-head is 56-24-12 over the 92 diverging positions (56-24-40 over all 120), while the 51-5-6 figure is a selection-conditioned subset, not a win rate, and all of it is agreement with our rule, not evidence of better teaching. This connects the gap-density measurement, the frontier-mirror measurement, the prompt-controlled test at 1.7B and 4B, and the grand evaluation.

## The Evidence (Knowledge Tree)

This is the verified evidence behind the claims above. Each entry lists its objective facts, a short plain-language summary, and a link. About 120 sources were reviewed across the full effort, and the highest-leverage ones are collected here, grouped by topic. Every load-bearing and against-consensus fact was checked against a primary source, with no invented citations.

### A. Distillation and small-model specialization

**Knowledge distillation and step-by-step distillation (Hinton, Vinyals, Dean 2015; Hsieh et al., ACL Findings 2023)**

- Fact: knowledge distillation transfers a large model's soft-target hidden knowledge to a smaller student.
- Fact: distilling step-by-step let a 770M model beat a few-shot 540B model while using about 80 percent of the data.
- In plain words: distillation can move a specific ability into a much smaller model, which is the mechanism the project bets on for the move-choosing behavior.
- Source: [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

**Small-model parity and specialization (Qwen3 Technical Report 2025; NVIDIA SLM position, Belcak et al. 2025; Finetuner's Fallacy 2026)**

- Fact: a 1.7B base model reached parity with a 2.5B to 3B base model, though this is a pretraining-parity result, distinct from distillation.
- Fact: a position paper argues small models are sufficient and economical for specialized agent tasks, and a separate result shows a 1B specialized model beating a 3B standard model on under-represented areas through specialized pretraining.
- In plain words: small models can match larger ones on narrow targets, but the strongest results lean on specialized pretraining rather than fine-tuning alone.
- Source: [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

### B. Prompting versus fine-tuning for reliability

**Alignment and constraint adherence (Ouyang et al. 2022; structured-output reliability 2026)**

- Fact: a 1.3B aligned model's outputs were preferred over a 175B model's, so bigger is not automatically better at following intent.
- Fact: naive prompting reached high task accuracy but zero valid structured output in one study, while prompt-optimization, not fine-tuning, brought a frontier model to about 95 percent valid output.
- In plain words: reliability and format adherence are often won by alignment and prompt design rather than by size, which is the mainstream expectation the project's own move test was built to challenge for the leveled move-selection behavior specifically.
- Source: [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

### C. Distillation failure modes

**Model collapse and small-student limits (Shumailov et al., Nature 2024; Small Model Learnability Gap, ACL Findings 2025; distillation traps 2026)**

- Fact: training on recursively generated data erases the tails of the distribution, and preserving those tails needs real human data.
- Fact: small models, at or below about 3B, learn better from shorter and simpler reasoning chains, and tail noise plus a teacher-student gap can drive overconfident invention.
- In plain words: distilling a frontier teacher's confident chess writing into a small student risks copying confident wrongness, which is why writing is kept out of the trained objective and the move, which cannot be invented, is trained instead.
- Source: [nature.com/articles/s41586-024-07566-y](https://www.nature.com/articles/s41586-024-07566-y)

**Adapter forgetting (LoRA intruder dimensions, NeurIPS 2025)**

- Fact: low-rank adapters introduce intruder dimensions and forget more of pretraining than full fine-tuning, and still trail full fine-tuning on some measures. (LoRA is a cheap fine-tuning method that adds small trainable pieces called adapters, rather than retraining the whole model.)
- In plain words: cheap fine-tuning has a real cost in retained general ability, which reinforces keeping the fine-tune narrow, as a move-chooser, and late.
- Source: [arxiv.org/abs/2410.21228](https://arxiv.org/abs/2410.21228)

### D. Chess engines and human-move modeling

**Human-move prediction (Maia, McIlroy-Young et al., KDD 2020; Maia-2, NeurIPS 2024)**

- Fact: Maia predicts human moves about 46 to 52 percent of the time, against roughly 33 to 41 percent for engine-style predictors, with accuracy peaking near the training rating, and personalization can reach up to about 65 percent.
- Fact: the authors note that per-level models can be volatile and inconsistent across nearby ratings and are limited as teaching tools, and that the human-move ceiling is well below 100 percent.
- In plain words: human-move modeling is a strong descriptive level signal and one of the two automatic inputs that define the official tier move, which is why it is treated as descriptive rather than as the selector.
- Source: [maiachess.com](https://maiachess.com)

**Compact rating-conditioned prediction (Maia-3 / Chessformer, ICLR 2026)**

- Fact: a 79M rating-conditioned model reached about 57.1 percent human-move accuracy at under a quarter of the previous state-of-the-art parameter count.
- In plain words: human-move prediction is improving and getting cheaper, which strengthens the automatic tier rule, but it still describes behavior rather than prescribing what to teach.
- Source: [arxiv.org/abs/2605.19091](https://arxiv.org/abs/2605.19091)

### E. Language models playing and explaining chess

**Real chess ability of language models (Acher 2024; Karvonen 2024; reasoning-LLM chess 2025)**

- Fact: one older model plays around 1750 Elo with under 0.1 percent illegal moves at the move level but an illegal move in about 16 percent of full games, while reasoning models are illegal in the large majority of cases.
- Fact: chat and instruction tuning were found to degrade performance on the well-defined task of chess.
- In plain words: a frontier model is not a dependable chess reasoner by default, which is why the move is grounded in the engine and the tier rule rather than generated, and why the trained objective is held to the automatic move.
- Source: [arxiv.org/abs/2512.01992](https://arxiv.org/abs/2512.01992)

**Grounded small chess reasoning (C1, CSSLab 2026; faithful reasoning training 2026)**

- Fact: a 4B model trained on engine-grounded reasoning learned from a frontier teacher, then reinforced, reached about 48.1 percent puzzle accuracy, surpassing its teacher at roughly 40.8 percent, with about 100 times fewer tokens, improving from 42.3 percent after supervised training to 48.3 percent after reinforcement.
- Fact: separate work found best-move supervised training strong but reasoning sometimes unfaithful, while multi-move trajectory training was more faithful.
- In plain words: grounded small models can reason well at 4B, which is the size the production coach targets, and the finding that best-move training is strong while free-text reasoning is sometimes unfaithful is direct support for training the move and demoting the writing.
- Source: [arxiv.org/abs/2603.20510](https://arxiv.org/abs/2603.20510)

**Commentary invention and its evaluation (ACT-Eval 2026; CCC and GCC-Eval, Kim et al., NAACL 2025)**

- Fact: a strong frontier model without tools produced factually incorrect chess claims about 22 percent of the time and smaller open models more than 50 percent, and standard reference-based AI-grader scoring could not reliably detect these inventions, rating a false commentary highly.
- Fact: concept-guided generation that integrates an expert model with the language model produces more accurate commentary, and evaluation is more reliable when expert-model knowledge is folded into the grader.
- In plain words: fluent chess writing is frequently false and a lone grader misses it, which is why writing is the optional, checker-gated layer and the graded claim is the move, which cannot be invented.
- Source: [openreview.net/forum?id=nne0ti66KT](https://openreview.net/forum?id=nne0ti66KT)

### F. Grounded coaching products and shipped small tutors

**Engine-as-truth production systems (Play Magnus and Take Take Take; DecodeChess; Chess.com Game Review 2026)**

- Fact: production coaches use the engine as ground truth and detectors for structured concepts, with the language model limited to translating into English, a choice made because independent chess reasoning by a language model invents.
- Fact: one major platform's game review picks the most human among strong moves so the feedback feels like a real coach.
- In plain words: production coaches already limit the model to writing over an engine-chosen move, which supports the optional-writing-layer design and leaves the level-appropriate move choice, the trained behavior here, as the open axis.
- Source: [decodechess.com](https://decodechess.com)

**Shipped small fine-tuned tutors (community LoRA tutors 2026)**

- Fact: a LoRA fine-tune of a 4B model on distilled explanations reported high completeness and near-zero invention on a small 50-puzzle test set, and a 270M model was fine-tuned for offline move classification and rating prediction.
- In plain words: small fine-tuned chess models exist at the 4B size the production coach targets, but the strongest reports measure writing completeness rather than automatic level-appropriate move selection.
- Source: [huggingface.co](https://huggingface.co)

### G. Learning science

**Tutoring effectiveness (VanLehn 2011)**

- Fact: intelligent tutoring systems reached an effect size of about 0.76 against no tutoring, close to human tutoring at about 0.79, so structured computer tutoring can approach human tutoring.
- In plain words: a well-designed tutor can be nearly as effective as a human, which sets a real bar and motivates getting the level-appropriate move right, since the move is the load-bearing teaching choice.
- Source: [doi.org/10.1080/00461520.2011.611369](https://doi.org/10.1080/00461520.2011.611369)

**Cognitive load and expertise reversal (Sweller et al. 2019; expertise-reversal literature)**

- Fact: new information passes through a limited working memory, so instruction should minimize extra load, and guidance that helps novices can harm more advanced learners and must fade with proficiency.
- Fact: deliberate practice explains only about 21 to 26 percent of performance variance, less than once claimed.
- In plain words: showing a weaker player a more human-findable move rather than the engine's sharpest line is expertise reversal applied to the move itself, and keeping engine jargon out of the optional writing is load reduction.
- Source: [link.springer.com/article/10.1007/s10648-019-09465-5](https://link.springer.com/article/10.1007/s10648-019-09465-5)

### H. Evaluation: AI graders and sycophancy

**Grader validity and sycophancy (Zheng et al., NeurIPS 2023; Sharma et al., ICLR 2024)**

- Fact: strong AI graders reach over 80 percent agreement with humans but carry position, length, and self-enhancement biases.
- Fact: a preference model preferred a convincing sycophantic answer over a truthful one the large majority of the time, and sampling many candidates only partly reduced this.
- In plain words: AI graders are useful for style but unreliable for truth and biased toward their own family, which is why the graded move claim uses no grader and any writing score is cross-family and corrected.
- Source: [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

### I. The project's own measurements

These are the project's own internal measurements, not outside primary sources. For a claim about this specific system, such as whether this fine-tune beats this prompt on the move, a controlled experiment is the right primary source, because no outside literature can settle it. The move-selection axis is graded automatically on the engine and the human-move model with no grader, which is what makes these measurements clean.

**The move test, base versus fine-tuned versus best-prompted base, with a prompt comparison at 1.7B and 4B**

- Fact: holding the shipped grounding identical and grading the move automatically against the official rule, the fine-tune beat both its untuned base and the best engineered prompt on that base at the sizes where a prompt arm was run. On the earlier slice: at 1.7B tier-policy match 0.296 (base) and 0.389 (prompt) versus 0.463 (fine-tune), with coherence violation 0.500 and 0.611 versus 0.333; at 4B, 0.347 and 0.350 versus 0.386. On the 20-model grand slice: the 4B trio is base 0.353, prompted base 0.378, fine-tune 0.397, and the 1.7B fine-tune is 0.578 (base 0.358), second of 20.
- Fact: at 32B there is a base-versus-fine-tune result (0.347 versus 0.767) but no matched, same-model prompt comparison was run, so the claim that a prompt cannot reproduce the rule at 32B is a hypothesis, not a measured result.
- Fact: prompting failed on the graded axis at the small sizes rather than merely trailing. At 1.7B the engineered prompt pushed cross-level coherence violation to 0.611, worse than the untuned base, and at 4B it produced more varied but mis-directed moves and still lost on the rule match.
- In plain words: with grounding frozen so only the weights or the prompt change, fine-tuning reproduces the rule where a prompt on the same base does not, proven at 1.7B and 4B and hypothesized at 32B. Because the official rule already computes the same move at about 1.0 from that grounding, this is learnability, not necessity.
- Source: the project's own 1.7B and 4B prompt-controlled move test plus the 20-model grand evaluation (internal measurement)

**The 32B v4 all-scenario lead and selection-conditioned head-to-head, and the writing trade**

- Fact: on 120 held-out validation positions across three levels, the 32B v4 fine-tune reached tier-policy match 0.767 raw (0.789 as served through the shipped gate) versus about 0.49 to 0.55 for the frontier (best frontier Gemini 0.553), distinct moves across levels 0.730 (73 of 100 official beginner-not-equal-advanced chances) versus roughly 0.21 to 0.28, and raw move soundness 0.942. Across all positions where v4's move differs from the best frontier's, with no success filter, the head-to-head is 56-24-12 over the 92 diverging positions (56-24-40 over all 120). The 51 wins, 5 losses, 6 ties figure is selection-conditioned: it is computed only on the 62 of 120 positions where v4 already gives a distinct, sound, correctly graded move and differs from the frontier, so it is a subset figure, not a win rate over all positions.
- Fact: the same v4 version is the weakest recent model on writing, with a blinded instructiveness grade of 4.67, below the 4B fine-tune at 5.32 and the earlier 32B v3 at 6.35, and with about 40 percent of raw drafts failing the writing check before the gate. Writing is secondary to the evaluation claim (still present in the training loss), and the gate drives shipped checkable violations to zero, which is not certified truth.
- In plain words: the strongest-move version being the weakest-writing version follows from separating the trained move from the writing. v4 leads on agreement with our tier-selection rule, not on proven coaching, and the head-to-head win rate is the unbiased all-diverging figure, not the conditioned subset.
- Source: the project's own v4-centered honest evaluation and 4B evaluation (internal measurement)

**The definitive twenty-model grand evaluation, v4-centered**

- Fact: on the shipped held-out validation slice of 120 positions across three levels, scored across all twenty models with identical grounding and the single strict any-legal move extractor, the 32B v4 fine-tune leads the field on tier-policy match at 0.767, the 1.7B fine-tune is second at 0.578 (above every frontier), the fine-tuned versions take four of the top five, and the only frontier in the top five is Gemini at fourth (0.553). The official rule scores about 1.0 by construction on the same grounding and is the named ceiling.
- Fact: the same v4 version is intentionally weaker on the blinded cross-family writing panel, which is on-thesis because writing is secondary. The earlier v3 all-rounder sits near the middle of the field, and the faithfulness-filtered v5 retrain regressed to tier-policy match 0.536 without improving raw faithfulness (near 0.58), but the v5 run was tangled (about 27 percent less optimization and token exposure, contrastive triads broken by row-wise filtering, about 42 percent boilerplate-principle pollution, retrained from base not from v4, no checkpoint selection), so it does not isolate filtering as the cause. The whole grand evaluation cost about 54 dollars.
- Fact: the grand evaluation was audited for fairness on two points, both clean. The human-move model is symmetric across all twenty models, feeding the ground-truth tier move and every model's grounding equally, and the 120-position validation set has zero train-test leakage against the shipped model's fine-tuning data, an exact-board-key intersection of zero out of 120. Re-scoring the published outputs reproduces tier-policy match 0.767 and distinct moves 0.730 exactly.
- In plain words: the definitive twenty-model evaluation confirms the fine-tuned models reproduce the tier-selection rule better than the field on a clean, held-out, symmetrically grounded set, with the fixed rule as the ceiling. It does not prove teaching or necessity.
- Source: the project's own twenty-model grand evaluation (internal measurement)

**The earlier 803-position gap leaderboard across the model field**

- Fact: on a curated, zero-leakage set of 803 held-out positions, each discriminating so that the level-appropriate move differs from the engine's first choice for at least one level, scored across fifteen models with identical grounding, the fine-tuned models reached the highest tier-policy match in the field at about 53 percent, above every frontier model at about 43 to 48 percent, with the widest lead at the beginner and intermediate levels, and the frontier mirrored the engine's best move at every level a high fraction of the time.
- Fact: tier-policy match is weak across the whole field, most models between about a third and a half, precisely because it is a trained behavior rather than an emergent one. Faithfulness after the checker is a fairness floor at zero checkable violations (not certified truth) for all fifteen models and is deliberately not a scoring axis. The blinded writing panel's measured self-preference was about 1.44 rank, corrected in the reported ranking. The whole evaluation cost about 112 dollars.
- In plain words: the large evaluation confirms tier-policy match is the one axis where the small fine-tuned models lead the field while the field stays weak because the behavior is trained rather than emergent, all graded automatically against the rule with no grader. Leading on rule match is not the same as proven teaching.
- Source: the project's own 803-position gap evaluation (internal measurement)

**The check-and-rewrite faithfulness gate for the optional writing layer**

- Fact: the production gate, which re-samples an explanation up to four times and otherwise substitutes a verified engine-derived explanation, drove checkable writing violations from about 40 percent to zero for the small model and from about 7 percent to zero for a frontier model, and across the fifteen-model field drove every model to zero on the checkable set. This is zero checkable mechanical violations, not certified truth. Meaning-level falsehoods the detectors miss (relational pawn-square claims, forks, threats, negations, and evaluation claims) can still pass, and a cross-family AI-grader residual remains.
- Fact: on a writing failure, the post-fix gate changes only the explanation and never the move, keeping the model's own best-guess sound move and rewriting the explanation with a verified engine-derived explanation of that same move, so the served move equals the evaluated best-guess move. The small model fell back to the verified explanation about one time in ten.
- In plain words: a claim-level non-language-model gate with an automatic fallback guarantees zero checkable writing violations for any model (not certified truth) and never alters the served move, which makes mechanical faithfulness table stakes for the optional layer and, under the three-part framing, outside the graded move claim.
- Source: the project's own checker evaluation (internal measurement)

**Bigger open models on the same input**

- Fact: on the identical grounded input, larger open models invented between about 1 and 8 percent in writing, with a 27-billion-parameter open model at about 1 percent matching the frontier, while the fine-tuned small model sat near a third and the untuned base near an eighth.
- Fact: every open model was judged more instructive in writing than the small model but none reached the frontier, and the very largest model did not coach best, so training quality and size both matter more than raw parameter count for the writing voice, while the best locally runnable base was a model of about 27 to 32 billion parameters.
- In plain words: the small model's writing deficit is closed for free by model size, not by the data intervention, and a mid-size open base is the natural stronger starting point for a local coach, none of which touches the graded move claim.
- Source: the project's own open-model benchmark (internal measurement)

**Base evaluation and the move-versus-writing split**

- Fact: an untuned 4-bit small base model, scored by a strong frontier grader, reached move soundness 1.00 while writing truthfulness was zero and the no-engine-jargon rate was 0.11 on the same outputs at the same time.
- Fact: rebuilding the training data to be faithfulness-filtered, tier-aware, and more concrete cut grounded writing invention from about half to about a third and lifted writing instructiveness, but even the filtered fine-tune invents more than the untuned base, so explanation truth is not reliably learnable in the weights at small size while the move under the tier rule is.
- In plain words: on the same outputs the move was sound while the writing was unfaithful, which is the earliest evidence that the move and the writing are separate parts and that the move is the one to train.
- Source: the project's own base and retrain evaluations (internal measurement)

**Move-selection gap, density, and richer input at the moment of use**

- Fact: across the frontier models on held-out positions, level-differentiation averaged about a fifth to a third, the frontier repeated the engine's best move across the levels about 77 percent of the time, and on a large curated set about two-thirds of the decidable positions were discriminating, so the leveled-move gap is common in ordinary play rather than a niche.
- Fact: replacing the trained writing grounding with a fuller structured board state at the moment of use raised the small model's writing invention from about 40 percent to about 56 percent, while the same change barely moved a frontier model, which is format-agnostic.
- In plain words: the frontier is weak at leveled move selection but strong at not lying about the board in writing, the leveled-move behavior is exercised in most normal positions, and a small fine-tune is tied to its trained input format, so any explanation faithfulness is fixed by the checker and model size while the trained move is what leads the field.
- Source: the project's own gap-density and rich-grounding analyses (internal measurement)

**Reward design and the automatic main signal**

- Fact: the training and evaluation loop uses a fully automatic main reward (level-appropriate move, distinct moves across levels, move soundness, well-formed, no engine jargon) and, only for the optional writing layer, a held-out blinded cross-family instructiveness panel of three frontier graders that the model never trains against, self-preference-corrected by leaving out each grader's own family.
- In plain words: because the trained behavior is a move, the main reward is grader-free and un-gameable, and the learned writing grader is quarantined to the optional layer, which is the strongest realization of the anti-gaming rule.
- Source: the project's own training and evaluation harness (internal measurement)

**The Stage-4 corrected-benchmark re-evaluation, and the DPO and distillation stretch results**

- Fact: the benchmark labels were rebuilt under deeper Stockfish-17 search plus Syzygy endgame tables, with the 120 held-out test positions unchanged and only the official and engine-best targets re-derived, and every fine-tuned model was re-scored in one controlled run with identical best-guess decoding and the same strict any-legal extractor. (Stockfish is the chess engine that rates moves. Syzygy endgame tables are perfect-play lookups for positions with few pieces.) Grounded tier-policy match on the 120 held-out test is base 0.428, shipped v4 0.861, the preference-tuned v6-dpo 0.881, and the stronger, level-targeted v6-dpo2 0.892, with move soundness 0.983 and distinct moves 0.987 identical across v4, v6-dpo, and v6-dpo2. Grounded fine-tuned-minus-base is 0.433 for v4 and 0.453 for v6-dpo, so the base-versus-fine-tuned gap is preserved under the correction.
- Fact (stretch, DPO): preference-tuning the shipped v4 on tier-move pairs sharpened the advantage on held-out data without regressing it, and the stronger, level-targeted v6-dpo2 (checkpoint step 200) is now the best DPO result and the model the live demo serves, replacing the earlier v6-dpo. (DPO, short for Direct Preference Optimization, is a training method that teaches a model to prefer a better answer over a worse one by showing it pairs.) v6-dpo2 posts overall tier-policy 0.892, which is 0.031 above v4 and 0.011 above v6-dpo, and the entire gain is in the intermediate level at 0.842 versus v4 0.750 and versus v6-dpo 0.808, on positions the adapter never trained on. Beginner at 0.858 and advanced at 0.975 are byte-identical to v4 and v6-dpo, because both levels already sit at their ceiling under grounding, so v6-dpo2 is a stronger v6-dpo, not a beginner or advanced breakthrough. Soundness at 0.983 and distinct moves at 0.987 are unchanged, names-a-move is nominally higher at 0.986, and format at 0.925 lands between v4 at 0.939 and v6-dpo at 0.919, a token-cap writing-length artifact rather than a move or soundness change. This is why v4 remains the base model behind the full evaluation here, while v6-dpo2, the best preference-tuned refinement of v4, is now the model the live demo serves.
- Fact (stretch, distillation): stripping the engine and Maia grounding from the prompt collapses the untuned base to tier-policy match 0.022 and names-a-move 0.250, because without the sound list it invents illegal or unsound moves, while the engine-distilled adapter recovers the tier rule from its weights to tier-policy 0.325 and names-a-move 0.983. The honest limit is the advanced level at 0.217, its weakest, because reproducing the sharpest engine-best move from the weights alone, without the grounding the condition removes, is genuinely hardest, and grounding-free soundness at 0.653 stays below the deployable grounded 0.98.
- Fact (corrected field re-score): a free, cached re-score of the full fifteen-model field against the corrected v6 labels, with no model re-run, keeps OURS on top of the advantage and preserves the cross-family order of OURS above frontier above open. OURS-v2 is still first at tier-policy match 0.509, which is 0.042 above the best frontier, and fine-tuned-over-base is preserved at 0.151 for the 1.7B pair and 0.162 for the 32B pair. The internal frontier order reshuffled, so Claude Opus 4.8 now edges Gemini 3.1 Pro as the single strongest frontier coach. Scope caveat: these are the v4-era-grounding cached outputs judged by the sharper v6 targets, so the absolute numbers are lower than a fresh-grounding evaluation, which makes the table valid for the relative and ranking read, not for each model's ceiling.
- In plain words: on the corrected held-out benchmark the shipped-model story is unchanged, and the two stretch experiments extend it honestly. Preference tuning gives the best refinement, now v6-dpo2 and the model the live demo serves, that sharpens only the mid-level advantage with no regression while beginner and advanced stay at their grounded ceiling, and engine distillation is a behavior-in-the-weights proof that the tier rule genuinely moves into the weights for the human-like levels while the sharpest level still needs grounding. The cached full-field re-score against the corrected labels keeps OURS on top of the advantage with the cross-family order intact, so none of these results changes the necessity or the teaching conclusion.
- Source: the project's own Stage-4 corrected-benchmark re-evaluation and full-field corrected re-score (internal measurement; `RESULTS_STAGE4_CORRECTED.md`, `RESULTS_FULL_EVAL_803.md`)

### J. Economics and local deployment (secondary, low-confidence)

**Cheap fine-tuning and local inference (QLoRA and on-device runtimes)**

- Fact: quantized low-rank fine-tuning of a small model is inexpensive in cost and hardware, and on-device runtimes keep data local for privacy, though the specific cost and speed figures come from vendors and practitioners and were not independently verified.
- In plain words: the form-factor advantages of a small local move-chooser are real in kind, so they are used only as the honest deployment win and never as a load-bearing number.
- Source: [unsloth.ai](https://unsloth.ai)
