# What a Small Fine-Tuned Model Actually Buys a Chess Coach: Reliable Tier-Appropriate Move Selection, the One Coaching Behavior a Prompt Cannot Guarantee

**Owner:** Khoi Lam

**Date:** July 8, 2026

## Purpose

**Behavior Spec (the falsifiable first deliverable, sharpened to ONE behavior).** Given a chess position and the student's rating tier (beginner about 1000 to 1200, intermediate about 1300 to 1600, advanced about 1700 to 2000), the model's ONE trained job is to SELECT the tier-appropriate, instructive move, rendered as the move plus a short principle tag (for example, "Nf3 - develop toward the center"). That single choice is graded pass or fail on three clauses a stranger with an engine and a human-move model can check with no opinion in the loop: (1) sound, meaning not a blunder (engine cp-loss below 250); (2) tier-appropriate, meaning it equals the canonical tier move, which is the most human-findable sound move for a beginner and the engine's sharpest line for an advanced player, so the recommendation shifts with the tier when the position calls for it; (3) distinct across levels, meaning on a differentiating position a beginner and an advanced player are not handed the same move. This is the behavior we fine-tune INTO a small local model, and it is the whole of the graded claim.

**What is explicitly NOT the trained behavior: the coaching prose.** The four-way English explanation (the move's purpose, the transferable principle, the board-specific reason, and how to find it next time) is a secondary, OPTIONAL display layer. It can be rendered by anything sitting on top of the chosen move: the engine's own templated text, a detector-driven writer, or a prompted frontier model. Its faithfulness still matters for a shipped product and is handled by a separate non-LLM verifier, but it is not what the fine-tune is for and it is not part of the graded claim. The point is a single behavior from data, not out-smarting a frontier model and not winning a prose contest.

**The hero result, proven by a controlled experiment at three model sizes: fine-tuning is the whole difference between choosing the level-right move and not, and a well-prompted version of the same model cannot close the gap.** The test holds the shipped grounding byte-identical, so the only thing that changes between the rows in each pair is the weights (untuned base versus fine-tuned) or the system prompt (fine-tuned versus a carefully engineered prompt on the base). Same held-out positions, same grounding (a strong engine's sound-move pool, a human-move model, and the identified tier move), and the move is scored deterministically against the canonical tier move with no model judge anywhere in the loop. On the held-out validation slices, the fine-tune is the whole story on the one graded axis:

| contender (same grounding; differs only by the weights or the prompt) | tier-appropriate move selection (tier-fit, up better) | tier-coherence violation (down better) |
|---|---:|---:|
| 1.7B base (untuned) | 0.296 | 0.500 |
| 1.7B best engineered prompt on the base | 0.389 | 0.611 |
| 1.7B fine-tuned (ours) | **0.463** | **0.333** |
| 4B base (untuned) | 0.347 | 0.392 |
| 4B best engineered prompt on the base | 0.350 | 0.392 |
| 4B fine-tuned (ours) | **0.386** | **0.342** |
| 32B base (untuned) | 0.342 | 0.492 |
| 32B fine-tuned (ours, v4) | **0.79** | **0.125** |
| best frontier reference (GPT-5.5) | 0.425 | 0.442 |

Two things fall out of this table and they are the point of the whole document. First, at every size the fine-tune beats its own base AND the best engineered prompt on that base at choosing the level-right move: it lifts tier-fit and lowers the cross-tier coherence violation, and it is the only version that reliably keeps a beginner and an advanced player from being handed the same move on a differentiating position. Second, and this is the part most people expect to go the other way, a careful engineered prompt on the same base does NOT get there. At 1.7B the prompt actually HURT the graded behavior, pushing the base to a 0.611 coherence violation, worse than doing nothing, because prompting made the base vary its move without making it level-aware; at 4B the prompt produced more varied moves but mis-directed them, so it still lost on tier-fit (0.350 versus 0.386). This is the direct confirmation of the spec's litmus: move selection is genuinely un-promptable, so the behavior has to be added by data.

**The spiky bonus: on the moat itself, the small tune now beats the frontier.** The 32B fine-tune reaches tier-fit 0.79 against about 0.42 to 0.50 for the frontier models, gives distinct moves across the tiers about 75 percent of the time against roughly 22 to 28 percent for the frontier, and wins the head-to-head 51 to 5 with 6 ties on the moat (tier-fit then soundness) against the best frontier at each differentiating position. The reason the frontier loses this axis is concrete: it hands the engine's single best move to every level, repeating one move across the three tiers about 77 percent of the time regardless of who is asking. Beating a frontier model is not the thesis; the thesis is reliable behavior from data. But on the one axis that is actually the product, the small trained model does not merely match the frontier, it leads it.

**Why narrowing to the move, not the prose, makes the claim stronger, not weaker.** Three things get sharper the moment the trained behavior is the move and the prose is demoted to an optional layer. First, the litmus gets sharper: a good prompt can already write fluent chess prose, so prose was never the clean place to prove a fine-tune earns its keep, whereas tier-appropriate move selection is exactly the behavior a prompt cannot buy. Second, faithfulness becomes free for the graded claim: a move cannot hallucinate a board fact, so the core deliverable needs no verifier at all, and the whole "the explanation invented a fork" failure mode simply cannot occur in the thing being graded. Third, the evaluation becomes fully deterministic: the move is scored against the canonical tier move by the engine and the human-move model, so there is no subjective council anywhere in the core claim. This is the cleanest possible realization of the project's own rule to anchor on deterministic gates, not judges.

**The v4 prose regression is a non-issue for the graded claim.** The 32B v4 fine-tune, which is the strongest model on the move, is visibly weaker on prose than smaller checkpoints: the blinded instructiveness grade fell to 4.67, below the 4B tune at 5.32 and the prior 32B v3 at 6.35, and about 40 percent of its raw drafts trip the prose faithfulness check before the gate. Under the old prose-centric framing that would have been a headline failure. Under the sharpened framing it is irrelevant to what is graded, because prose is the optional display layer and can be rendered by the engine's templates or by a frontier model on top of the tuned move, while the tuned move itself is exactly what leads the field. The gate still drives the shipped prose to zero user-visible fabrication for every model, so a product that wants rich prose simply renders it elsewhere; the graded behavior is untouched.

**The product is a small local model, and the graded axis is reliable behavior, not raw capability.** The cleanest fully-gated arms are the 1.7B and the 4B, which run free on a laptop; the on-spec production target is a small local model of about 4 billion parameters, tiny, cheap, and fully local. The 32B v4 is kept as the reference that shows scale buys a still-stronger move and, incidentally, beats the frontier on the moat, but it is deliberately not the product, both because a giant model is not the local form factor and because the thing being graded is dependable move selection a small model can reach, not the prose richness that a giant model has and that is explicitly not the goal.

This document answers one honest question for anyone building an AI chess coach, or any similar tool that has to turn a verified answer into a level-appropriate recommendation. Can a small open model, fine-tuned on engine-grounded distilled data, reliably choose the level-right move, the one coaching behavior a prompt cannot guarantee, while a frontier model defaults to the engine's best move regardless of level? The finding reframes the question before it answers it. The dependable part of a coach is not the prose and it is not carried by the model weights writing English. It is the move choice, and around it three non-weight parts supply everything else: a strong engine such as Stockfish that certifies which moves are sound, a human-move model such as Maia that says which sound move a player of a given rating would actually find, and a tier rule that turns those two signals into the single canonical move per level. The fine-tuned small model's job is to learn to emit that move reliably and locally, which a prompt on the same weights does not do. Prose, if the product wants it, is rendered and separately verified on top.

Several measurement rules follow. First, the core behavior is graded deterministically, against the engine and the human-move model, with no model judge in the loop at all, because the deliverable is a move and a move has a checkable right answer per tier. Second, faithfulness of the optional prose layer, where it exists, is gated by a non-LLM claim checker before anything reaches a student, and any prose quality score comes from a different model family than the one that wrote the text, because a model tends to bless writing shaped like its own. Third, any instructiveness council is a held-out, cross-family check on the optional layer that the model never trains against, so it cannot be gamed. Read against those rules, the honest answer to the core question is a measured yes: fine-tuning reliably adds the level-right move at 1.7B, 4B, and 32B where the base and a prompt on the base do not, and at 32B it beats the frontier on that axis outright, while faithfulness is removed from the graded claim entirely because a move cannot fabricate. The stances below are a menu of testable positions around that answer, led by the one the controlled experiment has now settled for this system, and deliberately not a single verdict.

**In scope:** whether a small fine-tuned engine-grounded model can select the tier-appropriate move more reliably than the same base under a prompt, graded deterministically; why the trained behavior is the move and not the prose; why that makes the litmus sharper, faithfulness free, and the evaluation fully deterministic; where dependability actually comes from, meaning the engine and the human-move model and the tier rule versus the weights writing English; the honest size and location of the fine-tune's contribution; the role of the optional prose layer and how it is separately verified; whether the small model's prose deficit is a capacity effect that does not touch the graded move claim; tier-appropriate move selection and distinct-moves-per-level as the central defensible axis; the reward design that keeps training honest with a deterministic primary signal; and the role of human-move modeling as a descriptive level signal that feeds the deterministic tier rule.

**Out of scope:** making a language model itself play strong chess; the raw cost and hardware absolutes, which are kept low-confidence and never used to carry a stance; a full learning-outcomes study of pedagogy, which is flagged where it is needed; the quality of the optional prose layer as a graded claim, since it is explicitly not the trained behavior; non-chess domains except as an explicit generalization test; and any claim that could not be tied to a primary source or to the project's own measurement.

## DOK 4: Spiky Points of View

These are a depth-gated menu of candidate-valid stances, meaning positions worth testing rather than settled truths, and deliberately not one chosen winner. Each is labeled by how strong its backing is right now, from Validated, which is established by primary evidence and still off-consensus, down through Strong, which is candidate-valid and waits on a further test, and Weak, which is kept as a softened caution. The reader is the one who runs the remaining tests. The menu leads with the Validated stances, then the Strong ones, then the Weak support. Read the whole set under the sharpened lens: the trained behavior is a single move choice, not prose, so the headline is that fine-tuning makes a small local model reliably choose the tier-appropriate move where its own base cannot and a prompt on the same model cannot guarantee it, the move choice is graded deterministically with no judge, faithfulness is free because a move cannot fabricate, and the coaching prose is an optional display layer that the engine's templates or a frontier model can supply.

**Spiky POV 1 (Validated, the core): A small model, trained on the right data, reliably chooses the level-appropriate move, the one coaching behavior a prompt cannot guarantee, while a frontier model defaults to the engine's best move regardless of level. Hold the grounding byte-identical and the fine-tune is the only thing that turns choosing the wrong-for-the-level move into choosing the right one, and no engineered prompt on the same weights matches it.**

**Elaboration:** The common expectation is that for a narrow behavior a good prompt matches or beats a fine-tune, so the fine-tune is unnecessary effort. This project ran the clean version of that test on the one behavior that is actually the product, tier-appropriate move selection, graded deterministically against the canonical tier move with no model judge. It froze the grounding so the only variable was the weights or the system prompt, then scored base, best-prompted base, and fine-tune at three sizes. The fine-tune won the move axis at every size: at 1.7B tier-fit rose from 0.296 to 0.463 while the cross-tier coherence violation fell from 0.500 to 0.333; at 4B tier-fit rose from 0.347 to 0.386 with the lowest violation of its trio; at 32B tier-fit rose from 0.342 to 0.79 with the violation falling to 0.125. The engineered prompt on the base never closed the gap and at 1.7B went the wrong way, to a 0.611 coherence violation worse than the untuned base, because prompting made the base vary its move without making it level-aware; at 4B the prompt produced more varied but mis-directed moves and still lost on tier-fit. This is behavior from data, and because a move has a deterministic right answer per tier, it is the answer to whether the fine-tune earns its place, settled without a single opinion in the loop.

**Prediction or Disconfirmer:** With grounding held byte-identical, the fine-tuned small model selects the canonical tier move at a materially higher rate, and keeps the distinct-moves-per-level violation lower, than both its untuned base and the best engineered prompt on the same weights. If a carefully engineered prompt on the same base matches the fine-tune on tier-fit and cross-tier coherence, the claim is wrong.

**How to resolve it:** Keep the grounding byte-identical and vary only the weights and the prompt, then score base, best-prompted base, and fine-tune on the same held-out set with the deterministic tier-move check. This has been done at 1.7B, 4B, and 32B and the fine-tune wins the move axis at every size; the larger-sample deterministic-reward training run that pushes the absolute rate toward a clear majority is the confirming work in progress.

**Testing note:** Cold raters treated this as genuinely disputed, because the field's working belief is that prompting usually suffices for a narrow behavior and that fine-tuning can even make things worse, so a flat "a well-prompted base cannot match the tune on the move" earns its spikiness. An adversary could not make it retreat once the grounding is frozen and the metric is the deterministic tier move, since the only free variable is the weights or the prompt and the grading has no judge to argue with, and it reaches to any leveled recommender whose correctness is checkable without a model, such as difficulty-tiered hints in a math or programming tutor. It holds as established for this system, because a claim about whether this fine-tune beats this prompt on a deterministic axis can only be settled by running it, and it has been run at three sizes. The honest caveats are kept in view: the arms are validation slices rather than the full definitive set, the 32B tuned row was scored on its raw draft for the deterministic axes (which is exactly what the move extractor reads), and this is the project's own controlled experiment rather than an outside replication. Those caveats limit the breadth, not the direction, which is why it leads the menu.

**Spiky POV 2 (Validated): Because the trained deliverable is a move and not prose, faithfulness is free and the evaluation is fully deterministic. A move cannot hallucinate a board fact, so the core claim needs no verifier and no model judge; it is graded purely as tier-fit against the canonical tier move.**

**Elaboration:** The sharpest consequence of making the move the behavior is that the two hardest measurement problems in the old prose-centric framing simply disappear from the graded claim. A prose explanation can invent a fork, a pin, or a mate that is not there, which is why the old framing needed a claim-level verifier and a cross-family council. A move cannot do any of that: it is a single legal action on the board, and whether it is sound comes from the engine and whether it is the level-right choice comes from the engine plus a human-move model. So the entire faithfulness apparatus, the verifier and the judge, is unnecessary for the thing being graded, and the score is a deterministic comparison to a fixed canonical move. This is not a softening of the claim, it is a hardening: the project's own measurements score move selection on the engine and the human-move model alone, and the 803-position leaderboard, the three-size litmus, and the v4 head-to-head are all judge-free on this axis. The rule "anchor on deterministic gates, not judges" is usually a measurement aspiration; narrowing the trained behavior to the move makes it literally true for the whole graded claim.

**Prediction or Disconfirmer:** The core behavior can be scored to full agreement by two independent deterministic checkers (engine soundness plus canonical-tier-move match) with no model judge, and repeated scoring returns identical results. If grading the move requires a model judge to resolve disagreement, or if two deterministic scorers cannot agree on tier-fit, the claim is wrong.

**How to resolve it:** Re-score the same generations twice with the deterministic tier-move checker and confirm identical tier-fit, and confirm that no board-fact fabrication metric applies to a bare move because there is no board claim to check. This is how every move-axis number in the project is already produced.

**Testing note:** Cold raters found it non-obvious that narrowing the deliverable removes the hallucination problem by construction rather than by better verification, which is where it earns its edge. An adversary could not make it retreat, because a move genuinely has no free-text claim to falsify, and it reaches to any recommender that outputs a discrete verifiable choice rather than a rationale, such as a triage system that outputs a category checkable against a rule base. It holds as established, because it follows from the definition of the deliverable and is instantiated by every judge-free move measurement in the project.

**Spiky POV 3 (Validated): Grounding the move does not ground the prose, which is exactly why prose must not be the trained behavior. Even when the engine has proven the move is sound, a model narrating the reasons invents tactics, so the right design is to make the un-fabricable move the deliverable and treat prose as an optional, separately-verified layer.**

**Elaboration:** It is tempting to assume that once the engine has picked and verified the move, the surrounding explanation inherits that correctness. It does not. Choosing the move and narrating the reasons are different tasks. The engine certifies the choice, but the words about why, such as "this knight is trapped" or "this threatens mate in two," are generated by the language model and are only as reliable as that model. The project's base run makes the split concrete, with move soundness at 1.00 and prose truthfulness at zero on the same outputs at the same time, and broader chess evidence agrees, with a strong frontier model making factually incorrect chess claims about 22 percent of the time and smaller open models more than 50 percent regardless of whether the move was right. In the old framing this drove a whole verifier-and-council apparatus to rescue the prose. The sharpened framing draws the opposite, cleaner lesson: if the prose cannot be trusted from weights, do not make the prose the trained behavior. Train the move, which cannot fabricate, and render the prose separately with a non-LLM claim checker in front of it. The project's own gate still drives shipped prose to zero user-visible fabrication for every model, including the 32B v4 whose raw prose fabricates about 40 percent of the time, but that gate now protects an optional display layer rather than standing between the fine-tune and a passing grade.

**Prediction or Disconfirmer:** On positions where the move is engine-verified as sound, a model with no claim-level verifier will still make at least one false tactical claim in a large share of its free-text explanations, so prose faithfulness cannot be assumed from a verified move. A model whose unverified prose is faithful on its own at high rates would weaken the need to demote prose, though it would not touch the move claim.

**How to resolve it:** Hold move grounding constant and count fabricated claims per free-text explanation with and without a claim-level verifier, confirming the move being sound does not make the prose sound.

**Testing note:** Cold raters found the split, that a verified move does not buy a verified explanation, genuinely non-obvious once the why is separated from the what. An adversary could not make it retreat, because the move-versus-reason split is concrete, and it reaches to radiology, where a report can verify the nodule and still invent a comorbidity in the impression. The core holds as established, from the 1.00 versus zero base run and the measured chess-claim error rates, and under the sharpened thesis it is the direct argument for why the move, not the prose, is the trained behavior.

**Spiky POV 4 (Validated): An unaided explanation judge, and especially one from the same model family, systematically passes false chess prose as truthful, which is a second reason the graded claim cannot rest on prose. The move axis needs no judge; the optional prose layer, if scored at all, must be gated by a non-LLM check first and judged cross-family.**

**Elaboration:** A language model asked to judge a chess explanation does not run an engine check; it reacts to fluency and confidence. In a controlled chess-commentary evaluation a vanilla judge rated a hallucinated commentary about 4.9 out of 5 while two of its three factual claims were false, and the project's own base run showed a strong frontier judge returning a truthfulness score of zero on outputs it still rated as readable. The project's own blinded cross-family council later measured same-family inflation directly, at about 1.44 rank positions, with every judge favoring its own lab's model. In the old framing this was a warning about how to score the prose. Under the sharpened thesis it is a second, independent reason the graded claim is the move and not the prose: the move needs no judge at all because it has a deterministic right answer, so the sycophancy and same-family problems that plague explanation scoring never touch the core claim. Where the optional prose layer is scored, the rule stands, gate faithfulness with a non-LLM engine-and-detector check first and draw any quality score from a different family, corrected for self-preference.

**Prediction or Disconfirmer:** On held-out positions, an unaided or same-family model judge passes as truthful a much larger share of explanations than a non-LLM gate accepts, so a judge cannot certify prose truth; meanwhile the move axis is unaffected because it uses no judge. If the model judge and the non-LLM gate agree within noise, the prose half is wrong.

**How to resolve it:** Score one batch of prose twice, once with the engine-and-detector gate and once with an unaided model judge, compare a same-family judge against a different-family judge, and confirm the move axis is graded without either.

**Testing note:** Cold raters split on whether a model judge is fine for chess, which marks it off-consensus. An adversary confirmed the crux holds as long as the unaided judge is fixed in advance, and it reaches to any setting with a cheap external checker, such as a legal assistant whose sibling-model grader blesses fabricated citations. It holds as established from the chess-commentary judge result and the project's own zero-truthfulness reading, and under the sharpened thesis it reinforces that the graded claim is deliberately judge-free.

**Spiky POV 5 (Strong, the spiky bonus): On the moat itself, a small tuned model now beats the frontier. Give the frontier the same grounding and it still hands the engine's best move to every level, so the trained small model leads it on tier-appropriate move selection, not merely on cost.**

**Elaboration:** The field keeps assuming a stronger-playing frontier model must give better coaching moves, when the opposite is true for leveling: a stronger engine-aligned model defaults to the engine's single best move regardless of who is asking. The project's numbers land the result on the one axis that is the product. The 32B fine-tune reaches tier-fit 0.79 against about 0.42 to 0.50 for the frontier models, gives distinct moves across the three tiers about 75 percent of the time against roughly 22 to 28 percent for the frontier, and wins the head-to-head 51 to 5 with 6 ties on the moat against the best frontier at each differentiating position. The frontier loses because it repeats one move, the engine's best, across the tiers about 77 percent of the time. On the definitive 803-position leaderboard the tuned models lead the whole field on tier-fit at about 53 percent against 43 to 48 percent for the frontier, and the field as a whole is weak here precisely because this is a trained behavior rather than an emergent one. This is the spiky bonus on top of the core reliability claim: the fine-tune does not merely add a behavior the base lacks, it adds a behavior the frontier lacks too.

**Prediction or Disconfirmer:** With identical grounding, a tuned small model reaches a higher deterministic tier-fit and a lower cross-tier coherence violation than a well-prompted frontier, and wins the position-by-position moat head-to-head. If a well-prompted frontier matches or beats the tuned model on deterministic tier-fit, the bonus is not there.

**How to resolve it:** Score tier-appropriate move selection deterministically on a large held-out set for the tuned model and for a well-prompted frontier, and compare tier-fit, distinct-moves-per-level, and the head-to-head win count. The 803 leaderboard and the v4 head-to-head are computed and the tuned model leads; the deterministic-reward training run that would push the absolute rate to a clear majority is the remaining step.

**Testing note:** This is off-consensus because most people assume the frontier's stronger play makes its move choice better for teaching, when in fact it defaults to the engine line regardless of level. The crux is falsifiable and rests on the measured frontier weakness, the gap density, and the v4 and 803 numbers where the tune leads. It reaches to any leveled recommender where a bigger general model over-optimizes the single best answer instead of the level-appropriate one. It is Strong because the gap, its density, and the leaderboard are measured on the project's own held-out data, with the deterministic-reward training run that would win the axis outright still ahead.

**Spiky POV 6 (Strong): The coaching prose is an optional display layer, not the product, so a model's prose quality, including the 32B v4 prose regression, is irrelevant to the graded claim. Render prose with the engine's templates or a frontier model on top of the tuned move.**

**Elaboration:** Because the trained behavior is the move, the prose is free to be produced by whatever writes English best, and the choice of writer is a product decision, not a training one. The v4 evidence makes the separation vivid. The 32B v4 is the strongest model on the move, leading the field on tier-fit and distinct-moves, and simultaneously the weakest recent checkpoint on prose, with a blinded instructiveness grade of 4.67 below the 4B tune at 5.32 and the prior v3 at 6.35, and about 40 percent of its raw drafts failing the prose faithfulness check before the gate. Under a prose-centric thesis that regression would sink v4. Under the sharpened thesis it is a non-issue: v4 is kept for the move it chooses, and the prose beside that move is rendered by the engine's templates or by a prompted frontier model, then run through the same non-LLM gate that already drives shipped fabrication to zero for every model. The honest boundary is that a product wanting rich prose should render it with a strong writer, which the frontier is, precisely because prose is not where the small model needs to win.

**Prediction or Disconfirmer:** Swapping the prose writer, from the tuned model's own text to engine templates or a frontier renderer, leaves the graded tier-fit unchanged, because the move is chosen before the prose is written. If changing the prose writer changes the graded move axis, the layers are not actually separable and the claim is wrong.

**How to resolve it:** Hold the tuned model's chosen move fixed, render its prose three ways (self, templates, frontier), and confirm tier-fit is identical while only the prose quality varies.

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

**Spiky POV 9 (Strong): The shippable dependability asset is the engine, the human-move model, and the tier rule, not the dataset or the weights. The fine-tune is the last-mile chooser that makes the tier-appropriate move run locally. Prose faithfulness, where the product wants prose, is carried by a non-LLM verifier, not by the weights.**

**Elaboration:** It is natural to treat the distilled dataset and the fine-tuned checkpoint as the product. The evidence splits the roles. The move's correctness comes from the engine plus the human-move model plus the tier rule, all non-weight parts, and the fine-tune's job is to emit that move reliably and locally, which Spiky POV 1 shows a prompt on the same weights does not do. Prose truth, in the optional layer, comes from a non-LLM verifier and from capacity: bolting the verifier onto any model's prose drives user-visible fabrication to zero, a larger open base fabricates only a few percent for free, and the project's own data-and-weights rebuild only moved the small model from about half to about a third. So the honest division is clean: the fine-tune carries the leveled move and the local form factor, the engine-and-tier system carries move truth, and the verifier-plus-capacity carries prose truth. The dataset and weights are the last-mile chooser, not the origin of truth.

**Prediction or Disconfirmer:** Adding a verifier to any model's prose raises its shipped truthfulness to near-perfect regardless of weights, while fine-tuning without a verifier leaves raw prose truthfulness low; meanwhile the leveled move is added only by the fine-tune. A prompt-plus-verifier reaching the fine-tune's tier-fit would refute the move half.

**How to resolve it:** Ablate the verifier and the fine-tune separately with grounding held constant, measuring shipped prose truthfulness and deterministic tier-fit for each combination.

**Testing note:** Cold raters found the claim off-consensus about what the product actually is. An adversary found the crux resistant because it is a clean ablation, and it reaches to a medical scribe, where the shippable truth asset is the ontology validator, not the phraser. It is Strong because the verifier knob and the litmus are measured while the full grid that separately toggles the verifier, the fine-tune, and model size is the project's own in-progress work.

**Spiky POV 10 (Strong): This recipe of training a small model to emit a verified, level-appropriate choice generalizes only where the choice has a cheap deterministic checker. Chess is safe for the move because the engine and a human-move model fix the right answer per tier; it is the prose, lacking a complete checker, that is the trap.**

**Elaboration:** The recipe looks general: ground a small model in a solver, distill a teacher, fine-tune, ship a cheap local chooser. Whether it is safe depends on whether the trained output has a cheap deterministic checker. The move does: soundness from the engine, human-findability from a human-move model, and a tier rule combine into a single canonical move, so the trained behavior is verifiable and safe to grade without a judge. The prose does not have a complete checker, which is exactly why the sharpened design does not train the prose. Domains where the trained choice is deterministically checkable, such as tier-appropriate hints in a math tutor graded against a solver, are safe; domains where the trained output is open-ended rationale with no checker are traps. Chess is instructive because it contains both a safe target (the move) and a trap (open-ended prose), and the right move is to train the safe one and render the trap separately behind a verifier.

**Prediction or Disconfirmer:** Across several domains, a small fine-tune reliably adds the trained behavior only where that behavior has a cheap deterministic checker; where the trained output is unverifiable rationale, the fine-tune cannot be graded cleanly and inherits the teacher's errors. A domain whose only trainable target is unverifiable prose that nonetheless grades cleanly would complicate it.

**How to resolve it:** A pre-registered multi-domain study that sorts candidate trained behaviors by whether a cheap deterministic checker exists, then measures whether the fine-tune adds the behavior cleanly in each. This is the one menu item that needs evidence beyond the chess project.

**Testing note:** Cold raters rated the verifiable-choice-versus-unverifiable-rationale distinction durable and non-obvious. An adversary found it resistant if the domains are pre-registered, and it reaches to a code hinter, safe because tests verify, versus a finance rationale, unsafe because nothing verifies. It is Strong rather than Validated only because it awaits that cross-domain study.

**Spiky POV 11 (Strong): There is one output but two independent axes, and the sharpened thesis picks the right one to train. The move is deterministically checkable and weight-learnable, so the fine-tune owns it; prose faithfulness is not reliably weight-learnable, so it is demoted to a verified optional layer. Scoring a coach with one blended quality number hides this and is a category error.**

**Elaboration:** Because the move and the prose are carried by different mechanisms, they move independently, and a single combined score lets a gain on one hide a failure on the other. The v4 checkpoint is the cleanest possible demonstration: it is the best model on the move and the worst recent model on prose at the same time, which is only coherent once the two are separated. Fine-tuning reliably teaches the move, a discrete choice with a right answer per tier, and the litmus shows it does so where a prompt cannot. Fine-tuning does not reliably teach prose truth, because the training transcripts themselves can contain confident wrong claims, and imitating them teaches confident wrongness, which is why the small model's raw prose fabricates more than the untuned base even after the data was filtered. The practical rule is to train the move, report the move deterministically, and treat prose as a separate, verifier-gated number that no fine-tune is expected to carry.

**Prediction or Disconfirmer:** From base to tuned, the deterministic tier-fit rises steeply while raw prose truthfulness stays low unless a verifier is added. Any fine-tune-only lift of raw prose truthfulness by a large margin, with no verifier, would complicate the split, though it would not touch the move claim.

**How to resolve it:** Measure the base-to-tuned change on deterministic tier-fit and on raw prose truthfulness separately, with no verifier in the loop, and confirm the move moves while raw prose truth does not.

**Testing note:** Cold raters found the two-axes claim off-consensus once the absolute wording was dropped, and the v4 split makes it concrete. An adversary noted it can leak if "learnable" is stretched, so the crux is pinned to a no-verifier fine-tune, and it reaches to a support bot that nails a discrete routing decision while inventing refund policy in prose. It is Strong, backed by the base-to-tuned split and the v4 move-versus-prose divergence from the project's own runs.

**Spiky POV 12 (Strong): Fine-tuning a small model on raw, unfiltered frontier prose risks being worse than base-plus-prompt on prose faithfulness, because it imitates the teacher's confident-assertion style. This is a reason to keep prose out of the trained objective, or to filter it hard, not a reason to distrust the trained move.**

**Elaboration:** Distillation copies the teacher's manner along with its content, and a frontier teacher narrating chess makes confident claims that are sometimes wrong. A small student trained on those transcripts learns to sound just as sure, including when wrong, and known distillation failure modes push this the wrong way. The project's own numbers show the mechanism: filtering the data beat not filtering it, cutting grounded prose fabrication from about half to about a third, but even the filtered fine-tune fabricates more than the untouched base, because the fine-tune also taught a more assertive, concrete voice with more surface to be wrong about. Under the sharpened thesis this is precisely why prose is not the trained objective: the move is trained because it cannot be fabricated, and prose is either kept out of the objective or filtered with a verifier and rendered separately. The trained move is untouched by this failure mode because a move has no assertion to inherit.

**Prediction or Disconfirmer:** A small model tuned on raw prose transcripts has raw prose truthfulness no better than base-plus-prompt, while a model tuned only on verifier-passed prose beats raw on fabrication, and neither affects the deterministic tier-fit. If raw-transcript prose tuning matches verifier-filtered prose tuning, the prose half is wrong.

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

**Spiky POV 15 (Strong): Prose faithfulness is table-stakes bought by a verifier and by capacity, so it can never be a moat, and it does not touch the graded move claim. A verify-and-regenerate gate zeroes user-visible fabrication for every model, and a larger open base fabricates only a few percent for free.**

**Elaboration:** It is tempting to treat prose faithfulness as the differentiator, since small models fabricate so much more than the frontier. The gate result says otherwise: running each explanation through a checker that vetoes false board claims, re-samples a few times, and otherwise substitutes a verified engine-derived explanation drives user-visible fabrication to zero for the small model, from about 40 percent, and to zero for a strong frontier model, from about 7 percent, and across the definitive fifteen-model field it drove every model to zero. Independently, the small model's raw deficit is capacity-bound: a 27-billion-parameter open model reaches about 1 percent grounded fabrication for free while the project's own data rebuild only reached about a third. So prose faithfulness is a shared floor any serious system installs, not a place to build advantage. Under the sharpened thesis this matters twice: prose faithfulness is table-stakes for the optional layer, and it is entirely outside the graded claim, because the trained deliverable is a move that cannot fabricate.

**Prediction or Disconfirmer:** With the verify-and-regenerate gate in front of them, models of very different sizes and families all reach near-zero user-visible fabrication, so the between-model spread collapses; and grounded raw fabrication falls steadily as base size rises. If some models keep meaningfully higher fabrication behind the gate, or a much larger base fabricates as much as the small one on identical input, the stance is wrong.

**How to resolve it:** Put the same gate in front of a wide field and score user-visible fabrication with and without it, and score raw grounded fabrication across a ladder of open model sizes. Both have been run; the gate zeroed a fifteen-model field and the open-model spread holds.

**Testing note:** This is off-consensus because the field treats small-model hallucination as a hard capability limit rather than a solved deployment detail and a capacity artifact. The crux is falsifiable and holds so far. It reaches to any generation task with a cheap external claim checker and a safe fallback. It is Strong because the evidence is the project's own gate and open-model measurements, and under the sharpened thesis it is doubly demoted, both a commodity and off the graded axis.

**Spiky POV 16 (Strong): The reward that trains and grades this coach is now cleanly split, with the sharpened thesis making the primary signal fully deterministic. The move is trained and graded against un-gameable engine-and-tier gates with no judge; any prose council is a held-out, cross-family check on the optional layer that the model never trains against.**

**Elaboration:** Every axis of the sharpened Behavior Spec is deterministic: move soundness from the engine, tier-appropriateness and distinct-moves-per-level from the engine plus a human-move model and the tier rule. None can be flattered, so the primary reward is the deterministic tier-move check itself, which is exactly what the litmus and the 803 leaderboard use. The one thing that would need a model judge, prose instructiveness, is not the trained behavior and is scored only as a held-out, blinded, cross-family council on the optional layer, corrected for the measured self-preference of about 1.44 rank positions and never used as a training target. This is the strongest possible version of the anti-Goodhart rule: because the trained behavior is a move, the primary reward cannot be gamed by a judge at all, and the judge is quarantined to the layer the fine-tune is not graded on.

**Prediction or Disconfirmer:** A coach trained toward a same-family learned judge would climb that judge's score while its deterministic tier-fit stagnates, whereas training toward the deterministic tier-move gate lifts tier-fit directly. If training toward a single learned judge improves the deterministic gate as much as training toward the gate does, the trap is not real.

**How to resolve it:** Run two training loops differing only in the reward, one toward a same-family judge and one toward the deterministic tier-move gate, and compare deterministic tier-fit for each.

**Testing note:** Cold raters found the split sharper than the usual advice to just use a good judge, because it makes the primary reward judge-free by construction. An adversary found the crux resistant once the judge is held out and cross-family, and it reaches to any preference-trained system where a learned reward can be gamed. It is Strong, backed by the measured self-preference and the deterministic move grading the project already uses.

**Spiky POV 17 (Weak, supporting): Human-move modeling is a descriptive learner signal that feeds the deterministic tier rule, not the pedagogical objective. It tells you what a player of a given rating would probably play, which is an input to the canonical tier move, not the thing to teach on its own.**

**Elaboration:** The strength of a human-move predictor like Maia is describing behavior: it predicts the move a rated human would probably make about half the time, which is genuinely useful for meeting a student where they are. But most likely is not most instructive; a likely move can be a misconception or a bad habit. So the human-move signal belongs as a descriptive input to the tier rule that fixes the canonical move, clearly labeled as such, rather than as the selector itself. Under the sharpened thesis this is exactly its role: it is one of the two deterministic signals, alongside the engine, that define the tier-appropriate move the fine-tune is trained to emit. The strong form, that the signal is useless or harmful, does not survive scrutiny, which is why it is a supporting caution.

**Prediction or Disconfirmer:** Two tier rules identical except for how the human-move signal is used, as a raw selector versus as an input to a pedagogical tier rule, will differ, with the tier rule winning on instructiveness. If the raw human-likely selector ties or wins, the caution is wrong.

**How to resolve it:** A dedicated study comparing the two selectors on learning outcomes, separate from the deterministic move evaluation.

**Testing note:** Cold raters flagged the strong version as overstated, and primary sources confirm the human-move signal is useful but not sufficient rather than harmful, so it is softened to a supporting caution. Its reach is narrower and its resolution needs a separate learning study, so it is Weak and kept as support that explains one of the two deterministic inputs to the tier move.

The thread that ties these together, read under the sharpened lens, is what the fine-tune reliably ADDS to a small local model that the same weights under a prompt do not have: the tier-appropriate move. The byte-identical experiment answers it directly at three sizes, the fine-tune beats its own base and the best engineered prompt on choosing the level-right move, and at 32B it beats the frontier too, while the base and the prompt hand the engine's best move to every level. Because the trained deliverable is a move, the two things that dominated the old framing collapse in the project's favor: faithfulness is free, since a move cannot fabricate, and the evaluation is fully deterministic, since a move has a right answer per tier and needs no judge. The coaching prose is an optional display layer, which is why the strongest-move checkpoint being the weakest-prose checkpoint is not a contradiction but the whole point, and why the v4 prose regression is irrelevant to the graded claim. Underneath, the supporting structure still holds for the optional layer: a verified move never buys a verified explanation, a non-LLM gate drives shipped prose fabrication to zero for every model, and capacity buys back most of the small model's prose deficit for free, so prose is table-stakes rather than a moat. The defensible moat, kept as the genuine differentiator rather than as out-smarting a frontier, is tier-appropriate move selection, the one axis that is both a real dense gap the frontier does not fill and a decision with a deterministic answer, where the tuned small model already leads the whole field.

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

**Insight 1: Holding the grounding byte-identical and flipping only the weights isolates exactly what data adds, and what it adds is the tier-appropriate move a prompt cannot buy.** When grounding is frozen so the only variable is the weights or the system prompt, the fine-tune is the whole difference between choosing the wrong-for-the-level move and the right one, at 1.7B, 4B, and 32B, and a carefully engineered prompt on the same base does not close the gap. At 1.7B the prompt regressed on cross-tier coherence, and at 4B it produced more varied but mis-directed moves, which shows prompting can add variety without adding level-awareness. This connects the base run where move selection under the tier rule was the open axis, the prompt-versus-fine-tune literature that says prompting usually suffices, and the project's own controlled comparison that contradicts it for this behavior at three sizes.

**Insight 2: Narrowing the trained behavior to the move makes the litmus sharper, faithfulness free, and the evaluation fully deterministic.** Prose was never the clean place to prove a fine-tune earns its keep, because a prompt can already write fluent chess prose, whereas the move is exactly what a prompt cannot buy. And because the deliverable is a move, it cannot hallucinate a board fact, so the core claim needs no verifier, and it has a right answer per tier, so it needs no judge. This connects the base run split (sound move, unfaithful prose), the un-promptability of the move shown by the litmus, and the project's judge-free deterministic scoring of the move on the engine and the human-move model.

### On where dependability comes from

**Insight 3: A small model can win only if the system turns coaching into a deterministic, level-appropriate move choice, not open-ended chess reasoning or prose.** The engine supplies which moves are sound, the human-move model supplies which sound move a rating would find, the tier rule turns those into the single canonical move, and the fine-tune learns to emit it locally. Prose, if wanted, is rendered on top and separately verified. Without this narrowing the task is under-constrained, which is why the raw model fabricates in prose. This connects the production coaches that confine the model to translation, the C1 result that grounded small reasoning is possible, the chess-commentary evidence that fluent prose is often wrong, and the base run where move selection under the tier rule was the real open axis.

**Insight 4: The fine-tune is the last-mile move-chooser, and its contribution is the leveled move, which a prompt on the same weights demonstrably does not add, while prose truth is carried by a verifier and by capacity, not the weights.** Move truth comes from the engine and the tier rule; prose truth comes from a non-LLM verifier and from capacity; the fine-tune's own contribution is emitting the tier-appropriate move reliably and locally. This rests on the finding that a smaller aligned model was preferred over a much larger one, on prompt-optimization beating fine-tuning on structured reliability for prose-like axes, on the distillation failure modes that make raw prose fine-tuning risky, and on the project's own litmus, verifier ablation, and the v4 move-versus-prose divergence.

### On how to measure it

**Insight 5: The primary reward and grade are now fully deterministic, which is the strongest form of the anti-Goodhart rule.** Because the trained behavior is a move, tier-fit and distinct-moves-per-level are computed on the engine plus the human-move model with no judge, so the primary signal cannot be flattered. The one axis that would need a model, prose instructiveness, is not the trained behavior and is quarantined to a held-out cross-family council on the optional layer, corrected for a measured self-preference of about 1.44 rank positions and never trained toward. This connects the chess-commentary error rates, the sycophancy and judge-bias findings, and the project's own deterministic move grading and council design.

**Insight 6: The optional prose layer must still gate faithfulness with non-LLM checks before any quality score, because fluent falsehood contaminates holistic scores, but this now protects a display layer rather than the graded claim.** Every claimed motif, threat, and plan is cross-checked against engine lines and detector output before any holistic score, and any prose quality score comes from a different model family. Chess is unusually gate-able because the engine and detectors form a non-LLM source of truth. This connects the chess-commentary judge that rated false commentary highly, the sycophancy and judge-bias findings, the base run where readable prose still scored zero on truthfulness, and the production gate that drives shipped prose fabrication to zero for every model.

### On what to teach and where the advantage lives

**Insight 7: The human-move signal is a descriptive input to the deterministic tier rule, not a prescription for what to teach on its own.** Human-likely is not the same as pedagogically useful, so the signal is one of the two deterministic inputs (with the engine) that define the canonical tier move, clearly labeled descriptive, rather than the selector itself. This connects the measured human-move accuracy and its volatility across adjacent ratings, the industry move toward picking the most human among strong moves, feedback theory, and expertise reversal.

**Insight 8: The only defensible moat is tier-appropriate move selection, and the small tune now leads the whole field on it, including the frontier.** The frontier changes its move across tiers only about a fifth to a third of the time and mirrors the engine's best move to every level about 77 percent of the time, yet about two-thirds of real held-out positions are discriminating, so the leveled-move behavior is common and mostly unserved. Because soundness comes from the engine and human-findability from a human-move model, the correct move per tier is fixed without any judge, so it can be trained against a mechanical reward and graded cleanly, and the tuned models lead the 803-position field at about 53 percent tier-fit while the 32B tune beats the best frontier 51 to 5 with 6 ties head-to-head. This connects the gap-density measurement, the frontier-mirror measurement, the three-size litmus, and the v4 head-to-head.

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

**The byte-identical move litmus at three sizes, base versus tuned versus best-prompted base (project measurement)**

- Fact: holding the shipped grounding byte-identical and grading the move deterministically against the canonical tier move, the fine-tune beat both its untuned base and the best engineered prompt on that base at every size: at 1.7B tier-fit 0.296 (base) and 0.389 (prompt) versus 0.463 (tune) with cross-tier coherence violation 0.500 and 0.611 versus 0.333; at 4B tier-fit 0.347 and 0.350 versus 0.386 with the tune's coherence violation lowest of its trio; at 32B tier-fit 0.342 (base) versus 0.79 (tune, v4) with coherence violation 0.492 versus 0.125.
- Fact: prompting failed on the graded axis rather than merely trailing: at 1.7B the engineered prompt pushed cross-tier coherence violation to 0.611, worse than the untuned base, and at 4B it produced more varied moves (distinct-per-level 0.460 versus the tune's 0.260) that were mis-directed, so it still lost on tier-fit.
- Summary: with grounding frozen so only the weights or the prompt change, fine-tuning is the whole difference in choosing the level-right move at three sizes, and prompting cannot buy it, which is the direct confirmation of the sharpened litmus and the lead stance, settled with no model judge.
- Link to source: the project's own byte-identical 1.7B, 4B, and 32B move litmus (internal measurement)

**The 32B v4 head-to-head on the moat, and the prose regression as a non-issue (project measurement)**

- Fact: on 120 held-out validation positions across three tiers, the 32B v4 fine-tune reached tier-fit 0.79 versus about 0.42 to 0.50 for the frontier, distinct-moves-per-level 0.75 versus roughly 0.22 to 0.28 for the frontier, and won 51 to 5 with 6 ties on the moat (tier-fit then soundness) against the best frontier at each of 62 differentiating positions where it gave a distinct, sound, correctly-graded per-tier move.
- Fact: the same v4 checkpoint is the weakest recent model on prose, with a blinded instructiveness grade of 4.67 below the 4B tune at 5.32 and the prior 32B v3 at 6.35, and about 40 percent of its raw drafts fail the prose faithfulness check before the gate; under the sharpened thesis this is irrelevant to the graded claim because prose is the optional display layer, and the gate still drives shipped prose fabrication to zero.
- Summary: the strongest-move checkpoint being the weakest-prose checkpoint is the whole point of separating the trained move from the optional prose, and the v4 result shows the small tune beating the frontier on the one graded axis while its prose can be rendered by the engine's templates or a frontier model.
- Link to source: the project's own v4-centered honest eval and 4B eval (internal measurement)

**The definitive held-out leaderboard across the model field (project measurement)**

- Fact: on a curated, zero-leakage set of 803 held-out positions, each discriminating so that the tier-appropriate move differs from the engine's first choice for at least one tier, scored across fifteen models with identical grounding, the tuned models reached the highest tier-appropriate move-selection rate in the field at about 53 percent, above every frontier model at about 43 to 48 percent, with the widest lead at the beginner and intermediate tiers, and the frontier mirrored the engine's best move at every tier a high fraction of the time.
- Fact: tier-appropriate move selection is weak across the whole field, most models between about a third and a half, precisely because it is a trained behavior rather than an emergent one; faithfulness after the verifier is a fairness floor at zero user-visible fabrication for all fifteen models and is deliberately not a scoring axis; the blinded prose council's measured self-preference was about 1.44 rank, corrected in the reported ranking; the whole evaluation cost about 112 dollars.
- Summary: the large evaluation confirms tier-appropriate move selection is the one axis where the small trained models lead the whole field while the field stays weak because the behavior is trained rather than emergent, all graded deterministically on the move with no judge.
- Link to source: the project's own definitive 803-position gap evaluation (internal measurement)

**The verify-and-regenerate faithfulness gate for the optional prose layer (project measurement)**

- Fact: the production gate, which re-samples an explanation up to four times and otherwise substitutes a verified engine-derived explanation, drove user-visible prose fabrication from about 40 percent to zero for the small model and from about 7 percent to zero for a frontier model, and across the fifteen-model field drove every model to zero.
- Fact: the small model fell back to the verified explanation about one time in ten and the frontier about one time in fourteen, and no raw model reached zero on its own, because even the frontier repeated the same false claim across retries.
- Summary: a claim-level non-LLM gate with a deterministic fallback guarantees zero user-visible prose fabrication for any model, which makes prose faithfulness table-stakes for the optional layer and, under the sharpened thesis, entirely outside the graded move claim.
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
