# What a Small Fine-Tuned Model Actually Buys a Chess Coach: Grounding Carries Dependability, the 1.7B Carries Form Factor

**Owner:** Khoi Lam

**Date:** July 6, 2026

## Purpose

This document answers one honest question for anyone building an AI chess coach, or any similar tool that has to explain a verified answer in plain human terms. Can a small open model of about 1.7 billion parameters, fine-tuned on engine-grounded distilled data, deliver reliably level-calibrated coaching, meaning the most instructive move for a student's rating explained without leaking engine internals, more dependably than a well-prompted frontier model? The finding reframes the question before it answers it. Dependability in this kind of system is not carried by the model weights. It is carried by three parts that sit outside the language model: a strong engine such as Stockfish that certifies which move is best, tactical and positional detectors that expose the concrete features of the position, and a non-LLM verifier that checks each explanation claim against the engine and the detectors before anything reaches the student. The fine-tuned 1.7B model is the last-mile compressor. Its job is to take that already grounded and already verified behavior and render it locally, cheaply, privately, and in a steady low-variance voice. It is not the origin of dependability.

Two measurement rules follow from this. First, "more dependable" has to be measured as the worst-case rate at which every constraint passes at once under repeated sampling at the temperature the coach will actually run at, not as an average quality score. Second, faithfulness has to be gated by the non-LLM checks first, and any remaining tone-or-level scoring has to come from a different model family than the one that produced the text, because a model tends to bless writing shaped like its own. The honest answer to the core question is a conditioned yes. With grounding held constant, a small fine-tuned model can match a well-prompted frontier model on faithful level-fit, and it can win on form factor, meaning cost, latency, privacy, and offline use, and on register consistency, meaning staying in plain no-engine-speak voice with low variance. It is not more dependable than a frontier model in general, and the sharper claim that it wins on the worst-case tail is a real bet that only the project's own grounding-held-constant comparison can settle. The stances below are a menu of testable positions around that answer, not a single verdict.

**In scope:** whether a small fine-tuned engine-grounded model can coach more dependably than a well-prompted frontier model; where dependability actually comes from, meaning grounding and detectors and a verifier versus the weights; how to measure it, meaning worst-case pass-all-constraints under sampling, a non-LLM faithfulness gate, and a different-family judge; the honest size and location of the fine-tune's contribution, meaning form factor and register; and the role of human-move modeling as a descriptive level signal rather than a teaching target.

**Out of scope:** making a frontier model itself play strong chess; the raw cost and hardware absolutes, which are kept low-confidence and never used to carry a stance; a full learning-outcomes study of pedagogy, which is flagged where it is needed; non-chess domains except as an explicit generalization test; and any claim that could not be tied to a primary source or to the project's own measurement.

## DOK 4: Spiky Points of View

These are a depth-gated menu of candidate-valid stances, meaning positions worth testing rather than settled truths, and deliberately not one chosen winner. Each is labeled by how strong its backing is right now, from Validated, which is established by outside primary evidence and still off-consensus, down through Strong, which is candidate-valid and waits on the project's own grounding-held-constant comparison, and Weak, which is kept as a softened caution. The reader is the one who runs the tests. The menu leads with the two Validated stances, then the Strong ones, then the Weak support, and the whole set sits under one lens: in this system, truth is supplied by the engine and the detectors and a verifier, while the small fine-tune supplies the voice and the form factor.

**Spiky POV 1 (Validated): An unaided explanation judge, and especially one from the same model family, systematically passes false chess coaching as truthful, so faithfulness must be gated by a non-LLM engine-and-detector check before any preference or pedagogy score is allowed to count.**

**Elaboration:** Chess is unusual because the truth of a claim about a position, such as whether a move really is a fork, whether a threat really exists, or whether a line really wins material, can be checked mechanically against the engine's principal variation and against motif detectors. An LLM asked to judge the same explanation does not run that check; it reacts to fluency and confidence. The evidence is direct. In a controlled chess-commentary evaluation, standard reference-based LLM-as-a-judge scoring could not reliably detect hallucinations, and a vanilla judge rated a hallucinated commentary about 4.9 out of 5 while two of its three factual claims were false. The project's own base run shows the same failure from the other side, where a strong frontier judge returned a truthfulness score of 0.00 on outputs it still rated as readable, which means the readable-looking text was not truthful and only a non-LLM gate caught it. Same-family judging makes this worse, because the teacher that produced the text and a judge from its own family share the same blind spots. The design rule is a hard ordering: run every claimed motif, threat, and line through the engine and the detectors first, discard whatever fails, and only then let any model score tone or level-fit, and draw that scorer from a different family.

**Prediction or Disconfirmer:** On held-out positions, an unaided or same-family LLM judge will pass as truthful a much larger share of explanations than the non-LLM gate accepts, including many the gate rejects; if the LLM judge and the non-LLM gate agree within noise, the claim is wrong.

**How to resolve it:** Score one batch of coaching outputs twice, once with the engine-and-detector gate and once with an unaided LLM judge, then also compare a same-family judge against a different-family judge, and measure the disagreement.

**Testing note:** Cold raters did not treat this as obvious; they split on whether an LLM judge is fine for chess, which marks it as off-consensus. An adversary confirmed the crux holds as long as "unaided judge" is fixed in advance, and it reaches beyond chess to any setting with a cheap external checker, such as a legal assistant whose sibling-model grader blesses fabricated citations. Against primary sources it holds as established, from the chess-commentary judge result and the project's 0.00 truthfulness reading, which is why it leads the menu.

**Spiky POV 2 (Validated): Grounding the move does not ground the explanation. Even when the engine has proven the recommended move is best, a small model narrating the reasons will still invent tactics, so what removes the fabrication is a claim-level non-LLM check, not a verified move.**

**Elaboration:** It is tempting to assume that once the engine has picked and verified the move, the surrounding explanation inherits that correctness. It does not. Choosing the move and narrating the reasons are different tasks. The engine certifies the choice, but the words about why, such as "this knight is trapped," "this pins the queen," or "this threatens mate in two," are generated by the language model and are only as reliable as that model. The project's base run makes the split concrete, with move soundness at 1.00 and truthfulness at 0.00 on the same outputs at the same time, which means every recommended move was correct while the explanations still carried false tactical claims. Broader chess evidence agrees: a strong frontier model without tools made factually incorrect chess claims about 22 percent of the time, and smaller open models more than 50 percent, regardless of whether the final move was right. The fix therefore cannot be to verify the move harder. It has to be a separate verifier that checks each individual claim in the explanation against the engine and the detectors and vetoes the ones that fail.

**Prediction or Disconfirmer:** On positions where the move is engine-verified as best, a small model with no claim-level verifier will still make at least one false tactical claim in a large share of its explanations, and only adding claim-level verification lifts truthfulness clearly above one half; a tuned model with no verifier reaching that level on its own would refute it.

**How to resolve it:** Hold move grounding constant, then compare explanation truthfulness with and without a claim-level verifier bolted on, counting fabricated claims per output.

**Testing note:** Cold raters found the sharp version, that a verified move does not buy a verified explanation, genuinely non-obvious once the "why" is separated from the "what." An adversary could not make it retreat, because the move-versus-reason split is concrete, and it reaches to radiology, where a report can verify the nodule and still invent a comorbidity in the impression. The core holds as established, from the 1.00 versus 0.00 base run and the measured chess-claim error rates; the narrower remedy that "only a claim verifier fixes it at 1.7B" is well supported but still pending the project's own larger run, so the core leads and the exact remedy is treated as strong-but-pending.

**Spiky POV 3 (Strong): When a small local coach appears to beat a frontier model, what it actually wins is form factor, not dependability. Give a prompted frontier model the same grounding, schema, and rubric, and it will match or beat the small tuned model on truthful level-fit; the small model's real edge is cost, latency, privacy, and offline use.**

**Elaboration:** The field keeps reporting small-model wins without holding grounding constant, which quietly turns an economics result into a capability headline. The honest framing separates two things a coach can win on. One is truthful level-fit, meaning it recommends the instructive move for the rating and explains it correctly and in plain language. The other is form factor, meaning it runs cheaply, privately, offline, and fast. A shipped grounded coach already delivers faithful, level-appropriate explanation using a prompted frontier model plus an engine, which shows the behavior is not exclusive to a fine-tune. So if the frontier model is given the same structured inputs and the same output rubric, the expected result is parity on truthful level-fit, and the small model's remaining advantage is the deployment envelope. That is still a real and valuable win, but it is a different claim from "more dependable," and calling it dependability oversells it.

**Prediction or Disconfirmer:** With identical grounding, schema, and rubric, a prompted frontier model matches or beats the tuned 1.7B on truthful level-fit under repeated sampling; if the tuned 1.7B clearly beats the equally grounded frontier on truthful level-fit, the claim is wrong.

**How to resolve it:** The project's own base-versus-tuned comparison with grounding held constant, run against an equally grounded prompted frontier, scoring truthful level-fit separately from cost and latency.

**Testing note:** Cold raters put the core near the mainstream, since few dispute that grounding helps both, so it earns its edge from the sharp reframing that the field mislabels an economics win as a capability win. An adversary found the crux resistant once grounding is equalized, and it reaches to on-device speech-to-intent, where the win is latency and privacy rather than accuracy. Its decisive resolution waits on the grounding-held-constant run, so it is Strong rather than Validated.

**Spiky POV 4 (Strong): At fixed grounding, the small tuned model's honest capability win is register consistency, not truth. A narrowly tuned 1.7B is the lowest-variance renderer of plain, no-engine-speak coaching voice, and it beats a prompted frontier on staying in that register even where the frontier ties or wins on faithfulness.**

**Elaboration:** There are two separable qualities in the output. One is faithfulness, which the grounding and the verifier carry. The other is register, meaning never leaking centipawns or deep engine lines, holding a steady voice for a given rating, and not drifting in tone across many outputs. Register is exactly the kind of narrow stylistic behavior that fine-tuning compresses well into a small model, and a model tuned on one task has little room to wander. A prompted frontier model, by contrast, is more prone to occasional drift out of the requested register even when it is factually fine. The base run hints at the size of the prize: no-engine-speak sat at 0.11, meaning the untuned small model almost always leaked engine talk, so this is the axis with the most headroom for the fine-tune to own. The claim is deliberately modest and winnable: the fine-tune's defensible edge is variance of style, measured across many samples, not correctness.

**Prediction or Disconfirmer:** Under repeated sampling at deployment temperature, the tuned 1.7B beats an equally grounded prompted frontier on no-engine-speak and register-consistency pass rates, even in cases where the frontier ties or wins on truthfulness; if the grounded frontier matches or beats the tuned model on register consistency under sampling, the claim is wrong.

**How to resolve it:** Sample both systems many times at deployment temperature with grounding held constant, and compare the pass rates and the variance of the register metrics, kept separate from the truthfulness metric.

**Testing note:** Cold raters treated this as a real empirical bet rather than a truism, since it names a specific axis where small could genuinely win. An adversary found it resistant as long as the register metrics are frozen in advance, and it reaches to on-device command-grammar parsing, where a tiny model adheres to a fixed format with lower variance. It needs the repeated-sampling measurement to settle, so it is Strong and explicitly awaiting that number.

**Spiky POV 5 (Strong): The shippable dependability asset is the detector-and-verifier layer, not the dataset or the weights. The fine-tune is the last-mile compressor that makes a proven-grounded behavior run locally; if a prompt plus constrained decoding plus a verifier reach the same faithfulness on untouched weights, the fine-tune is not what carries the thesis.**

**Elaboration:** It is natural to treat the distilled dataset and the fine-tuned checkpoint as the product. The evidence points elsewhere. Dependability here comes from three non-weight parts: the engine for move truth, motif and threat detectors that expose the concrete features of the position, and a non-LLM verifier that checks each explanation claim before it ships. The fine-tune's job is narrower and later: take that already grounded and already verified behavior and compress it into a small model that is cheap, private, offline, and stylistically steady. The base run supports the ordering, because the failures were missing truth checks and missing register control, with truthfulness at 0.00 and no-engine-speak at 0.11, not missing move quality, which was already at 1.00, and those are system gaps rather than weight gaps. The test that matters is an ablation: if adding detectors and a verifier to the untouched base lifts faithfulness a lot, and adding more fine-tuning transcripts without a verifier does not, then the layer is the deliverable and the fine-tune is the finisher.

**Prediction or Disconfirmer:** Adding a detector-and-verifier layer to the untuned weights raises held-out truthfulness to a high level, roughly six in ten or better, while fine-tuning on more transcripts with no verifier leaves truthfulness low; a tuned-but-verifier-less small model reaching high truthfulness on its own would refute it.

**How to resolve it:** An ablation that separately toggles the verifier layer and the fine-tune, with grounding held constant, measuring held-out truthfulness for each combination.

**Testing note:** Cold raters found the claim off-consensus about what the product actually is, which is where it earns its spikiness. An adversary found the crux resistant because it is a clean ablation, and it reaches to a medical scribe, where the shippable asset is the ontology validator, not the phraser. It is Strong because the decisive numbers come from the project's own ablation, which has not been run yet.

**Spiky POV 6 (Strong): This recipe of translating verified truth into leveled language is safe to ship only where the reasoning, not just the answer, has a cheap non-LLM verifier. Chess is deceptive because answer-verifiability, where the engine confirms the move, masquerades as rationale-verifiability, where nothing automatically confirms the explanation.**

**Elaboration:** The recipe looks general: ground a small model in a solver, distill a frontier teacher's explanations, fine-tune, and ship a cheap local coach. Whether it is safe depends on a distinction that is easy to miss. Answer-verifiability means a machine can confirm the final answer, which chess has through the engine. Rationale-verifiability means a machine can confirm the explanation of why, which chess has only partially, through motif and threat detectors, and not for open-ended strategic talk. Domains where both hold, such as code with unit tests, are safe, because a failing explanation is caught automatically. Domains with answer-verifiability but no rationale checker, such as personal-finance advice, are traps, because the model can produce a defensible-looking answer wrapped in an unverifiable and possibly false rationale. Chess sits in between and is dangerous precisely because the strong answer checker creates false confidence about the unverifiable rationale.

**Prediction or Disconfirmer:** Across several domains, small tuned coaches reach high explanation faithfulness, above roughly seven in ten, only where a cheap rationale verifier exists, while answer-only-verifiable domains stay low no matter how much the model is tuned; a domain with no rationale verifier reaching high faithfulness would refute it.

**How to resolve it:** A pre-registered multi-domain study that sorts domains by whether a cheap rationale verifier exists, then measures small-tuned-coach faithfulness in each. This is the one menu item that needs evidence beyond the chess project.

**Testing note:** Cold raters rated this the most durable and genuinely non-obvious of the set, because the answer-versus-rationale distinction is not commonly drawn. An adversary found it resistant if the domains are pre-registered, and it reaches directly to a code explainer, which is safe because tests verify, versus a finance explainer, which is unsafe because nothing verifies. It is Strong rather than Validated only because it awaits that cross-domain study, not because any part is in doubt.

**Spiky POV 7 (Strong): There is one output but two independent axes. Register, meaning no engine-speak and a tier-appropriate voice, is weight-learnable and the fine-tune can own it; faithfulness is not reliably weight-learnable without an external verifier. Scoring a coach with one blended quality number hides this and is a category error.**

**Elaboration:** Because register and faithfulness are carried by different mechanisms, they move independently, and a single combined score lets a gain on one hide a failure on the other. Fine-tuning reliably teaches style: the model can be pushed from almost always leaking engine talk to almost never doing so. Fine-tuning alone does not reliably teach truth, because the training transcripts themselves contain confident wrong claims, and imitating them teaches confident wrongness. The base run shows the two axes at their extremes on the same outputs, with no-engine-speak low but improvable and truthfulness at the floor, which is the clearest possible sign they are not one quantity. The practical rule is to always report register and faithfulness as two numbers, and to expect fine-tuning to move the first far more than the second.

**Prediction or Disconfirmer:** From base to tuned, the no-engine-speak score rises steeply while truthfulness stays near the floor unless a verifier is added; any fine-tune-only lift of truthfulness by a large margin, with no verifier, would refute it.

**How to resolve it:** Measure the base-to-tuned change on register and on truthfulness separately, with no verifier in the loop, and check whether truthfulness moves on its own.

**Testing note:** Cold raters found the two-axes claim off-consensus once the absolute wording was dropped. An adversary noted it can leak if "learnable" is stretched, so the crux is pinned to a no-verifier fine-tune, and it reaches to a support bot that nails brand voice while inventing refund policy. It is Strong pending the base-to-tuned split from the project's own run.

**Spiky POV 8 (Strong): Fine-tuning a small model on raw, unfiltered frontier transcripts risks being worse than base-plus-prompt, because it imitates the teacher's confident-assertion style and can inherit and amplify its fabrications. The fix is faithfulness-filtered data, or faithfulness as a training reward, not more transcripts.**

**Elaboration:** Distillation copies the teacher's manner along with its content, and a frontier teacher narrating chess makes confident claims that are sometimes wrong. A small student trained on those transcripts learns to sound just as sure, including when it is wrong, and known distillation failure modes, such as inheriting and amplifying teacher hallucinations and the tendency of small students to struggle with long reasoning, push this the wrong way rather than the right way. So pouring in more raw transcripts can move the student backward on truthfulness even while it improves style. The remedy is to put the verifier upstream, in the data: keep only transcripts whose every claim passes the engine-and-detector check, or reward faithfulness during training, so the student imitates verified explanation rather than confident guessing.

**Prediction or Disconfirmer:** A small model tuned on raw transcripts has held-out truthfulness no better than base-plus-prompt, while a model tuned only on verifier-passed transcripts beats both; if raw-transcript tuning matches verifier-filtered tuning, the claim is wrong.

**How to resolve it:** Train two fine-tunes that differ only in whether the data was verifier-filtered, and compare held-out truthfulness against the base-plus-prompt baseline.

**Testing note:** Cold raters found the amplification claim off-consensus once softened away from "always net-negative." An adversary noted that "value" can be redefined, so the crux is fixed to held-out truthfulness, and it reaches to distilling a clinician's off-hand wrong guesses into a junior model that guesses just as confidently. It is Strong pending the filtered-versus-raw comparison.

**Spiky POV 9 (Strong): Leveling is largely handled already by grounding plus a simple tier rubric, so faithfulness, not level-calibration, is the hard open axis. Effort spent polishing human-likeness while fabrication goes unfixed is allocated backward.**

**Elaboration:** The surprising base-run reading is that the untuned small model already scored well on level calibration at 0.875 and on specification adherence at 0.875, while scoring zero on truthfulness. That says the level-fit part of the problem is mostly carried by the grounding and a plain rubric that maps rating bands to explanation depth, and it is not where the system is failing. The failing part is truthfulness. So the highest-value work is a claim verifier, not more sophistication in modeling exactly which move a human of a given rating would play. The point is about where the marginal effort should go, not that leveling is finished forever.

**Prediction or Disconfirmer:** Adding more leveling or human-likeness machinery, with no verifier, barely moves the all-constraints-usable rate, while adding a verifier produces the largest single jump; if leveling work beats the verifier on that rate, the claim is wrong.

**How to resolve it:** Compare the all-constraints-pass rate after a leveling-focused change versus after adding a verifier, on the same positions.

**Testing note:** Cold raters found "leveling is largely handled, faithfulness is the hard part" off-consensus once "solved" was removed, though its reach is only moderate. An adversary found it leaky if "essentially solved" is used, so it is backed by the base-run numbers instead. It is Strong pending the head-to-head effort comparison in the project's own evaluation.

**Spiky POV 10 (Strong): Because no complete verifier for chess explanations exists, the safer coach is coverage-bounded: assert only the claims the detectors can verify, and abstain otherwise. Saying less, truthfully, beats saying more, fluently.**

**Elaboration:** Motif and threat detectors cover many but not all of the things a coach might want to say, so there will always be claims the system cannot check. The design choice is what to do about the unverifiable remainder. A coverage-bounded coach speaks only inside the verifiable set and stays silent elsewhere, trading richness for truth in a way that can be measured. This tends to raise truthfulness toward the coverage limit while lowering how much is said, which for a coaching tool is usually the right trade, because a confident false explanation harms a learner more than a shorter true one. The claim is not that silence is always best, only that, given an incomplete verifier, bounding coverage is the safer default.

**Prediction or Disconfirmer:** Constraining output to detector-verified claims raises truthfulness toward the coverage rate while richness drops, and beats "say more" variants on truthfulness; if a "say more" variant matches the bounded coach on truthfulness while keeping higher richness, the claim is wrong.

**How to resolve it:** Compare a coverage-bounded configuration against richer, less-restricted variants on the same positions, measuring truthfulness and a richness proxy together.

**Testing note:** Cold raters found the "say less, truthfully" rule off-consensus once "only honest coach" was softened. An adversary found the crux resolvable by the truthfulness-versus-richness trade, and it reaches to a clinical bot that refuses anything outside the retrieved guideline. It is Strong pending that trade-off measurement.

**Spiky POV 11 (Strong, riskiest): "More dependable" should be defined as the worst-case, all-constraints-at-once pass rate under repeated deployment sampling, not mean quality. A frontier model can have a higher average while failing the full stack of constraints more often; the small tuned model's plausible edge lives on that tail.**

**Elaboration:** Dependability is a tail property. A coaching output is only good if many things hold at once: the move is sound, the explanation is truthful, nothing is fabricated, the level fits, there is a useful next step, and there is no engine-speak. Averaging quality across outputs hides how often the whole stack fails together. The right measurement samples each system many times at the temperature it will actually run at and asks how often every constraint passes at once, then compares the worst cases. The current base measurement is too small and too clean for this, at nine greedy samples, so the variance question is genuinely open. This is the riskiest item on the menu because its affirmative half, that the small tuned model wins on that tail, has no supporting evidence yet; what is solid is the measurement definition itself.

**Prediction or Disconfirmer:** At many samples and deployment temperature, the tuned 1.7B pass-all-constraints rate exceeds an equally grounded prompted frontier's, even if its mean is lower; if the grounded frontier's pass-all rate is at least the tuned model's, the affirmative is falsified, though the metric itself still stands.

**How to resolve it:** Repeated sampling at deployment temperature for both systems with grounding held constant, scoring the fraction of outputs that pass every constraint at once, comparing worst cases rather than averages.

**Testing note:** Cold raters accepted the worst-case framing as a real and non-obvious measurement choice. An adversary found the crux resistant once the metric is fixed, and it reaches to an aviation-checklist phraser graded on zero omissions across eight of eight runs. It is Strong on the definition but flagged the riskiest, because the small-model-wins half is a bet that the project's own sampling run has to settle.

**Spiky POV 12 (Weak, supporting): Human-move modeling is a descriptive learner signal, not the pedagogical objective. It tells you what a player of a given rating would probably play, not what should be taught, and using human-likelihood as the only selector can mislead by drilling likely misconceptions.**

**Elaboration:** The strength of a human-move predictor like Maia is describing behavior: it predicts the move a rated human would probably make about half the time, which is genuinely useful for meeting a student where they are. But most likely is not most instructive. A likely move can be a common misconception, a bad habit, or a stepping stone, and teaching toward it because it is human-likely can entrench the very error a coach should correct, which mirrors the expertise-reversal finding that guidance helpful at one level can harm at another. So the human-move signal belongs in the system as a descriptive input to the level model, clearly labeled as such, with a separate pedagogical decision layer choosing what to actually teach. The strong form of this claim, that the signal is useless or actively harmful, does not survive scrutiny, which is why it sits here as a supporting caution rather than a headline.

**Prediction or Disconfirmer:** Two coaches identical except for the selector, human-likely versus pedagogically chosen, will differ, with the pedagogical selector winning on instructiveness in a learner study or expert proxy; if the human-likely selector ties or wins, the caution is wrong.

**How to resolve it:** A dedicated study comparing the two selectors on learning outcomes, separate from the main faithfulness evaluation.

**Testing note:** Cold raters flagged the strong version as overstated, and checking it against primary sources confirmed the human-move signal is useful but not sufficient rather than harmful, so the claim was softened to a supporting caution. Its reach is narrower than the others and its resolution needs a separate learning study, so it is Weak and kept as support.

The thread that ties these together is where dependability lives. In an engine-grounded coach, truth is supplied by the engine, the detectors, and a claim verifier, and a single verified move never buys a verified explanation. The fine-tuned small model is the last-mile compressor that renders that verified content locally and in a steady voice, so its honest wins are form factor and register consistency, not dependability in general. Measured correctly, as the worst-case rate at which every constraint passes at once, with faithfulness gated by non-LLM checks and any remaining scoring done by a different model family, the answer to the core question is a conditioned yes: a small model can match a grounded prompted frontier on faithful level-fit and win on cost and consistency, while the claim that it is more dependable everywhere remains a bet the project's own comparison must settle.

## Experts

These are the voices worth following, including the ones who disagree with each other. The disagreement is the point.

**Asbjorn Steinskog and Anant Dole**

- Who: builders of the Take Take Take and Play Magnus chess coach.
- Focus: shipping a production coach where the engine is the source of truth and the language model only translates.
- Why follow: they argue from production that an LLM "can't calculate" and should be confined to translating engine and detector output into English, and their shipped coach uses a prompted frontier model plus grounding rather than a fine-tune, which is the strongest real-world signal for the system-not-the-weights view.
- Where: "Building a Chess Coach," AI Engineer, 2026 - [ai.engineer](https://ai.engineer)

**Zhenwei Tang and the CSSLab C1 team**

- Who: authors of C1, a 4B chess model trained on engine-grounded reasoning distilled from a frontier teacher.
- Focus: a small grounded model that reasons about chess and beats its teacher.
- Why follow: C1 reaches about 48.1 percent puzzle accuracy and surpasses its distillation teacher with far fewer tokens, which is the strongest opposing signal to the translate-only stance and shows grounded small models can go further than expected, though at 4B rather than 1.7B.
- Where: [arxiv.org/abs/2603.20510](https://arxiv.org/abs/2603.20510)

**Reid McIlroy-Young and Ashton Anderson**

- Who: creators of Maia, the human-move prediction models, at CSSLab.
- Focus: rating-conditioned modeling of what a human of a given strength would actually play.
- Why follow: Maia predicts human moves about half the time and peaks near its training rating, which makes it a strong descriptive level signal, and the authors' own caveat that per-level models can lack coherence as teaching tools is exactly why the human-move signal must be treated as descriptive, not prescriptive.
- Where: [maiachess.com](https://maiachess.com)

**Nathan Lambert**

- Who: researcher and writer on open models and post-training at Interconnects.
- Focus: the gap between benchmark scores and real deployment robustness.
- Why follow: he warns that open models are "very jagged," easy to overfit on benchmarks, and often not specialized enough, and that closed models tend to be more robust where users keep presenting new challenges, which directly pressures any "small model is more dependable" claim.
- Where: [interconnects.ai](https://interconnects.ai)

**Kevin Lu and Thinking Machines Lab**

- Who: authors of the on-policy distillation work.
- Focus: making small models strong in a trained domain while watching what training costs them elsewhere.
- Why follow: they show small models with strong domain training can outperform larger generalists, and they document that fine-tuning small models on new knowledge causes catastrophic forgetting of instruction-following, which is the mechanism behind treating the fine-tune as a narrow last-mile step.
- Where: [thinkingmachines.ai](https://thinkingmachines.ai)

**Mathieu Acher**

- Who: professor and strong chess player who benchmarks LLM chess play empirically.
- Focus: how well general and reasoning LLMs actually play legal, sound chess.
- Why follow: he shows one older model plays around 1750 Elo yet produces an illegal move in about 16 percent of games, and that reasoning models are illegal most of the time, which guts the assumption that a frontier model is a strong chess reasoner out of the box.
- Where: [blog.mathieuacher.com](https://blog.mathieuacher.com)

**Adam Karvonen**

- Who: researcher on the empirical chess ability of language models.
- Focus: measuring legal-move rates and playing strength across model families.
- Why follow: his work on the one model that plays strong chess, and the finding that chat and instruction tuning degrade a well-defined task, is a caution that fine-tuning can move behavior in the wrong direction if the objective is not held straight.
- Where: [adamkarvonen.github.io](https://adamkarvonen.github.io)

**Simon Willison**

- Who: widely read practitioner writer on applied LLMs.
- Focus: what actually works when building with models.
- Why follow: he found prompt-engineering results on chess more convincing than fine-tuning, and argues that tools combined with reasoning are the most powerful current technique, which is the case for carrying dependability with grounding rather than weights.
- Where: [simonwillison.net](https://simonwillison.net)

**Tim Dettmers**

- Who: author of QLoRA.
- Focus: cheap, low-memory fine-tuning of small and mid-size models.
- Why follow: QLoRA makes fine-tuning a small model nearly free in cost and hardware, which is what makes the last-mile-compressor role practical, while his own caution that chatbot benchmarks are untrustworthy reinforces the need for a non-LLM gate.
- Where: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

**Mrinank Sharma and Ethan Perez**

- Who: authors of the sycophancy study in language models.
- Focus: why models, including model judges, prefer convincing answers over truthful ones.
- Why follow: they show a preference model chose a convincing sycophantic answer over a truthful one a large majority of the time, which is the mechanism behind gating faithfulness before any preference score and never trusting a same-family judge.
- Where: [arxiv.org/abs/2310.13548](https://arxiv.org/abs/2310.13548)

**Lianmin Zheng and colleagues**

- Who: authors of the LLM-as-a-judge evaluation.
- Focus: how well a strong model judge agrees with humans, and where it is biased.
- Why follow: they establish that judges reach high human agreement but carry position, verbosity, and self-enhancement biases, which is why a chess coach's truth must be checked by an engine first and any tone scoring must come from a different family.
- Where: [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

**John Sweller**

- Who: originator of Cognitive Load Theory.
- Focus: minimizing extraneous load so limited working memory can build schemas.
- Why follow: the requirement that a coach never leak engine internals is a direct application of reducing extraneous load, which gives the no-engine-speak register a real learning-science justification rather than a stylistic one.
- Where: [link.springer.com/article/10.1007/s10648-019-09465-5](https://link.springer.com/article/10.1007/s10648-019-09465-5)

## DOK 3: Insights

These are the conclusions that fell out of connecting the sources. Each drew on facts that no single source stated together.

### On where dependability comes from

**Insight 1: A small model can win only if the system turns coaching into constrained faithful translation, not open-ended chess reasoning.** The engine supplies truth about the move, detectors expose the motifs and threats, a human-move signal describes behavior at a rating, and the model renders it at level. As currently built, with no motif detectors and no verifier, the task is under-constrained, which is why the model fabricates. This connects the production coaches that deliberately confine the model to translation, the C1 result that grounded small reasoning is possible but at larger scale and narrow scope, the chess-commentary evidence that fluent output is often wrong, and the base run where move selection was solved while truthfulness and register were not.

**Insight 2: The fine-tune is not the origin of dependability; it is the last-mile compressor whose value must survive an ablation.** Dependability comes from grounding, detectors, and verification, while the fine-tune mainly compresses a desired style into a small local model that is cheaper, private, offline, and steady in register. If constrained decoding plus a prompt plus a verifier reach the same gains on untouched weights, the fine-tune is not carrying the thesis. This rests on the finding that a smaller aligned model was preferred over a much larger one, on prompt-optimization beating fine-tuning on structured reliability, on the distillation failure modes that make raw fine-tuning risky, on the thin evidence at 1.7B specifically, and on the base run.

### On how to measure it

**Insight 3: The key metric is worst-case variance under stacked constraints, not mean coaching quality.** More dependable means fewer bad failures when sound move, truthful explanation, no fabrication, level-fit, a useful next step, and no engine-speak must all hold at once. A frontier model can post a higher mean and still fail the full stack more often, so a fat tail of confident wrongness is the real risk. This is currently unmeasured, at nine greedy samples, and needs many samples at deployment temperature scored on worst-case pass-all, and it connects the chess-commentary error rates, the sycophancy result, and expertise reversal.

**Insight 4: A valid evaluation must gate faithfulness with non-LLM checks before judging pedagogy, because fluent falsehood contaminates holistic scores.** Every claimed motif, threat, and plan should be cross-checked against engine lines and detector output before any holistic score, and then level-fit and pedagogy should be judged by a different model family, since a model that both generates and judges leaks preference toward its own style. Chess is unusually gate-able because the engine and motif detectors form a non-LLM source of truth. This connects the chess-commentary judge that rated false commentary highly, the sycophancy and judge-bias findings, and the base run where readable output still scored zero on truthfulness.

### On what to teach and what is actually missing

**Insight 5: The human-move signal is a descriptive level input, not a prescription for what to teach.** Human-likely is not the same as pedagogically useful, since a likely move can be a misconception, a stepping stone, or a bad habit, so the signal should be marked explicitly descriptive and paired with a separate pedagogical decision layer. This connects the measured human-move accuracy and its volatility across adjacent ratings, the industry move toward picking the most human among strong moves, feedback theory, and expertise reversal, along with the fact that the link from human-move modeling to coaching reliability has not been measured.

**Insight 6: The genuinely underfilled cell is the small, local, fine-tuned form factor for grounded and leveled coaching, not the behavior itself.** A shipped grounded coach already produces faithful, level-appropriate explanation using a prompted frontier model, so the behavior exists. The open bet is compressing that behavior into a small local model without losing faithfulness or pedagogy, which is an economics and deployment bet rather than a "nobody built this" claim. The missing mechanism is an interface where rating-conditioned signals control explanation register while the engine, detectors, and verifier control truth, all rendered locally. This connects the shipped grounded systems, the human-move models, and the local fine-tuning tooling, with cost kept secondary and low-confidence.

## DOK 2: Knowledge Tree

This is the verified evidence behind the stances above. Each entry lists its objective facts, a short plain-language summary, and a link. About 120 sources were reviewed across the full pipeline, and the highest-leverage ones are collected here, grouped by topic. Every load-bearing and off-consensus fact was checked against a primary source, with no fabricated or hallucinated citations.

### A. Distillation and small-model specialization

**Knowledge distillation and step-by-step distillation (Hinton, Vinyals, Dean 2015; Hsieh et al., ACL Findings 2023)**

- Fact: knowledge distillation transfers a large model's soft-target "dark knowledge" to a smaller student.
- Fact: distilling step-by-step let a 770M model beat a few-shot 540B model while using about 80 percent of the data.
- Summary: distillation can move a specific capability into a much smaller model, which is the mechanism the project is betting on.
- Link to source: [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

**Small-model parity and specialization (Qwen3 Technical Report 2025; NVIDIA SLM position, Belcak et al. 2025; Finetuner's Fallacy 2026)**

- Fact: a 1.7B base model reached parity with a 2.5B to 3B base model, though this is a pretraining-parity result, distinct from distillation.
- Fact: a position paper argues small models are sufficient and economical for specialized agentic tasks, and a separate result shows a 1B specialized model beating a 3B standard model on under-represented domains through specialized pretraining.
- Summary: small models can match larger ones on narrow targets, but the strongest results lean on specialized pretraining rather than fine-tuning alone, which keeps the 1.7B affirmative a bet rather than a settled fact.
- Link to source: [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

### B. Prompting versus fine-tuning for reliability

**Alignment and constraint adherence (Ouyang et al. 2022; structured-output reliability 2026)**

- Fact: a 1.3B aligned model's outputs were preferred over a 175B model's, so bigger is not automatically better at following intent.
- Fact: naive prompting reached high task accuracy but zero valid structured output in one study, while prompt-optimization, not fine-tuning, brought a frontier model to about 95 percent valid output.
- Summary: reliability and format adherence are often won by alignment and prompt design rather than by scale, which is why the fine-tune's contribution has to be isolated by ablation.
- Link to source: [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

### C. Distillation failure modes

**Model collapse and small-student limits (Shumailov et al., Nature 2024; Small Model Learnability Gap, ACL Findings 2025; distillation traps 2026)**

- Fact: training on recursively generated data erases the tails of the distribution, and preserving those tails needs real human data.
- Fact: small models, at or below about 3B, learn better from shorter and simpler reasoning chains, and tail noise plus a teacher-student gap can drive overconfident hallucination.
- Summary: distilling a frontier teacher's confident chess narration into a small student risks copying confident wrongness, which is the basis for filtering the data with a verifier.
- Link to source: [nature.com/articles/s41586-024-07566-y](https://www.nature.com/articles/s41586-024-07566-y)

**Adapter forgetting (LoRA intruder dimensions, NeurIPS 2025)**

- Fact: low-rank adapters introduce "intruder dimensions" and forget more of pretraining than full fine-tuning, and still trail full fine-tuning on some measures.
- Summary: cheap fine-tuning has a real cost in retained general ability, which reinforces keeping the fine-tune narrow and late.
- Link to source: [arxiv.org/abs/2410.21228](https://arxiv.org/abs/2410.21228)

### D. Chess engines and human-move modeling

**Human-move prediction (Maia, McIlroy-Young et al., KDD 2020; Maia-2, NeurIPS 2024)**

- Fact: Maia predicts human moves about 46 to 52 percent of the time, against roughly 33 to 41 percent for engine-style predictors, with accuracy peaking near the training rating, and personalization can reach up to about 65 percent.
- Fact: the authors note that per-level models can be volatile and incoherent across adjacent ratings and are limited as teaching tools, and that the human-move ceiling is well below 100 percent.
- Summary: human-move modeling is a strong descriptive level signal but an unreliable teaching selector on its own, which is why it is treated as descriptive.
- Link to source: [maiachess.com](https://maiachess.com)

**Compact rating-conditioned prediction (Maia-3 / Chessformer, ICLR 2026)**

- Fact: a 79M rating-conditioned model reached about 57.1 percent human-move accuracy at under a quarter of the previous state-of-the-art parameter count.
- Summary: human-move prediction is improving and getting cheaper, but it still describes behavior rather than prescribing what to teach.
- Link to source: [arxiv.org/abs/2605.19091](https://arxiv.org/abs/2605.19091)

### E. LLMs playing and explaining chess

**Empirical LLM chess ability (Acher 2024; Karvonen 2024; reasoning-LLM chess 2025)**

- Fact: one older model plays around 1750 Elo with under 0.1 percent illegal moves at the move level but an illegal move in about 16 percent of full games, while reasoning models are illegal in the large majority of cases.
- Fact: chat and instruction tuning were found to degrade performance on the well-defined task of chess.
- Summary: a frontier model is not a dependable chess reasoner by default, which both weakens the well-prompted-frontier arm and warns that tuning can hurt a clean objective.
- Link to source: [arxiv.org/abs/2512.01992](https://arxiv.org/abs/2512.01992)

**Grounded small chess reasoning (C1, CSSLab 2026; faithful reasoning training 2026)**

- Fact: a 4B model trained on engine-grounded reasoning distilled from a frontier teacher, then reinforced, reached about 48.1 percent puzzle accuracy, surpassing its teacher at roughly 40.8 percent, with about 100 times fewer tokens, improving from 42.3 percent after supervised training to 48.3 percent after reinforcement.
- Fact: separate work found best-move supervised training strong but reasoning sometimes unfaithful, while multi-move trajectory training was more faithful.
- Summary: grounded small models can reason well and even beat their teacher, but at 4B and on puzzle accuracy rather than level-calibrated coaching, so the 1.7B coaching case remains to be shown.
- Link to source: [arxiv.org/abs/2603.20510](https://arxiv.org/abs/2603.20510)

**Commentary hallucination and its evaluation (ACT-Eval 2026; CCC and GCC-Eval, Kim et al., NAACL 2025)**

- Fact: a strong frontier model without tools produced factually incorrect chess claims about 22 percent of the time and smaller open models more than 50 percent, and standard reference-based LLM-as-a-judge scoring could not reliably detect these hallucinations, rating a false commentary highly.
- Fact: concept-guided generation that integrates an expert model with the language model produces more accurate commentary, and evaluation is more reliable when expert-model knowledge is folded into the judge.
- Summary: fluent chess commentary is frequently false, and an unaided LLM judge misses it, which grounds both the faithfulness-gate stance and the move-does-not-ground-explanation stance.
- Link to source: [openreview.net/forum?id=nne0ti66KT](https://openreview.net/forum?id=nne0ti66KT)

### F. Grounded coaching products and shipped small tutors

**Engine-as-truth production systems (Play Magnus and Take Take Take; DecodeChess; Chess.com Game Review 2026)**

- Fact: production coaches use the engine as ground truth and detectors for structured concepts, with the language model confined to translating into English, a choice made because independent LLM chess reasoning hallucinates.
- Fact: one major platform's game review picks the most human among strong moves so the feedback feels like a real coach.
- Summary: the grounded, leveled coaching behavior already ships using prompted models plus engines, so the open question is form factor, not the behavior.
- Link to source: [decodechess.com](https://decodechess.com)

**Shipped small fine-tuned tutors (community LoRA tutors 2026)**

- Fact: a LoRA fine-tune of a 4B model on distilled explanations reported high completeness and near-zero hallucination on a small 50-puzzle test set, and a 270M model was fine-tuned for offline move classification and rating prediction.
- Summary: small fine-tuned chess explainers exist and look promising, but the strongest reports are small-sample and measure completeness or classification rather than level-calibrated coaching.
- Link to source: [huggingface.co](https://huggingface.co)

### G. Learning science

**Tutoring effectiveness (VanLehn 2011)**

- Fact: intelligent tutoring systems reached an effect size of about 0.76 against no tutoring, close to human tutoring at about 0.79, so structured computer tutoring can approach human tutoring.
- Summary: a well-designed tutor can be nearly as effective as a human, which sets a real bar for what a dependable coach should achieve.
- Link to source: [doi.org/10.1080/00461520.2011.611369](https://doi.org/10.1080/00461520.2011.611369)

**Cognitive load and expertise reversal (Sweller et al. 2019; expertise-reversal literature)**

- Fact: novel information passes through a limited working memory, so instruction should minimize extraneous load, and guidance that helps novices can harm more advanced learners and must fade with proficiency.
- Fact: deliberate practice explains only about 21 to 26 percent of performance variance, less than once claimed.
- Summary: no-engine-speak is a load-reduction requirement, and mis-calibrated leveling can actively backfire, which is why leveling is a real but bounded part of the problem.
- Link to source: [link.springer.com/article/10.1007/s10648-019-09465-5](https://link.springer.com/article/10.1007/s10648-019-09465-5)

### H. Evaluation: LLM-as-judge and sycophancy

**Judge validity and sycophancy (Zheng et al., NeurIPS 2023; Sharma et al., ICLR 2024)**

- Fact: strong LLM judges reach over 80 percent agreement with humans but carry position, verbosity, and self-enhancement biases.
- Fact: a preference model preferred a convincing sycophantic answer over a truthful one the large majority of the time, and sampling many candidates only partly reduced this.
- Summary: LLM judges are useful for style but unreliable for truth and biased toward their own family, which is why faithfulness is gated by non-LLM checks and scored across families.
- Link to source: [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

### I. The project's own base-model measurement

**Base evaluation of the untuned small model (project measurement)**

- Fact: an untuned 4-bit 1.7B base model, scored over nine greedy samples by a strong frontier judge, reached move soundness 1.00, specification adherence 0.875, and level calibration 0.875, but no-engine-speak 0.11, truthfulness 0.00, and task quality 0.00.
- Fact: the failures were concentrated in fabricated tactical claims and leaked engine talk, not in move selection.
- Summary: on the same outputs, move selection was solved while truthfulness and register failed, which is the strongest single piece of evidence that dependability is a system property and that register and faithfulness are separate axes.
- Link to source: the project's own base-model evaluation run (internal measurement)

### J. Economics and local deployment (secondary, low-confidence)

**Cheap fine-tuning and local inference (QLoRA and on-device runtimes)**

- Fact: quantized low-rank fine-tuning of a small model is inexpensive in cost and hardware, and on-device runtimes keep data local for privacy, though the specific cost and speed figures come from vendors and practitioners and were not independently verified.
- Summary: the form-factor advantages of a small local coach are real in kind, so they are used only as the honest deployment win and never as a load-bearing number.
- Link to source: [unsloth.ai](https://unsloth.ai)
