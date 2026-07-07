# 03 — Candidate Spiky POVs (Step 4)

Wide pool for the Testing Protocol. Each overlaps >=2 validated insights into one
prescriptive stance, sharpened against a named expert. Predictions anchored where
possible to the project's own base-vs-tuned eval.
**Anchor (base Qwen3-1.7B-4bit, n=9 greedy, judge gpt-5.5-pro):** move_sound 1.00,
no_engine_speak 0.11 (judge 0.25), truthfulness 0.00, task_quality 0.00,
spec_adherence 0.875, level_calibration 0.875.

---

### SPOV 1 — The system, not the fine-tune, is the deliverable
**Assertion:** In an engine-grounded chess coach, the dataset and fine-tune are NOT the deliverable — the detector + non-LLM verifier layer is; the fine-tune only compresses a proven-grounded behavior to run local.
**Core question:** What produces "dependable coaching" — the weights or the surrounding system?
**Disconfirmer:** Add detector + non-LLM verifier, leave weights untouched → held-out truthfulness rises to >=0.6; fine-tune more transcripts with no verifier → truthfulness stays <=0.25. A tuned-but-verifier-less 1.7B reaching >=0.6 truthfulness refutes it.
**Reach:** Medical scribe — the shippable asset is the ontology validator, not the phraser.
**Expert:** extends Steinskog/Play Magnus; disagrees with "distillation-via-FT is the product." Insights 1,2,6.

### SPOV 2 — Dependability = worst-case variance, not mean
**Assertion:** "More dependable" means worst-case under stacked constraints, not mean quality; a small FT model can win only on that tail (pass-ALL-constraints@k at deployment temperature).
**Core question:** By what number do we decide "more dependable"?
**Disconfirmer:** At k>=8/T>=0.7, tuned 1.7B pass-all@k exceeds equally-grounded prompted frontier even with lower mean. If grounded frontier's pass-all@k >= tuned 1.7B's, falsified.
**Reach:** Aviation checklist phraser graded "zero omissions, 8/8 runs." Insights 3,2.

### SPOV 3 — LLM judges can't score chess faithfulness
**Assertion:** You cannot evaluate a chess coach's faithfulness with an LLM judge at all — truth must be gated by the engine before any preference score, and same-family judging inflates the tune.
**Core question:** Is LLM-as-judge valid for the truthfulness dimension?
**Disconfirmer:** LLM judge passes as "truthful" >=30% of outputs the engine gate rejects; a different-family judge scores gpt-distilled outputs lower than a gpt-family judge. If engine-gate and LLM-judge agree within noise, wrong.
**Reach:** Legal-brief assistant graded by a sibling LLM blesses fabricated cites. Insights 6,3.

### SPOV 4 — Grounding the move != grounding the explanation
**Assertion:** Grounding the move does not ground the explanation — a small model's fabrication of the "why" is intrinsic; only a non-LLM verifier vetoing individual claims removes it.
**Core question:** If the move is engine-verified, why does the coaching still lie?
**Disconfirmer:** On 100%-sound-move held-out positions, 1.7B (base or tuned, no rationale-verifier) fabricates >=1 tactical claim in >=50% of outputs; only claim-level veto lifts truthfulness >0.5. Tuned-no-verifier >0.5 refutes it.
**Reach:** Radiology report verifies the nodule but invents a comorbidity in the impression. Insights 1,6.

### SPOV 5 — Maia is descriptive, not prescriptive (and using it as pedagogy harms)
**Assertion:** Maia tells you what a player would PLAY, never what to TEACH — using human-likelihood as the pedagogical selector actively harms learners by drilling likely misconceptions.
**Core question:** Should Maia drive "which move to teach"?
**Disconfirmer:** Two coaches, identical but for the selector (Maia-likely vs pedagogically-selected) → pedagogical selector wins on instructiveness (learner study/expert proxy). If Maia-likely ties-or-wins, wrong.
**Reach:** Language tutor drilling the most-likely-next error entrenches it. Insight 4. (Needs its own study — external Test 3.)

### SPOV 6 — The honest win is form factor, not dependability
**Assertion:** The honest win of a small local chess coach is FORM FACTOR (offline/cheap/private/low-latency) at PARITY faithfulness — not "more dependable than a frontier model"; the field mislabels an economics win as a capability win.
**Core question:** When the small model "wins," what did it win?
**Disconfirmer:** Give a prompted frontier the SAME schema + rubric → it matches/beats the tuned 1.7B on truthful level-fit; the 1.7B's only edge is $/latency/offline/privacy. If the tuned 1.7B beats the equally-grounded frontier on truthful level-fit (surviving k-sampling), wrong.
**Reach:** On-device speech-to-intent — the win is latency/privacy, not accuracy. Insights 5,2.

### SPOV 7 — Two axes: register is weight-learnable, faithfulness is not
**Assertion:** There is one output but two independent axes — register (no-engine-speak, tier voice) is weight-learnable and the FT can own it; faithfulness is structurally not weight-learnable — a single "coach quality" score is a category error.
**Core question:** What can fine-tuning change, and what can it never change?
**Disconfirmer:** Base→tuned delta large on no_engine_speak (0.11→>=0.8) but ~0 on truthfulness (<=0.1) without a verifier. Any FT-alone (no verifier) lift of truthfulness by >=0.3 refutes it.
**Reach:** Support bot nails brand voice but invents refund policy. Insights 1,2.

### SPOV 8 — The verifier belongs in the training data, or the FT is negative value
**Assertion:** Fine-tuning a 1.7B on raw frontier transcripts is NEGATIVE value (imitates confident-assertion style, inherits/amplifies fabrication); the fix is faithfulness-as-reward / data filtration, not more transcripts.
**Core question:** Where must the verifier sit for the FT to help not hurt?
**Disconfirmer:** 1.7B FT on RAW transcripts has held-out truthfulness <= base+prompt; 1.7B FT on ONLY verifier-passed transcripts beats both. If raw-transcript FT >= verifier-filtered FT, wrong.
**Reach:** Distilling a clinician's off-hand wrong guesses into a junior model that guesses just as confidently. Insights 2,6,1.

### SPOV 9 — Leveling is ~solved; faithfulness is the sole open axis
**Assertion:** Level-calibration is already essentially solved by grounding + a tier rubric (untuned base scores it high), so faithfulness is the SOLE open axis, and the field allocates effort backwards by polishing Maia-leveling while fabrication goes unfixed.
**Core question:** Which sub-problem is actually hard?
**Disconfirmer:** Further leveling/Maia work (no verifier) raises all-constraints usable rate <0.1; adding a verifier produces the largest jump. If leveling work beats the verifier, wrong.
**Reach:** Tax assistant with fine tone but wrong numbers. Insights 3,6,4.

### SPOV 10 — The only honest coach is coverage-bounded ("say less, truthfully")
**Assertion:** Because no complete verifier for chess EXPLANATIONS exists, the only honest coach is coverage-bounded — assert only detector-verifiable claims, abstain otherwise; "say less, truthfully" beats "say more, fluently."
**Core question:** Given an incomplete verifier, what may the coach say?
**Disconfirmer:** Constrain to detector-verified claims → truthfulness rises to ~coverage while richness drops (quantifiable tradeoff), beating "say more" variants on truthfulness. If a "say more" variant matches it on truthfulness with higher richness, wrong.
**Reach:** Clinical bot that refuses anything not in the retrieved guideline. Insights 6,1,4.

### SPOV 11 — The recipe generalizes only where the RATIONALE is verifiable
**Assertion:** A "translate verified truth into leveled language" fine-tune is safe to ship only where the RATIONALE (not just the answer) has a cheap non-LLM verifier; chess is deceptive because answer-verifiability (Stockfish) masquerades as rationale-verifiability.
**Core question:** Where does this SLM-coach recipe generalize, and where is it a trap?
**Disconfirmer:** Across >=3 domains, small-FT coaches reach faithfulness >0.7 iff a cheap rationale-verifier exists; answer-only-verifier domains stay <=0.4 regardless of FT scale. A no-rationale-verifier domain hitting >0.7 refutes it.
**Reach:** Code explainer (unit tests verify → safe) vs personal-finance explainer (no verifier → fabricates). Insights 1,6. (Needs multi-domain external Test 3.)

### SPOV 12 — At fixed grounding, small wins on register variance
**Assertion:** At FIXED grounding, a narrowly-tuned 1.7B is the lowest-register-variance renderer and beats a prompted frontier on no-engine-speak and stylistic consistency (not faithfulness) — the FT's honest, winnable capability claim.
**Core question:** Is "going small" only an economics concession, or a real reliability advantage somewhere?
**Disconfirmer:** At k>=8/T>=0.7, tuned 1.7B no-engine-speak + register-consistency pass rates beat the equally-grounded prompted frontier's, even where the frontier ties/wins on truthfulness. If the grounded frontier matches/beats on register-consistency under sampling, wrong.
**Reach:** On-device wake-word/command-grammar parsing — tiny model adheres with lower variance. Insights 2,3,5.

---
**Shared project-native disconfirmer** (1/4/6/7/8/12): the base-vs-tuned eval with grounding held constant. SPOVs 5 & 11 need external studies. The ~1.7B affirmative is a candidate bet; variance claims (2,12) need k-sampling at deployment temperature; no SPOV leans on unverified economics.
