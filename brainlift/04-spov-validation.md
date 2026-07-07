# 04 — SPOV Validation (Testing Protocol)

## Protocol self-check (decoy)
Decoy **D** ("a chess coach should adapt to the student's level") was run cold
alongside the 12. Both testers rated it agreement 9 / MAINSTREAM = correctly caught
as a truism. The protocol is detecting truisms, not rhetoric.

## Test 1 — Cold spikiness (2 independent testers, SPOV sentence only, no evidence)
Pass = off-consensus/disputed (NOT "true"). Both testers converged.

| SPOV | Verdict | Note |
|---|---|---|
| 1 system-not-fine-tune is the deliverable | PASS (clean) | off-consensus on "what the product is" |
| 2 dependability = worst-case variance | PASS | sharpen "can ONLY win on the tail" (overreach) |
| 3 no LLM judge for faithfulness "at all" | PASS (via contrarian half) | absolute "at all" — soften/back in Test 2 |
| 4 grounding move != grounding explanation | PASS (contrarian half) | lead with "the why needs a non-LLM checker" |
| 5 Maia "tells you nothing / actively harms" | PASS but LIKELY WRONG | overstated; rescue by softening to "not sufficient / can mislead" |
| 6 honest win is form-factor, not dependability | BORDERLINE (near-mainstream core) | lead with "field mislabels economics as capability" |
| 7 register weight-learnable, faithfulness "structurally cannot" | PASS | absolute "structurally cannot" — soften/back |
| 8 raw-transcript distillation is "net-negative" | PASS (contrarian half) | foreground amplifies-teacher-fabrication |
| 9 leveling "basically solved", faithfulness sole axis | PASS | "solved" overstated; back with base-eval numbers |
| 10 only honest coach is coverage-bounded ("say less") | PASS | absolute "only" — soften |
| 11 recipe safe only where RATIONALE is verifiable | PASS (strongest, durable) | answer- vs reasoning-verifiability; genuinely non-obvious |
| 12 small = lowest register-variance renderer | PASS (empirical bet) | needs k-sample measurement, not rewording |

**Carry-forward:** durable/insight-spiky = 1, 2, 9, 11 (+6's reframe). Overstated-absolute
(fix wording or back hard, else fail defensibility) = 3, 5, 7, 8, 10. Empirical bet needing
measurement = 12. #5 flagged likely-indefensible as stated.

## Test 2 — Defensibility (crux hunt + depth gate), agent-to-agent red-teamer
Each SPOV: crux (one checkable sentence) + retreat-resistance + reach.

| SPOV | Verdict | Crux (falsifiable) | Retreat | Reach |
|---|---|---|---|---|
| 1 | PASS | grounding-constant, detector+verifier on untuned weights -> truthfulness >=0.6; tuned-no-verifier <0.6 | resistant | yes |
| 2 | PASS-IF-SOFTENED | tuned 1.7B pass-all@k > equally-grounded frontier at deployment sampling | resistant if metric fixed | yes |
| 3 | PASS-IF-SOFTENED | unaided/same-family LLM judge passes engine-rejected explanations as truthful at higher rate than the non-LLM gate | leaky ("LLM judge" redefinable) | yes |
| 4 | PASS | on 100%-sound positions, no-verifier 1.7B has >=1 false claim in >=50%; only claim-verification lifts truthfulness >0.5 | resistant | yes |
| 5 | FAIL as stated | (Maia-likely selector worse learning than pedagogical selector) — "actively harms/nothing" too strong | leaky | shallow |
| 6 | PASS | identical grounding+schema+rubric: prompted frontier matches/beats tuned 1.7B on truthful level-fit; 1.7B edge is cost/latency/privacy | resistant | yes |
| 7 | PASS-IF-SOFTENED | FT-alone (no verifier) improves truthfulness >=0.3 -> collapses | leaky ("not perfectly learnable") | yes |
| 8 | PASS-IF-SOFTENED | raw-transcript FT truthfulness <= base+prompt; verifier-filtered FT beats both | leaky ("value" moves) | yes |
| 9 | PASS-IF-SOFTENED | verifier work > leveling work on pass-all gain in the project eval | leaky ("essentially solved") | weak |
| 10 | PASS-IF-SOFTENED | a "say more" variant matches detector-bounded truthfulness with higher richness -> collapses | leaky | yes |
| 11 | PASS | across domains, small-FT faithfulness >0.7 iff cheap rationale-verifier exists; answer-only <=0.4 | resistant if pre-registered | yes |
| 12 | PASS | tuned 1.7B beats equally-grounded frontier on register-consistency under k-sampling | resistant if frozen | yes |

**Strongest final-menu = #6, #12, #1, #11.** #5 -> softened supporting only. #2/#3/#7/#8/#9/#10 kept as supporting after de-absolutizing.

## Test 3 — Crux verification against primary sources
- **VALIDATED (HOLDS-ESTABLISHED + still off-consensus):**
  - **#3** — an unaided/same-family LLM judge over-passes chess faithfulness. Settled by ACT-Eval (vanilla judge 4.9/5 on 66.7%-false commentary; "judges insensitive to factual errors") + the base run (judge truthfulness 0.00).
  - **#4-core** — grounding the move does not ground the explanation. Settled by ACT-Eval (frontier 22% wrong, OSS >50%) + the base run (move_sound 1.00, truthfulness 0.00 with concrete fabrications). (The 1.7B-specific "only claim-verification fixes it" remedy stays Strong/pending, n=9.)
- **STRONG (HOLDS-PENDING — the project's grounding-held-constant base-vs-tuned eval is the disconfirmer):** #1, #6, #7, #8, #9, #10, #12; **#11** (needs an external multi-domain study); **#2** (riskiest — its affirmative "1.7B beats frontier on pass-all@k" has no supporting primary evidence yet).
- **DROPPED:** #5 strong form REFUTED (Maia is useful-but-not-sufficient, not "actively harms/nothing"); softened "can mislead as sole selector" retained as Weak/supporting.

## Final tiered menu (for the editor)
- **Validated:** #3, #4.
- **Strong (lead with):** #6, #12, #1, #11 — then #7, #8, #9, #10, #2.
- **Weak/supporting:** #5 (softened).
No hallucinated/miscited sources across all three tests. Decoy caught. Protocol sound.
