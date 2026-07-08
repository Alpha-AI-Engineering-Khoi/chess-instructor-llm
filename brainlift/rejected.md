# Rejected — chess-coach-behavior-thesis

Everything cut at any gate, with the reason. (Shipping doc is `brainlift.md`.)

## DOK 1 — facts cut by the citation gate
(none yet)

## DOK 2 — summary/Knowledge-Tree claims dropped in synthesis
(none yet)

## DOK 3 — insights cut in critique
- Insight 2 "C1-vs-Play-Magnus resolves as benchmark-vs-deployment" — REJECTED (debate: skeptic-validator + MAS): (a) truism — "benchmarks != deployment" draws nods, not dispute; (b) its "same-lab contradiction" premise is factually false (C1 = CSSLab; Play Magnus/Take Take Take is a separate product team that merely USES CSSLab's Maia); (c) the invoked deployment exemplar ships a PROMPTED frontier translator, so it argues against fine-tune-necessity rather than for the thesis.

## DOK 4 — SPOV candidates cut by the Testing Protocol
- SPOV 5 (strong form) "Maia tells you NOTHING about what to teach and using human-likelihood as the selector ACTIVELY HARMS learners" — REJECTED (Test 2 leaky + Test 3 crux REFUTED): primary evidence shows human-likelihood models (Maia/Maia-2) are useful-but-not-sufficient teaching signals (surface likely mistakes, meet the learner at level); the "nothing / actively harms" absolute is unsupported. The softened claim ("Maia should be a descriptive learner-model feature, not the sole pedagogical objective; optimizing directly for likely moves can reinforce misconceptions") is retained as a Weak/supporting claim.
- Decoy (protocol self-check) "A chess coach should adapt its explanations to the student's skill level" — correctly CAUGHT as a truism by Test 1 (both cold testers: agreement 9, mainstream). Confirms the protocol tracks spikiness, not rhetoric.

## Superseded / complicated by new evidence (v2 + benchmark + open-model + verifier + gap, 2026-07-07)
_These are not menu deletions (the refined stances stay on the menu); logged so the change is traceable._

- SPOV 9 strong reading "faithfulness, not level-calibration, is THE hard/sole open axis" — SUPERSEDED by the project's own new evidence: faithfulness is now table-stakes (verify-and-regenerate gate → 0% user-visible fabrication for EVERY model incl. frontier; a 27B open base reaches 1–8% grounded for free). The hard/defensible axis is now tier-appropriate MOVE SELECTION (frontier 22.7% vs ours 39.2%; 67% of positions discriminate). SPOV 9 retained in refined form ("leveling is two axes; faithfulness table-stakes; move-selection is the hard axis").
- Hypothesis "richer/structured board-state grounding at inference lowers small-model fabrication" — REFUTED by the project's own rich-grounding A/B: OURS-v2 fabrication rose 40%→56% (+16 pts; clean stratum 24%→65%) because the new layout is off-distribution for the fine-tune, while the frontier is format-agnostic (0%→7%). Lever is the verifier, not the prompt; if you want structured grounding, TRAIN it in.
- Strong reading of SPOV 8 remedy "a verifier-filtered fine-tune beats base-plus-prompt on faithfulness" — COMPLICATED: filtered data beat RAW data (v1→v2 grounded fabrication 50%→33%), but even the filtered fine-tune still fabricates MORE than the untuned base on grounded input (33–38% vs 13–15%), because the fine-tune learned a more assertive/concrete voice (bigger fabrication surface). So "filtered beats raw" holds; "filtered-FT beats base+grounding" does NOT at 1.7B — the decisive fix is capacity + the inference verifier. SPOV 8 retained, refined.
- Implicit claim "our data intervention closes the small-model faithfulness gap" — COMPLICATED/REFRAMED: the v2 data rebuild only moved grounded fabrication 50%→33% (still far above frontier ~3% and above the untuned base), while a 27B open model reaches 1–8% for free → the deficit is a CAPACITY artifact, not a data-quality gap. Reframed as new candidate SPOV 14.
