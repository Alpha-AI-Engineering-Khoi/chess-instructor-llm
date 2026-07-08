"""HONEST base-vs-tuned evaluation: isolate *training* as the only variable.

The 803-gap eval (:mod:`scripts.gap803_*`) grounds every model with the same
Stockfish pool + Maia + verified facts, but its generation phase does NOT run the
shipped **faithfulness gate** (verify-and-regenerate + verified fallback) — it
measures fabrication as an objective, then treats the gate as a fairness floor.

This package closes that gap. It runs the base and the tuned model through the
*byte-identical* shipped pipeline — same grounding AND the same gate
(:func:`src.teacher.coach_gate.run_gate`, the exact code in
:mod:`src.api.server`) — so the ONLY difference between them is the model
weights. On top of that it adds:

* the **hard test** — a prompt-engineered base ("train by prompting"): an
  automated loop that iterates a base *system prompt* to maximise the eval score,
  the litmus for "a well-prompted base can't already do this reliably"
  (:mod:`src.eval.honest.promptopt`);
* a **6-dimension instructiveness rubric** graded by the blinded cross-family
  council, and a **tier-coherence** check that flags positions whose recommended
  moves across tiers are incoherent (:mod:`src.eval.honest.rubric`).

Nothing here touches the running platform or the Modal workspace; it does all its
own generation (local MLX for the 1.7B, TrueFoundry for the 32B + frontier +
judges) against the pre-computed gap803 scenarios.
"""
