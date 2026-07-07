# BrainLift intake — chess-coach behavior thesis

## Core question
Can a small (~1.7B) open model, fine-tuned on engine-grounded distilled data,
deliver **reliably level-calibrated chess coaching** — recommending the most
*instructive* move for a student's rating and explaining it in plain human terms
**without leaking engine internals** (centipawns, deep lines) — **more dependably
than a well-prompted frontier model**?

## Context
Project fine-tunes Qwen3-1.7B on data distilled from a frontier teacher (GPT-5.5)
grounded in Stockfish (soundness) + Maia (human-likelihood at a rating). The spiky
claim to stress-test: for one narrow behavior, a small fine-tuned model can match
the teacher's *quality* while exceeding a prompted model's *reliability* (no drift,
no engine-speak, consistent leveling). Base-model eval already shows a prompted
Qwen3-1.7B fails badly (11% clean of engine-speak; fabricates tactics).

## First-principles map (foundations -> adjacent -> specific)
1. **Knowledge distillation / SFT / LoRA-QLoRA** — teacher->student transfer, what
   transfers, capability ceilings, data-centric AI.
2. **Prompting vs fine-tuning for reliability** — instruction drift, format/constraint
   adherence, sycophancy, hallucination, consistency across inputs.
3. **Small-model specialization vs frontier generality** — scaling laws, narrow-task
   parity, when a small tuned model matches/loses to a big prompted one.
4. **Learning science** — Zone of Proximal Development, scaffolding, cognitive load,
   calibrated/formative feedback, deliberate practice, expertise development.
5. **Chess pedagogy** — how players improve by rating, engine use in coaching,
   "best move" vs "instructive move", human-like play (Maia).
6. **Tool-augmented / neurosymbolic LLMs** — engine-in-the-loop grounding, offloading
   capability to a solver, faithfulness to provided evidence.
7. **Evaluation methodology** — LLM-as-judge (+ its sycophancy failure modes),
   behavioral/reliability metrics, base-vs-tuned deltas.
8. **Economics / deployment** — local cheap/private/low-latency inference vs frontier
   API cost, and where that advantage is decisive.
9. **Distillation dynamics (the spiky core)** — student capability <= teacher, yet
   filtered distillation can make the student *more consistent* than the prompted
   teacher on the narrow target.
