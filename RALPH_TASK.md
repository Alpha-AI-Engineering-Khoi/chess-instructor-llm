---
task: Ship v3 — a genuinely BETTER chess-coach fine-tuned from Qwen3-32B on the
  larger contrastive dataset, trained on Modal, converted to 4-bit MLX, then
  re-evaluated on the definitive 803-position benchmark and reflected everywhere.
  Run autonomously as a resumable RALPH loop (state in files, checkpoint per phase,
  learn guardrails from failures). Do NOT disturb the LIVE v2 platform.
completion_criteria:
  - v3 dataset built: teacher (GPT-5.5 via TrueFoundry) labels on the v3_candidates
    contrastive bank, faithfulness-filtered to 0% false labels, split into
    data/dataset/{train_v3,valid_v3}.jsonl (report counts + contrastive coverage).
  - v3 trained: QLoRA fine-tune of Qwen3-32B on Modal (A100-80GB/H100), merged to
    16-bit, downloaded, converted to 4-bit MLX at models/mlx/chess-coach-v3 (runs
    locally). v2 artifacts untouched.
  - v3 evaluated: the 803-position benchmark re-run with OURS->v3, producing an
    apples-to-apples v2->v3 delta on tier-fit (the moat), instructiveness (council),
    move-safety, no-jargon, fabrication. Write RESULTS_V3.md + RESULTS_FULL_EVAL_803_v3.md.
  - Evals updated everywhere: HF Space + benchmark dataset + model card, and
    FINDINGS.md + README.md/SUBMISSION.md + BrainLift (both copies). Code/docs pushed
    to GitHub (secret-safe; gitignore protects data/models).
  - A short honest summary of whether v3 improved the moat/instructiveness and by
    how much, plus total cost.
deadline: this session (resumable — never lose work).
---

## What we're building (v3)
Same behavior spec as v1/v2 (engine-grounded, tier-calibrated coaching that selects
the most INSTRUCTIVE move for the student's tier and explains it in plain human terms
— no engine-speak, no fabrication, one transferable takeaway). The v3 intervention is
TWO things at once, on purpose:
  1. BIGGER BASE: fine-tune Qwen3-32B (the best locally-runnable base per
     RESULTS_FULL_EVAL_803.md) instead of Qwen3-1.7B. The 803 eval showed OURS-v2
     (1.7B) already LEADS the moat (tier-fit 53%) but is weak on instructiveness
     (council rank ~9.4/14) and high on fabrication (30%, neutralized at serve time
     by the verifier). A 20x-larger base should keep the moat while gaining coaching
     capacity + faithfulness.
  2. LARGER + CONTRASTIVE DATA: build the v3 SFT set from data/positions/
     v3_candidates.jsonl (2,423 curated contrastive multi-tier positions, motif-tagged,
     Stockfish sound pools) — verified ZERO overlap with the 803 eval and with
     train_v2/valid_v2. Deterministic tier-aware move selection (src.teacher.tier_select)
     + grounded, method-teaching GPT-5.5 labels + faithfulness filter.

## Locked design decisions
- TRAINING -> Modal ONLY. TrueFoundry is an inference gateway (cannot train). 32B QLoRA
  needs a bigger GPU than v2's A10G -> A100-80GB (or H100). Recipe = v2's Unsloth QLoRA,
  scaled (4-bit base load, LoRA on attn+MLP, grad-accum for memory, longer timeout).
- TEACHER (dataset labels) + EVAL judges/frontier/open models -> TrueFoundry gateway
  (openai-group/gpt-5.5 for teacher; council = GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro).
- Teacher = GPT-5.5 via TFY (chat.completions + response_format json_object + reasoning
  high — PROVEN working this session). base=Qwen3-32B, tuned=our v3 output.
- LOCAL inference = 4-bit MLX (mlx_lm.convert of the merged 16-bit). The Modal PEFT/bnb
  adapter is NOT MLX-loadable — must merge->download->convert.
- "unpaid invoice" AND "PING timed out"/unavailable are TRANSIENT -> retry/resume, never stop.

## Guardrails (do not violate)
- Do NOT disrupt the LIVE v2 platform (ports 8000/3000). v3 uses its own path/files.
  Do NOT auto-switch the platform to v3 — leave that to the user.
- Everything v3-suffixed; never overwrite v1/v2 artifacts, datasets, or models.
- Secrets are env-only (.env). Never print/commit them. gitignore protects data/models.
- Cost-aware but substantial spend approved ("make it really good"). Report total cost.
- If approaching a time/resource limit: WRITE partial state + a clear "done vs remaining"
  note in .ralph/progress.md and return so it can be resumed.

## Phases (sequential; checkpoint + update .ralph after each)
1. Dataset: plan from v3_candidates (Maia per-tier picks) -> generate GPT-5.5 labels
   (resumable, costed) -> filter --faithfulness reject --target-format v2
   --dedup-key fen_tier_move -> split -> train_v3/valid_v3.
2. Train: adapt train_modal_v2 -> train_modal_v3 (Qwen3-32B, A100-80GB). Smoke -> full ->
   merge 16-bit -> download -> mlx_lm.convert -q 4 -> models/mlx/chess-coach-v3.
3. Eval: gap803 harness with OURS->v3 (BENCH_OURS_MODEL) -> objective + safety + council
   -> v2->v3 delta -> RESULTS_V3.md + RESULTS_FULL_EVAL_803_v3.md.
4. Update everywhere: HF Space + dataset + model card; FINDINGS/README/SUBMISSION;
   BrainLift (both copies). git push (secret-safe).

## Known-good baseline (from RESULTS_FULL_EVAL_803.md, the moat = tier-fit)
- OURS-v2 (Qwen3-1.7B tuned): tier-fit 53% (field-leading), tier-diff 44%, direction 54%,
  instructiveness council rank 9.36/14 (top1 9%), safety 99%, no-jargon 100%, fab 30%.
- Qwen3-32B BASE (open, untuned): tier-fit 37%, instr rank 8.58/14, fab 6%, local yes.
- Honest framing: moat = tier-appropriate move selection; faithfulness = a verifier at
  serve time; do NOT overclaim beating frontier on raw coaching.
