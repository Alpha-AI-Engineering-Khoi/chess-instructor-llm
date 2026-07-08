---
task: Iteratively train a small (4B) open chess-coach model to maximum RELIABLE quality by improving the DATA, autonomously, until the completion criteria are met.
hero_model: Qwen3-4B-Instruct  # small + locally runnable + on-spec ("tiny local specialist"). 1.7B (v2) and 32B (v3/v4) are REFERENCES only.
max_iterations: 20
loop: /loop wakes the parent on a heartbeat to advance one iteration with a FRESH context; state lives in .ralph/ (gitignored), not in context.
---

## Mission
Make a 4B Qwen3 coach that **reliably** does the behavior — not a bigger model that out-scores frontier. The spec's grade is `tuned > base` on a Behavior Spec; "beat frontier" is a bonus, not the goal. Keep improving the DATA (the lever) and retraining until the model hits the completion criteria. **Do NOT chase capability benchmarks.**

## Behavior Spec (the falsifiable target)
Given a position + student tier (beginner ≈1000–1200 / intermediate ≈1300–1600 / advanced ≈1700–2000), return exactly ONE move + a short explanation that passes every clause: (1) one move; (2) sound (Stockfish cp-loss < 250, never a blunder); (3) **tier-appropriate — and DISTINCT across tiers when the position calls for it** (a beginner and an advanced player should NOT get the same move on a differentiating position; if they do, the model is wrong); (4) explained four ways — purpose, transferable principle, board-specific reason, how-to-find-it-next-time; (5) grounded (0 false board facts post-gate); (6) tier-calibrated voice; (7) zero engine-speak.

## Reward design (anchor on the un-gameable signals)
- **PRIMARY — deterministic, objective (drives training):**
  - tier-fit: pick == canonical tier move (`src/teacher/tier_select.select_tier_move`).
  - **distinct-moves-per-level:** on differentiating positions, beginner ≠ advanced; flag & penalize nonsensical B==A≠I collapses.
  - move-soundness (no blunders), well-formed output, 0 engine-speak, 0 post-gate fabrication.
- **SECONDARY — subjective instructiveness (validation only, NEVER trained toward directly):**
  - blinded cross-family frontier council (TrueFoundry: GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) ranks instructiveness on a **HELD-OUT set the model never trains on** (guards against Goodhart-ing the judges).
- Technique ladder: SFT (data-first) → DPO on council/deterministic preference pairs → optional GRPO on the deterministic reward. All cheap + local on a 4B.

## Completion criteria (the honest "100%" = reliability, not a perfect judge score)
1. Deterministic gates on the held-out eval: **100% move-soundness, 100% no-engine-speak, 100% well-formed, 0% post-gate fabrication.**
2. tier-fit ≥ 60% and **distinct-moves-per-level ≥ 95%** on differentiating positions (≈0 nonsensical B==A collapses).
3. Reliably **beats the untuned Qwen3-4B base** on every axis with identical tools+gate (only training differs).
4. **Beats the best prompt-engineered 4B base** (the spec's litmus) on the deterministic axes.
5. Council instructiveness **plateaus** (no gain over 3 iterations) or reaches parity with the best *prompted* model on the tier axis.

## Compute rotation (cheap → overflow) — SECRETS NEVER IN THIS REPO
Small 4B QLoRA runs are cheap (~$2–5). **kim-lam is BILLING-BLOCKED (overdue invoice) — DO NOT USE IT for anything.** Use in order, rotate when a workspace errors on credits:
1. **chess-instructor-2** ($30 fresh) — start here (no v4 competition).
2. **chess-instructor** (~$28, shared with the running v4 32B — do NOT disturb v4; the 4B run is small enough to coexist).
Modal auth is via named CLI profiles (`--profile <name>`) in `~/.modal.toml` and tokens in the **gitignored** `.env` — never write tokens into RALPH_TASK.md, .ralph/, or any committed file.

## Guardrails (see .ralph/guardrails.md for the living list)
- Do NOT stop or touch v4 (32B) training on chess-instructor — the user wants to see it finish.
- REUSE the hard-test eval harness (base+tuned share identical tools+gate; deterministic + held-out council) — do not rebuild it.
- Datasets live on the Modal volume / Hugging Face, NOT local disk (keep local storage clean).
- Data must be rich, high-quality, and correctness-checked against Stockfish/board facts (per `data/analysis/principle_library_v5.md`) — never parrot wrong commentary heuristics (e.g. "trade when behind" is inverted).
- One iteration per fresh context (Ralph malloc/free). Commit meaningful checkpoints (dataset/config/eval/model refs) with clear messages; keep .ralph/ gitignored to avoid churn.
