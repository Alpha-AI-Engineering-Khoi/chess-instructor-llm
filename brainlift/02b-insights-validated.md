# 02b - Insights Validated

Post-critique revision after four isolated critics (cogsci, MAS-engineer,
skeptic-validator, red-teamer) converged on one correction. The honest thesis is:

**Reliability is carried by Stockfish grounding, tactical/position-feature
detectors, and a non-LLM faithfulness verifier; the fine-tuned ~1.7B model is the
last-mile compressor that may make the system local, cheap, private, and
stylistically stable.**

Receipt from the project's own base run: **100% move-sound** (grounding carries
selection) but **truthfulness 0.0** and **no-engine-speak 11%** — i.e. the failures
are missing detectors + faithfulness checks + register control, not primarily weights.

---

## Update — v2 + benchmark + open-model + verifier + gap evidence (2026-07-07)

New project measurements sharpen the thesis. The reframe now is:

**Faithfulness has become table-stakes (a non-LLM verify-and-regenerate gate zeroes
user-visible fabrication for EVERY model, frontier included; and a 27B open base
reaches ~1–8% grounded fabrication for free). Explanation instructiveness is the
frontier's to win (it still out-coaches everyone even at equal grounding). So the
one axis that is both a real, dense gap AND deterministically gradeable — tier-appropriate
MOVE SELECTION — is where the only defensible moat can live.**

Receipts (all internal measurements):
- **Verifier gate:** user-visible fabrication OURS 40%→**0%**; gpt-5.5 7%→**0%**.
  Fallback rate OURS ~10%, frontier ~7%. No raw model self-corrects to 0% (even
  frontier fabrications are "sticky" — repeated across retries), so the deterministic
  fallback, not regeneration, closes the residual.
- **Capacity closes faithfulness for free:** every 27B+ open model 1–8% grounded
  fabrication (Gemma-3-27B **1%**, best) vs OURS-v2 **33–38%**, BASE 13–15%. Our own
  data intervention only moved 50%→33%; **size** is the lever, not our data.
- **Frontier still out-teaches:** grounded council ours 3.68 vs frontier avg ~2.21
  (5-model); on the 10-model field best open ~5.2 vs GPT-5.5 **2.53** vs OURS 7.95.
- **Self-preference (our own same-family-judge measurement):** +0.43 mean signed
  (0.44 magnitude); GPT +0.66, Gemini +0.64, Claude −0.01. Directly instantiates
  Insight 6 / SPOV-faithfulness-gate.
- **Move-selection gap is DENSE + frontier-weak:** 67% of decidable held-out
  positions discriminate (tier-move ≠ engine #1); frontier tier-differentiation
  22.7%, engine-mirror-at-every-tier 68.7%. v2 lifted ours 27.5%→**39.2%** and
  **corrected the direction** (beginner Maia-match 39%→62%), but not yet won (target >50%).
- **Rich grounding backfires for the small FT** (40%→56%, off-distribution); frontier
  format-agnostic (0%→7%). Lever is the verifier, not the prompt.

**v3 direction implied:** bigger open base (Gemma-3-27B, local 4-bit on 64GB Mac) +
contrastive multi-tier data (v1/v2 had 0%/348 contrastive FENs) + a
deterministic-reward training loop for tier-appropriate move selection.

**In-flight (fold in later):** a definitive 803-position held-out leaderboard and an
all-14-model verifier sweep.

## Validated DOK 3 insights

**Insight 1 — A small model can only win if the system turns coaching into constrained faithful translation, not open-ended chess reasoning.** Stockfish gives truth, detectors expose motifs/threats, Maia describes human behavior, and the model renders it at level. As-built (no motif detectors, no verifier) the task is UNDER-constrained, which is why it fabricates. Supported by: product convergence (Play Magnus/DecodeChess/chess-coach-mcp); C1 4B (grounded small reasoning possible, but larger + narrow); ACT-Eval/CCC (fluent-but-wrong); base run (move-sound solved, truthfulness/register not). **Status: candidate BET.**

**Insight 2 — The fine-tune is not the origin of dependability; it is the last-mile compressor whose value must survive ablation.** Dependability comes from grounding+detectors+verification; FT mainly compresses a desired style into a small local model (fewer tokens, steady no-engine-speak, consistent register, cheap/private/offline). If constrained decoding + prompt + verifier get the same gains, FT isn't carrying the thesis. Supported by: InstructGPT; prompt-optimization; distillation traps/collapse/learnability-gap/LoRA-forgetting; the 1.7B gap; base run. **Status: established connection + candidate BET.**

**Insight 3 — The key metric is worst-case variance under stacked constraints, not mean coaching quality.** "More dependable" = fewer bad failures when sound-move + truthful-explanation + no-fabrication + level-fit + useful-next-step + no-engine-speak must all hold at once. A frontier model may have a higher mean yet a fatter tail of confident wrongness. Currently unmeasured (greedy n=9) — needs k-samples at deployment temperature, worst-case not mean. Supported by: ACT-Eval 22%/judge 4.9-5; sycophancy; expertise-reversal; citation gaps. **Status: candidate BET.**

**Insight 4 — Maia is a descriptive level signal, not a prescription for what to teach.** Human-likely != pedagogically useful (a likely move may be a misconception, a stepping stone, or a bad habit). Mark Maia explicitly descriptive; require a separate pedagogical decision layer. Supported by: Maia 46-52%, volatility/ceiling; Chess.com Torch Human; Hattie; expertise-reversal; the unmeasured Maia->reliability link. **Status: established connection + candidate BET.**

**Insight 5 — The genuinely underfilled cell is the small/local/fine-tuned FORM FACTOR for grounded+leveled coaching, not the behavior itself.** Play Magnus already ships grounded, Maia-leveled explanation via a PROMPTED frontier — so the behavior exists. The bet is compressing it into a small local model without losing faithfulness/pedagogy. It's an economics/deployment bet, not a "nobody built the behavior" claim. The missing mechanism: an interface where rating-conditioned signals control explanation REGISTER while Stockfish/detectors/verifier control TRUTH, rendered locally. Supported by: shipped grounded systems; Maia; QLoRA/MLX/local (economics kept secondary/low-confidence). **Status: candidate BET.**

**Insight 6 — A valid eval must gate faithfulness (non-LLM) before judging pedagogy, because fluent falsehood contaminates holistic scores.** Cross-check every claimed motif/threat/plan against engine PVs + detector output BEFORE any holistic score; then judge level-fit/pedagogy with a DIFFERENT-family judge (gpt-5.5 generating AND judging = preference leakage). Chess is unusually gate-able (Stockfish + motif detectors = non-LLM truth). Supported by: ACT-Eval; CALM/sycophancy; base run (move-sound insufficient; truthfulness still fails). Now also: our own council self-preference +0.43 (GPT +0.66 / Gemini +0.64 / Claude −0.01) directly measures the same-family bias. **Status: established connection + own measurement.**

**Insight 7 — Faithfulness is now TABLE-STAKES (verifier + capacity), so it cannot be a moat.** A claim-level verify-and-regenerate gate drives user-visible fabrication to 0% for every model (ours 40%→0%, gpt-5.5 7%→0%); the guarantee comes from the deterministic fallback, not the model (even frontier fabrications are sticky). Independently, a 27B open base reaches ~1–8% grounded for free while our data rebuild only got 50%→33% — the deficit is capacity-bound, not a data problem. Supported by: verifier eval; open-model benchmark; rich-grounding A/B (enriching input made ours worse); base/v1/v2 runs. **Status: candidate BET (own measurement); external all-model sweep in-flight.**

**Insight 8 — The one axis that is both a real, dense gap AND deterministically gradeable is tier-appropriate MOVE SELECTION — where the only defensible moat can live.** Frontier tier-differentiation ~22.7%, engine-mirror-at-every-tier 68.7%, yet 67% of decidable held-out positions discriminate (tier-move ≠ engine #1) → common in normal play, mostly unserved. Ground truth is deterministic (Stockfish sound pool + Maia + tier rule), so it trains against a mechanical reward and grades with NO model judge. v2 moved ours 27.5%→39.2% and corrected direction, but not yet won (target >50%). Supported by: gap-density report; frontier gap report; divergence report (weak/mis-directed, 0% contrastive data); v2 retrain. **Status: candidate BET (own measurement); 803-position leaderboard + deterministic-reward training in-flight.**

## Residual gaps to carry forward
- ~1.7B affirmative is now reframed: faithfulness deficit is capacity-bound (27B fixes it free), so the real v3 bet is a bigger open base (Gemma-3-27B) + move-selection training, not the 1.7B.
- Small-FT vs prompted-frontier COACHING eval is now DONE in single-sample form (frontier out-instructs at equal grounding; ours 3.68 vs ~2.21). Still pending: the k-sampled, variance-first version at deployment temperature (SPOV 11/register-variance).
- Move-selection moat NOT yet won: v2 at 39.2% (target >50%, correctly directed); needs the deterministic-reward training loop + the 803-position leaderboard.
- Maia->pedagogy unproven (Weak SPOV, external study).
- SPOV 6 (rationale-verifiability generalization) still needs a multi-domain external study.
- Economics stays secondary/low-confidence.
- In-flight and to fold in: a definitive 803-position held-out leaderboard + an all-14-model verifier sweep.
- Honest caveat on tiering: the new evidence is the project's OWN measurement, which strengthens candidate-validity but is NOT an external primary source, so the new stances stay Strong, not Validated.
