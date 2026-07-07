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

## Validated DOK 3 insights

**Insight 1 — A small model can only win if the system turns coaching into constrained faithful translation, not open-ended chess reasoning.** Stockfish gives truth, detectors expose motifs/threats, Maia describes human behavior, and the model renders it at level. As-built (no motif detectors, no verifier) the task is UNDER-constrained, which is why it fabricates. Supported by: product convergence (Play Magnus/DecodeChess/chess-coach-mcp); C1 4B (grounded small reasoning possible, but larger + narrow); ACT-Eval/CCC (fluent-but-wrong); base run (move-sound solved, truthfulness/register not). **Status: candidate BET.**

**Insight 2 — The fine-tune is not the origin of dependability; it is the last-mile compressor whose value must survive ablation.** Dependability comes from grounding+detectors+verification; FT mainly compresses a desired style into a small local model (fewer tokens, steady no-engine-speak, consistent register, cheap/private/offline). If constrained decoding + prompt + verifier get the same gains, FT isn't carrying the thesis. Supported by: InstructGPT; prompt-optimization; distillation traps/collapse/learnability-gap/LoRA-forgetting; the 1.7B gap; base run. **Status: established connection + candidate BET.**

**Insight 3 — The key metric is worst-case variance under stacked constraints, not mean coaching quality.** "More dependable" = fewer bad failures when sound-move + truthful-explanation + no-fabrication + level-fit + useful-next-step + no-engine-speak must all hold at once. A frontier model may have a higher mean yet a fatter tail of confident wrongness. Currently unmeasured (greedy n=9) — needs k-samples at deployment temperature, worst-case not mean. Supported by: ACT-Eval 22%/judge 4.9-5; sycophancy; expertise-reversal; citation gaps. **Status: candidate BET.**

**Insight 4 — Maia is a descriptive level signal, not a prescription for what to teach.** Human-likely != pedagogically useful (a likely move may be a misconception, a stepping stone, or a bad habit). Mark Maia explicitly descriptive; require a separate pedagogical decision layer. Supported by: Maia 46-52%, volatility/ceiling; Chess.com Torch Human; Hattie; expertise-reversal; the unmeasured Maia->reliability link. **Status: established connection + candidate BET.**

**Insight 5 — The genuinely underfilled cell is the small/local/fine-tuned FORM FACTOR for grounded+leveled coaching, not the behavior itself.** Play Magnus already ships grounded, Maia-leveled explanation via a PROMPTED frontier — so the behavior exists. The bet is compressing it into a small local model without losing faithfulness/pedagogy. It's an economics/deployment bet, not a "nobody built the behavior" claim. The missing mechanism: an interface where rating-conditioned signals control explanation REGISTER while Stockfish/detectors/verifier control TRUTH, rendered locally. Supported by: shipped grounded systems; Maia; QLoRA/MLX/local (economics kept secondary/low-confidence). **Status: candidate BET.**

**Insight 6 — A valid eval must gate faithfulness (non-LLM) before judging pedagogy, because fluent falsehood contaminates holistic scores.** Cross-check every claimed motif/threat/plan against engine PVs + detector output BEFORE any holistic score; then judge level-fit/pedagogy with a DIFFERENT-family judge (gpt-5.5 generating AND judging = preference leakage). Chess is unusually gate-able (Stockfish + motif detectors = non-LLM truth). Supported by: ACT-Eval; CALM/sycophancy; base run (move-sound insufficient; truthfulness still fails). **Status: established connection.**

## Residual gaps to carry forward
- ~1.7B crux under-evidenced (grounded wins are 4B/7B/8B; <=3B learnability-gap headwind).
- No direct small-FT vs prompted-frontier COACHING eval (the project's base-vs-tuned IS the disconfirmer).
- Level-calibrated coaching not directly measured by comparables.
- Maia->pedagogy unproven.
- Economics stays secondary.
- Current eval too small (greedy n=9) for variance claims.
