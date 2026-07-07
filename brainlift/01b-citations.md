# 01b — Citation gate

**Headline:** every load-bearing/spiky fact + every 2026 arXiv preprint was checked
against primary sources. **Zero hallucinated/fabricated facts. Zero DROPs.** A few
FIXes (precision/attribution) and several load-bearing GAPS (the affirmative crux is
under-evidenced at exactly ~1.7B).

## Per-cluster verdict (all VERIFIED unless noted)
- **A. Distillation/small-model:** Hinton KD, Distilling-step-by-step (770M>540B PaLM), Orca (>113% BBH), Bucher&Martini, Gorilla, Qwen3 base-parity, NVIDIA SLM, Finetuner's Fallacy — all VERIFIED (see FIX 2,3).
- **B. Prompt vs FT reliability:** InstructGPT, Dong et al. 61.8% cousin-prompt drop, MulDimIF 80.82->36.76, structured-output 0% JSON -> 95.2% — VERIFIED (FIX 4). Practitioner "gap smaller than ever / distillation-via-FT standard" = opinion, low-DOK.
- **C. Distillation failure modes:** Stanton KD-reality, Distillation Traps (ACL 908/2604.18963), model collapse (Nature), Small Model Learnability Gap (2502.12143), LoRA intruder dims (2410.21228), CaOPD miscalibration (2604.16830), Incomplete Learning 15.3% (2604.10079) — VERIFIED (FIX 6).
- **D. Chess engines/Maia:** Stockfish/NNUE, Maia 46-52% (KDD2020), Maia-2, Maia-3 — VERIFIED (FIX 1).
- **E. LLM chess + commentary hallucination:** Acher/Karvonen (FIX 5), 2512.01992, ACT-Eval GPT-5.4 22% (⭐), CCC/GCC-Eval, 2604.05134, C1 48.1% > teacher (⭐) — VERIFIED.
- **F. Products (engine=truth, LLM=translator):** Play Magnus/Take Take Take, DecodeChess, chess-coach-mcp, Chess.com Torch Human — VERIFIED.
- **G. Shipped tutors:** felixmanojh Qwen3-4B (96%, zero-halluc small-n), NAKST Gemma-270M, ToastyDreams T5 — VERIFIED (spot-check ToastyDreams card).
- **H. Learning science:** ZPD/scaffolding, CLT, Hattie feedback, deliberate practice 21-26%, expertise-reversal — VERIFIED (FIX 7).
- **I. ITS + LLM-tutor:** VanLehn d=0.76/0.79, Ma g=.42, LLM-tutor 35% bad hints / 56.6% correct — VERIFIED.
- **J. LLM-judge + sycophancy:** Zheng >80-85%, Sharma 95%/45%/75% (⭐), ACT-Eval judge 4.9/5 on false commentary — VERIFIED.
- **K. Economics:** QLoRA cost curve, MLX-only-local, Bridgewater/Parsed numbers — NOT independently verified (vendor/practitioner); do not lean an SPOV on the absolutes.
- **Double-edged:** MATE 8B +24.2% (2411.06655), Qwen2.5-0.5B >25 F1 RE (2606.22606) — VERIFIED (caveats preserved).

## FIXes (real but mis-stated — applied to 01-experts-and-facts.md where load-bearing)
1. Maia-3 57.1% = 79M (23M = 56.6%); ICLR 2026. [applied]
2. Qwen3-1.7B≈Qwen2.5-3B is a PRETRAINING parity result; strong-to-weak distillation is separate. Decouple.
3. Finetuner's Fallacy mechanism is specialized PRETRAINING; it frames finetuning-only as the fallacy — double-edged, not clean support.
4. Structured-output paper "When Correct Isn't Usable" (2605.02363): author "Galeone" unconfirmed — verify attribution.
5. LLM-chess illegal rates: <0.1% move-level (Karvonen); ~16% game-level for gpt-3.5-turbo-instruct (Acher); >90% for o3/o4-mini. [applied]
6. Inherit/amplify hallucination -> cite "Teacher's Pet", NOT Smoothed KD (which shows KD reduces hallucination). [applied]
7. ZPD-conflation paper is Xi 2021 (not 2020).

## GAPS (load-bearing; treat as the SPOV crux / the user's experiment, not citation failures)
1. **~1.7B is under-evidenced** — grounded wins are 4B (C1) / 7-8B (MATE); ≤3B learnability gap is a headwind. The bet: engine grounding offloads capability so 1.7B suffices. This is the crux to TEST, not a proven fact.
2. **No direct measure of "level-calibrated coaching"** — comparables measure move/puzzle accuracy or commentary completeness, not instructiveness-at-level.
3. **Maia-leveling -> coach-reliability link unmeasured.**
4. **No head-to-head small-FT vs prompted-frontier on COACHING** — the thesis crux is only inferred from adjacent evidence. (This is exactly the project's base-vs-tuned eval.)
5. **Economics (K) unsourced** — mark vendor numbers low-confidence.

**Decision:** proceed to synthesis. Gaps 1-4 are the inherent novelty of the spiky
claim (their disconfirmer = the user's own base-vs-tuned experiment), not citation
failures, so NO gap-filling research pass is spawned; synthesis/SPOVs must frame the
1.7B affirmative as a candidate bet whose crux is tested by the project's eval, and
must not lean on the unsourced economics absolutes.
