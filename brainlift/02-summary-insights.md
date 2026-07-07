# 02 — Summary (DOK 2 Knowledge Tree) + DOK 3 Insights

Core question: can a small (~1.7B) fine-tuned, engine-grounded model deliver
reliably level-calibrated chess coaching more dependably than a well-prompted
frontier model? (Affirmative crux is a testable bet, not settled — see 01b gaps.)

## DOK 2 — Knowledge Tree

### 1. Distillation dynamics
- **What transfers:** KD "dark knowledge"; 770M > few-shot 540B PaLM w/ rationales; Orca 13B > Vicuna 113% BBH; Qwen3-1.7B-Base ≈ Qwen2.5-3B-Base (pretraining parity, NOT proof for coaching); distilled Qwen3-8B 70% AIME'24 in ~150 steps; C1 4B (Stockfish-grounded CoT + RL) 48.1% puzzles, surpassed its Gemini teacher.
- **Where it fails:** high-fidelity KD hard on synthetic data (Stanton); distillation traps -> overconfident hallucination + self-correction collapse; model collapse erases tails; Small Model Learnability Gap (≤3B); LoRA intruder dimensions/forgetting; SFT leaves ~15.3% instances unlearned (rare/compositional); students inherit/amplify teacher hallucinations.
- **FT is behavior-shaping, not capability creation:** 1.3B InstructGPT > 175B GPT-3 on intent; reliability drops 61.8% under cousin prompts, targeted FT helps; instruction tuning can make chess WORSE (Dynomight).

### 2. Prompt vs fine-tuning reliability
- Prompting is a serious baseline: prompt-opt fixed GPT-4o structured output 0% -> 95.2%; prompt-eng recovered much of GPT-4o chess (Dynomight); tools+reasoning strong (Willison).
- But prompting is brittle under nuanced multi-constraint stacks (correct concept + level language + no engine-speak + no fabricated tactic + useful feedback), all at once.
- **The head-to-head claim is untested:** no direct small-FT vs prompted-frontier coaching eval; ~1.7B under-evidenced (grounded wins are 4B/8B). Resolved by the project's own base-vs-tuned eval.

### 3. Small-model ceilings
- Small specialist wins are real but narrow: FT small BERT > zero-shot GPT-4/Claude; Gorilla 7B > GPT-4 API; MATE 8B +24.2% move-choice; Qwen2.5-0.5B > frontier on relation extraction. None is coaching/pedagogy.
- 1.7B is the hardest part: C1=4B/puzzles, MATE=8B/move-choice; ≤3B learnability gap. The bet: engine grounding offloads calculation so the LM's job shrinks.

### 4. Chess engine + Maia leveling primitives
- Stockfish = objective tactical truth (used as judge across DecodeChess, chess-coach-mcp, Play Magnus).
- Maia = human-likelihood + rating conditioning: 46-52% (peaks near training rating) vs Stockfish 33-41%; Maia-2 skill-aware, no search; Maia-3 57.1% (79M; 23M=56.6%); ceiling far below 100%, volatile across ratings; Maia4All personalizes w/ 20 games.
- **Human-likelihood != instructiveness:** the teaching move can differ from human-likely, engine-best, or natural-looking (Chess.com Torch Human; Hattie; expertise reversal).

### 5. LLM chess ability + commentary hallucination
- Frontier LLMs unreliable chess reasoners without tools: gpt-3.5-turbo-instruct ~1750 Elo (<0.1% move-level illegal; ~16% of games illegal); o3/o4-mini >90% illegal in some evals; Kaggle Arena "poor chess"; chat FT degrades the task.
- Commentary hallucinates even when prose sounds expert: ACT-Eval GPT-5.4 22% incorrect (OSS >50%); CCC/GCC-Eval; judges share the hallucinations (4.9/5 on false commentary); best-move SFT strong but RL -> unfaithful reasoning.

### 6. "Engine = truth, LLM = translator" architecture
- Shipped systems converge on the neurosymbolic split (Play Magnus, DecodeChess, chess-coach-mcp): engine+detectors produce truth; LLM translates. Argument for GROUNDING (not necessarily fine-tuning).
- The narrow translator role is where a small FT model MAY suffice — controlled pedagogical paraphrase over structured features — but unproven at 1.7B.

### 7. Learning-science pedagogy of leveled feedback
- Target = calibrated next-step feedback, not chess strength (ZPD; scaffolding; CLT "no engine-speak" = extraneous-load reduction; Hattie goal/progress/next-step).
- **Miscalibration isn't neutral:** over-explaining / above-level / wrong concept can HARM learning (expertise reversal; LLM-tutor bad-hint rates). "Dependable calibration" is a higher bar than "usually correct."

### 8. Eval + judge validity
- LLM judges useful but unsafe alone: >80-85% human agreement, but position/verbosity/self-enhancement bias (CALM), sycophancy 95%/45%/75% (Sharma), chess-hallucinating judges (ACT-Eval).
- The eval must SEPARATE engine-truth, level-fit, and pedagogy — not collapse into one vibe score.

### 9. Economics / local
- QLoRA/MLX/Unsloth make local small-model FT cheap/private/offline — but vendor absolutes unverified; economics is a SECONDARY advantage, not the load-bearing claim.

## DOK 3 — Insights

**Insight 1.** The most plausible reason a 1.7B model could beat a frontier model is not that it learns chess reasoning, but that engine grounding turns the task into constrained pedagogical translation. (Play Magnus translator-only + FT-for-behavior evidence + narrow-specialist wins -> the small model's role is "stable no-engine-speak translator of structured truth," not "mini grandmaster.")

**Insight 2.** The same-lab contradiction resolves if C1 proves small models can solve grounded chess puzzles while Play Magnus proves production coaching shouldn't trust an LLM to decide what's true. (C1 = benchmark-condition grounded pattern recognition; Play Magnus = deployment reliability in open-ended coaching; Lambert's benchmark≠deployment caveat bridges them.)

**Insight 3.** The empty product quadrant is "Maia-calibrated but mute" + "C1-explanatory but not level-calibrated"; the thesis lives in the missing COMPOSITION, not either component. (No cited source demonstrates dependable fine-tuned level-calibration + grounded explanation together.)

**Insight 4.** "More dependable than frontier" is less about average answer quality and more about VARIANCE under constraint stacking. (Frontier is strong-but-variable across the simultaneous constraints; a small FT model may win by reducing variance on the bundle even with lower raw capability.)

**Insight 5.** Maia's human-likelihood signal is necessary for calibration but dangerous if mistaken for pedagogy. (A likely human move may be a misconception to correct, a stepping stone, or a bad habit; the coach must turn Maia's DESCRIPTIVE level signal into a PRESCRIPTIVE teaching decision.)

**Insight 6.** The fine-tune may be the least load-bearing component of the system, even though it's the component being tested. (Reliability comes from Stockfish/Maia/detectors/eval gates; FT can worsen/forget/overfit; judge the tuned 1.7B as a controllable RENDERER over grounded features, not the origin of dependability.)

**Insight 7.** The evaluation must be adversarial to persuasive prose, because both LLM tutors and LLM judges reward fluent wrongness. (Decompose claims into engine-verifiable truth + level-fit + pedagogy BEFORE any holistic preference; sycophancy + chess-hallucinating judges otherwise select coach-sounding falsehoods.)
