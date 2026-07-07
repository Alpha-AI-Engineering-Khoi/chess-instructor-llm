# 01 — Experts & Facts (pooled DOK 1)

Core question: can a small (~1.7B) open model, fine-tuned on engine-grounded
distilled data, deliver reliably level-calibrated chess coaching (instructive move
at the student's level, no engine-speak) more dependably than a well-prompted
frontier model?

## First-principles map
KD/SFT/LoRA-QLoRA · prompting-vs-fine-tuning reliability · small-model
specialization vs frontier generality · learning science (ZPD, scaffolding,
cognitive load, feedback, deliberate practice) · chess pedagogy (best vs
instructive move; Maia human-likelihood) · tool-augmented/neurosymbolic grounding ·
eval methodology (LLM-judge + sycophancy) · economics/local deployment ·
distillation dynamics (student <= teacher capability, yet can exceed prompted
teacher on consistency).

Facts are tagged by lane: [foundations] [advocate] [adversary] [cartographer]
[frontier] [edge]. Pending citation gate (Step 1b).

## A. Distillation & small-model specialization
- [foundations] KD transfers a large model's "dark knowledge" via softened targets (Hinton, Vinyals, Dean 2015, arxiv 1503.02531).
- [advocate] Distilling step-by-step: a 770M model beat a few-shot 540B PaLM with 80% of data (Hsieh et al., ACL Findings 2023).
- [advocate] Orca (13B) learning GPT-4 explanation traces surpassed Vicuna-13B >100% on BBH (Mukherjee et al. 2023).
- [advocate] Fine-tuned small BERT-style models beat zero-shot GPT-4/Claude Opus on text classification (Bucher & Martini 2024, arxiv 2406.08660).
- [advocate] Gorilla (fine-tuned LLaMA-7B) surpasses GPT-4 on API calls with fewer hallucinations (Patil et al., NeurIPS 2024).
- [cartographer/foundations] Qwen3-1.7B-Base ≈ Qwen2.5-3B-Base via strong-to-weak distillation (Qwen3 Tech Report 2025, arxiv 2505.09388).
- [cartographer] NVIDIA position: SLMs are sufficient/economical for specialized agentic tasks (Belcak et al. 2025, arxiv 2506.02153).
- [foundations] "Finetuner's fallacy": a 1B specialized model beat a 3B standard model on under-represented domains (2026, arxiv 2603.16177).

## B. Prompting vs fine-tuning for reliability / constraint adherence
- [foundations] 1.3B InstructGPT outputs preferred over 175B GPT-3; bigger != better at following intent (Ouyang et al. 2022).
- [foundations] Instruction-following reliability drops up to 61.8% under nuanced prompt variants; targeted FT on "cousin prompts" beat Alpaca-style FT (Dong et al., ACL 2026).
- [advocate] Constraint adherence: avg accuracy falls 80.82%->36.76% from simple to hard constraints; training on framework data improves it (MulDimIF, ACL Findings 2026).
- [advocate] Structured-output reliability: naive prompting hits 85% task acc but 0% valid-JSON output acc; prompt-optimization (not FT) fixed GPT-4o to 95.2% (Galeone et al. 2026, arxiv 2605.02363).
- [adversary] Practitioner 2026: gap between well-prompted and fine-tuned frontier "smaller than ever"; FT "can make things worse" (masterprompting.net 2026).
- [frontier] Practitioner consensus: "fine-tune for how the model behaves"; FT when a model is "capable but inconsistent — drifts from format/tone" (AgentsCamp 2026); FT beats prompts on format adherence "at scale" (viqus 2026).
- [frontier] "Distillation via fine-tuning" (small open model on frontier outputs at ~100x lower cost) is "a standard production pattern" (Towards AI 2026); OpenAI's own guidance: hit target on big model, log, fine-tune small on logs (llm-stats 2026).

## C. Distillation failure modes (the risk cluster)
- [adversary] "Does KD really work?": high fidelity is hard with synthetic/augmented data and doesn't always generalize (Stanton et al., NeurIPS 2021).
- [adversary/edge] "Distillation traps": tail noise + teacher-student gap cause overconfident hallucinations, self-correction collapse (ACL 2026 long 908 / arxiv 2604.18963).
- [adversary] Model collapse: training on recursively generated data erases distribution tails; tails need real human data (Shumailov et al., Nature 2024; "Curse of Recursion" 2023).
- [frontier] Students can inherit/amplify teacher hallucinations (Google "Teacher's Pet", Lukasik et al., TMLR). [citation-fix: do NOT cite Smoothed KD 2502.11306 here — it shows KD can REDUCE hallucination]
- [adversary] "Small models struggle to learn from strong reasoners" — the Small Model Learnability Gap; ≤3B do better on shorter/simpler chains (Li et al., ACL Findings 2025).
- [adversary] LoRA introduces "intruder dimensions" -> forgets more of pretraining vs full FT (NeurIPS 2025, arxiv 2410.21228); LoRA still trails full FT (LoRA-Pro, ICLR 2025).
- [edge] On-policy distillation can trap models in overconfidence (calibration != capability) (Salesforce CaOPD 2026, arxiv 2604.16830).
- [edge] SFT "Incomplete Learning": ~15.3% of supervised instances remain unlearned; failures cluster in rare/compositional cases (2026, arxiv 2604.10079).

## D. Chess: engines, ratings, Maia human-move prediction
- [cartographer] Stockfish = superhuman strength (handcrafted->NNUE, Nasu 2018, ported SF12 2020); AlphaZero self-play; the two are complementary to Maia.
- [foundations/cartographer/advocate] Maia predicts human moves 46-52% (vs Stockfish/Leela ~33-41%); 9 models for rating bins 1100-1900; accuracy peaks near the training rating (McIlroy-Young et al., KDD 2020).
- [cartographer/advocate] Maia-2 unifies skill levels with skill-aware attention; most accurate human-move predictor (NeurIPS 2024).
- [edge] Maia-3 (Chessformer, ICLR 2026) reaches 57.1% human-move accuracy (79M model; 23M variant = 56.6%) at <1/4 the prior 355M SOTA's params; rating-conditioned (arxiv 2605.19091). [citation-fix: 57.1% is the 79M, not 23M]
- [adversary] But: Maia-2 says the human-move-prediction ceiling is "far below 100%," overall gain ~2pp, predictions volatile/incoherent across adjacent ratings (NeurIPS 2024).

## E. LLM chess (in)ability + commentary hallucination
- [cartographer/frontier] gpt-3.5-turbo-instruct plays ~1750 Elo (<0.1% illegal at MOVE level; ~16% of GAMES contain an illegal move, Acher); reasoning models o3/o4-mini >90% illegal; chat FT degrades a well-defined task (Karvonen; Acher 2024). [citation-fix: denominators corrected]
- [adversary] Only the strongest reasoning LLMs beat a random agent even when given legal moves; o3-low ~758 Elo; format-sensitive (arxiv 2512.01992, 2025).
- [frontier/edge] ACT-Eval: GPT-5.4 no-tools gives incorrect chess claims 22% of the time; small OSS >50%; tools help but strategic coverage limited (OpenReview 2026).
- [advocate/adversary] Chess commentary is prone to hallucination even with expert-model grounding (CCC/GCC-Eval, NAACL 2025); the LLM judge "often exhibits the same hallucinations" (ACT-Eval).
- [edge] Best-move SFT gave strong perf but RL elicited unfaithful reasoning; multi-move trajectory training was more faithful (ICML 2026, arxiv 2604.05134).
- [edge] C1 (4B, SFT on Stockfish-grounded CoT distilled from Gemini + RL) hit 48.1% puzzle acc, surpassing its teacher and most frontier models with ~100x fewer tokens (2026, arxiv 2603.20510; CSSLab/C1).

## F. Chess-coaching products: "engine = truth, LLM = translator"
- [frontier/edge] Play Magnus / Take Take Take: Stockfish = ground truth, detectors extract structured concepts, LLM ONLY translates to English — chosen because independent LLM chess reasoning hallucinates (Weldon 2026; AI Engineer Europe talk 2026).
- [frontier] DecodeChess: XAI explains Stockfish NNUE (depth ~24), for players up to ~2000 Elo; premise "engines say what, not why."
- [frontier] chess-coach-mcp: "Stockfish is the judge, the LLM is the explainer"; moves classified by win-% drop (Lichess model), not raw centipawns.
- [frontier] Chess.com (Mar 2026) replaced Stockfish with "Torch Human" in Game Review — among strong moves, picks the most human, to feel like a real coach.

## G. Direct comparables (shipped fine-tuned chess tutors)
- [frontier] felixmanojh/Qwen3-4B Lichess Puzzle Tutor: LoRA on Qwen3-4B, distilled Claude explanations, MLX-trained; reports 96% completeness, "zero hallucinations" on a 50-puzzle test set (small n).
- [frontier] NAKSTStudio/chess-gemma-commentary: Gemma 3 270M, LoRA, 25k positions, offline/mobile, move classification + Elo prediction.
- [frontier] ToastyDreams/chess-commentary-t5: T5-small grounded on engine-derived features; degrades if features missing.

## H. Learning science (the pedagogy foundations)
- [foundations/cartographer] ZPD = distance between independent and guided performance (Vygotsky); scaffolding = expert controls elements beyond learner capacity (Wood, Bruner, Ross 1976).
- [adversary] But ZPD<->scaffolding conflation is "problematic"/"dilutes" the theory (JTSB 2020); ZPD empirics called "trivial" (Newman & Latifi 2021).
- [foundations/cartographer] Cognitive Load Theory: limited working memory; minimize extraneous load for schema acquisition (Sweller 1988; 2019 update). ("No engine-speak" ~ extraneous-load reduction.)
- [foundations] Feedback works when it addresses goal/progress/next-step; effect varies by type (Hattie & Timperley 2007).
- [foundations] Deliberate practice matters but explains only ~21-26% of variance in games/music (Macnamara et al. 2014) — less than once claimed.
- [adversary] Expertise-reversal: guidance that helps novices can HARM experts; must fade with proficiency (Kalyuga & Renkl 2010). (Mis-calibrated leveling can backfire.)

## I. ITS effect sizes + LLM-tutor reliability
- [cartographer/advocate] ITS ≈ human tutoring: VanLehn 2011 d=0.76 (ITS) vs 0.79 (human); Bloom's "2-sigma" is contested (~0.3-0.8 modern range).
- [advocate] ITS meta-analysis: g=.42 vs teacher-led, no sig. diff vs human tutoring (Ma et al. 2014; Steenbergen-Hu & Cooper 2014).
- [adversary] LLM tutors unreliable: 35% of hints too general/incorrect/give away answer (IJAIED 2025); only 56.6% of math-tutoring dialogues fully correct (arxiv 2503.16460); over-validate wrong solutions (BEA 2026).

## J. LLM-as-judge validity + sycophancy (measurement risk)
- [cartographer] Strong LLM judges reach >80-85% human agreement (Zheng et al., NeurIPS 2023) — but with position/verbosity/self-enhancement bias (12 biases catalogued, CALM 2024).
- [cartographer/adversary] Sycophancy: Claude 2 PM preferred a convincing sycophantic answer over a truthful one 95% of the time (45% on hardest); best-of-N still 75% (Sharma et al., ICLR 2024).
- [adversary] In chess, the judge hallucinated like the model: a hallucinated commentary scored 4.9/5 while 2/3 claims were false (ACT-Eval).

## K. Economics / local deployment
- [frontier] QLoRA fine-tune of a 7-8B model ~$0.35-5, 45min-4hr; cost fell from ~$300 (2024) to <$5 (2026) with Unsloth; free on Apple Silicon.
- [frontier] MLX is the only local runtime with native LoRA/QLoRA on-device; keeps data on-device (privacy); ~15-30% faster than Ollama at equal quant.
- [frontier] Bridgewater fine-tune: 84.66% vs frontier ~78.2% prompted, 13.8x cheaper; Parsed: 50-80% cheaper, 2-3x faster.

## Double-edged (pro-thesis, surfaced by adversary — must reconcile)
- MATE: fine-tuned LLaMA-3-8B beat GPT/Claude/Gemini on chess move selection by 24.2% with strategy+tactic annotations (arxiv 2411.06655) — but 8B, human data, move choice not explanation.
- Sub-billion Qwen2.5-0.5B beat frontier by >25 F1 on relation extraction (arxiv 2606.22606) — narrow extraction, not pedagogy.

## Candidate experts (deduped)
- KD/PEFT: Geoffrey Hinton, Jeff Dean, Tim Dettmers, Luke Zettlemoyer, Edward Hu, Cheng-Yu Hsieh, Subhabrata Mukherjee.
- Scaling/small models: Jared Kaplan, Jordan Hoffmann, Peter Belcak (NVIDIA), Nathan Lambert.
- Reliability/faithfulness: Miles Turpin (CoT unfaithfulness), Mrinank Sharma/Ethan Perez (sycophancy), Samuel Stanton (KD reality), Ilia Shumailov (model collapse).
- Chess ML: Reid McIlroy-Young, Ashton Anderson, Jon Kleinberg, Siddhartha Sen, Zhenwei Tang (Maia/Maia-2/3, C1), Daniel Monroe (Chessformer), Anian Ruoss (searchless chess), Adam Karvonen / Mathieu Acher (empirical LLM chess).
- Learning science: John Sweller (CLT), Wood/Bruner (scaffolding), Kurt VanLehn (ITS effect sizes), Slava Kalyuga (expertise reversal), Ken Koedinger / Vincent Aleven (ITS), John Hattie (feedback), K. Anders Ericsson / Brooke Macnamara (deliberate practice).
- Eval: Lianmin Zheng (LLM-as-judge), + ACT-Eval / CCC authors (chess commentary eval).
- Practitioners: Anant Dole & Asbjorn Steinskog (Play Magnus coach), Thinking Machines Lab (on-policy distillation), Daniel & Michael Han (Unsloth), Felix Manojh (Qwen3-4B tutor).

## Child-brainlift candidates
Flagged by multiple lanes; deduped + decision recorded in `child-brainlifts.md`.
