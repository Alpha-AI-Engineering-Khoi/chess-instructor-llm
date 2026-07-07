# 01c — Expert-Signal Scan (Step 1c)

**Core question:** Can a small (~1.7B) fine-tuned, engine-grounded model deliver reliably level-calibrated chess coaching more dependably than a well-prompted frontier model?

**Method:** For each named expert, recent PUBLIC output was located and quoted. Every item is a DOK 1 fact about *what the person publicly stated* (never laundered into established truth). Tags: `[measured]` = backed by their own experiments/benchmarks; `[opinion]` = stated view; `[public-talk]/[public-post]/[research-blog]/[vendor-claim]/[author-claim]`.

---

### Anant Dole & Asbjorn Steinskog — "Building a Chess Coach," AI Engineer (Take Take Take / Play Magnus)
Stance: oppose (LLM-reasoning) / support (engine grounding). Talk 2026-05-13.
- Steinskog (2026-05-13): "LLMs often hallucinate because obviously they're trained on language, they can't calculate." `[public-talk][opinion]`
- Steinskog (2026-05-13): "the LLM's job is only to translate this information into English... we really don't want it to try to figure out too much on its own, because it quickly leads to hallucination." `[public-talk][opinion]`
- Production coach does NOT ask the LLM to reason: Stockfish scores moves, Maia predicts the human move at a rating, detectors feed structured context, LLM only translates. `[public-talk]`
- Shipped system uses a PROMPTED frontier model (Gemini Flash), not a fine-tune; passes ~75% of 16 eval scenarios at sub-3s latency. `[measured]`

### Kevin Lu & Thinking Machines Lab — "On-Policy Distillation" (2025-10-27)
Stance: support.
- "smaller models with stronger training often outperform larger, generalist models in their trained domains of expertise." `[company-blog][opinion]`
- Distilled Qwen3-8B to 70% AIME'24 in ~150 steps, matching a larger teacher at a fraction of RL compute. `[measured]`
- Caveat: fine-tuning small models on new knowledge causes catastrophic forgetting of instruction-following; "LoRA learns less and forgets less" but "still insufficient for preserving IF-eval." `[measured]`

### Nathan Lambert (Interconnects)
Stance: neutral (double-edged).
- On-policy distillation is now "a core post-training optimization technique." `[public-post][opinion]`
- Benchmarks "easier to overfit," open models "very jagged in performance," "most domain-specific models of today... are actually not specialized enough." `[opinion]`
- "closed [models tend to] be more robust and useful than similarly scoring open models" due to "hard-to-measure qualities," esp. "where an individual user constantly presents new challenges." `[opinion]`

### Reid McIlroy-Young & Ashton Anderson (CSSLab — Maia)
Stance: support (level-conditioning) / caveat (accuracy ceiling).
- Maia's best move-matching ~"over 52%," worst 46%; "Over half the time, Maia 1900 predicts the exact move a 1900-rated human played." `[research-blog][measured]`
- Base Maia ~50%, personalized "up to 65%." `[measured]`
- Anderson framed Maia as a teaching tool to identify "the mistakes... that are the most costly." `[opinion]`
- Maia-2: prior per-level models "lack coherence... limited as teaching tools"; Maia-2 "does not yet incorporate search." `[author-claim]`
- Maia4All (2025): personalizing now needs "only 20 games" vs 5,000. `[measured]`

### Zhenwei Tang (C1 / Master Distillation — CSSLab) (2026-03-20)
Stance: support (strongly).
- 4B C1 "advances from a near-zero baseline to 48.1% accuracy, outperforming all open-source models and most frontier proprietary systems." `[preprint][author-claim]`
- "C1 surpasses its distillation teacher" Gemini-3-Flash (48.1% vs 40.8%). `[measured]`
- "unlike prior neural chess approaches that predict only best moves, C1 generates explainable solutions revealing strategic reasoning." `[opinion]`
- Recipe: SFT on Stockfish-grounded CoT distilled from Gemini-3-Flash, then DAPO RL on Qwen3-4B-Instruct; SFT 42.3% -> RL 48.3%. `[author-claim]`

### Simon Willison
Stance: oppose (fine-tuning) / lean prompting+tools.
- (2024-11-21) "most LLMs are terrible chess players with the exception of gpt-3.5-turbo-instruct." `[opinion]`
- Found Dynomight's prompt-engineering results "more convincing" than their fine-tuning. `[opinion]`
- (2025-06-06) "tools combined with reasoning is the most powerful technique in all of AI engineering right now." `[opinion]`

### "Dynomight"
Stance: neutral (measured), lean prompting.
- Only gpt-3.5-turbo-instruct plays strong chess; chat/instruction-tuned models "terrible." `[measured]`
- "additional instruction tuning makes the model worse" at chess. `[opinion]`
- Prompt "regurgitation" recovered most of GPT-4o's chess quality; prompt-eng > fine-tuning. `[measured]`

### Mathieu Acher (Professor, FIDE 2341)
Stance: oppose (frontier LLM chess).
- gpt-3.5-turbo-instruct ~1750 Elo but illegal move in ~16% of games. `[measured]`
- "training for chat makes GPT worse on a well-defined problem (chess)." `[opinion+measured]`
- o3/o4-mini "not able to play legal moves in a vast majority of cases"; "no apparent progress in chess in the world of (general) reasoning LLMs." `[measured]`
- (2025) a 4-move sequence forces GPT-5 into an illegal move; Kaggle Game Arena "showed none of these models are good at chess yet." `[measured]`

### Tim Dettmers (QLoRA, 2023)
Stance: support (feasibility).
- Finetune a quantized 4-bit model "without any performance degradation," 65B on one 48GB GPU. `[measured]`
- QLoRA "18x cheaper"; 4-bit Guanaco "99.3% of ChatGPT" on Vicuna after 24h on one GPU. `[measured]`
- Caveat: "current chatbot benchmarks are not trustworthy." `[opinion]`

### Daniel & Michael Han (Unsloth)
Stance: support (cost/practice).
- Fine-tune/RL "for free... with just 3GB VRAM"; "start with QLoRA." `[vendor-claim]`
- "LoRA can match full fine-tuning performance while using 4x less VRAM." `[vendor-claim]`
- Caveats: too-large rank "causes overfitting"; "if LoRA fails, don't assume FFT will magically fix it." `[opinion]`

### Neutral signal — Kaggle Game Arena (Aug 2025)
- 8 frontier LLMs generated all moves themselves, no engine; illegal moves penalized. `[measured]`
- o3 won, beating Grok 4; GM commentary: even the winner played "poor chess," Grok "blundering... losing its queen repeatedly." `[journalistic]`

---

## Sharpest expert tensions (prime SPOV raw material)
1. **Same lab, opposite verdicts:** Take Take Take (2026-05-13) says LLMs "can't calculate," confine to translation — while CSSLab's own C1 (2026-03-20) is a 4B model that *reasons* about chess and *beats its teacher*. Central load-bearing tension.
2. **Benchmark win != dependable calibration (Lambert):** a small model can win the eval while being less dependable in deployment — directly targets "reliably... more dependably."
3. **Practitioner vote against fine-tuning:** Willison + Dynomight found prompting > fine-tuning; the shipped Take Take Take product uses a prompted Gemini Flash + engine grounding, not a fine-tune.
4. **Frontier chess is measurably weak (Acher, Kaggle):** guts the "well-prompted frontier" arm — but the prescribed remedy splits (prompt vs fine-tune).
5. **The ~50% ceiling vs the coaching need:** Maia is reliably level-calibrated but MUTE; C1 explains but is scored on puzzles, not calibration. **No shipped system combines dependable level-calibration with grounded explanation** — the exact gap the thesis targets.
6. **Engine grounding, not the fine-tune, may buy dependability:** TML catastrophic-forgetting + Dynomight "tuning makes it worse" + Unsloth overfitting warnings suggest the grounding carries calibration, reframing the thesis.
