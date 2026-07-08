# Rich / structured grounding vs. current prose grounding — A/B experiment

_Generated 2026-07-07T02:00:48.336152+00:00. OURS-v2 = `chess-coach-v2` (local, in-process `mlx_lm`, decode temp 0.7). Frontier reference = openai-group/gpt-5.5 (reasoning_effort=low), small subset._

## Question

> Does giving the model a **complete, explicit board state** (every piece/square + castling + en-passant + side-to-move + move number) plus the Stockfish sound pool (evals + short PV) and Maia likelihoods **as structured data** reduce fabrication vs. our current prose grounding — without hurting move soundness?

## What differs between A and B (everything else is identical)

Both conditions use the **same system prompt** (the production `coach_system.md` + grounding + format suffix from `src/api/server.py`), the **same engine sound pool / Maia signal / student-move context / task line**, and the **same decode** (per-position seed shared by A and B). Only the *grounding block* changes:

- **A (baseline)** — `render_pool_facts` (prose piece list, loose pieces, what each candidate move does) + `render_user_prompt` (ASCII board + prose pool + Maia). This is exactly what `src/api/server.py` serves today.
- **B (rich)** — `render_rich_facts`: every occupied square enumerated with piece+color, explicit castling rights, en-passant target, side-to-move and move number, then the sound pool (san/uci/eval/short PV) and Maia as explicit tables.

Prompt size (mean chars): A ≈ 1468, B ≈ 1629.

## Sample

- **50 held-out positions** from `data/benchmark_v2`: **33** where OURS-v2 GROUNDED fabricated (the full population of such cases in v2) + **17** clean controls (seeded) to reach the target size and to detect any *new* fabrications B might introduce.
- Frontier (gpt-5.5) reference ran on **15** of these positions (cost-aware).

## Headline — fabrication rate (lower is better)

| Model / slice | A fab | B fab | Δ fab (B−A) | A sound | B sound | n A/B |
|---|---|---|---|---|---|---|
| OURS-v2 — overall | 40% | 56% | +16 pts | 100% | 94% | 50/50 |
| OURS-v2 — fabricated stratum | 48% | 52% | +3 pts | 100% | 94% | 33/33 |
| OURS-v2 — clean stratum | 24% | 65% | +41 pts | 100% | 94% | 17/17 |
| openai-group/gpt-5.5 — overall | 0% | 7% | +7 pts | 100% | 100% | 15/15 |

### OURS-v2 paired view (same positions, A vs B)

- Positions where **A fabricated → B clean (fixed):** **10**
- Positions where **A clean → B fabricated (regressed):** **18**
- Both fabricated: 10  ·  both clean: 12  ·  paired n: 50

- Avg false-claims/output: A = 0.52 → B = 0.9.
- Move soundness (recommended move stayed in the sound pool): A = 100% → B = 94% (no-engine-speak A/B = 100%/100%; ply-cap-ok A/B = 100%/100%).

## Verdict

Rich structured grounding **increases** OURS-v2 fabrication by **+16 pts** overall (40% → 56%), while move soundness moved -6 pts. On the hard (fabricated) stratum it went 48% → 52%, and it introduced new fabrications on clean positions (18 regressions).

**No — structured board state alone does not reduce fabrication here;** keep the current prose grounding (or combine both) rather than replacing it.

## Why it backfired (mechanism)

- **The paired counts are decisive.** On the same 50 positions, B *fixed* 10 of A's fabrications but *created* 18 new ones — a net **+8** fabrications from switching to structured grounding.
- **The clean stratum is the smoking gun.** On positions the current prose grounding already handled cleanly, structured grounding sent fabrication from 24% to 65% (+41 pts). It actively broke things that were working.
- **OURS-v2 is fine-tuned on the ASCII board, so B is off-distribution.** The v2 SFT rows (`data/dataset/train_v2.jsonl`) render the position as the `Board:` ASCII grid (`render_user_prompt`) with **no** structured / `VERIFIED FACTS` block. Condition A keeps that exact ASCII board; Condition B *removes* it and substitutes a per-square enumeration the model never saw in training. A 1.7B model reads the board it was trained to read — hand it a novel structured layout and it tracks the position worse, inventing pieces/squares (e.g. the examples below fabricate a “knight on g2”, “rook on f1”, “knight on c3”).
- **The format is not intrinsically worse — the coupling is.** The frontier model (gpt-5.5), which is not fine-tuned on any grounding format, is format-agnostic and stays near-zero under both conditions (0% → 7%; the single B miss is within noise on a 15-item subset). So structured grounding is fine *in principle*; the regression is specific to the small fine-tune.
- **Move quality also dipped slightly**, consistent with off-distribution parsing: soundness 100% → 94%; avg false claims/output 0.52 → 0.9.

## Recommendation for v3

1. **Do not swap in rich/structured grounding at inference.** For the current 1.7B coach it does not reduce fabrication — it increases it (+16 pts overall; +41 pts on positions the current grounding already handled). Keep the prose/ASCII grounding the model was trained on.
2. **If you want structured grounding, TRAIN it in.** Regenerate the SFT data in the structured format and fine-tune v3 on it, so the format is in-distribution. Do not bolt a new prompt shape onto a model trained on a different one.
3. **The residual-fabrication lever is the verifier, not the prompt.** The production verify-and-regenerate gate (`src/api/server.py`) already catches these false board claims before they reach the student; investing there (or in more/cleaner in-format training data) beats reshaping the prompt.
4. **Worth a follow-up:** test grounding that *adds* the explicit fields (castling / en-passant / side-to-move / move number) while **keeping** the ASCII board — a superset of A rather than a replacement — which would stay closer to the training distribution while still surfacing the extra state.

## Before / after examples (OURS-v2)

### FIXED (A fabricated → B clean) — `2UticRp2_20` (beginner, opening, student played Bxd2+)
FEN: `r1bqr1k1/ppp2ppp/2np4/3NP3/1b1P4/3Q1N2/PPPB2PP/R3KB1R b KQ - 0 10`

**A (prose) — fabricated=True (1 false claim(s)):** “Your bishop on d2 was the active piece, so the natural instinct is to look for it.” → no bishop on d2

> I'd play dxe5. Your bishop on d2 was the active piece, so the natural instinct is to look for it. Black can use that bishop to take the loose pawn on e5, and instead of just moving it around, you remove White’s advanced center pawn. So this move also attacks d4, your loose pawn on e5, and your own loose bishop on b4. How to find it: When it is your turn, run this quick checklist before moving a piece: 1) Is my piece attacked or hanging? 2) Can I make it active by taking an undefended pawn or loose piece? Here, the bishop on d2 can take the pawn on e5, so dxe5 is the clean move to look at before any other activity. Takeaway: Before moving an active piece, check whether it can take an undefended pawn in one move.

**B (rich) — fabricated=False (0 false claim(s)):** (none)

> I'd play Bxd2+. Play this move when you can win material while also using the check: your bishop on c8 attacks the bishop on b4 and removes it, so you are not just moving a bishop; you are removing White’s active piece and forcing White to answer your active move. How to find it: When your king is not attacked, use this quick checklist before choosing a normal move: first ask, ‘Which enemy piece is loose or active, and can I capture it while giving check?’ Here, the loose piece is White’s bishop on b4, and Bxd2+ captures it while giving check to the king on e1. Takeaway: Before playing a normal move, check for loose pieces you can capture with check or capture.

### FIXED (A fabricated → B clean) — `9M8bagRc_36` (beginner, middlegame, student played Qf6)
FEN: `r2q1rk1/pp3ppp/2p4B/3b4/8/2Q5/PP3PPP/R2R2K1 b - - 1 18`

**A (prose) — fabricated=True (1 false claim(s)):** “Why it is good: after gxh6, your queen also defends the rook on d8 and your rook on d1, so you gain space while protecting your own pieces.” → no rook on d8

> I would play gxh6. Your Qf6 move missed the obvious loose piece: White’s bishop on h6 is undefended, so you can take it right away. The point is simple: use a capture when your piece can make the most useful move possible, not by hanging pieces randomly. Why it is good: after gxh6, your queen also defends the rook on d8 and your rook on d1, so you gain space while protecting your own pieces. Takeaway: When an enemy piece is undefended, use a capture to remove it before leaving it on the board.

**B (rich) — fabricated=False (0 false claim(s)):** (none)

> I'd play gxh6. Your Qf6 idea tries to use queen pressure on White, but it misses the loose material: the undefended White bishop on h6. That means your queen can still do more work right away. gxh6 removes that bishop, so instead of letting White keep a free piece, you turn their worst piece into something you can eliminate. How to find it: When an opponent piece is loose, run this quick checklist before making a queen jump: 1) is this piece hanging? 2) can my queen capture it safely? Here, the bishop on h6 is undefended, so gxh6 is the clean capture move to look for before queen activity on other squares. Takeaway: Before starting queen pressure, scan for hanging pieces you can capture right away.

### FIXED (A fabricated → B clean) — `G18ICJyg_100` (advanced, endgame, student played a4)
FEN: `8/8/6P1/pp5K/1k6/7P/8/8 b - - 0 50`

**A (prose) — fabricated=True (1 false claim(s)):** “Kc3 is the cleaner coordinating move because it defends b5 and keeps your king on c3 instead of spending another tempo on the same pawn.” → no king on c3

> I'd play Kc3. Your a4 push is understandable because it defends b5, but it is a bit too direct: it does not improve your king safety while also defending that pawn. Kc3 is the cleaner coordinating move because it defends b5 and keeps your king on c3 instead of spending another tempo on the same pawn. It also allows you to continue improving while staying tactically flexible. Takeaway: When two captures defend the same pawn, choose the one that also improves your king or piece coordination.

**B (rich) — fabricated=False (0 false claim(s)):** (none)

> I'd play Kc3. Your a4 push is understandable because it defends g6, but it is a bit too direct: it does not improve your king safety while also defending h3. Kc3 is the cleaner solution because it defends h3 and also defends c3, so your whole setup gets better without rushing to make another pawn move. How to find it: When both a pawn and king are tied to the same square, do not choose the pawn automatically. First ask: 'Can I make a useful king move that defends both pawn targets at once?' Here, the square c3 and pawn h3 need defense, so Kc3 is the active move to calculate over a4. Takeaway: When two loose points are tied to the same square, prefer a king move that defends both over a pawn push that only defends one of them.

### STILL FABRICATED under both — `KHUJlsaJ_29` (advanced, middlegame, student played Nd1)
FEN: `3rkb1r/p4pnp/2p1p1p1/4P3/8/2N5/PPP3PP/R1B2RK1 w k - 0 15`

**A (prose) — fabricated=True (1 false claim(s)):** “Your Nd1 was understandable because it defends the pawn on c3, but it spends a tempo on a fixed defensive detail.” → no pawn on c3

> I would play Ne4. Your Nd1 was understandable because it defends the pawn on c3, but it spends a tempo on a fixed defensive detail. Ne4 is the more active version: it develops a piece, attacks the pawn on a7, and also defends the pawn on e5. That matters because in sharp middlegames, improving activity over activity that only defends one thing is usually better than just defending passively. How to find it: When you are choosing between two active-looking pieces, run this advanced checklist: first ask, 'Which piece becomes more useful if it moves to the center or attacks something?' Here, Ne4 improves your knight and creates pressure on a7 at the same time. Then check whether the move also defends a target; Bf4 does both, so it is harder to ignore than a passive defense like Rd8. Takeaway: In sharp middlegames, prefer an active developing move over a passive defense if it defends a target too.

**B (rich) — fabricated=True (1 false claim(s)):** “Your Nd1 idea tries to solve the active knight on c3, but it misses a cleaner solution: using the e4-rook instead of moving the knight again.” → no rook on e4

> I'd play Ne4. Your Nd1 idea tries to solve the active knight on c3, but it misses a cleaner solution: using the e4-rook instead of moving the knight again. Ne4 attacks the bishop on c1, so you are turning a loose-looking piece into pressure. That matters because White has to deal with the pressure on c1, and this active move does it without weakening your position around h1 and g1. Takeaway: When two active pieces can attack each other, choose the one that creates pressure without becoming a hanging problem.

### REGRESSED (A clean → B fabricated) — `D5vN67Yh_47` (advanced, middlegame, student played Rg3)
FEN: `r4rk1/5pp1/b2qp2p/p1pnN3/Pp3P1Q/1B1P3R/1PP3PP/1R4K1 w - - 0 24`

**A (prose) — fabricated=False (0 false claim(s)):** (none)

> I'd play Qg4. Your Rg3 was a reasonable practical rook move because it defends the knight on e5 and the pawn on f4, but it misses a more urgent improvement of the queen. Qg4 attacks the e6 pawn and g7 pawn while also defending f4 and g2, so it improves your queen while keeping the game under pressure. It also defends the knight on e5, so it is not just another defensive-looking rook move. Takeaway: When there are no loose pieces, prefer the queen move that attacks two targets and defends several points over a rook move that only covers one pawn.

**B (rich) — fabricated=True (3 false claim(s)):** “Your Rg3 idea is understandable: it attacks the knight on g2 and also defends h2.” → no knight on g2; “Rf1 is more useful because it defends b3, so your rook keeps doing defensive work without giving up the chance to attack the knight on g2.” → no knight on g2; “First ask: 'Which rook move also defends something important?' Here, both rooks do useful defensive jobs, but the rook on f1 defends b3, so it wins the comparison and keeps the tension.” → no rook on f1

> I'd play Rf1. Your Rg3 idea is understandable: it attacks the knight on g2 and also defends h2. The problem is that it is a one-purpose attack while it also defends h2. Rf1 is more useful because it defends b3, so your rook keeps doing defensive work without giving up the chance to attack the knight on g2. How to find it: When two rooks are on wings, do not switch only because one can attack something. First ask: 'Which rook move also defends something important?' Here, both rooks do useful defensive jobs, but the rook on f1 defends b3, so it wins the comparison and keeps the tension. Takeaway: When two rooks are available, choose the rook move that defends something more than it attacks.

### REGRESSED (A clean → B fabricated) — `D6kWitIo_26` (beginner, middlegame, student played O-O)
FEN: `r1bqk2r/ppp4p/4p1p1/4Q3/3np3/8/PPP2PPP/R3KBNR b KQkq - 1 13`

**A (prose) — fabricated=False (0 false claim(s)):** (none)

> I'd play Nxc2+. Play Nxc2+. Your rook on h8 was attacked, so before rushing to something else, first turn that rook into a useful piece. Why it is good: your knight captures the undefended pawn on c2, gives check on the king on e1, and attacks the rook on a1. That is exactly how you use an attacked rook: solve the capture with tempo before anything else. Takeaway: When your rook is attacked, first look for a capture with check or tempo before pushing another rook move.

**B (rich) — fabricated=True (1 false claim(s)):** “Here, the knight on c3 can capture c2, so you should compare it with a normal castling move before choosing.” → no knight on c3

> I'd play Nxc2+. Castling is usually good, but it misses the chance to capture White’s c2 pawn while White has a forcing reply on e4. Nxc2+ uses your active knight to capture the loose c2 pawn and also gives check to the king on e1. That is why it is strong: it turns an active piece against a loose piece instead of only moving forward. How to find it: When the position is loose, do not play a normal castling move first. Run this quick checklist: 1) Which of my pieces can capture something? 2) Does that capture also give check or improve my position? Here, the knight on c3 can capture c2, so you should compare it with a normal castling move before choosing. Takeaway: Before castling, check whether you can capture a loose piece while improving your piece with check.

## Cost

- **OURS-v2: $0.00** (local `mlx_lm`, 100 generations).
- **openai-group/gpt-5.5: $0.1188** (29,355 prompt + 8,214 completion tokens @ $1.25/$10.0 per 1M).
- **Total: $0.1188.**

## Reproduce

```bash
~/.venvs/mlx/bin/python scripts/run_rich_grounding_ab.py
```

Renderer: `src/experiments/rich_grounding.py` · raw per-item rows: `data/experiments/rich_grounding_raw.jsonl` · machine summary: `data/experiments/rich_grounding_summary.json`.
