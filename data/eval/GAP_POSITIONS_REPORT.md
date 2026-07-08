# Gap-Position Set — curated positions that expose the tier-move selection gap

The target behavior (see `data/analysis/GAP_REPORT.md`): for a stated ELO tier, recommend the move that is **sound** (in Stockfish's tolerance pool) **and human-findable at that tier** (top Maia move *inside* the sound pool) — which is often NOT the engine's #1. A position is **discriminating** for a tier when that tier-appropriate move differs from the engine's #1; those are the only positions where 'just give the best move' (what the frontier does) is provably wrong for a lower tier.

## Method

- **Candidate pool:** `data/positions/positions_v1.jsonl` — real rated Lichess decision positions (reused from the existing polite sample; no new API load).
- **Engines:** Stockfish 18 (`sound_pool`, tolerance 150cp, multipv 8, movetime 300ms, Threads=1) + Maia (`maia-1100/1500/1900`) per tier via lc0 `nodes=1`.
- **Tier-appropriate move:** the sound-pool move with the highest Maia policy at that tier (same definition as `frontier_gap.py`).
- **Gates:** legal/valid FEN, not terminal, engine OK, not trivially decided (no forced mate & |best eval| < 800cp), ≥2 sound moves, discriminates for ≥1 tier.
- **Dedup:** board (placement + side-to-move) absent from `train_v2` + `valid_v2` + `benchmark_v2` and unique within this build.

## 1. Headline counts

| Metric | Value |
|---|---|
| Positions analyzed | 5999 |
| Decidable (non-trivial, ≥2 sound) | 4816 |
| **Discriminating (tier-move ≠ engine #1, ≥1 tier)** | **3226 (67.0% of decidable)** |
| Strong discriminating (findable ≥10%, gap ≥5%) | 2794 (58.0%) |
| Eligible held-out discriminating (deduped) | 3226 |
| **EVAL set → `data/eval/gap_positions.jsonl`** | **803** |
| **TRAINING pool → `data/positions/v3_candidates.jsonl`** | **2423** |

## 2. How discriminating — % where the tier-move ≠ engine #1

Per tier, over the decidable analyzed positions (this is the crux rate):

| Tier | discriminating | strong |
|---|---|---|
| beginner | 2766 (57.4%) | 2086 (43.3%) |
| intermediate | 2702 (56.1%) | 2179 (45.2%) |
| advanced | 2671 (55.5%) | 2248 (46.7%) |

Number of tiers a position discriminates for (of 3):

| tiers | positions |
|---|---|
| 1 | 510 |
| 2 | 519 |
| 3 | 2197 |

Distinct tier-appropriate moves across the 3 tiers (2–3 ⇒ the move itself should change with level — the highest-value contrastive positions):

| distinct moves | positions |
|---|---|
| 1 | 1479 |
| 2 | 1560 |
| 3 | 187 |

## 3. EVAL composition

**Phase** (EVAL):

| phase | positions |
|---|---|
| opening | 344 |
| middlegame | 374 |
| endgame | 85 |

**Source rating tier** (EVAL):

| tier | positions |
|---|---|
| beginner | 241 |
| intermediate | 265 |
| advanced | 297 |

**Primary motif** (EVAL):

| motif | positions |
|---|---|
| fork_shot | 304 |
| enemy_hanging | 190 |
| own_hanging | 68 |
| check_available | 64 |
| capture_available | 59 |
| pin | 45 |
| passed_pawn | 28 |
| quiet_positional | 28 |
| promotion | 17 |

**All motif tags** (EVAL, positions may carry several):

| motif | positions |
|---|---|
| capture_available | 639 |
| fork_shot | 435 |
| check_available | 406 |
| own_hanging | 293 |
| passed_pawn | 235 |
| enemy_hanging | 195 |
| pin | 146 |
| quiet_positional | 115 |
| promotion | 17 |

## 4. TRAINING composition

**Phase** (TRAINING):

| phase | positions |
|---|---|
| opening | 1132 |
| middlegame | 1214 |
| endgame | 77 |

**Source rating tier** (TRAINING):

| tier | positions |
|---|---|
| beginner | 720 |
| intermediate | 805 |
| advanced | 898 |

**Primary motif** (TRAINING):

| motif | positions |
|---|---|
| fork_shot | 963 |
| enemy_hanging | 596 |
| own_hanging | 198 |
| capture_available | 193 |
| check_available | 161 |
| pin | 136 |
| quiet_positional | 70 |
| passed_pawn | 64 |
| promotion | 42 |

**All motif tags** (TRAINING, positions may carry several):

| motif | positions |
|---|---|
| capture_available | 2030 |
| fork_shot | 1361 |
| check_available | 1150 |
| own_hanging | 932 |
| enemy_hanging | 608 |
| passed_pawn | 588 |
| pin | 469 |
| quiet_positional | 327 |
| promotion | 42 |

## 5. Dedup / leakage confirmation

- Held-in board keys (train_v2 + valid_v2 + benchmark_v2): **2006**.
- Analyzed rows whose board leaked into v2 corpora: **0** (dropped).
- Duplicate boards within the pool: **0** (dropped).
- EVAL ∩ TRAINING (board overlap): **0** (must be 0).
- Every emitted position is held-out and unique by board — **zero leakage**.

## 6. Example discriminating positions (EVAL)

Positions where the tier-appropriate move genuinely CHANGES across tiers (distinct moves ≥ 2) are the sharpest illustration of the gap — the model must give a *different* move by level, not the engine's #1:

- `8MGanXJy_20` [opening/pin] engine#1 **g5** — B:Bg4  I:a6  A:c6
- `AtgA9tf7_36` [middlegame/fork_shot] engine#1 **Ba6** — B:Nf6  I:e5  A:c4
- `EYrWAPbF_139` [endgame/passed_pawn] engine#1 **Bh7** — B:Be6  I:Be4  A:Bd3
- `bx4pn43R_22` [opening/fork_shot] engine#1 **Nh2** — B:Nh2  I:Nf6  A:Qh4
- `3oxrrH7B_47` [middlegame/capture_available] engine#1 **Qc2** — B:Rfc1  I:Nd3  A:Ne2
- `6yFJ5GyX_97` [endgame/check_available] engine#1 **Rd1+** — B:Rd1+  I:Ra1  A:g4
- `24dMHvVW_25` [opening/fork_shot] engine#1 **Rf2** — B:Rc1  I:Rf2  A:Re1
- `Rex6dtFE_46` [middlegame/enemy_hanging] engine#1 **Qd8** — B:Qb7  I:Qa5  A:Qb4
- `26MZFDEq_141` [endgame/check_available] engine#1 **d5** — B:Rf7+  I:d5  A:Rf6
- `vKRHem32_37` [opening/fork_shot] engine#1 **Qg4** — B:Qg4  I:Bf4  A:Ne2
- `O3cLGska_56` [middlegame/passed_pawn] engine#1 **g6** — B:g6  I:Kh8  A:Kg8
- `pjw4t40x_59` [endgame/fork_shot] engine#1 **Kg5** — B:Re8+  I:Rf6  A:Kg5
- `QsTYZO9F_27` [opening/fork_shot] engine#1 **Bc4** — B:Bb5  I:Bc4  A:Ng6
- `zOt1AUhj_47` [middlegame/quiet_positional] engine#1 **Kf1** — B:Ne5  I:Nb5  A:Kf1
