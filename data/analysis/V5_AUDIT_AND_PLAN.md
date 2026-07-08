# v5 Dataset Audit + Curation Plan (instructiveness-focused)

Read-only audit of the SFT datasets against the 6-element instructiveness rubric,
tier-coherence diagnosis, a correctness-checked principle library
(`principle_library_v5.md`), and a data-first v5 curation/generation plan sized to
move the three weak axes (**beginner tier-fit, how-to-find-it quality,
truthfulness**). v4 is training now; this feeds v5.

Reproduce: `scripts/audit_v5_instructiveness.py` (coverage + coherence),
`audit_v5_quality.py` (method/principle/trade-correctness), `audit_v5_artifacts.py`
(target-text artifacts), `audit_v5_coherence.py` (collapse decomposition),
`v5_demo_generate.py` (small TFY A/B proof). Machine output: `data/analysis/v5_audit.json`.

---

## 1. Element coverage (train_v4, 6,767 unique rows after dropping 1,978 oversample dups)

| Element | Overall | beginner | intermediate | advanced |
|---|---:|---:|---:|---:|
| E1 move + concrete purpose | 98.8% | 97.2% | 99.6% | 99.6% |
| **E2 named STRATEGIC principle** | **85.8%** | **68.7%** | 92.7% | 96.3% |
| **E2 principle IN THE TAKEAWAY** | **58.4%** | **42.1%** | 66.5% | 67.0% |
| E2 incl. tactical motif (fork/pin/hang…) | 94.2% | 87.7% | — | — |
| E3 board-specific reason (≥2 square refs) | 96.1% | 93.4% | 96.8% | 98.0% |
| **E4 "how to find it" present** | **100%** | 100% | 100% | 100% |
| E4 method is a *genuine reusable routine* | 98.7% | — | — | — |
| E6 no engine-speak | 100% | 100% | 100% | 100% |
| takeaway present | 100% | 100% | 100% | 100% |
| **ALL 6 together** | **81.9%** | — | — | — |
| median words | 175 | 140 | 177 | 208 |

**Read-out.**
- **E4 is saturated by construction** — `render_assistant_target_v2` always appends
  `How to find it: <method>`, and 98.7% of methods are genuine checklists (median
  405 chars). So #4's eval weakness is **not data absence**; it is (a) the small
  model failing to *reproduce* the routine, and (b) the routines being
  **formulaic** ("run this checklist: first… then…" repeated), which hurts
  generalization. Lever = *diversity + brevity + tie-to-principle*, not "add more."
- **E2 is the real weak axis, worst at beginner** — only 68.7% of beginner rows
  name a strategic principle and only **42.1% put a principle in the takeaway**
  (the most memorable line). ~12% of beginner rows name *no* principle at all
  (strategic or tactical); the rest lean on procedural narration ("does two jobs
  at once") without crystallizing a named, reusable principle.
- E1/E3/E6/takeaway are effectively solved.

## 2. Tier-appropriateness of the LABELS (the moat signal, per prompt-parsed Maia/pool)

| tier | rec == engine-best | rec == top human (Maia) | median words |
|---|---:|---:|---:|
| beginner | 13.2% | 76.7% | 140 |
| intermediate | 43.0% | 54.0% | 177 |
| advanced | 95.0% | 20.7% | 208 |

The **labels are correctly directed** (beginner picks the human move, advanced the
sharp move). The eval's beginner tier-fit regression (v2 47.9% → v3 29.6%) is the
**32B reverting to its engine-best prior at inference**, *not* bad labels — so the
v5 lever is a **stronger within-position contrastive signal**, not relabeling.

## 3. Tier coherence (positions taught at ≥2 tiers)

train_v4: 2,301 boards, 2,289 multi-tier, **2,177 taught at all 3 tiers**;
**0 intra-(board,tier) inconsistency** (labels are deterministic).

| pattern across the 3 tiers | count | % of all-3 |
|---|---:|---:|
| B=I ≠ A (human/blend share the move, advanced sharper) | 844 | 38.8% |
| I=A ≠ B (beginner differs, int+adv sharp) | 723 | 33.2% |
| full gradient B≠I≠A | 316 | 14.5% |
| **all-same B=I=A** | **239** | **11.0%** |
| **COLLAPSE B=A ≠ I** (the pathological one) | **55** | **2.5%** |

**Diagnosis.**
- **~86% of all-3 boards differentiate correctly** (B=I≠A and I=A≠B are *valid*
  gradients, not defects).
- **all-same (11%)** is mostly **benign convergence** — the most human move *is*
  the engine-best move, so every tier correctly lands on it. (All 239 have pool≥2,
  but that only means differentiation was *possible*, not *pedagogically right*.)
  It is not a bug to fix per-board; it just dilutes the "tier changes the move"
  gradient, so **down-weight** these in the mix.
- **COLLAPSE B=A≠I (2.5%, 55 boards) is a real defect, and its cause is
  mechanical:** in `select_tier_move`, when the engine-best move is *also* the
  top-Maia move, **beginner (pure Maia) and advanced (pure eval) both land on it**,
  while the **intermediate 50/50 blend** can maximize on a *different* "compromise"
  move that wins neither pure axis. Example (train_v4):
  `pool [Nxg3, Bxc5, Qc8]` → beginner=Nxg3, advanced=Nxg3, **intermediate=Bxc5**.
  Fix in `select_tier_move` (below), not by hand.

## 4. Fixable target-text artifacts (no teacher re-spend — a render/regex fix + rebuild)

| artifact | rows | % |
|---|---:|---:|
| recommended SAN repeated ≥4× | 2,179 | 32.2% |
| **dangling leading connector** ("I'd play d5. — and in fact…") | 647 | 9.6% |
| **restated move** ("I'd play Ke2. THE MOVE: Play Ke2." / "The move is Qxd1.") | 407 | 6.0% |

`_strip_leading_move_restatement` only strips `Play X` / `Consider X` / `I'd play X`;
it misses `THE MOVE:`, `The move is X`, `This is the move`, and leaves a **dangling
dash** when it strips a restatement mid-sentence. ~**15.6%** of rows carry a
leading artifact — this is almost certainly a chunk of the eval's reported
"~4–5% malformed leading-garble outputs," taught straight from the data.

## 5. Tier calibration (E5) + correctness

- **Beginner forbidden-vocab leakage = 66 rows, and it is ENTIRELY the word
  "tempo."** No prophylaxis/outpost/zugzwang etc. leaked. A trivial scrub.
- Ply-cap violations: **0** across all tiers (already gated).
- **"Trade when behind" is NOT present in the data at scale.** Of 511 rows
  mentioning trade/exchange/simplify, the 15 that advocate a trade while the side
  to move is worse are all **sound recaptures** (Rxh1, Bxc6, Qxd6…), where the
  negative eval predates the trade. The anti-heuristic to reject lives in the
  *principle library / commentary*, not the examples — handled in
  `principle_library_v5.md` §D.

## 6. Small v5 A/B proof (sanctioned, sparing TFY — 4 calls of `openai-group/gpt-5.5`, $0.03)

Reused **identical grounded prompts** from train_v4; only upgraded the teacher
instruction to the v5 spec (named principle in takeaway; ≤2-sentence reusable
method; clean lead; beginner vocab cap). Same board `r..qk..r / pp..bppp / ...`:

| tier | canonical move | OLD | NEW |
|---|---|---|---|
| beginner | **Ne4** | 147w, principle-in-takeaway ✓ | 103w, **clean**, contrastive |
| intermediate | **Rb8** | 182w, principle-in-takeaway ✗ | 122w, **clean**, principle ✓ |
| advanced | **O-O** | 204w, principle-in-takeaway ✗ | **99w**, **clean**, principle ✓ |

Wins: **artifacts eliminated, length cut ~40%, a genuine contrastive triad**
(different tier-appropriate move + explanation at each level). Key finding:
**even with an explicit instruction, gpt-5.5 put a keyword-detectable named
principle in the takeaway only 2/4 times** → prompting alone is insufficient; v5
needs a **deterministic takeaway-principle gate** (below).

---

## 7. v5 curation & generation plan (data-first; prefer improving DATA over hparams)

**Shape:** v5 = v4 gates + these upgrades, re-derived from `candidates_v3.jsonl`
where free, with a **small, targeted teacher top-up** only for slices that need
new text. No full ~7k re-spend.

### Tier 0 — free re-render fixes (rebuild targets, no teacher calls)
1. **Fix `_strip_leading_move_restatement`** to also strip `THE MOVE:`,
   `The move is X`, `This is the move`, `Play X.`/`Consider X.` restatements AND
   any dangling leading connector (`— and/but/so/then/in fact`, `, and`). Target
   artifacts 15.6% → **<1%**.
2. **Beginner vocab scrub:** substitute "tempo/with tempo" → "gain time / for
   free" in beginner targets (66 rows). Target leakage → **0**.
3. **Down-weight non-differentiating boards** (all-same B=I=A, 11%) and
   **up-weight genuinely contrastive boards** in the train mix, to strengthen the
   "tier changes the move" gradient (weak axis A) without fabricating differences.

### Tier 1 — `select_tier_move` fix (re-derive labels; regenerate only changed text)
4. **Kill the B=A≠I blend artifact:** when `beginner_pick == advanced_pick`, force
   `intermediate_pick` to that same move (a position whose best move is also the
   most human should be taught consistently). Removes the 55 collapses.
   Only the intermediate rows whose move changes need coaching regeneration
   (~tens–low-hundreds of calls).

### Tier 2 — named-principle + method gates (the E2/E4 quality levers)
5. **Deterministic takeaway-principle GATE:** every kept row's takeaway must name
   a principle from the controlled vocabulary (`principle_library_v5.md` §A/§B).
   Rows failing → regenerate the takeaway (cheap, targeted). Target
   E2-in-takeaway: beginner **42% → ≥85%**, overall **58% → ≥90%**.
6. **Method-diversity + brevity gate:** enforce ≤2-sentence methods and rotate
   ≥5 distinct routine shapes (not the single "run this checklist: first…then…"),
   and **tie the method to the named principle** so #2 and #4 reinforce. Protects
   the 98.7% floor while de-formulaizing it (helps the model generalize #4).

### Tier 3 — beginner tier-fit boost (weak axis A, the big one)
7. **Add NEW beginner-discriminating positions** (most-human sound move ≠
   engine-best AND pedagogically distinct) rather than duplicating (v4 oversampled
   2×; duplication risks memorization). Target: raise the beginner-discriminating
   share by ~50%.
8. **Explicit contrastive triads** (same position, all 3 tiers) where B/I/A moves
   genuinely differ, and the coaching **names the contrast** ("a stronger player
   might go for the sharper O-O; at your level Ne4 is just as sound and easier to
   handle"). Proven feasible in §6.

### Tier 4 — truthfulness (weak axis C)
9. **Apply the wide-checker → LLM-judge exclusion** (build_v4 already *stages*
   wide-flagged kept rows; v5 should actually run the judge and pass
   `--exclude-ids`). Drop judge-confirmed semantic fabrications.
10. **Swap `principles.md` → `principle_library_v5.md`** so the teacher stops
    parroting wrong heuristics ("trade when behind", "passed pawns must be
    pushed", "space is always good"), improving the semantic-truth residual.

### Targeted teacher top-up (sanctioned, sparing)
Only regenerate: takeaway-gate failures + coherence-changed intermediate rows +
new beginner contrastive triads. Estimate **~1–2k calls** at medium effort ≈
**$8–15** (demo measured $0.0075/call), vs v3's $141 full run.

### Validate BEFORE training (cheap gate — prefer data over hparams)
Re-run `audit_v5_instructiveness.py` on v5 and require:
- E2-in-takeaway ≥ 90% overall, ≥ 85% beginner
- artifacts (dangling+restate) < 1%; beginner forbidden vocab = 0
- B=A≠I collapse = 0; beginner-discriminating share ≥ target
- method routines ≥ 5 distinct shapes; median words ≤ v4

Only then spend on Modal QLoRA. Downstream targets: beginner tier-fit back toward
/ above **47%**, instructiveness rank improved, truthfulness residual narrowed.

## 8. Priority order (impact × cost)
1. Tier 0 artifact + vocab re-render (near-zero cost, fixes ~15.6% + all vocab). 
2. Tier 2 takeaway-principle gate (biggest instructiveness lever; small top-up).
3. Tier 3 beginner contrastive triads (the moat's weak tier).
4. Tier 1 `select_tier_move` fix (removes structural collapse).
5. Tier 4 truthfulness (principle swap is free; judge-exclusion is small).
