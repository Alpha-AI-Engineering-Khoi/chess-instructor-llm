# Version Comparison (scope-labeled)

A single, honest map of every chess-coach model version and exactly which eval each
number comes from. The one rule that makes this document trustworthy: **numbers live
in different eval scopes and must not be compared across scopes.** Every figure below
is tagged with its scope and its source doc. Where a cross-scope confusion is likely,
there is a one-line caution.

If you read nothing else, read the scope key next, then the caution box at the end.

---

## Scope key (define once, tag everywhere)

Each number in this doc carries one of these scope tags. They differ on which
positions, which label era, and which grounding was used, so a number is only
comparable to other numbers with the SAME tag.

| Tag | Scope name | What it is | Label era / grounding | Comparable to | Source |
|---|---|---|---|---|---|
| **(a)** | 803 re-score field | Full 803 gap positions x 3 tiers (2409 scenarios; the 3 frontier refs on a 150-subset x 3 = 450). OLD cached generations, re-scored against corrected v6 labels with the vendored extractor. No model was re-run. | old (v4-era) generations judged by corrected v6 labels | Only other (a) numbers, and only as a RELATIVE / ranking read, not a ceiling | `RESULTS_FULL_EVAL_803.md` |
| **(b)** | corrected 120 TEST | 120 held-out TEST positions x 3 tiers = 360 scenarios. FRESH generations against the corrected v6 grounding, one controlled Modal session, grounded. | corrected v6 labels, fresh corrected grounding | Only other (b) numbers | `RESULTS_STAGE4_CORRECTED.md` |
| **(c)** | matched-frontier 120 | Same 360 grounded prompts as (b); OURS reused, the 3 frontier coaches generated fresh on the same prompts (TrueFoundry gateway). | corrected v6 labels, fresh corrected grounding | Only other (c) numbers | `data/benchmark_gap803/stage4_frontier/scores.json` (commit 9c64749) |
| **(d)** | v4-era 120-val | 120 held-out VAL positions x 3 tiers = 360. The original (pre-correction) benchmark. Home of the shipped v4 0.767 headline, the prompt-control, and the head-to-head. | old (v4-era) labels | Only other (d) numbers | `RESULTS_HONEST_EVAL_V4.md`, `RESULTS_PROMPT_CONTROL.md` |
| **(e)** | no-grounding | 120 TEST x 3, distillation prompt with NO engine and NO Maia. Tests behavior-in-weights. | corrected v6 labels, no grounding | Only other (e) numbers | `RESULTS_STAGE4_CORRECTED.md` |

Two auxiliary sets that are not tier-policy scopes but appear in the matrix:

- **adversarial**: 54 attack cases across five categories, two tracks (live deployed
  endpoint and raw base-vs-v4). Source: `RESULTS_ADVERSARIAL.md`.
- **prompt-control**: 360 scenarios (the (d) 120-val x 3), same-backend base weights
  with a spec-exact system prompt vs v4. Source: `RESULTS_PROMPT_CONTROL.md`.

Note on the 120 positions: the (b)/(c) corrected 120 TEST and the (d) v4-era 120-val
are the SAME held-out FENs (0 FEN changes across the v6 rebuild); only the labels and
grounding era differ. That is precisely why the same model shows a different number in
(b) vs (d) and they are NOT comparable (`RESULTS_STAGE4_CORRECTED.md`).

Metric names used below: **tier-policy match** = exact agreement with the preregistered
`select_tier_move` canonical move, averaged over the 3 tiers (agreement with a project
rule, not validated pedagogy). **distinct** = of positions whose canonical beginner and
advanced differ, the share where the model's beginner and advanced picks also differ.
**tier-diff** (v1/v2 era only) = picks at least one different move across the 3 tiers.

---

## 1. Lineage

Two untuned base models and the versions trained from them. "What changed" is relative
to the prior version in the same line.

| Version | Base model + size | What changed vs prior | Status |
|---|---|---|---|
| **BASE-1.7B** | Qwen3-1.7B-4bit (untuned) | n/a (starting point of the 1.7B line) | untuned baseline |
| **BASE-32B** | Qwen3-32B (untuned) | n/a (starting point of the 32B line) | untuned baseline |
| **v1** | Qwen3-1.7B QLoRA | First tune. GPT-5.5 teacher labels filtered for format/soundness but NOT faithfulness (6.3% false labels); no contrastive tiers. Fixed style/soundness/jargon; truthfulness stayed flat. | superseded (research) |
| **v2** | Qwen3-1.7B QLoRA | Data intervention: faithfulness reject-gate (0% false labels), deterministic tier-aware teacher (`tier_select`), 348 contrastive multi-tier FENs. Moved fabrication and tier-differentiation in the right direction. | shipped then superseded (was the live 1.7B coach) |
| **v3** | Qwen3-32B QLoRA (SFT) | The capacity bet: 20x larger base + larger cleaner contrastive set (7,128 rows, 0% false labels), LoRA r=32, 2 epochs. Best local instructiveness; field-leading moat; tripped the strict formatting gate. | superseded |
| **v4** | Qwen3-32B QLoRA (SFT) | The shipped 32B coach and the model the whole evaluation centers on; strongest on the move axis (leads the grand field), weakest recent on prose. Same 32B base as v3. | shipped; evaluation baseline-of-record (live serving later handed to v6-dpo2) |
| **v5** | Qwen3-32B QLoRA (SFT) | Faithfulness-filtered retrain that regressed under several tangled changes at once (about 27% less optimization, broken contrastive triads, about 42% boilerplate-principle pollution, retrained from base not from v4, no checkpoint selection). Isolates nothing. | confounded (abandoned) |
| **v6-dpo** | DPO on v4 | Preference tuning on tier-move pairs on top of shipped v4. Sharpened the moat without regressing; gain entirely in the intermediate tier, out of distribution. | superseded by v6-dpo2 |
| **v6-dpo2** | DPO on v4 | Stronger, tier-targeted preference pairs (checkpoint step 200). Best DPO variant; deepest mid-tier moat; beginner/advanced byte-identical to v4. | live-served coach (current) |
| **v6-distill** | Engine-distillation adapter (Qwen3-32B) | Distills the tier rule into the weights with NO engine/Maia grounding at inference. A behavior-in-weights proof, not a deployable coach (grounding-free soundness 0.653 below the grounded 0.98). | research |

Adapter repos: `khoilamalphaai/chess-coach-32b-v4-qlora`,
`chess-coach-32b-v6-dpo`, `chess-coach-32b-v6-dpo2`, `chess-coach-32b-v6-distill`
(`RESULTS_STAGE4_CORRECTED.md`).

Side ablation (not in the numbered lineage): a Qwen3-4B QLoRA tune ("OURS-4B") exists
as a capacity control; on scope (d) it reads tier-policy match 0.397 vs its base 0.353
and a prompted base 0.378, and instructiveness 5.32 (`RESULTS_HONEST_EVAL_V4.md`). It
supports "learnability tracks the data, not size" (the 1.7B tune 0.578 beats it) and is
otherwise out of scope here.

---

## 2. Stats per version (every number scope-labeled)

### Base models (untuned)

**BASE-1.7B (Qwen3-1.7B, untuned)**
- tier-policy match **0.358** [scope (a), 803 field, corrected labels] (`RESULTS_FULL_EVAL_803.md`); this is the base the 1.7B tune improves on.
- tier-policy match **0.358** [scope (d), v4-era 120-val, 20-model grand slice] (`RESULTS_HONEST_EVAL_V4.md`, `BRAINLIFT.md`).
- No no-grounding number was measured for the 1.7B base (scope (e) covers the 32B base only).

**BASE-32B (Qwen3-32B, untuned)**
- tier-policy match **0.428** (B/I/A 0.442 / 0.408 / 0.433), move-sound 0.969, distinct 0.303 [scope (b), corrected 120 TEST, grounded] (`RESULTS_STAGE4_CORRECTED.md`).
- tier-policy match **0.428** (same grounded run reused in the frontier panel) [scope (c), matched-frontier 120] (`data/benchmark_gap803/stage4_frontier/scores.json`).
- tier-policy match **0.347** [scope (d), v4-era 120-val, aws-bedrock backend, shipped-coach prompt] (`RESULTS_PROMPT_CONTROL.md`, `RESULTS_HONEST_EVAL_V4.md`).
- tier-policy match **0.300** as "Qwen3-32B (untuned v3 base)" [scope (a), 803 field] (`RESULTS_FULL_EVAL_803.md`).
- tier-policy match **0.022**, names-a-move 0.250, move-sound 0.081 [scope (e), no-grounding] (`RESULTS_STAGE4_CORRECTED.md`).
- Caution: the 32B base reads 0.428 (b/c), 0.347 (d), 0.300 (a), and 0.022 (e). All four are the untuned 32B; none is comparable to another. The spread is the label era, the grounding, and the position set, not the model changing.

### v1 (Qwen3-1.7B QLoRA)

- No tier-policy-match number exists (v1 predates the 803/120 tier-policy benchmarks).
- Evaluated only on a 15-scenario base->v1 check: move-sound 87% -> 100%, no-engine-speak 33% -> 100%, ply-cap 67% -> 100%; Claude-rubric truthfulness flat 0.13 -> 0.13 (`FINDINGS.md` section 2a, `RESULTS.md`).
- tier-differentiation **27.5%** [120 matched held-out, v4-era divergence harness] (`FINDINGS.md` section 2b, `RESULTS_V2.md`).

### v2 (Qwen3-1.7B QLoRA, shipped 1.7B)

- tier-policy match **0.509** (B/I/A 0.557 / 0.532 / 0.438), move-sound 0.898, distinct 0.477 [scope (a), 803 field, corrected labels] - #1 in the whole field, +0.042 over the best frontier (`RESULTS_FULL_EVAL_803.md`).
- tier-fit **53%** [historical 803 gap803_report pipeline, old labels] and **0.633** [scope (a) same field under the vendored extractor, old labels] (`RESULTS_FULL_EVAL_803.md`); use 0.509 as the current corrected-field number.
- tier-differentiation **39.2%** (from v1 27.5%), grounded fabrication 33% [v2-era grounded benchmark] (`RESULTS_V2.md`, `FINDINGS.md`).
- Learnability figure quoted elsewhere: the 1.7B tune **0.578** (from base 0.358), #2 of 20, above every frontier [scope (d), 20-model grand slice] (`RESULTS_HONEST_EVAL_V4.md`, `BRAINLIFT.md`).
- Caution: v2 at 0.509 [scope (a)] is NOT comparable to v6-dpo2 at 0.892 [scope (b)]. Different model size, different position set (803 vs 120), different label era, old cached gens vs fresh grounding. A "1.7B v2 50.9% vs 32B v6-dpo2 89.2%" line would be a scope error.

### v3 (Qwen3-32B QLoRA, SFT)

- tier-policy match **0.463** (B/I/A 0.463 / 0.441 / 0.484), move-sound 0.867, distinct 0.509 [scope (a), 803 field, corrected labels] - slips to #3 as Claude edges past under the correction (`RESULTS_FULL_EVAL_803.md`).
- tier-fit **53%** (advanced 84%, beginner 30%), corrected council instructiveness rank **6.93** of 15 (best local), balanced score 58.0 (1st raw, trips the strict formatting gate) [historical 803 gap803_report / grand eval, old labels] (`RESULTS_V3.md`, `RESULTS_FULL_EVAL_803.md`, `FINDINGS.md`).
- tier-policy match **0.558**, instructiveness 6.35 [scope (d), v4-era 120-val leaderboard] (`RESULTS_HONEST_EVAL_V4.md`).

### v4 (Qwen3-32B QLoRA, shipped, evaluation baseline)

- tier-policy match **0.767** raw (**0.789** as served through the shipped gate), distinct 0.730, move-sound raw 0.942; instructiveness 4.67 [scope (d), v4-era 120-val] (`RESULTS_HONEST_EVAL_V4.md`).
- Head-to-head vs best frontier: **56-24-12** over the 92 diverging positions (56-24-40 over all 120); the 51-5-6 over 62 is the v4-success-conditioned subset, NOT a win rate [scope (d)] (`RESULTS_HONEST_EVAL_V4.md`, `FINDINGS.md`).
- tier-policy match **0.861** (B/I/A 0.858 / 0.750 / 0.975), move-sound 0.983, distinct 0.987, format 0.939 [scope (b), corrected 120 TEST, grounded] (`RESULTS_STAGE4_CORRECTED.md`).
- tier-policy match **0.861** [scope (c), matched-frontier 120] (`data/benchmark_gap803/stage4_frontier/scores.json`).
- Base-vs-tuned (the spec's core requirement): **v4 - base = +0.433** (0.861 vs 0.428) [scope (b)]; distinct +0.684 (`RESULTS_STAGE4_CORRECTED.md`).
- Not evaluated on scope (a) 803 field (see section 5).
- Caution: v4 = 0.767 [scope (d)] and v4 = 0.861 [scope (b)] are the SAME model on the SAME 120 FENs; the difference is old v4-era labels vs fresh corrected grounding. Do not read +0.094 as an improvement.

### v5 (Qwen3-32B QLoRA, SFT, confounded)

- tier-policy match **0.536**, raw faithfulness roughly flat (near 0.58) [grand-eval slice] - regressed under several tangled changes at once (about 27% less optimization, broken contrastive triads, about 42% boilerplate-principle pollution, retrained from base not v4, no checkpoint selection), so it isolates no single cause (`BRAINLIFT.md`, `docs/EVAL_REVIEW.md`).
- Treat 0.536 as a confounded data point, not a clean version result.

### v6-dpo (DPO on v4)

- tier-policy match **0.881** (B/I/A 0.858 / 0.808 / 0.975), move-sound 0.983, distinct 0.987, format 0.919 [scope (b), corrected 120 TEST, grounded] (`RESULTS_STAGE4_CORRECTED.md`).
- tier-policy match **0.881** [scope (c), matched-frontier 120] (`data/benchmark_gap803/stage4_frontier/scores.json`).
- Base-vs-tuned: **v6-dpo - base = +0.453** (0.881 vs 0.428) [scope (b)] (`RESULTS_STAGE4_CORRECTED.md`).
- Gain over v4 (+0.0195) is entirely the intermediate tier (0.808 vs 0.750); beginner, advanced, soundness, distinct identical to v4 [scope (b)].
- Instructiveness council (secondary): does not regress vs v4 (overlapping CIs; mean-rank advantage +0.104, CI straddles zero) [90-item stratified TEST sample, grounded] (`RESULTS_STAGE4_CORRECTED.md`).

### v6-dpo2 (DPO on v4, live-served)

- tier-policy match **0.892** (B/I/A 0.858 / 0.842 / 0.975), move-sound 0.983, distinct 0.987, names 0.986, format 0.925 [scope (b), corrected 120 TEST, grounded] (`RESULTS_STAGE4_CORRECTED.md`).
- tier-policy match **0.892** (89.2%) [scope (c), matched-frontier 120] (`data/benchmark_gap803/stage4_frontier/scores.json`, commit 9c64749).
- Best overall of the DPO ladder: +0.031 vs v4, +0.011 vs v6-dpo; the entire lift is the intermediate tier (0.842 vs v4 0.750); beginner (0.858) and advanced (0.975) byte-identical to v4 [scope (b)] (`RESULTS_STAGE4_CORRECTED.md`).
- Not evaluated on scope (a) 803 field (a Colab notebook exists to run it; see section 5).

### v6-distill (engine-distillation)

- tier-policy match **0.325** (B/I/A 0.358 / 0.400 / 0.217), names-a-move 0.983, move-sound 0.653, distinct 0.461 [scope (e), no-grounding] (`RESULTS_STAGE4_CORRECTED.md`).
- Recovers the tier rule from weights vs the base's collapse (base no-grounding 0.022 -> 0.325, about 15x); honest limit is the advanced tier (0.217, the sharpest move genuinely needs grounding); grounding-free soundness 0.653 is below the deployable grounded 0.98 [scope (e)].
- Only measured on scope (e); it is not a grounded coach, so (a)/(b)/(c) do not apply.

---

## 3. Matched-frontier panel (scope (c), 120 held-out TEST, fresh grounding)

The one panel where OURS and the 3 frontier coaches were scored on the SAME 360 grounded
prompts and corrected v6 labels (`data/benchmark_gap803/stage4_frontier/scores.json`,
commit 9c64749). Percentages match the live Benchmark Space panel.

| Model | tier-policy match | B / I / A | move-sound | distinct | names |
|---|---:|---|---:|---:|---:|
| OURS-v6-dpo2 | 0.892 (89.2%) | 0.858 / 0.842 / 0.975 | 0.983 | 0.987 | 0.986 |
| OURS-v6-dpo | 0.881 (88.1%) | 0.858 / 0.808 / 0.975 | 0.983 | 0.987 | 0.983 |
| OURS-v4 (shipped) | 0.861 (86.1%) | 0.858 / 0.750 / 0.975 | 0.983 | 0.987 | 0.983 |
| Claude Opus 4.8 | 0.614 (61.4%) | 0.558 / 0.575 / 0.708 | 1.000 | 0.303 | 1.000 |
| Gemini 3.1 Pro | 0.614 (61.4%) | 0.558 / 0.575 / 0.708 | 1.000 | 0.171 | 1.000 |
| GPT-5.5 | 0.578 (57.8%) | 0.583 / 0.533 / 0.617 | 1.000 | 0.224 | 1.000 |
| BASE (Qwen3-32B untuned) | 0.428 (42.8%) | 0.442 / 0.408 / 0.433 | 0.969 | 0.303 | 0.975 |

v6-dpo2 leads the best frontier by +27.8 points on tier-policy match. The frontier is
100% sound and 100% names-a-move but differentiates tiers on only 17-30% of positions vs
98.7% for OURS - the leveled-move moat. This is scope (c) only; do NOT line it up against
the scope (a) 803 field table (that field has the frontier at 0.404-0.467 on OLD cached
generations judged by corrected labels - a different scope entirely).

---

## 4. Eval-coverage matrix

Rows are versions and bases; columns are the six evals. "yes" = evaluated in that scope;
"no" = not evaluated. Scope tags map to section 1.

| Model | 803 field (a) | corrected-120 (b) | matched-frontier-120 (c) | no-grounding (e) | adversarial | prompt-control |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| BASE-1.7B | yes | no | no | no | no | no |
| BASE-32B | yes | yes | yes | yes | yes (raw track) | yes |
| v1 (1.7B) | no | no | no | no | no | no |
| v2 (1.7B) | yes | no | no | no | no | no |
| v3 (32B) | yes | no | no | no | no | no |
| v4 (32B) | no | yes | yes | no | yes (raw + deployed) | yes |
| v5 (32B) | no | no | no | no | no | no |
| v6-dpo (32B) | no | yes | yes | no | no | no |
| v6-dpo2 (32B) | no | yes | yes | no | no | no |
| v6-distill (32B) | no | no | no | yes | no | no |

Notes on the matrix:
- "prompt-control" is base-32B (spec-exact prompt) vs v4 on the (d) 120-val: base 0.428
  / 0.431 (optimized) vs v4 0.767, closing about 19% of the base-to-tune gap
  (`RESULTS_PROMPT_CONTROL.md`). It is a base-vs-v4 control, so only those two rows are
  "yes".
- "adversarial" is base (raw) and v4 (raw + deployed): deployed v4 held 52/54 with 0
  broke after the malformed-FEN fix; the fine-tune's clear win is injection resistance
  (raw base 4/12 broke, raw and deployed v4 0/12) (`RESULTS_ADVERSARIAL.md`).
- v1's only evaluation is the 15-scenario base->v1 rubric plus the v1/v2 divergence
  harness, neither of which is one of these six scopes, so all six are "no".

---

## 5. What is NOT fully evaluated, and why (compute / budget)

The single biggest coverage gap: **v4, v5, v6-dpo, v6-dpo2, and v6-distill were never
run on the full 803 field (scope (a)).** Only the two 1.7B tunes (v2), v3, and the bases
have full-803 tier-policy coverage, and even that is now a re-score of OLD generations,
not fresh grounding.

The reason is cost, stated plainly:

- The definitive **803 x 3 grand eval with the blinded cross-family council was a
  one-time run costing about $112.15** (open-model generation $28.17, frontier generation
  $21.71, council $62.27, local $0) (`RESULTS_FULL_EVAL_803.md` section 8). The council
  alone was $62.27 of that.
- That budget was spent while the lineage was at **v2 / v3**. Re-running the full 803 x 3
  generation for each later adapter (v4 -> v6) would repeat the gateway + GPU spend for a
  reference field whose ordering is already established, so later versions were instead
  scored on the **cheaper, representative 120 held-out TEST** (scope (b)/(c)) to conserve
  Modal / gateway credits. The adversarial and prompt-control runs were kept deliberately
  small for the same reason (about $3.45 and about $4.71 respectively;
  `RESULTS_ADVERSARIAL.md`, `RESULTS_PROMPT_CONTROL.md`).
- The corrected-label full-803 field table (scope (a)) was produced for **$0** as a
  cached re-score (no model re-run) precisely because re-generating the field was
  off-limits on cost (`RESULTS_FULL_EVAL_803.md`).
- To close the gap for the current live model without spending project credits, a
  resumable Colab notebook exists to run **v6-dpo2 on the full 803 on the user's own
  compute**: `notebooks/colab_803_eval_v6dpo2.ipynb` (with `notebooks/README_colab.md`
  and `scripts/precompute_grounded_prompts_803.py`; commit db1b1f0). It is present and
  tracked in git but has not yet been executed, so v6-dpo2 has no scope (a) number today.

Secondary gaps: v5 is confounded and was not carried onto the clean benchmarks; v1 has no
tier-policy-scope number at all; v6-distill is intentionally scope (e) only (it is a
behavior-in-weights proof, not a grounded coach); a corrected-label frontier re-score on
the full 803 (fresh grounding) was also skipped on cost, which is exactly why the
matched-frontier comparison lives on the 120 TEST (scope (c)) instead.

---

## 6. Cross-scope cautions (do not conflate)

- **v2 50.9% vs v6-dpo2 89.2% are NOT comparable.** v2 0.509 is scope (a) (1.7B, 803
  positions, old cached gens, corrected labels); v6-dpo2 0.892 is scope (b)/(c) (32B, 120
  fresh-grounded positions). Different size, position set, generation freshness, and label
  handling.
- **v4 0.767 vs v4 0.861 are the same model.** 0.767 is scope (d) (v4-era labels), 0.861
  is scope (b) (fresh corrected grounding) on the identical 120 FENs. The delta is the
  label/grounding era, not a model change.
- **Frontier 0.404-0.467 (scope a) vs frontier 0.578-0.614 (scope c) are different
  scopes.** Scope (a) is old cached frontier generations on 803 judged by corrected
  labels; scope (c) is fresh frontier generations on the 120 matched grounded prompts.
- **base-32B has four legitimate but non-comparable numbers**: 0.428 (b/c), 0.347 (d),
  0.300 (a), 0.022 (e). Always cite the scope.
- **803-field numbers are a ranking read, not a ceiling.** They are old generations
  scored by sharper labels, which lowers everyone's absolute exact-match; a fresh-grounding
  re-generation raises it (fresh v4 = 0.861 on 120 vs its 803-field would be lower)
  (`RESULTS_FULL_EVAL_803.md`).

---

## 7. Source index

| Source doc | What it anchors |
|---|---|
| `RESULTS_STAGE4_CORRECTED.md` | Scope (b) corrected 120 TEST (base/v4/v6-dpo/v6-dpo2 grounded; base/v6-distill no-grounding); base-vs-tuned +0.433 / +0.453; scope (e) |
| `RESULTS_FULL_EVAL_803.md` | Scope (a) 803 re-score field (v2 0.509, v3 0.463, bases, open, frontier); the $112.15 one-time cost |
| `data/benchmark_gap803/stage4_frontier/scores.json` (commit 9c64749) | Scope (c) matched-frontier 120 panel (exact per-model numbers) |
| `RESULTS_HONEST_EVAL_V4.md` | Scope (d) v4 0.767 / served 0.789; head-to-head 56-24-12; base-vs-tuned +0.433 / +0.453; grand-eval leaderboard |
| `RESULTS_PROMPT_CONTROL.md` | prompt-control 0.428 vs 0.767 (base-32B vs v4, same backend) |
| `RESULTS_ADVERSARIAL.md` | adversarial 52/54 held, injection resistance, malformed-FEN fix |
| `RESULTS_V2.md`, `RESULTS_V3.md`, `FINDINGS.md` | v1/v2/v3 lineage deltas, tier-differentiation, fabrication |
| `BRAINLIFT.md`, `docs/EVAL_REVIEW.md` | v5 confound (0.536); three-claim honest framing; talking points |
| `notebooks/colab_803_eval_v6dpo2.ipynb` (commit db1b1f0) | resumable full-803 v6-dpo2 eval on the user's own compute |

## 8. Verification note

Every figure above was checked against a committed source file or committed data artifact
in this repo; none is unverified or fabricated. Two provenance notes rather than
unverifiable claims: (1) v2's 803 number appears as 0.509 (current corrected re-score,
scope a), 0.633 (same field, old labels, vendored extractor), and 53% (historical
gap803_report pipeline) - all three are real and documented; this doc uses 0.509 as the
current figure. (2) v1 has no tier-policy-match number by design (it predates the tier
benchmarks); its only committed evaluation is the 15-scenario base->v1 rubric.
