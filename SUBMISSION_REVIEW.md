# Submission-Readiness Review — chess-instructor-llm

**Reviewed:** 2026-07-06 · **Brief:** *Train Your Own Small Learning Model* (find a gap where a frontier
model is unreliable → build a dataset → fine-tune a small model → **prove the behavior moved**).
**Deliverables:** dataset on HF · fine-tuned model on HF + running demo · eval harness (base-vs-tuned +
benchmark) · BrainLift · 3–5 min demo video · public repo.

**Verdict:** **~90% there.** The hard, easy-to-fail parts are genuinely done and done *honestly* — the
gap is measured before it's filled, and the behavior-moved proof is cross-family, blinded, held-out, with
a written pass bar and reported losses. What's left is **packaging/consistency drift**, not missing
substance: the live demo + committed launcher still point at **v1** while every doc says **v2**, the
BrainLift shipped inside the public repo is a **stale older version**, and the **demo video isn't recorded
yet**. All are fixable in well under a day.

> Verification done for this review: HTTP-checked all 5 live artifacts (200 OK), read both HF cards
> (model + move-review dataset are v2), queried the HF Space runtime (`static`, **RUNNING**), inspected
> both git repos, and read the running-server terminal metadata. I did **not** touch
> `data/benchmark_gap803/` or `RESULTS_FULL_EVAL_803.md`, and did not restart/kill any server.

---

## Submission-readiness checklist

| # | Deliverable | Status | Concrete artifact (path / URL) | One-line quality note |
|---|---|:---:|---|---|
| 1 | **Dataset on HF** | ✅ **DONE** | Training set 🤗 [`datasets/khoilamalphaai/chess-coach-move-review`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-move-review) (v2 default + v1 config, 4,034 rows, updated ~2h ago) · Benchmark 🤗 [`datasets/khoilamalphaai/chess-coach-benchmark`](https://huggingface.co/datasets/khoilamalphaai/chess-coach-benchmark) (~5.8k rows) | Excellent, honest, versioned cards; the *primary* dataset deliverable (the SFT set) is published — but README/SUBMISSION undersell it (see Gap 5). |
| 2a | **Fine-tuned model on HF** | ✅ **DONE** | 🤗 [`khoilamalphaai/qwen3-1.7b-chess-coach-mlx`](https://huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx) (QLoRA→merged→4-bit MLX, ~968 MB) | Model card is v2, with base-vs-tuned + v1→v2 + frontier tables and honest caveats. Strong. |
| 2b | **Running demo** | ⚠️ **PARTIAL** | Local "Analysis Room" `./run_platform.sh` (`src/api/server.py` :8000 + `web/` :3000) · HF Space 🤗 [`spaces/khoilamalphaai/chess-coach-benchmark`](https://huggingface.co/spaces/khoilamalphaai/chess-coach-benchmark) (RUNNING, `static`) | **Two problems:** the live local backend + the *committed* launcher default both serve **v1**, not v2 (Gap 1); and the HF "demo" is a **static results dashboard**, not an interactive model demo (the interactive coach is local-only). |
| 3a | **Eval harness (base-vs-tuned)** | ✅ **DONE** | `src/eval/evaluate.py` → [`RESULTS.md`](RESULTS.md) / [`RESULTS_V2.md`](RESULTS_V2.md); protocol + pass bar in [`docs/EVAL_AND_ITERATE.md`](docs/EVAL_AND_ITERATE.md) | Cross-family (GPT-5.5 teacher → Claude judge), held-out, deterministic objective checks + rubric. Re-runnable, one command. |
| 3b | **Benchmark (ours vs frontier)** | ✅ **DONE** | `src/eval/benchmark/` + `scripts/frontier_gap*.py` + `scripts/divergence_*.py` → [`RESULTS_BENCHMARK_v2.md`](RESULTS_BENCHMARK_v2.md), `data/analysis/GAP_REPORT.md`, `DIVERGENCE_REPORT.md`; bonus 9-open-model run `RESULTS_OPEN_MODELS.md` | Blinded 5-model council, grounded/ungrounded, self-preference measured (+0.43). Gap proven *first*. The strong open-model extension is **untracked** (Gap 4). |
| 4 | **BrainLift** | ⚠️ **PARTIAL** | Newer (480-line) 🡒 [`../brainlifts/chess-coach-behavior-thesis/brainlift.md`](../brainlifts/chess-coach-behavior-thesis/brainlift.md) · **In-repo/committed** (389-line, older) `brainlift/brainlift.md` | Content is excellent (DOK-4/3/2, 15 SPOVs, ~120 sources, primary-source-tied). **But** the version *in the public repo is stale*, and README/SUBMISSION links point outside the repo — broken in a clone (Gap 2). |
| 5 | **Demo video (3–5 min)** | ❌ **MISSING** | Script only: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) (well-built ~4:30 shot list). No recording, no link in any doc. | The one hard deliverable not produced. Must record after the v2 relaunch (Gap 3). |
| 6 | **Public repo** | ⚠️ **PARTIAL** | 🡒 [`github.com/kimkhoi2202/chess-instructor-llm`](https://github.com/kimkhoi2202/chess-instructor-llm) (nested repo, 2 commits) | Live with core v2 docs committed, but **behind actual work**: v1-default launcher, stale brainlift, and untracked v2 evidence / Space code / harness+UI edits (Gaps 1, 2, 4). |

**Legend:** ✅ DONE · ⚠️ PARTIAL · ❌ MISSING.

### Does it satisfy the brief's core emphasis?

- **"Prove the gap exists" — ✅ strong.** `GAP_REPORT.md` measures the frontier (GPT-5.5 / Claude Opus 4.8 /
  Gemini 3.1 Pro) on the narrow behavior with byte-identical grounding, held-out: tier-differentiation
  **22.7%**, engine-mirroring at every tier **68.7%**, beginner findable-pick **20.5%**. The gap is real and
  measured *before* any claim to fill it. It even reports the honest counter-finding (the frontier is
  *good* at truthfulness, ~3%).
- **"Prove the behavior moved" — ✅ strong on the controllable axes, honestly partial elsewhere.**
  Base→tuned (cross-family, held-out): no-engine-speak **33%→100%**, move-sound **87%→100%**, ply-cap
  **67%→100%**, spec-adherence **0.47→0.93**, level-calibration **0.60→1.13**. v1→v2 (matched, grounded):
  fabrication **50%→33%**, tier-differentiation **27.5%→39.2%** with the *direction corrected*, council rank
  **4.13→3.68**. The submission does *not* overclaim — it says plainly it doesn't out-teach the frontier and
  that truthfulness is carried by grounding + the verifier, not the weights. This honesty is a grading asset,
  not a liability.
- **Net:** the intellectual core (gap → dataset → fine-tune → measured move, with a fixed re-runnable
  yardstick) is complete and credible. The risks below are about *what a reviewer actually sees* — the demo,
  the repo, and the video — not about the science.

---

## Gaps & risks to fix before submission (ranked by importance)

### 1. [HIGH] The demo runs **v1**, but the whole submission says **v2** is shipped
- **Evidence:** the running backend was launched with `COACH_MODEL_PATH=models/mlx/chess-coach-v1` (terminals
  `123369–123374`, live uvicorn pid `82651`), and the **committed** `run_platform.sh` still defaults to
  `models/mlx/chess-coach-v1`. Meanwhile README, `SUBMISSION.md`, the HF cards, and `docs/DEMO_SCRIPT.md`
  all say the platform "serves `chess-coach-v2`." A reviewer who clones the repo and runs `./run_platform.sh`
  — or a demo recorded against the current server — gets the **older model**, silently contradicting every v2
  claim and the "Tuned coach (v2)" badge the demo script tells you to confirm.
- **Fix:** (a) commit the already-made local edit flipping the default to `chess-coach-v2` and push; (b) before
  recording, relaunch the platform on v2 (`COACH_MODEL_PATH=models/mlx/chess-coach-v2 ./run_platform.sh`) and
  confirm the backend log prints `chess-coach-v2`. *(I did not restart the servers per your constraint — this
  relaunch is yours to do.)*

### 2. [HIGH] The BrainLift **in the public repo is the stale version**, and the doc links break in a clone
- **Evidence:** `brainlift/brainlift.md` (committed, on GitHub) is the **older 389-line** draft ("*Grounding
  Carries Dependability…*"). The current **480-line** BrainLift ("*Faithfulness Is Table-Stakes, the Moat Is
  Tier-Appropriate Move Selection*" — 15 SPOVs, open-model + capacity + verifier-gate evidence) lives at
  `../brainlifts/chess-coach-behavior-thesis/brainlift.md`, **outside** the `chess-instructor-llm` repo.
  README:75 and SUBMISSION.md:32 both link to `../brainlifts/...`, which **does not exist** in a clone of
  `kimkhoi2202/chess-instructor-llm` → broken link, and the reviewer instead finds the stale copy.
- **Fix:** copy the current 480-line BrainLift over `brainlift/brainlift.md` (and its supporting `01*/02*/03*/04*`
  files if you want the full trail), repoint README/SUBMISSION links to the in-repo path (`brainlift/brainlift.md`),
  commit + push. Verify the title in the repo matches "*…the Moat Is Tier-Appropriate Move Selection*."

### 3. [HIGH] Demo video not recorded (hard deliverable)
- **Evidence:** only `docs/DEMO_SCRIPT.md` exists; no video file in the repo and no video link in README/SUBMISSION.
- **Fix:** after fixes 1–2, record the ~4:30 walkthrough per the script (hook → live coach + tier toggle →
  base-vs-tuned + benchmark → HF artifacts → honest v2 arc), upload (unlisted YouTube/Loom is fine), and add the
  link to README + SUBMISSION deliverable #5. Use a position from `web/public/library_differentiated.json` so the
  Beginner→Advanced toggle visibly changes the move on camera.

### 4. [MEDIUM] The public repo is **behind the actual work** (strong evidence is untracked)
- **Evidence (untracked / uncommitted):** `RESULTS_OPEN_MODELS.md` (the 9-bigger-open-model benchmark — genuinely
  strong supporting evidence), `scripts/space/` (the HF Space source), `data/benchmark_open/`, plus uncommitted
  edits to `src/eval/benchmark/config.py`, `src/eval/benchmark/council.py`, and `web/src/components/Studio.tsx`.
  Only 2 commits exist; the v2 core docs are committed, but this newer layer is not.
- **Fix:** commit + push the intended-public subset (at minimum `RESULTS_OPEN_MODELS.md`, `scripts/space/`, and the
  benchmark/UI edits), respecting `.gitignore` for large `data/` blobs. Confirm the pushed repo matches what the
  video and SUBMISSION describe.

### 5. [MEDIUM] README / SUBMISSION **undersell the published training dataset**
- **Evidence:** README "Live artifacts" (lines 14–18) lists only the *benchmark* dataset; `SUBMISSION.md`
  deliverable **1b** calls the training set "Local … *(gitignored)*." In reality the SFT set — the brief's "real
  deliverable" — **is published** as `chess-coach-move-review` (v2 default, versioned, updated ~2h ago). The docs
  make the strongest artifact look local-only.
- **Fix:** add `chess-coach-move-review` to README "Live artifacts," and change SUBMISSION deliverable 1/1b to
  "✅ Published (v2)" with the URL. (This is an easy *upgrade* — you already shipped it.)

### 6. [LOW] Numeric / labeling / metadata nits (quick copy-edits)
- **Fabrication 33% vs 38%:** headline v2 fabrication is **33%** (`RESULTS_BENCHMARK_v2.md`) but the newer,
  slightly stricter verifier reads **38%** (`RESULTS_OPEN_MODELS.md`). This *is* disclosed (open-models note +
  brainlift "33–38%"), so it's fine — just make sure the video/README say "~33–38% depending on verifier
  strictness" rather than a bare 33% if you cite the open-model table.
- **Typo:** `RESULTS_BENCHMARK_v2.md` Setup row labels OURS as `chess-coach-v1` (should be v2); tables below are
  correct.
- **License mismatch:** model card tag is `apache-2.0` while the dataset card states `CC-BY-NC-4.0`. Pick one story
  (the dataset is distilled from GPT-5.5 + Lichess, so NC is the safer claim) and make the model card consistent.
- **HF display quirk:** the model shows "0.3B params" (HF misreads packed 4-bit MLX U32 weights). Harmless; a one-line
  note in the card avoids a reviewer thinking it's not a 1.7B.

### 7. [LOW / OPTIONAL] "Running demo" on HF is a dashboard, not an interactive model
- **Evidence:** the HF Space is a `static` benchmark viewer; the only *interactive* coach is the local platform.
  If the grader reads "running demo" strictly as a hosted, clickable model, the static dashboard may not fully
  satisfy it.
- **Fix (optional, only if time):** either (a) make the video unambiguously carry the "interactive demo"
  requirement (clearly show the live local coach + tier toggle), or (b) stand up a small Gradio Space that calls the
  model for a truly hosted interactive demo. (a) is sufficient for most rubrics.

---

## Fix order (fastest path to "record-ready")
1. Commit `run_platform.sh` v1→v2 default; relaunch platform on v2; confirm badge/log = v2. *(Gap 1)*
2. Copy the current BrainLift into `brainlift/brainlift.md`; fix README/SUBMISSION links to the in-repo path. *(Gap 2)*
3. Add `chess-coach-move-review` to README "Live artifacts" + SUBMISSION 1b; fix the 3 low-risk nits. *(Gaps 5, 6)*
4. Commit + push `RESULTS_OPEN_MODELS.md`, `scripts/space/`, harness/UI edits. *(Gap 4)*
5. Record the ~4:30 video; add the link to README + SUBMISSION #5. *(Gap 3)*
