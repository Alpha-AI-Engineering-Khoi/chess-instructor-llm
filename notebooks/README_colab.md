# Colab: full 803 x 3 evaluation of chess-coach v6-dpo2 (resumable)

`colab_803_eval_v6dpo2.ipynb` runs the **full 803-position gap benchmark x 3 tiers
(2409 scenarios)** evaluation of the `chess-coach-32b-v6-dpo2` adapter on your own
Colab Pro/Pro+ GPU and prints **tier-policy match** numbers that are directly comparable
to `RESULTS_STAGE4_CORRECTED.md` / `RESULTS_FULL_EVAL_803.md`, so v6-dpo2 can be added to
the 803 field.

It is engine-free (no Stockfish / Maia install) and **resumable**: every batch is
checkpointed to Google Drive, so a disconnect or a compute-unit-out loses nothing.

## What makes it faithful (comparable, not a re-implementation)

- **Grounding**: the GROUNDED prompts are precomputed offline and are **byte-identical**
  to the Stage-4 eval. They are verified against the committed `stage4_eval_inputs.jsonl`
  (the 360 held-out TEST prompts match byte-for-byte) and reproduce all 2409 committed
  canonical labels. They live on Hugging Face so Colab does not need a chess engine:
  **`https://huggingface.co/datasets/khoilamalphaai/chess-coach-803-eval-prompts`**
  (file `eval803_grounded_prompts.jsonl`, 2409 rows). Rebuild locally any time with
  `python scripts/precompute_grounded_prompts_803.py`.
- **Decode**: greedy, byte-identical to `scripts/stage4_eval_v6dpo2.py`
  (`do_sample=False`, `repetition_penalty=1.15`, `no_repeat_ngram_size=4`,
  `max_new_tokens=256`) over the exact base `unsloth/Qwen3-32B-unsloth-bnb-4bit`
  plus the published v6-dpo2 LoRA (`khoilamalphaai/chess-coach-32b-v6-dpo2`).
- **Scoring**: the extractor is copied **verbatim** from `src.eval.evaluate`
  (the one `scripts/reproduce_v4.py` asserts), scored against the committed corrected
  labels.
- **Built-in proof**: the 120 held-out TEST positions are a strict subset of the 2409,
  so the notebook reproduces the published Stage-4 v6-dpo2 numbers on that subset
  (tier-policy **0.8917**; B/I/A 0.8583/0.8417/0.9750; sound 0.9833; distinct 75/76).
  The notebook prints a PASS/CHECK line for this self-validation.

## Hardware requirement (read this first)

The faithful 4-bit base (`unsloth/Qwen3-32B-unsloth-bnb-4bit`, unsloth's dynamic quant)
is about **38 GB resident**, so it needs an **A100-80GB** runtime:

- **A100-80GB**: recommended. Fits with KV headroom; batch 24 (what Stage-4 used).
- **A100-40GB**: very tight. The ~38 GB base barely fits, leaving little for the KV
  cache; use a small batch (~6) and expect possible out-of-memory. Not recommended.
- **L4 / T4 (<= 24 GB)**: cannot hold the base at all. Do not use for the faithful run.

The notebook auto-detects GPU memory and warns / sets a safe batch size accordingly.

## Step by step

1. **Open in Colab.** Easiest: go to Colab, `File -> Upload notebook`, and upload
   `notebooks/colab_803_eval_v6dpo2.ipynb`. (If this repo is on GitHub you can instead
   open `https://colab.research.google.com/github/<owner>/<repo>/blob/<branch>/notebooks/colab_803_eval_v6dpo2.ipynb`.)
2. **Pick the GPU.** `Runtime -> Change runtime type -> A100 GPU` (High-RAM). Confirm it
   is the 80 GB SKU when Step 0 prints the detected memory.
3. **Run Step 0** (GPU check) and **Step 1** (install; a few minutes).
4. **Run Step 2** and paste your **Hugging Face token** at the prompt (a read token is
   enough; the base, adapter, and prompts are public). The token is read via `getpass`
   and never printed.
5. **Run Step 3** to mount Google Drive (approve the popup). Checkpoints go to
   `MyDrive/chess_coach_803_eval/`.
6. **Run Steps 4-6**: download prompts + adapter, define the vendored scorer, load the
   model (first run also downloads the ~39 GB base; cached afterwards).
7. **(Recommended) Smoke test first.** In Step 2 set `LIMIT = 24`, run Steps 4 and 7 once
   to read the printed `gen/s`, then set `LIMIT = 0` and delete the smoke line from the
   checkpoint (or just start fresh) for the full run. This gives you a real,
   machine-specific time/compute estimate before committing.
8. **Run Step 7** (generate). It prints progress and an ETA and checkpoints every batch.
9. **Run Step 8** to score and print the final table + the self-validation.
10. **(Optional) Step 9** to publish generations + scores to Hugging Face.

## Resuming after a disconnect or running out of units

Just reopen the notebook and re-run the cells top to bottom (Steps 0-6 to reinstall and
reload the model, then Step 7). Step 7 reads the Drive checkpoint, skips every scenario
already generated, and continues. Because decoding is greedy (deterministic), any single
in-flight batch that was lost is regenerated identically, so the final numbers are
unaffected. You can stop and resume as many times as you like.

## Runtime and compute-unit estimate

These are **estimates** (I could not benchmark on Colab hardware directly; run the Step 7
smoke test above for an exact number on your machine). On an **A100-80GB** with batch 24
and 256 new tokens per scenario:

- Generation: roughly **1.5 to 3 s per scenario** amortized in batches, so the full 2409
  scenarios take about **1.0 to 2.0 hours**.
- One-time overhead on a fresh runtime: install ~3-5 min, ~39 GB base download ~5-15 min
  (network dependent), model load ~3-8 min.
- **Total first run: about 1.5 to 2.5 hours.**

Colab bills the A100 at roughly **10-13 compute units per hour**, so the full run is on
the order of **~15 to 35 compute units** end to end. Because it is fully resumable, an
underestimate only means you resume in a later session; you never lose completed work.

## Honesty and comparability

- **No verify-gate during scoring, on purpose.** The published Stage-4 / 803 field
  tier-policy numbers are computed on single greedy generations (no live
  verify-and-regenerate loop). The gate is a serving-time faithfulness floor applied
  equally to every model (`RESULTS_FULL_EVAL_803.md` section 4), not part of tier-policy
  scoring, so reproducing the eval means not gating. Adding it would move the numbers away
  from the comparable tables.
- **Scope.** These are corrected-v6 **fresh-grounding** numbers, directly comparable to the
  Stage-4 120-TEST table. The 803 field table in `RESULTS_FULL_EVAL_803.md` is a re-score
  of OLD (v4-era-grounding) generations against the corrected labels, a different scope;
  use it for ranking context only, not as a same-grounding ceiling.
- **Tier-policy match** is agreement with a preregistered project rule (learnability), not
  a claim of validated pedagogy.

## Troubleshooting

- **Out of memory**: lower `EVAL_BATCH` in Step 2 (try 12, then 6), re-run Step 6 if
  needed, then re-run Step 7 (it resumes). If you are on a 40 GB A100 this may still OOM;
  switch to an 80 GB A100.
- **Install resolution errors**: the pinned recipe (transformers 4.57.1, the torch-matched
  xformers, unsloth) is known-good for Qwen3 on Colab as of mid-2026. If Unsloth publishes
  a breaking change, use their current Colab install snippet from
  `https://unsloth.ai/docs/get-started/install/google-colab` and keep the rest of the
  notebook unchanged.
- **Gated/token errors**: the base, adapter, and prompts are public; a plain read token
  works. Re-run Step 2 to re-enter it.
