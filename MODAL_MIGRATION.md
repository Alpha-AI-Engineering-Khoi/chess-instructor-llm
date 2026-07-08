# Modal Migration Manifest — chess-instructor-llm

**Purpose:** the current Modal account (`kim-lam`) is running low on credits and is
being migrated to a new account. This document is a complete manifest of everything
that lived on the current Modal account, where each artifact is now durably backed
up, and a step‑by‑step runbook to re‑provision a **fresh Modal account** from those
backups.

- **Generated:** 2026‑07‑07
- **Source Modal account:** `kim-lam` (workspace that owns volume `chess-coach-lora`)
- **Backup principle:** durable, account‑independent **Hugging Face** backup preferred
  over local disk (local disk was at 97% / ~31 GB free — too tight to hold copies).
- **Safety:** all Modal operations used were **read‑only** (`volume list/ls/get`,
  `app list`). **Nothing was deleted on Modal.** All live jobs were **not disrupted**:
  `chess-coach-v3-serve` (deployed), `chess-coach-v3-vllm` (deployed), and
  `chess-coach-qlora-v4` (training) were all still running after the export.
- **Secrets:** Modal + HF credentials were read only from `.env`
  (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`) and
  are **never** printed or committed here.

---

## 1. TL;DR — where everything is now

| Artifact | Size | On old Modal | Backed up at | Action needed |
|---|---|---|---|---|
| **v3 LoRA adapter** (Qwen3‑32B) | 1.0 GiB | `chess-coach-v3/adapter` | ✅ HF `khoilamalphaai/qwen3-1.7b-chess-coach-mlx` → `v3-lora-qwen3-32b/` | none (already on HF) |
| **Published 1.7B MLX coach** | ~1.8 GiB | (built from v2) | ✅ HF `khoilamalphaai/qwen3-1.7b-chess-coach-mlx` (root) | none |
| **v1/v2/v3 datasets** | small | (local, mounted) | ✅ HF dataset `khoilamalphaai/chess-coach-move-review` (`v1/ v2/ v3/`) | none |
| **Benchmark dataset** | small | (local) | ✅ HF dataset `khoilamalphaai/chess-coach-benchmark` | none |
| **v1 LoRA adapter** (Qwen3‑1.7B) | 66.5 MiB | `chess-coach-v1/adapter` | ✅ HF `khoilamalphaai/chess-coach-modal-backup` → `v1-lora-qwen3-1.7b/` | **exported now** |
| **v2 LoRA adapter** (Qwen3‑1.7B) | 66.5 MiB | `chess-coach-v2/adapter` | ✅ HF `…/chess-coach-modal-backup` → `v2-lora-qwen3-1.7b/` | **exported now** |
| **v1 merged fp16** (Qwen3‑1.7B) | 3.2 GiB | `chess-coach-v1/merged_16bit` | ✅ HF `…/chess-coach-modal-backup` → `v1-merged-16bit-qwen3-1.7b/` | **exported now** |
| **v2 merged fp16** (Qwen3‑1.7B) | 3.2 GiB | `chess-coach-v2/merged_16bit` | ✅ HF `…/chess-coach-modal-backup` → `v2-merged-16bit-qwen3-1.7b/` | **exported now** |
| **v3 generations** `ours_v3_gen.jsonl` | 2.4 MiB | `chess-coach-v3/ours_v3_gen.jsonl` | ✅ HF `…/chess-coach-modal-backup` → `v3-gen/` | **exported now** |
| **v4 training dataset** | 39 MiB | (local only, gitignored) | ✅ HF dataset `khoilamalphaai/chess-coach-move-review` → `v4/` | **exported now** |
| **Repo source snapshot** (incl. local‑only v4 + serve/vLLM code) | ~6 MiB | (local only) | ✅ HF `…/chess-coach-modal-backup` → `repo-src-snapshot.tar.gz` | **exported now** |
| **v1/v2/v3 trainer checkpoints** | ~3.7 GiB | `chess-coach-v*/_trainer/checkpoint-*` | ⏭️ not exported (redundant) | see §5 |
| **v4 checkpoint‑40** (partial, in‑progress) | ~1.54 GiB | `chess-coach-v4/_trainer/checkpoint-40` | ✅ HF `…/chess-coach-modal-backup` → `v4-checkpoints/checkpoint-40/` | **exported now (safety net)** |
| **Base model caches** | n/a | (not on volume) | ♻️ re‑downloadable free from HF | none |

**New durable backup repo created:** `khoilamalphaai/chess-coach-modal-backup` (private, model repo).

---

## 2. Modal account inventory (read‑only)

### 2.1 Volumes

Only one volume exists: **`chess-coach-lora`** (created 2026‑07‑06 by `kim-lam`).
Total file bytes inventoried: **~11.2 GiB** across v1–v3 (v4 dir is empty except an
in‑progress `_trainer/`).

Full recursive tree (sizes as reported by `modal volume ls`):

```
chess-coach-lora/
├── chess-coach-v1/                                   (~3.5 GiB total)
│   ├── adapter/                                      → HF backup v1-lora-qwen3-1.7b/
│   │   ├── adapter_model.safetensors        66.5 MiB
│   │   ├── adapter_config.json / tokenizer.* / vocab / merges / chat_template / README
│   ├── _trainer/
│   │   ├── checkpoint-20/   (adapter 66.5M + optimizer 34.2M + tokenizer …  ≈116 MiB)
│   │   └── checkpoint-344/  (≈116 MiB)  ← final step of v1 run
│   └── merged_16bit/
│       ├── model.safetensors                 3.2 GiB → HF backup v1-merged-16bit-qwen3-1.7b/
│       └── config.json / tokenizer.* / vocab / merges / chat_template
├── chess-coach-v2/                                   (~3.6 GiB total)
│   ├── adapter/  (adapter_model.safetensors 66.5 MiB) → HF backup v2-lora-qwen3-1.7b/
│   ├── _trainer/
│   │   ├── checkpoint-20/   (≈116 MiB)
│   │   ├── checkpoint-500/  (≈116 MiB)
│   │   └── checkpoint-616/  (≈116 MiB)  ← final step of v2 run
│   └── merged_16bit/
│       └── model.safetensors                 3.2 GiB → HF backup v2-merged-16bit-qwen3-1.7b/
├── chess-coach-v3/                                   (~4.1 GiB total)
│   ├── adapter/  (adapter_model.safetensors 1.0 GiB) → ALREADY on HF (qwen3-1.7b-chess-coach-mlx/v3-lora-qwen3-32b/)
│   ├── _trainer/
│   │   ├── checkpoint-400/  (adapter 1.0 GiB + optimizer 523 MiB ≈ 1.54 GiB)
│   │   └── checkpoint-424/  (≈ 1.54 GiB)  ← final step of v3 run
│   └── ours_v3_gen.jsonl                      2.4 MiB → HF backup v3-gen/
└── chess-coach-v4/                                   (in progress — training LIVE)
    └── _trainer/
        └── checkpoint-40/  (adapter 1.0 GiB + optimizer 521 MiB ≈ 1.54 GiB) → HF backup v4-checkpoints/checkpoint-40/
            # checkpoints rotate: SAVE_TOTAL_LIMIT=2 keeps only the latest 2 (40 will be
            # superseded by 80/120 …). The grabbed checkpoint-40 is a partial-but-usable
            # v4 adapter + resume state captured as a safety net.
```

Notes:
- **v3 has no `merged_16bit` / `mlx_4bit` on the volume** — v3 is served as base‑4bit
  + LoRA adapter (no merged 32B was kept). The deployable v3 artifact is the adapter
  (already on HF).
- Base weights (`unsloth/Qwen3-1.7B`, `unsloth/Qwen3-32B-unsloth-bnb-4bit`,
  `mlx-community/Qwen3-1.7B-4bit`) are **not** stored on the volume — they re‑pull from
  HF for free, so they are intentionally **not** backed up.

### 2.2 Apps

| App (description) | State | Notes |
|---|---|---|
| `chess-coach-v3-serve` | **deployed (LIVE)** | scale‑to‑zero A100‑40GB transformers+peft endpoint (`src/serve/serve_v3_modal.py`); do not disrupt |
| `chess-coach-v3-vllm` | **deployed (LIVE)** | vLLM v3 endpoint (`src/serve/serve_v3_vllm_modal.py`), 2 tasks; reuses the same v3 adapter on the volume; do not disrupt |
| `chess-coach-qlora-v4` | **running (ephemeral, detached)** | the in‑progress v4 QLoRA training (`src/train/train_modal_v4.py`); do not disrupt |
| `chess-coach-qlora-v4` (×6) | stopped | earlier v4 attempts (preempted / restarted) |
| `chess-coach-qlora`, `-v2`, `-v3` | (aged out of list) | v1/v2/v3 training runs — complete; outputs on volume/HF |
| `chess-coach-eval-v3`, `-v4` | (aged out of list) | eval runs — no durable artifacts to keep |

Modal secret in use: **`chess-hf`** (holds `HF_TOKEN` + `HUGGING_FACE_HUB_TOKEN`), used
by the serve/train images to pull the base model.

---

## 3. Hugging Face backup layout (durable, account‑independent)

**`khoilamalphaai/chess-coach-modal-backup`** (private model repo — NEW):
```
v1-lora-qwen3-1.7b/           # v1 LoRA adapter (base: unsloth/Qwen3-1.7B)
v2-lora-qwen3-1.7b/           # v2 LoRA adapter (base: unsloth/Qwen3-1.7B)
v1-merged-16bit-qwen3-1.7b/   # v1 merged fp16 model
v2-merged-16bit-qwen3-1.7b/   # v2 merged fp16 model
v3-gen/ours_v3_gen.jsonl      # v3 generation dump
v4-checkpoints/checkpoint-40/ # v4 partial (step 40) adapter + optimizer — safety net
repo-src-snapshot.tar.gz      # full repo source (excl. data/models/.git/node_modules)
```

**`khoilamalphaai/qwen3-1.7b-chess-coach-mlx`** (existing model repo):
```
(root)                        # published 1.7B MLX coach (deployable)
v3-lora-qwen3-32b/            # v3 LoRA adapter (base: unsloth/Qwen3-32B-unsloth-bnb-4bit)
```

**`khoilamalphaai/chess-coach-move-review`** (existing dataset repo):
```
v1/  v2/  v3/                 # train.jsonl + validation.jsonl per version
v4/                           # NEW — v4 train.jsonl + validation.jsonl
```

**`khoilamalphaai/chess-coach-benchmark`** (existing dataset repo): the 803× eval benchmark.

**Code:** GitHub `https://github.com/kimkhoi2202/chess-instructor-llm` (branch `main`).
⚠️ See §5 — the v4 pipeline and the v3 serve app are **not yet pushed** to GitHub; they
are captured in `repo-src-snapshot.tar.gz` on HF as a stopgap.

---

## 4. Re‑provisioning runbook (NEW Modal account)

All commands assume the repo root and the mlx venv Modal/HF CLIs:
`export PATH="/Users/khoilam/.venvs/mlx/bin:$PATH"` (or `pip install modal huggingface_hub`).

### 4.1 Point the CLI at the NEW account
1. Create a new token in the new Modal workspace and put it in `.env`:
   ```
   MODAL_TOKEN_ID=...        # new account
   MODAL_TOKEN_SECRET=...    # new account
   ```
2. Load env + verify:
   ```bash
   set -a; . ./.env; set +a
   modal profile current      # or: modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
   modal app list             # should be empty on the fresh account
   ```

### 4.2 Recreate the HF secret (needed by serve + train images)
```bash
set -a; . ./.env; set +a
modal secret create chess-hf HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"
```

### 4.3 Recreate the volume + restore adapters from HF
The train scripts auto‑create the volume (`create_if_missing=True`), but you can also:
```bash
modal volume create chess-coach-lora
```
Restore the **v3 adapter** (required for the serve app) from HF into the volume:
```bash
hf download khoilamalphaai/qwen3-1.7b-chess-coach-mlx --include 'v3-lora-qwen3-32b/*' \
  --local-dir /tmp/restore_v3
modal volume put chess-coach-lora /tmp/restore_v3/v3-lora-qwen3-32b /chess-coach-v3/adapter
rm -rf /tmp/restore_v3
```
(Optional) restore v1/v2 adapters + merged models from the backup repo:
```bash
hf download khoilamalphaai/chess-coach-modal-backup --include 'v1-lora-qwen3-1.7b/*' --local-dir /tmp/r
modal volume put chess-coach-lora /tmp/r/v1-lora-qwen3-1.7b /chess-coach-v1/adapter
# repeat for v2-lora-qwen3-1.7b, v1-merged-16bit-qwen3-1.7b, v2-merged-16bit-qwen3-1.7b …
rm -rf /tmp/r
```

### 4.4 Redeploy the serve app(s)
```bash
set -a; . ./.env; set +a
modal deploy src/serve/serve_v3_modal.py         # app: chess-coach-v3-serve (transformers+peft)
modal deploy src/serve/serve_v3_vllm_modal.py    # app: chess-coach-v3-vllm (vLLM) — if you want the vLLM endpoint too
```
Both bake base `unsloth/Qwen3-32B-unsloth-bnb-4bit` into the image and load the v3
adapter from `chess-coach-lora:/chess-coach-v3/adapter` (restored in §4.3).

### 4.5 Restart / continue v4 training
1. Ensure the v4 dataset is present locally (pull from HF if needed):
   ```bash
   hf download khoilamalphaai/chess-coach-move-review --repo-type dataset \
     --include 'v4/*' --local-dir /tmp/v4data
   mkdir -p data/dataset
   cp /tmp/v4data/v4/train.jsonl      data/dataset/train_v4.jsonl
   cp /tmp/v4data/v4/validation.jsonl data/dataset/valid_v4.jsonl
   ```
2. Launch (auto‑creates the volume, resumes from any committed checkpoint):
   ```bash
   set -a; . ./.env; set +a
   modal run src/train/train_modal_v4.py            # full train (adapter‑only, resumable)
   # smoke first if desired: modal run src/train/train_modal_v4.py --smoke
   ```
   > Note: v4 on the OLD account had **no committed checkpoint yet**, so the new
   > account starts v4 from scratch (this is expected — see §5).

### 4.6 Env / secrets checklist (from `.env`)
- **Required for serve + train:** `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (new acct),
  `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` → also mirrored into Modal secret `chess-hf`.
- **Used elsewhere in the pipeline (teacher/eval/data), not needed to redeploy:**
  `OPENAI_API_KEY`, `TEACHER_MODEL`, `TEACHER_REASONING_EFFORT`, `LICHESS_TOKEN`,
  `TFY_API_KEY`, `TFY_BASE_URL`, `TFY_TEACHER_MODEL`, `TFY_JUDGE_MODEL`.

---

## 5. Not exported / caveats (with reasons)

1. **v1/v2/v3 `_trainer/checkpoint-*` (~3.7 GiB total)** — *intentionally not exported.*
   These are mid‑training states (optimizer/scheduler/RNG) for **completed** runs. The
   final LoRA adapters (the actual outputs) are all backed up, so the checkpoints are
   redundant. If you ever want them, they can be pulled read‑only from the old volume
   while it still exists:
   ```bash
   modal volume get chess-coach-lora /chess-coach-v2/_trainer/checkpoint-616 ./modal_export/
   ```

2. **v4 in‑progress checkpoint — grabbed as a safety net.** At the start of the export
   the v4 run (restarted several times) had not yet committed a checkpoint. Once
   `checkpoint-40` appeared it was captured read‑only and pushed to HF at
   `khoilamalphaai/chess-coach-modal-backup/v4-checkpoints/checkpoint-40/` (~1.54 GiB:
   the partial step‑40 LoRA adapter + optimizer/scheduler for resume). Because
   checkpoints rotate (`SAVE_TOTAL_LIMIT=2`), this specific one will be superseded on the
   volume — the HF copy preserves it.    To grab a fresher one later while the old account
   is alive (read‑only get, then push to HF):
   ```bash
   set -a; . ./.env; set +a
   modal volume ls chess-coach-lora /chess-coach-v4/_trainer          # find newest checkpoint-N
   modal volume get chess-coach-lora /chess-coach-v4/_trainer/checkpoint-N ./modal_export/
   hf upload khoilamalphaai/chess-coach-modal-backup \
     ./modal_export/checkpoint-N v4-checkpoints/checkpoint-N --repo-type model
   ```
   Even without any checkpoint, v4 is fully reproducible from the **v4 dataset**
   (HF `chess-coach-move-review/v4`) + the **v4 training code**
   (`repo-src-snapshot.tar.gz`), since it trains adapter‑only and is resumable.

3. **⚠️ Local‑only code not on GitHub.** `git` verification showed the v4 pipeline
   (`src/train/train_modal_v4.py`, `src/eval/eval_modal_v4.py`,
   `src/teacher/build_v4_dataset.py`) and the serve app (`src/serve/serve_v3_modal.py`)
   are **not tracked/pushed** to `origin/main` (last pushed commit is v3‑era). The vLLM
   serve app `src/serve/serve_v3_vllm_modal.py` is likewise local‑only. Per instructions
   no git commit was made. All of these are backed up in `repo-src-snapshot.tar.gz` on HF
   as a stopgap. **Action for you:**
   ```bash
   git add -A && git commit -m "v4 pipeline + v3 serve app" && git push
   ```

4. **Base‑model caches** — not backed up on purpose; they re‑download free from HF.

5. **Nothing was deleted on Modal.** Once the new account is verified working, you can
   optionally free the old account (only if you choose to): `modal volume rm …` etc.
   That is out of scope here and was **not** performed.
