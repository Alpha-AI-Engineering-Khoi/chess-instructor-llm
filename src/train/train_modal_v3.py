#!/usr/bin/env python3
"""QLoRA fine-tune **Qwen3-32B** into the chess-coach **v3** specialist on Modal.

Same Unsloth QLoRA recipe as v2 (``train_modal_v2.py``), scaled from a 1.7B base on
an A10G to a **32B** base on an **A100-80GB**: 4-bit base load, LoRA on attention +
MLP, small per-device batch with large grad-accum, a longer timeout, and a bigger
LoRA rank to absorb the larger, contrastive v3 dataset. Points at
``data/dataset/{train_v3,valid_v3}.jsonl`` and writes to its own run dir
(``chess-coach-v3``) on the shared Volume so **nothing v1/v2 is overwritten**.

Because the disk-constrained Mac cannot hold the ~65GB fp16 merged 32B, the 4-bit
**MLX** conversion is done ON MODAL (``to_mlx``, mlx image) so only the ~18GB MLX
model is downloaded — verified viable by ``_mlx_modal_probe.py``.

Commands
--------
Smoke (<=20 rows, ~20 steps; proves the 32B loop on the A100 cheaply)::

    modal run src/train/train_modal_v3.py --smoke

Full train + merge + MLX convert + download the MLX model::

    modal run src/train/train_modal_v3.py

MLX-convert only (after a completed train)::

    modal run src/train/train_modal_v3.py --skip-train

Prereq (full): ``data/dataset/{train_v3,valid_v3}.jsonl`` present locally.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import modal

# --------------------------------------------------------------------------- #
# Names / paths (v3)
# --------------------------------------------------------------------------- #
APP_NAME: str = "chess-coach-qlora-v3"
VOLUME_NAME: str = "chess-coach-lora"           # shared volume; v3 uses its own run dir
RUN_NAME: str = "chess-coach-v3"                # this run's artifact dir (v3)

VOL_MOUNT: str = "/vol"
REMOTE_TRAIN: str = "/data/train_v3.jsonl"
REMOTE_VALID: str = "/data/valid_v3.jsonl"
ADAPTER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
MERGED_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/merged_16bit"
MLX_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/mlx_4bit"

if modal.is_local():
    _THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT: Optional[Path] = _THIS_DIR.parents[1]
    LOCAL_TRAIN: Optional[Path] = REPO_ROOT / "data" / "dataset" / "train_v3.jsonl"
    LOCAL_VALID: Optional[Path] = REPO_ROOT / "data" / "dataset" / "valid_v3.jsonl"
    LOCAL_OUT_DIR: Optional[Path] = REPO_ROOT / "models" / "adapters" / RUN_NAME
    LOCAL_MLX_DIR: Optional[Path] = REPO_ROOT / "models" / "mlx" / RUN_NAME
else:
    REPO_ROOT = LOCAL_TRAIN = LOCAL_VALID = LOCAL_OUT_DIR = LOCAL_MLX_DIR = None

# --------------------------------------------------------------------------- #
# Hyper-parameters (v2 recipe, scaled for a 32B base + larger dataset)
# --------------------------------------------------------------------------- #
BASE_MODEL: str = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
MAX_SEQ_LEN: int = 2048

LORA_R: int = 32                # up from 16 (1.7B): more capacity for the 32B + 7k rows
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.0
TARGET_MODULES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LEARNING_RATE: float = 2e-4     # QLoRA standard (paper uses 2e-4 up to 65B)
NUM_EPOCHS: float = 2.0
WARMUP_RATIO: float = 0.05
LR_SCHEDULER: str = "cosine"
WEIGHT_DECAY: float = 0.01
OPTIMIZER: str = "adamw_8bit"
PER_DEVICE_BATCH: int = 1      # smoke-validated; batch=2 OOM-killed the A100 worker mid-run
GRAD_ACCUM: int = 16           # eff batch 16 (up from v2's 8) for stability on 7k rows
LOGGING_STEPS: int = 1
#: Checkpoint to the shared Volume every N optimizer steps so a preempted/OOM-killed
#: worker ("Runner failed: Worker disappeared") resumes instead of restarting from 0.
SAVE_STEPS: int = 40
SAVE_TOTAL_LIMIT: int = 2
SEED: int = 3407

SMOKE_MAX_ROWS: int = 20
SMOKE_MAX_STEPS: int = 20

QWEN_INSTRUCTION_PART: str = "<|im_start|>user\n"
QWEN_RESPONSE_PART: str = "<|im_start|>assistant\n"

# --------------------------------------------------------------------------- #
# Modal infra
# --------------------------------------------------------------------------- #
GPU: str = "A100-80GB"
TIMEOUT_S: int = 5 * 3600       # generous: 32B, 2 epochs on ~7k rows + 16-bit merge
CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION: str = "3.11"

PIP_PACKAGES: list[str] = [
    "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
    "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
]
MLX_PIP: list[str] = [
    "mlx", "mlx-lm", "transformers", "huggingface_hub", "hf_transfer",
    "sentencepiece", "protobuf",
]


def _require_local_data() -> None:
    missing = [p for p in (LOCAL_TRAIN, LOCAL_VALID) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise SystemExit(
            "BLOCKED: missing v3 dataset shard(s):\n  "
            f"{names}\n"
            "Build them first:\n"
            "  python src/filter/filter.py --candidates data/generated/candidates_v3.jsonl "
            "--faithfulness reject --dedup-key fen_tier_move --target-format v2 "
            "--train-out data/dataset/train_v3.jsonl --rejects-out data/generated/rejects_v3.jsonl\n"
            "  python src/train/split_data.py --input data/dataset/train_v3.jsonl "
            "--train-out data/dataset/train_v3.jsonl --valid-out data/dataset/valid_v3.jsonl"
        )


train_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(*PIP_PACKAGES)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)
mlx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*MLX_PIP)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

if modal.is_local():
    _require_local_data()
    train_image = (
        train_image
        .add_local_file(LOCAL_TRAIN.as_posix(), REMOTE_TRAIN)
        .add_local_file(LOCAL_VALID.as_posix(), REMOTE_VALID)
    )

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Remote helpers (training)
# --------------------------------------------------------------------------- #
def _read_chat_rows(path: str, *, limit: Optional[int] = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _build_text_dataset(rows: list[dict], tokenizer: Any):
    from datasets import Dataset

    texts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        for row in rows
    ]
    return Dataset.from_list([{"text": t} for t in texts])


def _make_sft_config(**kwargs: Any):
    import inspect

    from trl import SFTConfig

    valid = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_seq_length" in kwargs and "max_seq_length" not in valid:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return SFTConfig(**filtered)


def _make_trainer(**kwargs: Any):
    import inspect

    from trl import SFTTrainer

    valid = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "tokenizer" in kwargs and "tokenizer" not in valid and "processing_class" in valid:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return SFTTrainer(**filtered)


def _print_gpu_banner() -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        print(f"[gpu] nvidia-smi: {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu] nvidia-smi unavailable: {exc}")

    import torch

    avail = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if avail else None
    print(f"[gpu] torch={torch.__version__} cuda_available={avail} device={name}")
    return name


@app.function(image=train_image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume})
def train(smoke: bool = False, merge_16bit: bool = True) -> dict:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    gpu_name = _print_gpu_banner()

    print(f"[load] base={BASE_MODEL!r} 4-bit max_seq_len={MAX_SEQ_LEN}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, target_modules=TARGET_MODULES, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, bias="none",
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )

    row_limit = SMOKE_MAX_ROWS if smoke else None
    rows = _read_chat_rows(REMOTE_TRAIN, limit=row_limit)
    if not rows:
        raise RuntimeError(f"No training rows found at {REMOTE_TRAIN}")
    dataset = _build_text_dataset(rows, tokenizer)
    print(f"[data] train_rows={len(rows)} (smoke={smoke})")
    print("[data] sample rendered row (first 700 chars):")
    print(dataset[0]["text"][:700])

    max_steps = SMOKE_MAX_STEPS if smoke else -1
    num_epochs = 1.0 if smoke else NUM_EPOCHS
    trainer_dir = f"{VOL_MOUNT}/{RUN_NAME}/_trainer"
    sft_config = _make_sft_config(
        output_dir=trainer_dir,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=PER_DEVICE_BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO, num_train_epochs=num_epochs, max_steps=max_steps,
        learning_rate=LEARNING_RATE, logging_steps=LOGGING_STEPS, optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY, lr_scheduler_type=LR_SCHEDULER, seed=SEED,
        # Checkpoint to the Volume so a Modal worker preemption ("Worker
        # disappeared, in-progress inputs will be re-scheduled") RESUMES instead of
        # restarting from step 0. Committed to the Volume after every save.
        save_strategy="steps", save_steps=SAVE_STEPS, save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(), report_to="none",
    )
    trainer = _make_trainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=sft_config)
    trainer = train_on_responses_only(
        trainer, instruction_part=QWEN_INSTRUCTION_PART, response_part=QWEN_RESPONSE_PART,
    )

    from transformers import TrainerCallback

    class _VolCommit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
            try:
                volume.commit()
                print(f"[ckpt] volume committed at step {state.global_step}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ckpt] volume commit failed: {exc}")

    trainer.add_callback(_VolCommit())

    # Resume from the latest committed checkpoint if a prior (preempted) run left one.
    import glob as _glob
    resume_ckpt = None
    if not smoke:
        volume.reload()
        ckpts = _glob.glob(f"{trainer_dir}/checkpoint-*")
        if ckpts:
            resume_ckpt = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
            print(f"[resume] found checkpoint -> {resume_ckpt}")

    print(f"[train] starting: max_steps={max_steps} epochs={num_epochs} "
          f"lr={LEARNING_RATE} r={LORA_R} eff_batch={PER_DEVICE_BATCH * GRAD_ACCUM} "
          f"resume={resume_ckpt}")
    try:
        train_output = trainer.train(resume_from_checkpoint=resume_ckpt)
    except Exception as exc:  # noqa: BLE001 - a corrupt resume must not doom the run
        if resume_ckpt is None:
            raise
        print(f"[resume] resume failed ({exc}); restarting fresh")
        train_output = trainer.train()

    losses = [
        {"step": d.get("step"), "loss": d["loss"]}
        for d in trainer.state.log_history if "loss" in d
    ]
    first_loss = losses[0]["loss"] if losses else None
    last_loss = losses[-1]["loss"] if losses else None
    print(f"[train] done. steps_logged={len(losses)} first_loss={first_loss} last_loss={last_loss}")

    print(f"[save] LoRA adapter -> {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    saved_merged = False
    if merge_16bit:
        print(f"[save] merged 16-bit model -> {MERGED_DIR}")
        model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
        saved_merged = True

    volume.commit()
    print("[save] volume committed.")

    return {
        "gpu": gpu_name, "smoke": smoke, "base_model": BASE_MODEL, "train_rows": len(rows),
        "lora_r": LORA_R, "max_steps": max_steps, "num_epochs": num_epochs,
        "steps_logged": len(losses), "first_loss": first_loss, "last_loss": last_loss,
        "train_metrics": getattr(train_output, "metrics", None),
        "adapter_dir": ADAPTER_DIR, "merged_dir": MERGED_DIR if saved_merged else None,
        "run_name": RUN_NAME,
    }


@app.function(image=mlx_image, timeout=3600, cpu=16.0, memory=131072, volumes={VOL_MOUNT: volume})
def to_mlx(q_bits: int = 4, q_group_size: int = 64) -> dict:
    """Quantize the merged 16-bit 32B to 4-bit MLX ON MODAL (avoids a 65GB local pull)."""
    import os
    import time

    volume.reload()
    if not os.path.isdir(MERGED_DIR) or not os.listdir(MERGED_DIR):
        raise RuntimeError(f"merged model not found at {MERGED_DIR}; run train(merge_16bit=True) first")

    shutil.rmtree(MLX_DIR, ignore_errors=True)
    cmd = [
        "python", "-m", "mlx_lm", "convert",
        "--hf-path", MERGED_DIR, "--mlx-path", MLX_DIR,
        "-q", "--q-bits", str(q_bits), "--q-group-size", str(q_group_size),
    ]
    print(f"[mlx] {' '.join(cmd)}")
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    print((out.stdout + out.stderr)[-3000:])
    if out.returncode != 0:
        raise RuntimeError(f"mlx_lm convert failed rc={out.returncode}")

    # Sanity: load + one short generation to confirm the MLX model is usable.
    from mlx_lm import generate, load
    model, tok = load(MLX_DIR)
    sample = generate(model, tok, prompt="Say OK.", max_tokens=8, verbose=False)
    files = sorted(os.listdir(MLX_DIR))
    volume.commit()
    return {"mlx_dir": MLX_DIR, "secs": round(time.time() - t0, 1),
            "files": files, "gen_sample": sample[:80]}


def _volume_get(remote_path: str, local_parent: Path) -> None:
    local_parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "modal", "volume", "get", "--force",
           VOLUME_NAME, remote_path, str(local_parent)]
    print(f"[download] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(smoke: bool = False, merge: bool = True, convert: bool = True,
         download: bool = True, skip_train: bool = False) -> None:
    print(f"=== {APP_NAME}: {'SMOKE' if smoke else 'FULL'} run ===")
    print(f"base model : {BASE_MODEL}")
    print(f"train data : {LOCAL_TRAIN}")
    print(f"gpu        : {GPU}")

    if not skip_train:
        result = train.remote(smoke=smoke, merge_16bit=merge)
        print("\n=== remote train() result ===")
        print(json.dumps(result, indent=2, default=str))

    if convert and merge and not smoke:
        print("\n=== converting merged 16-bit -> 4-bit MLX on Modal ===")
        mlx_res = to_mlx.remote()
        print(json.dumps(mlx_res, indent=2, default=str))

        if download:
            shutil.rmtree(LOCAL_MLX_DIR, ignore_errors=True)
            _volume_get(f"/{RUN_NAME}/mlx_4bit", LOCAL_MLX_DIR.parent / "_mlx_dl")
            # modal volume get nests the dir; normalize to models/mlx/chess-coach-v3
            nested = LOCAL_MLX_DIR.parent / "_mlx_dl" / "mlx_4bit"
            if nested.exists():
                shutil.rmtree(LOCAL_MLX_DIR, ignore_errors=True)
                nested.rename(LOCAL_MLX_DIR)
                shutil.rmtree(LOCAL_MLX_DIR.parent / "_mlx_dl", ignore_errors=True)
            print(f"\nDONE. MLX model under: {LOCAL_MLX_DIR}")

    # Always pull the small adapter for provenance (skip on smoke to save time).
    if download and not smoke:
        _volume_get(f"/{RUN_NAME}/adapter", LOCAL_OUT_DIR)
        print(f"[download] adapter -> {LOCAL_OUT_DIR}")
