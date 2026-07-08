#!/usr/bin/env python3
"""QLoRA fine-tune **Qwen3-4B-Instruct-2507** into the chess-coach **4B iter1**
specialist on Modal — the small, locally-runnable hero of the autonomous data
loop (see ``RALPH_TASK.md``).

Same Unsloth QLoRA recipe as ``train_modal_v4.py`` (the 32B reference) with the
knobs sized down for a 4B model on a cheap GPU:

* base = ``unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`` (4-bit, instruct /
  non-thinking — matches our non-``<think>`` targets and the served coach);
* GPU = **A10G** (24 GB is plenty for a 4B 4-bit QLoRA at seq 2048, per-device
  batch 1 x grad-accum 16); a ~9 k-row 2-epoch run is ~$2-4 and ~<1 h;
* trains on the v5-curated **``train_4b_iter1.jsonl``** (baked into the image so
  the detached run needs nothing local), writes to its OWN run dir
  ``chess-coach-4b-iter1`` on the shared Volume (nothing v1-v4 is overwritten);
* adapter-only by default (a 4B LoRA adapter is tiny), checkpoint/resume every 40
  steps to the Volume so a timeout/preemption just resumes.

WORKSPACE: launch on **chess-instructor-2** (kim-lam is billing-blocked; do NOT
use it). Select the workspace per-command WITHOUT switching the active profile
(which hosts the live v4 32B run on chess-instructor):

    # smoke (proves the loop cheaply, ~20 rows / ~20 steps):
    MODAL_PROFILE=chess-instructor-2 modal run src/train/train_modal_4b.py --smoke

    # full train, detached + resumable (survives local disconnect):
    MODAL_PROFILE=chess-instructor-2 modal run --detach src/train/train_modal_4b.py

Confirm the app URL prefix is ``chess-instructor-2--...`` (never ``kim-lam--``).

Prereq (full): ``data/dataset/{train_4b_iter1,valid_4b_iter1}.jsonl`` present
locally (built by ``python -m src.teacher.build_4b_dataset build``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import modal

# --------------------------------------------------------------------------- #
# Names / paths (4B iter1)
# --------------------------------------------------------------------------- #
ITER: str = os.environ.get("CHESS_4B_ITER", "iter1")   # "iter1" (default) | "iter2" | ...
APP_NAME: str = "chess-coach-qlora-4b" if ITER == "iter1" else f"chess-coach-qlora-4b-{ITER}"
VOLUME_NAME: str = "chess-coach-lora"            # shared volume; 4B uses its own run dir
RUN_NAME: str = f"chess-coach-4b-{ITER}"         # this run's artifact dir (iter-scoped)

VOL_MOUNT: str = "/vol"
REMOTE_TRAIN: str = f"/data/train_4b_{ITER}.jsonl"
REMOTE_VALID: str = f"/data/valid_4b_{ITER}.jsonl"
ADAPTER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
MERGED_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/merged_16bit"

if modal.is_local():
    _THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT: Optional[Path] = _THIS_DIR.parents[1]
    LOCAL_TRAIN: Optional[Path] = REPO_ROOT / "data" / "dataset" / f"train_4b_{ITER}.jsonl"
    LOCAL_VALID: Optional[Path] = REPO_ROOT / "data" / "dataset" / f"valid_4b_{ITER}.jsonl"
    LOCAL_OUT_DIR: Optional[Path] = REPO_ROOT / "models" / "adapters" / RUN_NAME
else:
    REPO_ROOT = LOCAL_TRAIN = LOCAL_VALID = LOCAL_OUT_DIR = None

# --------------------------------------------------------------------------- #
# Hyper-parameters (v4 recipe, sized for 4B — a clean DATA-only intervention)
# --------------------------------------------------------------------------- #
BASE_MODEL: str = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
MAX_SEQ_LEN: int = 2048

LORA_R: int = 32
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.0
TARGET_MODULES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LEARNING_RATE: float = 2e-4
NUM_EPOCHS: float = 2.0
WARMUP_RATIO: float = 0.05
LR_SCHEDULER: str = "cosine"
WEIGHT_DECAY: float = 0.01
OPTIMIZER: str = "adamw_8bit"
PER_DEVICE_BATCH: int = 1      # 1 x accum 16 = eff 16; memory-safe on A10G for 4B
GRAD_ACCUM: int = 16
LOGGING_STEPS: int = 1
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
GPU: str = "A10G"
TIMEOUT_S: int = 6 * 3600      # observed ~13.8s/step on A10G => ~1,148 steps (2 ep) ~4.4h + headroom
                                # (checkpoints every 40 steps to the Volume, so a timeout still resumes)
CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION: str = "3.11"

PIP_PACKAGES: list[str] = [
    "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
    "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
]


def _require_local_data() -> None:
    missing = [p for p in (LOCAL_TRAIN, LOCAL_VALID) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise SystemExit(
            "BLOCKED: missing 4B dataset shard(s):\n  "
            f"{names}\n"
            "Build them first:\n"
            "  python -m src.teacher.build_4b_dataset build"
        )


train_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(*PIP_PACKAGES)
    # Bake ITER into the image so the REMOTE container's module import resolves the
    # SAME iter-scoped paths as the local build (Modal does NOT propagate local env
    # vars to the container; without this the remote defaults to iter1 and can't find
    # the baked iter2 data file).
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "CHESS_4B_ITER": ITER})
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
def train(smoke: bool = False, merge_16bit: bool = False) -> dict:
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
    # Smoke checkpoints to a SEPARATE dir so a smoke run can never be picked up as a
    # resume point by the full run (which would restart from 20-row-overfit weights).
    trainer_dir = f"{VOL_MOUNT}/{RUN_NAME}/{'_trainer_smoke' if smoke else '_trainer'}"
    sft_config = _make_sft_config(
        output_dir=trainer_dir,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=PER_DEVICE_BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO, num_train_epochs=num_epochs, max_steps=max_steps,
        learning_rate=LEARNING_RATE, logging_steps=LOGGING_STEPS, optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY, lr_scheduler_type=LR_SCHEDULER, seed=SEED,
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


def _volume_get(remote_path: str, local_parent: Path) -> None:
    local_parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "modal", "volume", "get", "--force",
           VOLUME_NAME, remote_path, str(local_parent)]
    print(f"[download] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(smoke: bool = False, merge: bool = False, download: bool = False,
         spawn: bool = False) -> None:
    print(f"=== {APP_NAME}: {'SMOKE' if smoke else 'FULL'} run ===")
    print(f"base model : {BASE_MODEL}")
    print(f"train data : {LOCAL_TRAIN}")
    print(f"gpu        : {GPU}")
    print(f"run dir    : {VOLUME_NAME}:/{RUN_NAME}")

    # `--spawn` (with `modal run --detach`) submits the training server-side and
    # returns immediately, so the run survives the local process ending. This is
    # Modal's recommended primitive for detached/background work (a detached
    # `.remote()` can be canceled when the local caller disconnects).
    if spawn and not smoke:
        call = train.spawn(smoke=smoke, merge_16bit=merge)
        print(f"[spawn] submitted detached train; FunctionCall id = {call.object_id}")
        print("[spawn] the run continues on Modal independent of this local process. "
              "Checkpoints every 40 steps to the Volume; re-run with --spawn to resume.")
        return

    result = train.remote(smoke=smoke, merge_16bit=merge)
    print("\n=== remote train() result ===")
    print(json.dumps(result, indent=2, default=str))

    if download and not smoke:
        _volume_get(f"/{RUN_NAME}/adapter", LOCAL_OUT_DIR)
        print(f"[download] adapter -> {LOCAL_OUT_DIR}")
