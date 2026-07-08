#!/usr/bin/env python3
"""P2 — the autonomous 4B loop, DETACHED on Modal (no Mac, no Cursor agent).

A self-perpetuating cloud orchestrator: each iteration improves the DATA
deterministically -> trains (QLoRA on Modal) -> evaluates (the P1
``chess-coach-eval-4b`` app) -> records results -> decides -> re-spawns itself for
the next iteration, until the completion criteria (``RALPH_TASK.md``) are met, a
max-iteration cap, or a credit guard trips. ALL state lives on the
``chess-coach-lora`` Volume (``/loop/state.json``) — NOT the Mac's ``.ralph/`` —
so a laptop restart never stops it.

Why self-spawning: one Modal function can run at most ~24h, but 20 iterations of
(~4.4h train + ~1h eval) is far longer. So ``loop_step`` runs exactly ONE
iteration (well under the timeout), persists state, then ``.spawn()``s the next
step server-side and returns. If a step dies mid-way it is fully resumable (every
sub-step checks the Volume for its own artifact before redoing work), so re-
triggering the same iteration continues where it left off.

Functions (all on chess-instructor-2):
  * ``build_iter_dataset`` (CPU) — deterministic data recipe -> train/valid on Volume.
  * ``train_iter``        (GPU) — QLoRA (same recipe as train_modal_4b) -> adapter on Volume.
  * ``loop_step``         (CPU) — the controller for one iteration; self-spawns.

Launch (detached; survives Mac restart)::

    MODAL_PROFILE=chess-instructor-2 modal deploy src/orchestrate/loop_modal.py
    MODAL_PROFILE=chess-instructor-2 modal run src/orchestrate/loop_modal.py::start --start-iter 1

Check state::

    MODAL_PROFILE=chess-instructor-2 modal run src/orchestrate/loop_modal.py::status
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        force=True)

# --------------------------------------------------------------------------- #
# Names / infra
# --------------------------------------------------------------------------- #
APP_NAME = "chess-coach-loop-4b"
EVAL_APP = "chess-coach-eval-4b"
LORA_VOLUME = "chess-coach-lora"
EVAL_VOLUME = "chess-coach-eval"
LORA_MOUNT = "/lora"
EVAL_MOUNT = "/eval"

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
GPU = "A10G"
TRAIN_TIMEOUT_S = 6 * 3600
STEP_TIMEOUT_S = 23 * 3600           # one full iteration (train+eval) < 24h
DATA_TIMEOUT_S = 1 * 3600
CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION = "3.11"

ROOT_REMOTE = "/root"
STATE_PATH = f"{LORA_MOUNT}/loop/state.json"
CANDIDATES_VOL = f"{LORA_MOUNT}/datasets/candidates_v3.jsonl"

# Loop policy.
MAX_ITERATIONS = 20
# Rough per-iteration Modal cost (A10G train ~4.4h + eval gen ~0.5h). The council
# runs on the org-funded TrueFoundry gateway (NOT Modal credits).
EST_COST_PER_ITER_USD = 6.0
BUDGET_USD = 22.0                     # stop before exhausting CI2 headroom (~$24)

# QLoRA recipe (identical to src/train/train_modal_4b.py — a DATA-only loop).
MAX_SEQ_LEN = 2048
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2.0
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01
OPTIMIZER = "adamw_8bit"
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16
LOGGING_STEPS = 1
SAVE_STEPS = 40
SAVE_TOTAL_LIMIT = 2
SEED = 3407
QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
QWEN_RESPONSE_PART = "<|im_start|>assistant\n"

if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
else:
    REPO = None

SECRET = modal.Secret.from_name("chess-eval-secrets")
_PY_IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(
        "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
        "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
        "python-chess", "openai", "python-dotenv",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": ROOT_REMOTE})
    .workdir(ROOT_REMOTE)
)
if modal.is_local():
    image = (
        image
        .add_local_dir((REPO / "src").as_posix(), f"{ROOT_REMOTE}/src", copy=True, ignore=_PY_IGNORE)
        .add_local_dir((REPO / "config").as_posix(), f"{ROOT_REMOTE}/config", copy=True, ignore=_PY_IGNORE)
        .add_local_dir((REPO / "scripts").as_posix(), f"{ROOT_REMOTE}/scripts", copy=True, ignore=_PY_IGNORE)
        .add_local_dir((REPO / "prompts").as_posix(), f"{ROOT_REMOTE}/prompts", copy=True, ignore=_PY_IGNORE)
    )

lora_vol = modal.Volume.from_name(LORA_VOLUME, create_if_missing=True)
eval_vol = modal.Volume.from_name(EVAL_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# State helpers (on the Volume — cloud, never .ralph/)
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_state() -> Dict[str, Any]:
    p = Path(STATE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def _write_state(state: Dict[str, Any]) -> None:
    p = Path(STATE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    state["updated_ts"] = _now()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    lora_vol.commit()


def _init_state() -> Dict[str, Any]:
    return {
        "status": "running", "workspace": "chess-instructor-2",
        "current_iter": None, "next_iter": 1, "max_iterations": MAX_ITERATIONS,
        "budget_usd": BUDGET_USD, "est_cost_usd": 0.0,
        "started_ts": _now(), "iterations": [],
    }


# --------------------------------------------------------------------------- #
# Data step (CPU): deterministic recipe -> train/valid on Volume
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=DATA_TIMEOUT_S,
              volumes={LORA_MOUNT: lora_vol}, secrets=[SECRET])
def build_iter_dataset(iter_n: int, recipe: str) -> Dict[str, Any]:
    sys.path.insert(0, ROOT_REMOTE)
    os.chdir(ROOT_REMOTE)
    _setup_logging()
    from src.orchestrate import data_recipes as DR

    lora_vol.reload()
    train_path = f"{LORA_MOUNT}/datasets/train_4b_iter{iter_n}.jsonl"
    valid_path = f"{LORA_MOUNT}/datasets/valid_4b_iter{iter_n}.jsonl"
    if not Path(CANDIDATES_VOL).exists():
        raise RuntimeError(f"missing {CANDIDATES_VOL} — run scripts/bootstrap_cloud_loop.py")
    manifest = DR.build_dataset(recipe, CANDIDATES_VOL, train_path, valid_path, seed=SEED)
    lora_vol.commit()
    print(f"[data] iter={iter_n} recipe={recipe} -> {json.dumps({k: manifest[k] for k in ('train_rows','valid_rows','train_discriminating')})}")
    return {"train_path": train_path, "valid_path": valid_path, "manifest": manifest}


# --------------------------------------------------------------------------- #
# Train step (GPU): QLoRA (same recipe as train_modal_4b) reading Volume dataset
# --------------------------------------------------------------------------- #
def _read_chat_rows(path: str, *, limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


@app.function(image=image, gpu=GPU, timeout=TRAIN_TIMEOUT_S,
              volumes={LORA_MOUNT: lora_vol})
def train_iter(train_path: str, run_name: str, *, smoke: bool = False) -> Dict[str, Any]:
    """QLoRA-train the 4B on ``train_path`` (Volume) -> adapter at /lora/<run_name>/adapter."""
    import inspect

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    _setup_logging()
    lora_vol.reload()
    adapter_dir = f"{LORA_MOUNT}/{run_name}/adapter"
    trainer_dir = f"{LORA_MOUNT}/{run_name}/{'_trainer_smoke' if smoke else '_trainer'}"

    print(f"[train] run={run_name} data={train_path} adapter->{adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, target_modules=TARGET_MODULES, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, bias="none",
        use_gradient_checkpointing="unsloth", random_state=SEED)

    rows = _read_chat_rows(train_path, limit=20 if smoke else None)
    if not rows:
        raise RuntimeError(f"no training rows at {train_path}")
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
             for r in rows]
    dataset = Dataset.from_list([{"text": t} for t in texts])
    print(f"[train] rows={len(rows)} smoke={smoke}")

    def _cfg(**kw):
        valid = set(inspect.signature(SFTConfig.__init__).parameters)
        if "max_seq_length" in kw and "max_seq_length" not in valid:
            kw["max_length"] = kw.pop("max_seq_length")
        return SFTConfig(**{k: v for k, v in kw.items() if k in valid})

    sft = _cfg(output_dir=trainer_dir, dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
               per_device_train_batch_size=PER_DEVICE_BATCH, gradient_accumulation_steps=GRAD_ACCUM,
               warmup_ratio=WARMUP_RATIO, num_train_epochs=(1.0 if smoke else NUM_EPOCHS),
               max_steps=(20 if smoke else -1), learning_rate=LEARNING_RATE,
               logging_steps=LOGGING_STEPS, optim=OPTIMIZER, weight_decay=WEIGHT_DECAY,
               lr_scheduler_type=LR_SCHEDULER, seed=SEED, save_strategy="steps",
               save_steps=SAVE_STEPS, save_total_limit=SAVE_TOTAL_LIMIT,
               bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(), report_to="none")

    tkw = {"model": model, "train_dataset": dataset, "args": sft}
    tvalid = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in tvalid:
        tkw["processing_class"] = tokenizer
    else:
        tkw["tokenizer"] = tokenizer
    trainer = SFTTrainer(**tkw)
    trainer = train_on_responses_only(
        trainer, instruction_part=QWEN_INSTRUCTION_PART, response_part=QWEN_RESPONSE_PART)

    class _Commit(TrainerCallback):
        def on_save(self, args, state, control, **kw):  # noqa: ANN001
            try:
                lora_vol.commit()
            except Exception:  # noqa: BLE001
                pass

    trainer.add_callback(_Commit())

    resume = None
    if not smoke:
        ckpts = glob.glob(f"{trainer_dir}/checkpoint-*")
        if ckpts:
            resume = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
            print(f"[train] resume from {resume}")
    try:
        out = trainer.train(resume_from_checkpoint=resume)
    except Exception as exc:  # noqa: BLE001
        if resume is None:
            raise
        print(f"[train] resume failed ({exc}); fresh")
        out = trainer.train()

    losses = [d["loss"] for d in trainer.state.log_history if "loss" in d]
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    lora_vol.commit()
    res = {"adapter_dir": adapter_dir, "run_name": run_name, "rows": len(rows),
           "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
           "metrics": getattr(out, "metrics", None)}
    print(f"[train] DONE {json.dumps({k: res[k] for k in ('run_name','first_loss','last_loss')}, default=str)}")
    return res


# --------------------------------------------------------------------------- #
# The controller: one iteration, then self-spawn (CPU)
# --------------------------------------------------------------------------- #
def _adapter_ready(run_name: str) -> bool:
    return Path(f"{LORA_MOUNT}/{run_name}/adapter/adapter_model.safetensors").exists()


def _report_ready(iter_tag: str) -> Optional[Dict[str, Any]]:
    p = Path(f"{EVAL_MOUNT}/runs/{iter_tag}/report_4b.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


@app.function(image=image, timeout=STEP_TIMEOUT_S,
              volumes={LORA_MOUNT: lora_vol, EVAL_MOUNT: eval_vol}, secrets=[SECRET])
def loop_step(
    iter_n: int,
    *,
    max_val_positions: int = 0,
    optimize_rounds: int = 3,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Run ONE iteration (data -> train -> eval -> decide) then spawn the next."""
    sys.path.insert(0, ROOT_REMOTE)
    os.chdir(ROOT_REMOTE)
    _setup_logging()
    from src.orchestrate import data_recipes as DR

    lora_vol.reload()
    eval_vol.reload()
    state = _read_state() or _init_state()
    if state.get("status") not in (None, "running"):
        print(f"[loop] status={state.get('status')}; not continuing.")
        return {"stopped": True, "status": state.get("status")}
    state["current_iter"] = iter_n
    _write_state(state)

    run_name = f"chess-coach-4b-iter{iter_n}"
    iter_tag = f"iter{iter_n}"
    prev_report = state["iterations"][-1].get("report") if state.get("iterations") else None
    t_start = time.time()
    print(f"[loop] === iteration {iter_n} (run={run_name}) ===")

    # -- credit guard (deterministic; TrueFoundry council is org-funded) ----- #
    if iter_n >= 2 and (state.get("est_cost_usd", 0.0) + EST_COST_PER_ITER_USD) > state.get("budget_usd", BUDGET_USD):
        state["status"] = "paused_low_credits"
        state["pause_reason"] = (f"est cost ${state['est_cost_usd']:.1f} + ${EST_COST_PER_ITER_USD:.1f} "
                                 f"> budget ${state['budget_usd']:.1f}. Rotate to another workspace "
                                 f"(chess-instructor; never kim-lam) and re-deploy+re-trigger, or raise budget_usd.")
        _write_state(state)
        print(f"[loop] PAUSED (low credits): {state['pause_reason']}")
        return {"stopped": True, "status": "paused_low_credits"}

    rec: Dict[str, Any] = {"iter": iter_n, "run_name": run_name, "ts_start": _now()}

    # -- 1) DATA + 2) TRAIN (skip for iter-1: its adapter already exists) ---- #
    if iter_n == 1:
        if not _adapter_ready(run_name):
            raise RuntimeError(f"iter-1 adapter missing at /lora/{run_name}/adapter — cannot baseline.")
        rec["recipe"] = "(pretrained iter-1 adapter)"
        rec["train"] = "skipped (adapter present)"
    else:
        recipe = DR.recipe_for_iter(iter_n, prev_report)
        rec["recipe"] = recipe
        train_path = f"{LORA_MOUNT}/datasets/train_4b_iter{iter_n}.jsonl"
        if not Path(train_path).exists():
            data_res = build_iter_dataset.remote(iter_n, recipe)
            rec["dataset_manifest"] = data_res["manifest"]
            lora_vol.reload()
        else:
            print(f"[loop] dataset exists: {train_path} (resume)")
        if not _adapter_ready(run_name):
            train_res = train_iter.remote(train_path, run_name)
            rec["train"] = {k: train_res.get(k) for k in ("first_loss", "last_loss", "rows")}
            state["est_cost_usd"] = state.get("est_cost_usd", 0.0) + EST_COST_PER_ITER_USD
            lora_vol.reload()
        else:
            print(f"[loop] adapter exists: {run_name} (resume, skip train)")

    adapter_dir = f"{LORA_MOUNT}/{run_name}/adapter"
    rec["adapter_dir"] = adapter_dir

    # -- 3) EVAL via the P1 app (resumable) ---------------------------------- #
    report = _report_ready(iter_tag)
    if report is None:
        eval_gen = modal.Function.from_name(EVAL_APP, "eval_generate")
        eval_jr = modal.Function.from_name(EVAL_APP, "eval_judge_report")
        gen_manifest = eval_gen.remote(
            iter_tag, adapter_dir, max_val_positions=max_val_positions,
            optimize_rounds=optimize_rounds, batch_size=batch_size)
        rec["gen_manifest"] = gen_manifest
        report = eval_jr.remote(iter_tag, max_val_positions=max_val_positions)
        eval_vol.reload()
    else:
        print(f"[loop] eval report exists for {iter_tag} (resume)")
    rec["report"] = report
    rec["criteria"] = report.get("criteria") if report else None
    rec["eval_report_path"] = f"{EVAL_MOUNT}/runs/{iter_tag}/report_4b.json"

    # -- 4) RECORD + DECIDE -------------------------------------------------- #
    rec["ts_end"] = _now()
    rec["duration_min"] = round((time.time() - t_start) / 60, 1)
    state["iterations"].append(rec)

    criteria = (report or {}).get("criteria") or {}
    done = bool(criteria.get("all_met"))
    if done:
        state["status"] = "complete"
        state["next_iter"] = None
        _write_state(state)
        print(f"[loop] COMPLETE at iter {iter_n} — criteria met: {json.dumps(criteria)}")
        return {"stopped": True, "status": "complete", "iter": iter_n, "criteria": criteria}
    if iter_n >= state.get("max_iterations", MAX_ITERATIONS):
        state["status"] = "max_iters"
        state["next_iter"] = None
        _write_state(state)
        print(f"[loop] reached max_iterations={iter_n}; stopping.")
        return {"stopped": True, "status": "max_iters", "iter": iter_n}

    state["next_iter"] = iter_n + 1
    _write_state(state)
    print(f"[loop] iter {iter_n} done (criteria={json.dumps(criteria)}); spawning iter {iter_n + 1}")
    loop_step.spawn(iter_n + 1, max_val_positions=max_val_positions,
                    optimize_rounds=optimize_rounds, batch_size=batch_size)
    return {"stopped": False, "status": "running", "iter": iter_n, "next": iter_n + 1,
            "criteria": criteria}


# --------------------------------------------------------------------------- #
# Local entrypoints (trigger + inspect; the loop itself runs on Modal)
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def start(start_iter: int = 1, max_val_positions: int = 0, optimize_rounds: int = 3,
          batch_size: int = 16, reset: bool = False) -> None:
    """Kick off the detached loop server-side, then return (safe to close the Mac)."""
    if reset:
        _reset_state.remote()
    call = loop_step.spawn(start_iter, max_val_positions=max_val_positions,
                           optimize_rounds=optimize_rounds, batch_size=batch_size)
    print(f"SPAWNED loop_step(iter={start_iter}) call_id={call.object_id} on {APP_NAME}. "
          f"Detached on Modal — it self-spawns each next iteration; state at "
          f"{LORA_VOLUME}:{'/loop/state.json'}. A Mac restart will NOT stop it.")


@app.function(image=image, volumes={LORA_MOUNT: lora_vol})
def _reset_state() -> None:
    _write_state(_init_state())
    print("[loop] state reset.")


@app.function(image=image, volumes={LORA_MOUNT: lora_vol})
def read_state() -> Dict[str, Any]:
    lora_vol.reload()
    return _read_state()


@app.local_entrypoint()
def status() -> None:
    s = read_state.remote()
    if not s:
        print("no loop state yet.")
        return
    print(json.dumps({
        "status": s.get("status"), "current_iter": s.get("current_iter"),
        "next_iter": s.get("next_iter"), "est_cost_usd": s.get("est_cost_usd"),
        "n_iterations": len(s.get("iterations", [])),
        "last": (s.get("iterations") or [{}])[-1].get("criteria"),
    }, indent=2))
