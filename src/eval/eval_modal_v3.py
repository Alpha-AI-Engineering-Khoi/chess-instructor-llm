#!/usr/bin/env python3
"""Generate v3 coaching for the 803-position eval ON MODAL (base 4-bit + LoRA adapter).

The 4-bit MLX v3 is far too slow to generate 2,409 (803x3) coachings locally on the
Mac (a 32B is ~20x the v2 1.7B). So we generate the eval outputs on an A100: load
``unsloth/Qwen3-32B-unsloth-bnb-4bit`` in 4-bit + the trained LoRA adapter from the
shared Volume (``chess-coach-v3/adapter``), and coach every (position,tier) prompt.

Prompts are built LOCALLY (fast, no model) with the SAME grounding + system + format
as every other model in the benchmark (``scripts.gap803_gen`` seed +
``src.eval.benchmark.prompts``) and baked into the image as ``prompts_v3.jsonl``:
each line ``{id, pos_id, tier, phase, severity, system, user}``. Output rows are the
benchmark generation schema, written to the Volume and downloaded to
``data/benchmark_gap803/gen/ours_v3.jsonl``.

This is a genuine 4-bit fine-tuned-Qwen3-32B eval; the shipped local model is the
same LoRA fused into the MLX 4-bit base (near-identical behavior; spot-checked).

Commands
--------
    # 1) build prompts locally (no model), then upload into the image at build:
    python -m scripts.gap803_prompts_v3          # writes data/benchmark_gap803/prompts_v3.jsonl
    # 2) generate on Modal + download:
    modal run src/eval/eval_modal_v3.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import modal

APP_NAME = "chess-coach-eval-v3"
VOLUME_NAME = "chess-coach-lora"
RUN_NAME = "chess-coach-v3"
VOL_MOUNT = "/vol"
ADAPTER_DIR = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
REMOTE_PROMPTS = "/data/prompts_v3.jsonl"
REMOTE_OUT = f"{VOL_MOUNT}/{RUN_NAME}/ours_v3_gen.jsonl"

BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
GPU = "A100-80GB"
TIMEOUT_S = 4 * 3600
CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION = "3.11"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 32   # A100-80GB has ample KV headroom for a 4-bit 32B at seq<=3584

if modal.is_local():
    REPO_ROOT: Optional[Path] = Path(__file__).resolve().parents[2]
    LOCAL_PROMPTS: Optional[Path] = REPO_ROOT / "data" / "benchmark_gap803" / "prompts_v3.jsonl"
    LOCAL_OUT: Optional[Path] = REPO_ROOT / "data" / "benchmark_gap803" / "gen" / "ours_v3.jsonl"
else:
    REPO_ROOT = LOCAL_PROMPTS = LOCAL_OUT = None

# Reuse the SAME Unsloth CUDA image as the trainer so the eval model == the trained
# model exactly (Unsloth loads base 4-bit + the saved LoRA adapter dir), and the image
# layer is already cached on Modal from training (fast cold start).
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install("unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
                 "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)

if modal.is_local():
    if not LOCAL_PROMPTS.exists():
        raise SystemExit(
            f"BLOCKED: {LOCAL_PROMPTS} missing. Build it first:\n"
            "  /Users/khoilam/.venvs/mlx/bin/python -m scripts.gap803_prompts_v3"
        )
    image = image.add_local_file(LOCAL_PROMPTS.as_posix(), REMOTE_PROMPTS)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume})
def generate(limit: int = 0) -> dict:
    import time

    import torch
    from unsloth import FastLanguageModel

    volume.reload()
    print(f"[load] Unsloth base 4-bit + adapter={ADAPTER_DIR}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR, max_seq_length=3072, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    rows = [json.loads(l) for l in open(REMOTE_PROMPTS, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]

    done: set = set()
    if Path(REMOTE_OUT).exists():
        for l in open(REMOTE_OUT, encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["scenario_id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[gen] {len(todo)} pending of {len(rows)} ({len(done)} done)")

    from datetime import datetime, timezone
    t0 = time.time()
    written = 0
    with open(REMOTE_OUT, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            texts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": r["system"]},
                     {"role": "user", "content": r["user"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for r in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=3072).to("cuda")
            with torch.no_grad():
                # repetition_penalty + no_repeat_ngram_size stop the greedy-decode
                # degeneration (repeated chars / garbled numeric prefixes) that a
                # 32B QLoRA can fall into on ~9% of prompts; other (API) competitors
                # don't degenerate, so this levels the decoding quality fairly.
                gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     repetition_penalty=1.15, no_repeat_ngram_size=4,
                                     pad_token_id=tok.pad_token_id)
            for r, g, inp in zip(batch, gen, enc["input_ids"]):
                text = tok.decode(g[inp.shape[0]:], skip_special_tokens=True)
                out.write(json.dumps({
                    "scenario_id": r["id"], "model": "ours_v3", "condition": "grounded",
                    "tier": r["tier"], "phase": r["phase"], "severity": r["severity"],
                    "pos_id": r["pos_id"], "output": _strip_think(text),
                    "prompt_tokens": int(inp.shape[0]), "completion_tokens": 0,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            if (i // BATCH_SIZE) % 5 == 0 or i + BATCH_SIZE >= len(todo):
                dt = time.time() - t0
                n = i + len(batch)
                print(f"  {n}/{len(todo)} ({dt/max(1,n):.2f}s/it, eta {dt/max(1,n)*(len(todo)-n)/60:.0f}m)")
                volume.commit()
    volume.commit()
    return {"written": written, "total": len(rows), "secs": round(time.time() - t0, 1),
            "out": REMOTE_OUT}


@app.local_entrypoint()
def main(limit: int = 0, block: bool = False) -> None:
    # Use .spawn() (NOT .remote()) so generation runs to completion SERVER-SIDE and
    # survives a local disconnect. A plain .remote() call is canceled when the caller
    # drops (Modal: ".remote()/.map() in detached apps may be canceled when the local
    # caller disconnects") — that killed two prior attempts on network blips. The
    # function writes ours_v3_gen.jsonl to the Volume incrementally and is resumable
    # by scenario_id, so poll + download the Volume file separately (scripts do this).
    call = generate.spawn(limit=limit)
    print(f"SPAWNED generate call_id={call.object_id} — running detached on Modal; "
          f"poll {VOL_MOUNT}/{RUN_NAME}/ours_v3_gen.jsonl on the Volume for completion.")
    if block:
        res = call.get()
        print(json.dumps(res, indent=2, default=str))
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = LOCAL_OUT.parent / "_ours_v3_dl"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                        VOLUME_NAME, f"/{RUN_NAME}/ours_v3_gen.jsonl", str(tmp)], check=True)
        got = tmp / "ours_v3_gen.jsonl"
        if got.exists():
            shutil.move(str(got), str(LOCAL_OUT))
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"DONE -> {LOCAL_OUT}")
