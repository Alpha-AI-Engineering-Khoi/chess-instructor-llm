#!/usr/bin/env python3
"""Generate v4 coaching for the 803-position eval ON MODAL (base 4-bit + LoRA adapter).

Identical to ``eval_modal_v3.py`` but points at the **v4** adapter
(``chess-coach-v4/adapter`` on the shared Volume) and writes ``ours_v4``. Loads
``unsloth/Qwen3-32B-unsloth-bnb-4bit`` in 4-bit + the trained v4 LoRA adapter and
coaches every (position, tier) prompt.

Because v4 was trained on the EXACT served prompt (``build_grounded_user`` — the
same VERIFIED FACTS + user + FORMAT_INSTRUCTION the eval feeds), the train/serve
skew that produced v3's ~4-5% malformed leading fragments is gone. A light,
documented ``_clean_lead`` still strips any residual leading rating-range /
prompt-echo before the first "I'd play" (the same trivially-deployable cleanup
noted in RESULTS_V3), and BOTH the raw and cleaned outputs are recorded so the
malformed-output reduction can be reported honestly.

Prompts are the SAME grounded prompts every model gets (``prompts_v4.jsonl``, a
copy of the benchmark's ``build_grounded_user`` render). Output rows use the
benchmark generation schema, written to the Volume and downloaded to
``data/benchmark_v4/gen/ours_v4.jsonl``.

Commands
--------
    python -m scripts.gap803_prompts_v4          # writes data/benchmark_v4/prompts_v4.jsonl
    modal run src/eval/eval_modal_v4.py          # generate on Modal (spawned/detached)
    modal run src/eval/eval_modal_v4.py --block  # generate + wait + download
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "chess-coach-eval-v4"
VOLUME_NAME = "chess-coach-lora"
RUN_NAME = "chess-coach-v4"
VOL_MOUNT = "/vol"
ADAPTER_DIR = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
REMOTE_PROMPTS = "/data/prompts_v4.jsonl"
REMOTE_OUT = f"{VOL_MOUNT}/{RUN_NAME}/ours_v4_gen.jsonl"

BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
GPU = "A100-80GB"
TIMEOUT_S = 4 * 3600
CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION = "3.11"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 32

if modal.is_local():
    REPO_ROOT: Optional[Path] = Path(__file__).resolve().parents[2]
    LOCAL_PROMPTS: Optional[Path] = REPO_ROOT / "data" / "benchmark_v4" / "prompts_v4.jsonl"
    LOCAL_OUT: Optional[Path] = REPO_ROOT / "data" / "benchmark_v4" / "gen" / "ours_v4.jsonl"
else:
    REPO_ROOT = LOCAL_PROMPTS = LOCAL_OUT = None

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
            "  ~/.venvs/mlx/bin/python -m scripts.gap803_prompts_v4"
        )
    image = image.add_local_file(LOCAL_PROMPTS.as_posix(), REMOTE_PROMPTS)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


def _clean_lead(text: str) -> str:
    """Drop a leading garble/prompt-echo fragment before the first "I'd play".

    Deterministic + conservative: only strips when "I'd play" appears but not at
    the very start AND the junk prefix before it is short (<160 chars), which is
    exactly the v3 failure mode (a spurious leading rating-range like "(1000-1200)"
    or an echoed prompt fragment). A well-formed output (already starting with
    "I'd play") is returned unchanged.
    """
    t = text.strip()
    if t.startswith("I'd play") or t.startswith("I’d play"):
        return t
    idx = t.find("I'd play")
    if idx < 0:
        idx = t.find("I’d play")
    if 0 < idx <= 160:
        return t[idx:].strip()
    return t


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
    n_lead_cleaned = 0
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
                gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     repetition_penalty=1.15, no_repeat_ngram_size=4,
                                     pad_token_id=tok.pad_token_id)
            for r, g, inp in zip(batch, gen, enc["input_ids"]):
                raw = _strip_think(tok.decode(g[inp.shape[0]:], skip_special_tokens=True))
                cleaned = _clean_lead(raw)
                if cleaned != raw:
                    n_lead_cleaned += 1
                out.write(json.dumps({
                    "scenario_id": r["id"], "model": "ours_v4", "condition": "grounded",
                    "tier": r["tier"], "phase": r["phase"], "severity": r["severity"],
                    "pos_id": r["pos_id"], "output": cleaned, "output_raw": raw,
                    "lead_cleaned": cleaned != raw,
                    "prompt_tokens": int(inp.shape[0]), "completion_tokens": 0,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            if (i // BATCH_SIZE) % 5 == 0 or i + BATCH_SIZE >= len(todo):
                dt = time.time() - t0
                n = i + len(batch)
                print(f"  {n}/{len(todo)} ({dt/max(1,n):.2f}s/it, "
                      f"eta {dt/max(1,n)*(len(todo)-n)/60:.0f}m, lead_cleaned={n_lead_cleaned})")
                volume.commit()
    volume.commit()
    return {"written": written, "total": len(rows), "lead_cleaned": n_lead_cleaned,
            "secs": round(time.time() - t0, 1), "out": REMOTE_OUT}


@app.local_entrypoint()
def main(limit: int = 0, block: bool = False) -> None:
    call = generate.spawn(limit=limit)
    print(f"SPAWNED generate call_id={call.object_id} — running detached on Modal; "
          f"poll {VOL_MOUNT}/{RUN_NAME}/ours_v4_gen.jsonl on the Volume for completion.")
    if block:
        res = call.get()
        print(json.dumps(res, indent=2, default=str))
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = LOCAL_OUT.parent / "_ours_v4_dl"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                        VOLUME_NAME, f"/{RUN_NAME}/ours_v4_gen.jsonl", str(tmp)], check=True)
        got = tmp / "ours_v4_gen.jsonl"
        if got.exists():
            shutil.move(str(got), str(LOCAL_OUT))
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"DONE -> {LOCAL_OUT}")
