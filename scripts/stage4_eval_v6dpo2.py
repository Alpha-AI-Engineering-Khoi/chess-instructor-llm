#!/usr/bin/env python3
"""Stage-4 corrected-benchmark re-eval for v6-dpo2 (apples-to-apples with v4/v6-dpo).

A focused sibling of ``scripts/stage4_eval.py``: ONE Modal session loads the base
32B once, swaps adapters, and generates the GROUNDED coach condition over the SAME
120 held-out TEST positions x 3 tiers (``data/benchmark_gap803/stage4_eval_inputs.jsonl``)
with the byte-identical greedy decode Stage-4 used. Scoring is LOCAL + deterministic
with the SAME vendored extractor the shipped v4 report uses
(``src.eval.evaluate.extract_recommended_move``), so the v6-dpo2 row is directly
comparable to the v4 / v6-dpo rows in ``RESULTS_STAGE4_CORRECTED.md``.

By default it runs THREE grounded passes (v4, v6-dpo, v6-dpo2) in the one session so
the table is fully single-session and self-verifies that the pulled v4 / v6-dpo numbers
reproduce. Pass ``--passes v6dpo2`` to run only the new adapter (cheaper) and pull v4 /
v6-dpo from the doc.

Commands (ALWAYS scrub the bare kim-lam tokens + pin the FUNDED workspace)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    export MODAL_PROFILE=chess-instructor-2
    P=/Users/khoilam/.venvs/mlx/bin/modal
    $P run scripts/stage4_eval_v6dpo2.py --limit 6                 # smoke
    $P run scripts/stage4_eval_v6dpo2.py                          # full 360, all 3 grounded passes
    $P run scripts/stage4_eval_v6dpo2.py --passes v6dpo2          # only the new adapter
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME: str = "chess-coach-stage4-v6dpo2"
VOLUME_NAME: str = "chess-coach-lora"
VOL_MOUNT: str = "/vol"
STAGE4_REMOTE_DIR: str = f"{VOL_MOUNT}/stage4_v6dpo2"

REMOTE_INPUTS: str = "/data/stage4_eval_inputs.jsonl"
V4_ADAPTER_REMOTE: str = "/adapters/v4"
V6DPO_ADAPTER_REMOTE: str = "/adapters/v6-dpo"
V6DPO2_ADAPTER_REMOTE: str = "/adapters/v6-dpo2"

BASE_MODEL: str = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
MAX_SEQ_LEN: int = 3072
GROUNDED_MAX_NEW: int = 256   # identical to Stage-4 grounded
EVAL_BATCH: int = 24
TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")

# (name, adapter-key). adapter-key indexes ADAPTERS below.
ALL_PASSES: List[Tuple[str, str]] = [
    ("v4_grounded", "v4"),
    ("v6dpo_grounded", "v6_dpo"),
    ("v6dpo2_grounded", "v6_dpo2"),
]

GPU: str = "A100-80GB"
TIMEOUT_S: int = 2 * 3600
CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION: str = "3.11"
PIP_PACKAGES: List[str] = [
    "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
    "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_INPUTS = REPO_ROOT / "data" / "benchmark_gap803" / "stage4_eval_inputs.jsonl"
LOCAL_V4 = REPO_ROOT / "models" / "adapters" / "chess-coach-v4" / "adapter"
LOCAL_V6DPO = REPO_ROOT / "models" / "adapters" / "chess-coach-v6-dpo"
LOCAL_V6DPO2 = REPO_ROOT / "models" / "adapters" / "chess-coach-v6-dpo2"
LOCAL_OUT_DIR = REPO_ROOT / "data" / "benchmark_gap803" / "stage4_v6dpo2"

if modal.is_local():
    _need = [LOCAL_INPUTS, LOCAL_V4 / "adapter_model.safetensors",
             LOCAL_V6DPO / "adapter_model.safetensors",
             LOCAL_V6DPO2 / "adapter_model.safetensors"]
    _missing = [p for p in _need if not p.exists()]
    if _missing:
        raise SystemExit(
            "BLOCKED: missing input(s):\n  " + "\n  ".join(str(p) for p in _missing)
            + "\n(train + push v6-dpo2 first: scripts/train_dpo_v6dpo2.py)")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(*PIP_PACKAGES)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1"})
)
if modal.is_local():
    image = (
        image
        .add_local_file(LOCAL_INPUTS.as_posix(), REMOTE_INPUTS)
        .add_local_dir(LOCAL_V4.as_posix(), V4_ADAPTER_REMOTE)
        .add_local_dir(LOCAL_V6DPO.as_posix(), V6DPO_ADAPTER_REMOTE)
        .add_local_dir(LOCAL_V6DPO2.as_posix(), V6DPO2_ADAPTER_REMOTE)
    )

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

ADAPTERS: Dict[str, str] = {
    "v4": V4_ADAPTER_REMOTE, "v6_dpo": V6DPO_ADAPTER_REMOTE, "v6_dpo2": V6DPO2_ADAPTER_REMOTE,
}


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume})
def generate(pass_keys: List[str], limit: int = 0,
             grounded_max_new: int = GROUNDED_MAX_NEW) -> Dict[str, List[dict]]:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    import torch
    from peft import PeftModel

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"[gpu] torch={torch.__version__} cuda={torch.cuda.is_available()} device={name}", flush=True)
    os.makedirs(STAGE4_REMOTE_DIR, exist_ok=True)

    rows = [json.loads(x) for x in open(REMOTE_INPUTS, encoding="utf-8") if x.strip()]
    if limit:
        rows = rows[:limit]
    passes = [(nm, ad) for (nm, ad) in ALL_PASSES if ad in pass_keys]
    print(f"[eval] scenarios={len(rows)} passes={[p[0] for p in passes]}", flush=True)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    keys = list({ad for _, ad in passes})
    first = keys[0]
    print(f"[peft] loading adapters {keys}", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTERS[first], adapter_name=first)
    for k in keys[1:]:
        model.load_adapter(ADAPTERS[k], adapter_name=k)
    FastLanguageModel.for_inference(model)

    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": r["grounded_system"]},
             {"role": "user", "content": r["grounded_user"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for r in rows
    ]

    results: Dict[str, List[dict]] = {}
    for name_pass, adapter in passes:
        model.set_adapter(adapter)
        outs: List[dict] = []
        t0 = time.time()
        for i in range(0, len(texts), EVAL_BATCH):
            batch = texts[i:i + EVAL_BATCH]
            tok.padding_side = "left"
            tok.truncation_side = "left"
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SEQ_LEN).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=grounded_max_new, do_sample=False,
                                     repetition_penalty=1.15, no_repeat_ngram_size=4,
                                     pad_token_id=tok.pad_token_id)
            for j, (g, inp) in enumerate(zip(gen, enc["input_ids"])):
                raw = _strip_think(tok.decode(g[inp.shape[0]:], skip_special_tokens=True))
                outs.append({"i": i + j, "id": rows[i + j]["id"], "output": raw})
        dt = time.time() - t0
        print(f"[pass] {name_pass}: {len(outs)} gens in {dt:.0f}s ({dt/max(1,len(outs)):.2f}s/gen)", flush=True)
        results[name_pass] = outs
        with open(f"{STAGE4_REMOTE_DIR}/{name_pass}.json", "w", encoding="utf-8") as fh:
            json.dump(outs, fh)
        try:
            volume.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt] commit failed: {exc}", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Local scoring (byte-identical extractor to reproduce_v4.py / stage4_eval.py)
# --------------------------------------------------------------------------- #
def _load_inputs() -> List[dict]:
    return [json.loads(x) for x in LOCAL_INPUTS.read_text(encoding="utf-8").splitlines() if x.strip()]


def _extract(output: str, fen: str, student_uci: str) -> Optional[str]:
    from src.eval.evaluate import extract_recommended_move
    _san, uci = extract_recommended_move(output or "", fen, student_uci or "")
    return uci


def score_condition(outputs: List[dict], inputs: List[dict]) -> Dict[str, Any]:
    from statistics import mean

    by_id = {r["id"]: r for r in inputs}
    by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIERS}
    sound = [0, 0]
    named = [0, 0]
    fmt = [0, 0]
    preds_by_pos: Dict[str, Dict[str, Optional[str]]] = {}
    canon_by_pos: Dict[str, Dict[str, Optional[str]]] = {}
    for o in outputs:
        r = by_id.get(o["id"])
        if r is None:
            continue
        tier = r["tier"]
        uci = _extract(o["output"], r["fen"], r.get("student_uci") or "")
        if tier in by_tier:
            by_tier[tier][1] += 1
            if uci and uci == r.get("canonical_uci"):
                by_tier[tier][0] += 1
        sound[1] += 1
        if uci and uci in set(r.get("sound_ucis", [])):
            sound[0] += 1
        named[1] += 1
        if uci:
            named[0] += 1
        fmt[1] += 1
        text = (o["output"] or "")
        if uci and ("I'd play" in text or "I\u2019d play" in text) and "Takeaway:" in text:
            fmt[0] += 1
        preds_by_pos.setdefault(r["pos_id"], {})[tier] = uci
        canon_by_pos.setdefault(r["pos_id"], {})["beginner"] = r.get("canonical_beginner_uci")
        canon_by_pos[r["pos_id"]]["advanced"] = r.get("canonical_advanced_uci")

    per_tier = {t: (by_tier[t][0] / by_tier[t][1]) for t in TIERS if by_tier[t][1]}
    diff = dist = 0
    for pid, cd in canon_by_pos.items():
        cb, ca = cd.get("beginner"), cd.get("advanced")
        if not (cb and ca and cb != ca):
            continue
        diff += 1
        mb = preds_by_pos.get(pid, {}).get("beginner")
        ma = preds_by_pos.get(pid, {}).get("advanced")
        if mb and ma and mb != ma:
            dist += 1
    return {
        "tier_policy_match": round(mean(per_tier.values()), 4) if per_tier else 0.0,
        "per_tier": {t: round(v, 4) for t, v in per_tier.items()},
        "per_tier_counts": {t: by_tier[t] for t in TIERS if by_tier[t][1]},
        "move_sound": round(sound[0] / sound[1], 4) if sound[1] else 0.0,
        "named_rate": round(named[0] / named[1], 4) if named[1] else 0.0,
        "format_rate": round(fmt[0] / fmt[1], 4) if fmt[1] else 0.0,
        "distinct_rate": round(dist / diff, 4) if diff else 0.0,
        "distinct_counts": [dist, diff],
        "n": len(outputs),
    }


@app.local_entrypoint()
def main(limit: int = 0, passes: str = "all", grounded_max_new: int = GROUNDED_MAX_NEW) -> None:
    if passes == "all":
        pass_keys = ["v4", "v6_dpo", "v6_dpo2"]
    elif passes == "v6dpo2":
        pass_keys = ["v6_dpo2"]
    else:
        pass_keys = [p.strip() for p in passes.split(",") if p.strip()]
    print(f"=== {APP_NAME} ({'SMOKE limit=%d' % limit if limit else 'FULL 360'}) passes={pass_keys} ===")
    inputs = _load_inputs()
    keep_ids = {r["id"] for r in (inputs[:limit] if limit else inputs)}

    t0 = time.time()
    results = generate.remote(pass_keys=pass_keys, limit=limit, grounded_max_new=grounded_max_new)
    print(f"[remote] generate() returned in {time.time()-t0:.0f}s")

    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, outs in results.items():
        with open(LOCAL_OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for o in outs:
                fh.write(json.dumps(o, ensure_ascii=False) + "\n")

    scen = [r for r in inputs if r["id"] in keep_ids]
    scores = {name: score_condition(outs, scen) for name, outs in results.items()}
    report = {
        "benchmark": "scenarios_v6 (corrected labels), 120 held-out TEST x 3 tiers",
        "n_scenarios": len(scen),
        "decode": {"do_sample": False, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4,
                   "grounded_max_new": grounded_max_new},
        "scores": scores,
    }
    (LOCAL_OUT_DIR / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Stage-4 corrected-benchmark primary metrics (120 TEST) ===")
    print(f"{'model/condition':22} {'tier_fit':>8} {'sound':>7} {'distinct':>9} {'named':>7} {'format':>7}")
    for name, s in scores.items():
        print(f"{name:22} {s['tier_policy_match']:>8.4f} {s['move_sound']:>7.4f} "
              f"{s['distinct_rate']:>9.4f} {s['named_rate']:>7.4f} {s['format_rate']:>7.4f}  "
              f"per_tier={s['per_tier']}")
    print(f"\nwrote generations + scores -> {LOCAL_OUT_DIR}")
