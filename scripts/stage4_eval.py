#!/usr/bin/env python3
"""Stage-4 consolidating re-eval of all OURS models on the CORRECTED v6 benchmark.

ONE Modal session (base 32B loaded once, adapters swapped) generates every model
condition over the 120 held-out TEST positions x 3 tiers (360 scenarios), so the
comparison is perfectly apples-to-apples (identical decode, identical prompt per
condition). Scoring is LOCAL + deterministic with the SAME vendored extractor the
shipped v4 report uses (``src.eval.evaluate.extract_recommended_move``), against
the corrected v6 labels baked into ``stage4_eval_inputs.jsonl``.

Passes (base = bare Qwen3-32B via PEFT ``disable_adapter``):
  GROUNDED (deployable coach prompt: sound-pool + Maia + facts)
    - base   : the base-vs-tuned anchor
    - v4     : shipped coach (head-to-head reference)
    - v6-dpo : preference-tuned coach (the head-to-head that matters)
  NO-GROUNDING (distillation prompt: no engine, no Maia — behavior in the weights)
    - base    : distill thesis anchor
    - v6-distill

Every pass is committed to the shared Volume under ``/stage4`` right after it
finishes, so a timeout/credit-kill still leaves the completed passes on disk.

Commands (ALWAYS scrub the bare kim-lam tokens + pin the FUNDED workspace)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    export MODAL_PROFILE=chess-instructor-2
    P=/Users/khoilam/.venvs/mlx/bin/modal
    $P run scripts/stage4_eval.py --limit 6      # smoke (measure timing/cost)
    $P run scripts/stage4_eval.py                # full 360-scenario TEST slice
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# --------------------------------------------------------------------------- #
# Names / paths
# --------------------------------------------------------------------------- #
APP_NAME: str = "chess-coach-stage4-eval"
VOLUME_NAME: str = "chess-coach-lora"
VOL_MOUNT: str = "/vol"
STAGE4_REMOTE_DIR: str = f"{VOL_MOUNT}/stage4"

REMOTE_INPUTS: str = "/data/stage4_eval_inputs.jsonl"
V4_ADAPTER_REMOTE: str = "/adapters/v4"
V6DPO_ADAPTER_REMOTE: str = "/adapters/v6-dpo"
V6DISTILL_ADAPTER_REMOTE: str = "/adapters/v6-distill"

BASE_MODEL: str = "unsloth/Qwen3-32B-unsloth-bnb-4bit"

# decode: identical greedy recipe to the v4/DPO honest eval
MAX_SEQ_LEN: int = 3072
GROUNDED_MAX_NEW: int = 256   # room for "I'd play <MOVE>." + 2-4 sentences + Takeaway
NOG_MAX_NEW: int = 120        # the distill target is short
EVAL_BATCH: int = 24
TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")

# GROUNDED passes use grounded_system/grounded_user; NOG passes use nog_system/nog_user.
# (name, condition, adapter)  adapter=None -> bare base via disable_adapter()
PASSES: List[Tuple[str, str, Optional[str]]] = [
    ("base_grounded", "grounded", None),
    ("v4_grounded", "grounded", "v4"),
    ("v6dpo_grounded", "grounded", "v6_dpo"),
    ("base_nog", "nog", None),
    ("v6distill_nog", "nog", "v6_distill"),
]

# --------------------------------------------------------------------------- #
# Modal infra
# --------------------------------------------------------------------------- #
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
LOCAL_V6DISTILL = REPO_ROOT / "models" / "adapters" / "chess-coach-v6-distill"
LOCAL_OUT_DIR = REPO_ROOT / "data" / "benchmark_gap803" / "stage4"

if modal.is_local():
    _missing = [p for p in (LOCAL_INPUTS,
                            LOCAL_V4 / "adapter_model.safetensors",
                            LOCAL_V6DPO / "adapter_model.safetensors",
                            LOCAL_V6DISTILL / "adapter_model.safetensors") if not p.exists()]
    if _missing:
        raise SystemExit(
            "BLOCKED: missing input(s):\n  " + "\n  ".join(str(p) for p in _missing)
            + "\n(run scripts/stage4_build_inputs.py first; ensure v4 + v6-dpo + v6-distill adapters are local)"
        )

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
        .add_local_dir(LOCAL_V6DISTILL.as_posix(), V6DISTILL_ADAPTER_REMOTE)
    )

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Remote generation
# --------------------------------------------------------------------------- #
def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume})
def generate(limit: int = 0, grounded_max_new: int = GROUNDED_MAX_NEW,
             nog_max_new: int = NOG_MAX_NEW) -> Dict[str, List[dict]]:
    import unsloth  # noqa: F401  (import before peft/transformers so patches apply)
    from unsloth import FastLanguageModel

    import torch
    from peft import PeftModel

    def banner() -> Optional[str]:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        print(f"[gpu] torch={torch.__version__} cuda={torch.cuda.is_available()} device={name}", flush=True)
        return name

    banner()
    os.makedirs(STAGE4_REMOTE_DIR, exist_ok=True)

    rows = [json.loads(x) for x in open(REMOTE_INPUTS, encoding="utf-8") if x.strip()]
    if limit:
        rows = rows[:limit]
    print(f"[eval] scenarios={len(rows)} passes={[p[0] for p in PASSES]}", flush=True)

    # base loaded ONCE
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # adapters: v4 + v6-dpo + v6-distill all baked into the image
    print("[peft] loading adapters (v4, v6_dpo, v6_distill)", flush=True)
    model = PeftModel.from_pretrained(model, V4_ADAPTER_REMOTE, adapter_name="v4")
    model.load_adapter(V6DPO_ADAPTER_REMOTE, adapter_name="v6_dpo")
    model.load_adapter(V6DISTILL_ADAPTER_REMOTE, adapter_name="v6_distill")
    FastLanguageModel.for_inference(model)

    def render(cond: str) -> List[str]:
        sk, uk = ("grounded_system", "grounded_user") if cond == "grounded" else ("nog_system", "nog_user")
        return [
            tok.apply_chat_template(
                [{"role": "system", "content": r[sk]}, {"role": "user", "content": r[uk]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for r in rows
        ]

    tok.padding_side = "left"
    tok.truncation_side = "left"

    results: Dict[str, List[dict]] = {}
    for name, cond, adapter in PASSES:
        texts = render(cond)
        max_new = grounded_max_new if cond == "grounded" else nog_max_new
        outs: List[dict] = []
        t0 = time.time()

        def run_batches() -> None:
            for i in range(0, len(texts), EVAL_BATCH):
                batch = texts[i:i + EVAL_BATCH]
                # decoder-only: MUST left-pad for correct batched gen. Re-assert per
                # batch (unsloth/for_inference can reset the tokenizer's padding side).
                tok.padding_side = "left"
                tok.truncation_side = "left"
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=MAX_SEQ_LEN).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                         repetition_penalty=1.15, no_repeat_ngram_size=4,
                                         pad_token_id=tok.pad_token_id)
                for j, (g, inp) in enumerate(zip(gen, enc["input_ids"])):
                    raw = _strip_think(tok.decode(g[inp.shape[0]:], skip_special_tokens=True))
                    outs.append({"i": i + j, "id": rows[i + j]["id"], "output": raw})

        if adapter is None:
            with model.disable_adapter():
                run_batches()
        else:
            model.set_adapter(adapter)
            run_batches()
        dt = time.time() - t0
        print(f"[pass] {name}: {len(outs)} gens in {dt:.0f}s ({dt/max(1,len(outs)):.2f}s/gen)", flush=True)
        results[name] = outs
        # checkpoint this pass to the volume immediately
        with open(f"{STAGE4_REMOTE_DIR}/{name}.json", "w", encoding="utf-8") as fh:
            json.dump(outs, fh)
        try:
            volume.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt] commit failed: {exc}", flush=True)

    return results


# --------------------------------------------------------------------------- #
# Local scoring (byte-identical extractor to reproduce_v4.py) + report
# --------------------------------------------------------------------------- #
def _load_inputs() -> List[dict]:
    return [json.loads(x) for x in LOCAL_INPUTS.read_text(encoding="utf-8").splitlines() if x.strip()]


def _extract(output: str, fen: str, student_uci: str) -> Optional[str]:
    from src.eval.evaluate import extract_recommended_move
    _san, uci = extract_recommended_move(output or "", fen, student_uci or "")
    return uci


def score_condition(outputs: List[dict], inputs: List[dict]) -> Dict[str, Any]:
    """Deterministic primary metrics vs the corrected v6 labels."""
    from statistics import mean

    by_id = {r["id"]: r for r in inputs}
    by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIERS}
    sound = [0, 0]
    named = [0, 0]
    fmt = [0, 0]
    # for distinct: pos_id -> {tier: pred_uci}
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
    # distinct-moves-per-level: positions whose canonical beginner!=advanced (all-opportunity denom)
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
def main(limit: int = 0, grounded_max_new: int = GROUNDED_MAX_NEW,
         nog_max_new: int = NOG_MAX_NEW) -> None:
    print(f"=== {APP_NAME} ({'SMOKE limit=%d' % limit if limit else 'FULL 360'}) ===")
    inputs = _load_inputs()
    if limit:
        keep_ids = {r["id"] for r in inputs[:limit]}
    else:
        keep_ids = {r["id"] for r in inputs}

    t0 = time.time()
    results = generate.remote(
        limit=limit, grounded_max_new=grounded_max_new, nog_max_new=nog_max_new)
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
                   "grounded_max_new": grounded_max_new, "nog_max_new": nog_max_new},
        "scores": scores,
    }
    (LOCAL_OUT_DIR / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Stage-4 corrected-benchmark primary metrics (120 TEST) ===")
    hdr = f"{'model/condition':22} {'tier_fit':>8} {'sound':>7} {'distinct':>9} {'named':>7} {'format':>7}"
    print(hdr)
    for name, s in scores.items():
        print(f"{name:22} {s['tier_policy_match']:>8.4f} {s['move_sound']:>7.4f} "
              f"{s['distinct_rate']:>9.4f} {s['named_rate']:>7.4f} {s['format_rate']:>7.4f}  "
              f"per_tier={s['per_tier']}")
    print(f"\nwrote generations + scores -> {LOCAL_OUT_DIR}")
