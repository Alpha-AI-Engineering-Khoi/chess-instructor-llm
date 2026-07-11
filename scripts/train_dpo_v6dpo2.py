#!/usr/bin/env python3
"""STRONGER, tier-targeted DPO ("v6-dpo2") — a clean SUPERSET successor to v6-dpo.

Motivation
----------
v6-dpo (``scripts/train_dpo_v6.py``) sharpened the tier-appropriate MOVE moat over
v4, but on the corrected 120 held-out TEST the gain was ENTIRELY the intermediate
tier (+0.058); beginner and advanced did NOT move (see RESULTS_STAGE4_CORRECTED.md).
Diagnosis: v6-dpo's balanced pair set used the single ``dpo_rejected_uci`` contrast
(abundant for beginner, scarce for intermediate/advanced) and those beginner/advanced
negatives were not the *tempting wrong-tier* move. This run fixes the DATA, not the
recipe: HARD, tier-targeted negatives on the two weak tiers, while preserving the
proven intermediate pairs.

The key change: HARDER, tier-targeted preference pairs
-----------------------------------------------------
Style-matched pairs exactly as v6-dpo (chosen = the row's real v4-style assistant
prose; rejected = the SAME prose with ONLY the move SAN swapped), so DPO learns the
MOVE preference, not prose style. Each position carries a deep-verified canonical
move PER TIER (``select_tier_v6``); ``advanced`` canonical is always the engine-best
(sharpest) move, ``beginner``/``intermediate`` are the human-appropriate softer moves.
We exploit that to build the *tempting* negative for each weak tier:

* BEGINNER (weak): rejected = the ADVANCED / engine-best move — the too-sharp move a
  beginner should NOT be handed. Teaches "prefer the human-appropriate beginner move
  over the sharp one."
* ADVANCED (weak): rejected = the sound-but-not-best BEGINNER (and, where distinct,
  INTERMEDIATE) move. Teaches "prefer the sharpest engine-best move."
* INTERMEDIATE (preserve): the ORIGINAL v6-dpo ``dpo_rejected_uci`` pairs, kept to
  hold the +0.058 intermediate gain (do not drop it).

Negatives are SOUND (both moves sit in the engine sound pool; median cp gap ~11) —
they differ by tier-appropriateness, not blunder, which is the hard regime. Pairs are
weighted toward the discriminating positions (``high_conf_discriminating`` /
``distinct_moves``, i.e. where the tiers genuinely diverge and the wrong-tier move is
tempting) and stratified by phase, then balanced with per-tier targets weighted toward
beginner+advanced (default 750/450/800 = 2000 pairs, ~2.4x v6-dpo's 840). Override with
env ``V6DPO2_BEG`` / ``V6DPO2_INTER`` / ``V6DPO2_ADV``.

Method (identical, proven two-adapter TRL DPO on the 32B QLoRA)
--------------------------------------------------------------
Same base ``unsloth/Qwen3-32B-unsloth-bnb-4bit``. Policy(default)=v4 and
reference(frozen)=v4 (init BOTH from the v4 LoRA, NOT from v6-dpo -> clean superset,
avoids DPO-on-DPO drift). beta=0.1, lr=1e-5, ~1 epoch, checkpoint every 25 steps and
commit each to the shared Volume (a timeout / credit-kill leaves a usable adapter).

Selection (STRICT no-regression)
--------------------------------
Every checkpoint (+ the v4 baseline) is generated over ``valid_v6`` (game-disjoint dev)
with the SAME grounded decode as Stage-4 (greedy, rep-penalty 1.15, no-repeat-4,
256 new tokens so soundness AND the Takeaway format are measurable), scored LOCALLY
with the canonical ``extract_recommended_move``. We pick the checkpoint with the best
OVERALL tier-policy exact match SUBJECT TO a strict NO-REGRESSION floor vs v4's dev
numbers: every tier (beginner/intermediate/advanced), move-soundness, names-a-move and
format must all be >= v4. The 120-position held-out TEST is NEVER touched here (that is
Stage-4's job: ``scripts/stage4_eval.py`` / ``scripts/stage4_eval_v6dpo2.py``).

Commands (ALWAYS scrub the bare kim-lam tokens + pin the FUNDED workspace)::

    python scripts/train_dpo_v6dpo2.py                  # LOCAL dry-run: build+print pairs, exit
    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    export MODAL_PROFILE=chess-instructor-2
    P=/Users/khoilam/.venvs/mlx/bin/modal
    $P run scripts/train_dpo_v6dpo2.py --smoke              # tiny loop (proves 2-adapter DPO + times it)
    $P run --detach scripts/train_dpo_v6dpo2.py --skip-eval # full short pass (resumable, checkpoints as it goes)
    $P run scripts/train_dpo_v6dpo2.py --skip-train         # eval every ckpt + no-regression select + push
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# --------------------------------------------------------------------------- #
# Names / paths
# --------------------------------------------------------------------------- #
APP_NAME: str = "chess-coach-dpo-v6dpo2"
VOLUME_NAME: str = "chess-coach-lora"            # shared volume; v6-dpo2 uses its own run dir
RUN_NAME: str = "chess-coach-v6-dpo2"

VOL_MOUNT: str = "/vol"
REMOTE_PAIRS: str = "/data/dpo_pairs_v6dpo2.jsonl"
REMOTE_VALID: str = "/data/valid_v6.jsonl"
V4_ADAPTER_REMOTE: str = "/v4_adapter"           # baked-in v4 LoRA (policy + reference init)
ADAPTER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
TRAINER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/_trainer"

HF_ADAPTER_REPO: str = "khoilamalphaai/chess-coach-32b-v6-dpo2"

# --------------------------------------------------------------------------- #
# Hyper-parameters
# --------------------------------------------------------------------------- #
BASE_MODEL: str = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
MAX_SEQ_LEN: int = 2048
MAX_PROMPT_LEN: int = 1600

LEARNING_RATE: float = 1e-5      # low end: improve v4 without regressing it
DPO_BETA: float = 0.1            # KL/reference pressure vs the frozen v4 reference
NUM_EPOCHS: float = 1.0          # one short pass
WARMUP_RATIO: float = 0.1
LR_SCHEDULER: str = "cosine"
WEIGHT_DECAY: float = 0.0
OPTIMIZER: str = "adamw_8bit"
PER_DEVICE_BATCH: int = 1
GRAD_ACCUM: int = 8              # eff batch 8 (DPO forms chosen+rejected per example)
LOGGING_STEPS: int = 1
SAVE_STEPS: int = 25
SAVE_TOTAL_LIMIT: int = 20      # keep EVERY checkpoint for per-ckpt no-regression selection
SEED: int = 3407

# Per-tier pair targets (weighted toward the two weak tiers: beginner + advanced).
# Availability (clean, guarded): beginner 2010 / intermediate 514 / advanced 2146.
BEG_TARGET: int = int(os.environ.get("V6DPO2_BEG", "750"))
INTER_TARGET: int = int(os.environ.get("V6DPO2_INTER", "450"))
ADV_TARGET: int = int(os.environ.get("V6DPO2_ADV", "800"))
SMOKE_MAX_PAIRS: int = 24
SMOKE_MAX_STEPS: int = 8

# eval decode: identical grounded recipe to Stage-4 (so dev numbers are on the same
# footing as the TEST). 256 new tokens => soundness AND the Takeaway format are both
# measurable for the strict no-regression gate.
EVAL_MAX_NEW_TOKENS: int = 256
EVAL_BATCH: int = 24
TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")

ASSISTANT_MARKER: str = "<|im_start|>assistant\n"

# --------------------------------------------------------------------------- #
# Modal infra
# --------------------------------------------------------------------------- #
GPU: str = "A100-80GB"
TIMEOUT_S: int = 5 * 3600
CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION: str = "3.11"

PIP_PACKAGES: List[str] = [
    "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
    "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
]


# --------------------------------------------------------------------------- #
# Local DPO-pair construction (runs at import under modal.is_local())
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _rank_bucket(rank: Optional[int]) -> str:
    if rank is None:
        return "na"
    if rank <= 0:
        return "best"
    if rank <= 2:
        return "near"
    if rank <= 5:
        return "mid"
    return "tail"


def _swap_move(text: str, canonical_san: str, rejected_san: str) -> str:
    """Replace every whole-token occurrence of ``canonical_san`` with
    ``rejected_san`` (SAN tokens use letters/digits/+#=xO- so we guard those on both
    sides), producing a style-matched rejected response (byte-identical except move)."""
    pat = re.compile(
        r"(?<![A-Za-z0-9+#=x\-])" + re.escape(canonical_san) + r"(?![A-Za-z0-9+#=x\-])"
    )
    return pat.sub(rejected_san, text)


def _resolve_san(fen: str, uci: str, pool: List[dict]) -> Optional[str]:
    for p in pool:
        if p.get("uci") == uci and p.get("san"):
            return p["san"]
    try:
        import chess

        board = chess.Board(fen)
        return board.san(chess.Move.from_uci(uci))
    except Exception:  # noqa: BLE001
        return None


def _cp_of(prov: dict, uci: str) -> Optional[int]:
    for m in prov.get("sound_pool", []):
        if m.get("uci") == uci:
            return m.get("cp")
    return None


def _mk_pair(cell: dict, rej_uci: Optional[str], tier: str, kind: str) -> Optional[dict]:
    """Build ONE style-matched pair for a tier's row (``cell``): chosen = the row's
    canonical prose, rejected = same prose with the move swapped to ``rej_uci``.
    Returns None if the swap cannot be made cleanly (matches v6-dpo's guards)."""
    prov = cell["prov"]
    prose = cell["prose"]
    can_uci = prov.get("canonical_uci")
    can_san = prov.get("canonical_san") or ""
    if not rej_uci or rej_uci == can_uci:
        return None
    if not prose.startswith(f"I'd play {can_san}."):
        return None
    rej_san = _resolve_san(prov["fen"], rej_uci, prov.get("sound_pool", []))
    if not rej_san or rej_san == can_san:
        return None
    rejected = _swap_move(prose, can_san, rej_san)
    if rejected == prose or not rejected.startswith(f"I'd play {rej_san}."):
        return None
    cp_can, cp_rej = _cp_of(prov, can_uci), _cp_of(prov, rej_uci)
    cp_gap = (cp_can - cp_rej) if (cp_can is not None and cp_rej is not None) else None
    return {
        "system": cell["system"],
        "user": cell["user"],
        "chosen": prose,
        "rejected": rejected,
        "meta": {
            "tier": tier,
            "kind": kind,
            "pos_id": prov.get("pos_id"),
            "phase": prov.get("phase"),
            "rank_bucket": _rank_bucket(prov.get("canonical_pool_rank")),
            "weight": float(prov.get("weight", 1.0)),
            "high_conf": bool(prov.get("high_conf_discriminating")),
            "distinct_moves": int(prov.get("distinct_moves") or 0),
            "cp_gap": cp_gap,
            "canonical_uci": can_uci,
            "rejected_uci": rej_uci,
        },
    }


def build_tiered_pairs(train_path: Path) -> Dict[str, List[dict]]:
    """Tier-targeted HARD preference pairs grouped by tier.

    beginner : reject = advanced/engine-best (too sharp) — teach the softer move.
    advanced : reject = sound-but-not-best beginner (and distinct intermediate) — teach the sharpest.
    intermediate : reject = the original v6 ``dpo_rejected_uci`` — preserve the gain.
    """
    from collections import defaultdict

    by_pos: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for row in _iter_jsonl(train_path):
        prov = row["provenance"]
        msgs = row["messages"]
        if len(msgs) < 3 or msgs[2]["role"] != "assistant":
            continue
        by_pos[prov["pos_id"]][prov["tier"]] = {
            "prov": prov, "prose": msgs[2]["content"],
            "system": msgs[0]["content"], "user": msgs[1]["content"],
        }

    out: Dict[str, List[dict]] = {t: [] for t in TIERS}
    for _pid, d in by_pos.items():
        if not all(t in d for t in TIERS):
            continue
        beg, inter, adv = d["beginner"], d["intermediate"], d["advanced"]
        bc = beg["prov"].get("canonical_uci")
        ic = inter["prov"].get("canonical_uci")
        ac = adv["prov"].get("canonical_uci")
        # BEGINNER: prefer the softer move over the sharp advanced/engine-best move.
        if ac and bc and ac != bc:
            p = _mk_pair(beg, ac, "beginner", "beg_vs_sharp")
            if p:
                out["beginner"].append(p)
            # ADVANCED: prefer the sharpest move over the sound-but-not-best beginner move.
            p = _mk_pair(adv, bc, "advanced", "adv_vs_softbeg")
            if p:
                out["advanced"].append(p)
        # ADVANCED (harder, distinct): reject = the intermediate move where it differs.
        if ac and ic and ic != ac and ic != bc:
            p = _mk_pair(adv, ic, "advanced", "adv_vs_softinter")
            if p:
                out["advanced"].append(p)
        # INTERMEDIATE: original v6 contrast — preserve the proven intermediate gain.
        rj = inter["prov"].get("dpo_rejected_uci")
        if rj and rj != ic:
            p = _mk_pair(inter, rj, "intermediate", "inter_orig")
            if p:
                out["intermediate"].append(p)
    return out


def balance_tier(pairs: List[dict], target: int, seed: int = SEED) -> List[dict]:
    """Stratified sample to ``target`` within one tier: front-load the discriminating
    positions (where tiers genuinely diverge => the wrong-tier move is tempting), then
    spread across game phase so no phase is starved."""
    import random
    from collections import defaultdict

    if target <= 0 or not pairs:
        return []
    rng = random.Random(seed)
    want = min(target, len(pairs))
    strata: Dict[str, List[dict]] = defaultdict(list)
    for p in pairs:
        strata[p["meta"]["phase"] or "na"].append(p)
    # hardest first within each phase: high-confidence discriminating, more distinct
    # tier moves, higher provenance weight, then a stable jitter.
    for k in strata:
        strata[k].sort(key=lambda p: (
            not p["meta"]["high_conf"], -p["meta"]["distinct_moves"],
            -p["meta"]["weight"], rng.random(),
        ))
    keys = list(strata)
    idx = {k: 0 for k in keys}
    picked: List[dict] = []
    while len(picked) < want and any(idx[k] < len(strata[k]) for k in keys):
        rng.shuffle(keys)
        for k in keys:
            if len(picked) >= want:
                break
            if idx[k] < len(strata[k]):
                picked.append(strata[k][idx[k]])
                idx[k] += 1
    return picked


def _distribution(pairs: List[dict]) -> Dict[str, Any]:
    from collections import Counter

    gaps = [p["meta"]["cp_gap"] for p in pairs if p["meta"].get("cp_gap") is not None]
    return {
        "n": len(pairs),
        "tier": dict(Counter(p["meta"]["tier"] for p in pairs)),
        "kind": dict(Counter(p["meta"]["kind"] for p in pairs)),
        "phase": dict(Counter(p["meta"]["phase"] for p in pairs)),
        "high_conf": dict(Counter(p["meta"]["high_conf"] for p in pairs)),
        "cp_gap_median": (sorted(gaps)[len(gaps) // 2] if gaps else None),
    }


def _write_pairs(pairs: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


if modal.is_local():
    _THIS = Path(__file__).resolve().parent
    REPO_ROOT: Optional[Path] = _THIS.parent
    LOCAL_TRAIN: Optional[Path] = REPO_ROOT / "data" / "dataset" / "train_v6.jsonl"
    LOCAL_VALID: Optional[Path] = REPO_ROOT / "data" / "dataset" / "valid_v6.jsonl"
    LOCAL_V4: Optional[Path] = REPO_ROOT / "models" / "adapters" / "chess-coach-v4" / "adapter"
    LOCAL_PAIRS: Optional[Path] = REPO_ROOT / "data" / "dataset" / "_dpo_pairs_v6dpo2.jsonl"
    LOCAL_OUT_DIR: Optional[Path] = REPO_ROOT / "models" / "adapters" / RUN_NAME
    LOCAL_DEV_SCORES: Optional[Path] = REPO_ROOT / "data" / "dataset" / "_v6dpo2_dev_scores.json"

    _missing = [p for p in (LOCAL_TRAIN, LOCAL_VALID) if not p.exists()]
    _missing += [p for p in [LOCAL_V4 / "adapter_model.safetensors"] if not p.exists()]
    if _missing:
        raise SystemExit(
            "BLOCKED: missing input(s):\n  " + "\n  ".join(str(p) for p in _missing)
            + "\n(need v6 train/valid shards and the local v4 adapter)"
        )
    _cands = build_tiered_pairs(LOCAL_TRAIN)
    _avail = {t: len(_cands[t]) for t in TIERS}
    _bal = (
        balance_tier(_cands["beginner"], BEG_TARGET)
        + balance_tier(_cands["intermediate"], INTER_TARGET)
        + balance_tier(_cands["advanced"], ADV_TARGET)
    )
    import random as _random

    _random.Random(SEED).shuffle(_bal)
    _write_pairs(_bal, LOCAL_PAIRS)
    print(f"[pairs] available per tier: {_avail}")
    print(f"[pairs] targets beg/inter/adv = {BEG_TARGET}/{INTER_TARGET}/{ADV_TARGET}"
          f" -> balanced={len(_bal)}")
    print(f"[pairs] balanced distribution: {json.dumps(_distribution(_bal))}")
else:
    REPO_ROOT = LOCAL_TRAIN = LOCAL_VALID = LOCAL_V4 = LOCAL_PAIRS = None
    LOCAL_OUT_DIR = LOCAL_DEV_SCORES = None


image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(*PIP_PACKAGES)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1"})   # real-time logs (block-buffered stdout hides hangs)
)
if modal.is_local():
    image = (
        image
        .add_local_file(LOCAL_PAIRS.as_posix(), REMOTE_PAIRS)
        .add_local_file(LOCAL_VALID.as_posix(), REMOTE_VALID)
        .add_local_dir(LOCAL_V4.as_posix(), V4_ADAPTER_REMOTE)
    )

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Shared remote helpers
# --------------------------------------------------------------------------- #
def _gpu_banner() -> Optional[str]:
    import torch

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"[gpu] torch={torch.__version__} cuda={torch.cuda.is_available()} device={name}")
    return name


def _load_base_4bit(max_seq_len: int):
    """Load the v4 base 4-bit with Unsloth (so the v4 adapter sits on the SAME
    quantized base it was trained on), returning (model, tokenizer)."""
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=max_seq_len, load_in_4bit=True, dtype=None,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _render_pair(tok, system: str, user: str, chosen: str, rejected: str
                 ) -> Optional[Tuple[str, str, str]]:
    """(prompt, chosen_completion, rejected_completion) rendered EXACTLY as the v6
    SFT target: full-conversation chat template, split at the assistant marker."""
    def full(assistant: str) -> str:
        return tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user},
             {"role": "assistant", "content": assistant}],
            tokenize=False, add_generation_prompt=False,
        )
    fc = full(chosen)
    fr = full(rejected)
    ic = fc.rfind(ASSISTANT_MARKER)
    ir = fr.rfind(ASSISTANT_MARKER)
    if ic < 0 or ir < 0:
        return None
    prompt = fc[: ic + len(ASSISTANT_MARKER)]
    return prompt, fc[ic + len(ASSISTANT_MARKER):], fr[ir + len(ASSISTANT_MARKER):]


def _filter_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import inspect

    valid = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in valid}


def _resolve_adapter_dir(path: str) -> str:
    """A saved checkpoint may hold the adapter at the root or nested under the adapter
    name (``default``). Return the dir that actually has an adapter."""
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    for cand in ("default",):
        d = os.path.join(path, cand)
        if os.path.exists(os.path.join(d, "adapter_config.json")):
            return d
    return path


# --------------------------------------------------------------------------- #
# Remote: DPO train (two-adapter, reference = frozen v4)
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume})
def train(smoke: bool = False, beta: float = DPO_BETA, lr: float = LEARNING_RATE,
          epochs: float = NUM_EPOCHS, grad_accum: int = GRAD_ACCUM,
          save_steps: int = SAVE_STEPS) -> dict:
    import unsloth  # noqa: F401  (MUST precede trl/peft/transformers so its patches apply)
    from unsloth import is_bfloat16_supported

    import glob as _glob
    import time

    import torch  # noqa: F401
    from datasets import Dataset
    from peft import PeftModel
    from transformers import TrainerCallback
    from trl import DPOConfig, DPOTrainer

    gpu_name = _gpu_banner()
    # Smoke MUST NOT share the full run's dirs (its checkpoints would otherwise be
    # resumed by the full run and drag in the smoke's tiny-batch state + save cadence).
    run_trainer_dir = f"{TRAINER_DIR}_smoke" if smoke else TRAINER_DIR
    run_adapter_dir = f"{ADAPTER_DIR}_smoke" if smoke else ADAPTER_DIR
    t_load = time.time()
    model, tok = _load_base_4bit(MAX_SEQ_LEN)
    print(f"[peft] policy(default)=v4 + reference(frozen)=v4 from {V4_ADAPTER_REMOTE}")
    model = PeftModel.from_pretrained(model, V4_ADAPTER_REMOTE, is_trainable=True,
                                      adapter_name="default")
    model.load_adapter(V4_ADAPTER_REMOTE, adapter_name="reference")
    model.set_adapter("default")
    try:
        model.enable_input_require_grads()
    except Exception as exc:  # noqa: BLE001
        print(f"[peft] enable_input_require_grads: {exc}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    load_s = time.time() - t_load
    print(f"[peft] trainable params={n_train:,}  load_s={load_s:.0f}")

    rows = [json.loads(x) for x in open(REMOTE_PAIRS, encoding="utf-8") if x.strip()]
    if smoke:
        rows = rows[:SMOKE_MAX_PAIRS]
    recs: List[dict] = []
    for r in rows:
        rp = _render_pair(tok, r["system"], r["user"], r["chosen"], r["rejected"])
        if rp is None:
            continue
        prompt, chosen, rejected = rp
        recs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if not recs:
        raise RuntimeError("no DPO pairs rendered")
    dataset = Dataset.from_list(recs)
    print(f"[data] dpo_pairs={len(recs)} (smoke={smoke})")
    print("[data] sample prompt tail:\n" + recs[0]["prompt"][-320:])
    print("[data] sample chosen:  " + recs[0]["chosen"][:120].replace("\n", " "))
    print("[data] sample rejected:" + recs[0]["rejected"][:120].replace("\n", " "))

    max_steps = SMOKE_MAX_STEPS if smoke else -1
    eff_save = 4 if smoke else save_steps          # smoke: force a checkpoint to validate layout
    eff_accum = 1 if smoke else grad_accum
    cfg_kwargs = dict(
        output_dir=run_trainer_dir,
        beta=beta,
        model_adapter_name="default", ref_adapter_name="reference",
        max_length=MAX_SEQ_LEN, max_prompt_length=MAX_PROMPT_LEN,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=eff_accum,
        warmup_ratio=WARMUP_RATIO, num_train_epochs=(1.0 if smoke else epochs),
        max_steps=max_steps, learning_rate=lr, logging_steps=LOGGING_STEPS,
        optim=OPTIMIZER, weight_decay=WEIGHT_DECAY, lr_scheduler_type=LR_SCHEDULER,
        seed=SEED, save_strategy="steps", save_steps=eff_save,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0, dataset_num_proc=1,   # avoid fork deadlock (old kernel)
        remove_unused_columns=False, report_to="none",
    )
    dpo_config = DPOConfig(**_filter_kwargs(DPOConfig, cfg_kwargs))

    trainer_kwargs = dict(model=model, ref_model=None, args=dpo_config,
                          train_dataset=dataset, tokenizer=tok, processing_class=tok)
    trainer_kwargs = _filter_kwargs(DPOTrainer, trainer_kwargs)
    trainer = DPOTrainer(**trainer_kwargs)

    class _VolCommit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
            try:
                volume.commit()
                print(f"[ckpt] committed at step {state.global_step}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ckpt] commit failed: {exc}")

    trainer.add_callback(_VolCommit())

    resume = None
    if not smoke:
        volume.reload()
        ckpts = _glob.glob(f"{run_trainer_dir}/checkpoint-*")
        if ckpts:
            resume = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
            print(f"[resume] {resume}")
    print(f"[train] beta={beta} lr={lr} epochs={epochs} eff_batch={PER_DEVICE_BATCH*eff_accum} "
          f"max_steps={max_steps} save_steps={eff_save}")
    t_train = time.time()
    try:
        out = trainer.train(resume_from_checkpoint=resume)
    except Exception as exc:  # noqa: BLE001
        if resume is None:
            raise
        print(f"[resume] failed ({exc}); fresh restart")
        out = trainer.train()
    train_s = time.time() - t_train

    hist = trainer.state.log_history
    losses = [d["loss"] for d in hist if "loss" in d]
    accs = [d["rewards/accuracies"] for d in hist if "rewards/accuracies" in d]
    margins = [d.get("rewards/margins") for d in hist if "rewards/margins" in d]
    steps_done = len(losses)
    print(f"[train] steps={steps_done} first_loss={losses[0] if losses else None} "
          f"last_loss={losses[-1] if losses else None} last_acc={accs[-1] if accs else None} "
          f"train_s={train_s:.0f} s_per_step={train_s/max(1,steps_done):.1f}")

    print(f"[save] default adapter -> {run_adapter_dir}")
    model.save_pretrained(run_adapter_dir, selected_adapters=["default"])
    tok.save_pretrained(run_adapter_dir)
    volume.commit()

    volume.reload()
    ckpts = sorted(_glob.glob(f"{run_trainer_dir}/checkpoint-*"),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    return {
        "gpu": gpu_name, "smoke": smoke, "beta": beta, "lr": lr,
        "trainable_params": n_train, "n_pairs": len(recs),
        "steps": steps_done, "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "last_acc": accs[-1] if accs else None,
        "last_margin": margins[-1] if margins else None,
        "load_s": round(load_s), "train_s": round(train_s),
        "s_per_step": round(train_s / max(1, steps_done), 2),
        "adapter_dir": run_adapter_dir, "checkpoints": ckpts, "run_name": RUN_NAME,
        "metrics": getattr(out, "metrics", None),
    }


# --------------------------------------------------------------------------- #
# Remote: generate valid_v6 completions for a set of adapters (base loaded once)
# --------------------------------------------------------------------------- #
def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


def _clean_lead(text: str) -> str:
    t = text.strip()
    if t.startswith("I'd play") or t.startswith("I\u2019d play"):
        return t
    idx = t.find("I'd play")
    if idx < 0:
        idx = t.find("I\u2019d play")
    return t[idx:].strip() if 0 < idx <= 160 else t


@app.function(image=image, gpu=GPU, timeout=3 * 3600, volumes={VOL_MOUNT: volume})
def eval_valid(specs: Dict[str, str], limit: int = 0,
               max_new: int = EVAL_MAX_NEW_TOKENS) -> Dict[str, List[dict]]:
    """For each {name: adapter_dir}, greedy-generate over valid_v6 (grounded decode,
    identical to Stage-4) and return {name: [{"i", "output"}]} for LOCAL scoring."""
    import unsloth  # noqa: F401  (import before peft/transformers so its patches apply)
    from unsloth import FastLanguageModel

    import time

    import torch
    from peft import PeftModel

    volume.reload()
    rows = [json.loads(x) for x in open(REMOTE_VALID, encoding="utf-8") if x.strip()]
    if limit:
        rows = rows[:limit]
    prompts = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]
    print(f"[eval] valid rows={len(rows)} adapters={list(specs)} max_new={max_new}")

    model, tok = _load_base_4bit(MAX_SEQ_LEN + 1024)
    names = list(specs)
    dirs = {nm: _resolve_adapter_dir(specs[nm]) for nm in names}
    model = PeftModel.from_pretrained(model, dirs[names[0]], adapter_name=names[0])
    for nm in names[1:]:
        model.load_adapter(dirs[nm], adapter_name=nm)
    FastLanguageModel.for_inference(model)   # once (avoid per-adapter resets)

    results: Dict[str, List[dict]] = {}
    for nm in names:
        model.set_adapter(nm)
        outs: List[dict] = []
        t0 = time.time()
        for i in range(0, len(prompts), EVAL_BATCH):
            batch = prompts[i:i + EVAL_BATCH]
            texts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": s}, {"role": "user", "content": u}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for s, u in batch
            ]
            tok.padding_side = "left"        # decoder-only: MUST left-pad for correct batched gen
            tok.truncation_side = "left"
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SEQ_LEN + 1024).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new,
                                     do_sample=False, repetition_penalty=1.15,
                                     no_repeat_ngram_size=4, pad_token_id=tok.pad_token_id)
            for j, (g, inp) in enumerate(zip(gen, enc["input_ids"])):
                raw = _strip_think(tok.decode(g[inp.shape[0]:], skip_special_tokens=True))
                outs.append({"i": i + j, "output": _clean_lead(raw)})
        print(f"[eval] {nm}: {len(outs)} gens in {time.time()-t0:.0f}s")
        results[nm] = outs
        # Persist per-adapter so a client disconnect can't waste the GPU work:
        # the local scorer can recover via `--collect-only` (no GPU re-run).
        try:
            os.makedirs(f"{VOL_MOUNT}/{RUN_NAME}/_dev", exist_ok=True)
            with open(f"{VOL_MOUNT}/{RUN_NAME}/_dev/{nm}.json", "w", encoding="utf-8") as fh:
                json.dump(outs, fh)
            volume.commit()
            print(f"[eval] persisted {nm} -> volume")
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] persist {nm} failed: {exc}")
    return results


# --------------------------------------------------------------------------- #
# Remote: promote a chosen checkpoint's adapter to the run's adapter dir
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=1800, volumes={VOL_MOUNT: volume})
def promote(ckpt_dir: str) -> dict:
    volume.reload()
    src = ckpt_dir
    if not os.path.exists(os.path.join(src, "adapter_model.safetensors")):
        for cand in (os.path.join(ckpt_dir, "default"), ckpt_dir):
            if os.path.exists(os.path.join(cand, "adapter_model.safetensors")):
                src = cand
                break
    if not os.path.exists(os.path.join(src, "adapter_model.safetensors")):
        raise RuntimeError(f"no adapter_model.safetensors under {ckpt_dir}")
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    for fn in os.listdir(src):
        if fn.startswith("checkpoint-") or fn in ("optimizer.pt", "scheduler.pt"):
            continue
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(ADAPTER_DIR, fn))
    volume.commit()
    files = sorted(os.listdir(ADAPTER_DIR))
    print(f"[promote] {src} -> {ADAPTER_DIR}: {files}")
    return {"src": src, "adapter_dir": ADAPTER_DIR, "files": files}


@app.function(image=image, timeout=600, volumes={VOL_MOUNT: volume})
def _list_checkpoints() -> List[str]:
    import glob as _glob

    volume.reload()
    return sorted(_glob.glob(f"{TRAINER_DIR}/checkpoint-*"),
                  key=lambda p: int(p.rsplit("-", 1)[-1]))


# --------------------------------------------------------------------------- #
# Local scoring (canonical extractor) + orchestration
# --------------------------------------------------------------------------- #
def _score(outputs: List[dict], valid_rows: List[dict]) -> Dict[str, Any]:
    """Dev metrics vs canonical_uci: per-tier + overall tier-policy match, soundness,
    names-a-move, and the Takeaway format rate (all with the canonical extractor)."""
    from statistics import mean

    from src.eval.evaluate import extract_recommended_move

    by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIERS}
    sound = [0, 0]
    named = [0, 0]
    fmt = [0, 0]
    for o in outputs:
        row = valid_rows[o["i"]]
        prov = row["provenance"]
        tier = prov.get("tier")
        fen = prov["fen"]
        student_uci = (prov.get("student") or {}).get("uci") or ""
        text = o["output"] or ""
        _san, uci = extract_recommended_move(text, fen, student_uci)
        if tier in by_tier:
            by_tier[tier][1] += 1
            if uci and uci == prov.get("canonical_uci"):
                by_tier[tier][0] += 1
        named[1] += 1
        if uci:
            named[0] += 1
        pool = {p.get("uci") for p in prov.get("sound_pool", [])}
        sound[1] += 1
        if uci and uci in pool:
            sound[0] += 1
        fmt[1] += 1
        if uci and ("I'd play" in text or "I\u2019d play" in text) and "Takeaway:" in text:
            fmt[0] += 1
    per_tier = {t: (by_tier[t][0] / by_tier[t][1]) for t in TIERS if by_tier[t][1]}
    return {
        "tier_policy_match": round(mean(per_tier.values()), 4) if per_tier else 0.0,
        "per_tier": {t: round(v, 4) for t, v in per_tier.items()},
        "per_tier_counts": {t: by_tier[t] for t in TIERS if by_tier[t][1]},
        "move_sound": round(sound[0] / sound[1], 4) if sound[1] else 0.0,
        "named_rate": round(named[0] / named[1], 4) if named[1] else 0.0,
        "format_rate": round(fmt[0] / fmt[1], 4) if fmt[1] else 0.0,
        "n": len(outputs),
    }


def _no_regression(s: Dict[str, Any], v4: Dict[str, Any], eps: float = 1e-9) -> bool:
    """True iff ``s`` does not regress ANY tier / soundness / names-a-move / format
    versus the v4 dev numbers (the strict selection floor)."""
    for t in TIERS:
        if s["per_tier"].get(t, 0.0) < v4["per_tier"].get(t, 0.0) - eps:
            return False
    return (s["move_sound"] >= v4["move_sound"] - eps
            and s["named_rate"] >= v4["named_rate"] - eps
            and s["format_rate"] >= v4["format_rate"] - eps)


def _volume_get(remote_path: str, local_parent: Path) -> None:
    local_parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                    VOLUME_NAME, remote_path, str(local_parent)], check=True)


def _load_dev_from_volume(specs: Dict[str, str]) -> Dict[str, List[dict]]:
    """Download the per-adapter dev generations that eval_valid persisted and load
    them locally. Lets `--collect-only` finish scoring/selection after a client
    disconnect WITHOUT re-running the GPU eval."""
    dl = LOCAL_OUT_DIR.parent / "_v6dpo2_dev_dl"
    shutil.rmtree(dl, ignore_errors=True)
    _volume_get(f"/{RUN_NAME}/_dev", dl)
    base = dl / "_dev"
    src = base if base.exists() else dl
    out: Dict[str, List[dict]] = {}
    for nm in specs:
        f = src / f"{nm}.json"
        if f.exists():
            out[nm] = json.loads(f.read_text(encoding="utf-8"))
    return out


def _write_readme(local_dir: Path, selected: str, dev: Dict[str, Any]) -> None:
    (local_dir / "README.md").write_text(
        f"""---
base_model: {BASE_MODEL}
library_name: peft
tags: [chess, dpo, qlora, move-review, coaching]
---

# chess-coach-32b-v6-dpo2

STRONGER, tier-targeted DPO successor to `chess-coach-32b-v6-dpo`. A LoRA adapter on
`{BASE_MODEL}`, initialized from the shipped v4 LoRA
(`khoilamalphaai/chess-coach-32b-v4-qlora`) for BOTH policy and (frozen) reference —
a clean superset of v6-dpo (no DPO-on-DPO drift).

Trained with harder, tier-appropriate style-matched preference pairs: beginner learns
to prefer the human move over the sharp engine move; advanced learns to prefer the
sharpest engine-best move over the sound-but-softer move; intermediate keeps the
original v6 pairs to preserve its gain. See `scripts/train_dpo_v6dpo2.py`.

Selected checkpoint: `{selected}` (dev tier-policy match {dev.get('tier_policy_match')},
per-tier {dev.get('per_tier')}). Evaluate on the corrected 120 held-out TEST with
`scripts/stage4_eval_v6dpo2.py`.
""", encoding="utf-8")


def _push_hf(local_dir: Path, private: bool = True) -> str:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise SystemExit("BLOCKED: no HF_TOKEN in .env")
    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    api.create_repo(HF_ADAPTER_REPO, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(folder_path=str(local_dir), repo_id=HF_ADAPTER_REPO, repo_type="model")
    return f"https://huggingface.co/{HF_ADAPTER_REPO}"


@app.local_entrypoint()
def main(smoke: bool = False, beta: float = DPO_BETA, lr: float = LEARNING_RATE,
         epochs: float = NUM_EPOCHS, grad_accum: int = GRAD_ACCUM,
         save_steps: int = SAVE_STEPS,          skip_train: bool = False,
         skip_eval: bool = False, skip_push: bool = False, eval_limit: int = 0,
         eval_max_new: int = EVAL_MAX_NEW_TOKENS, collect_only: bool = False,
         eval_only: bool = False) -> None:
    print(f"=== {APP_NAME} ({'SMOKE' if smoke else 'FULL'}) ===")
    print(f"base={BASE_MODEL}  pairs={LOCAL_PAIRS}  v4_adapter={LOCAL_V4}")

    if not skip_train:
        res = train.remote(smoke=smoke, beta=beta, lr=lr, epochs=epochs,
                           grad_accum=grad_accum, save_steps=save_steps)
        print("\n=== train() ===\n" + json.dumps(res, indent=2, default=str))
        if smoke:
            return

    if skip_eval:
        print("[main] --skip-eval: stopping after train (run --skip-train to eval+select+push).")
        return

    # Evaluate v4 (baseline) + every DPO checkpoint on valid_v6, then no-regression select.
    # V6DPO2_SKIP_STEPS lets us drop known-bad checkpoints (e.g. a resume-reset final
    # step) from the dev eval to save budget.
    specs: Dict[str, str] = {"v4": V4_ADAPTER_REMOTE}
    _skip = {s for s in os.environ.get("V6DPO2_SKIP_STEPS", "").split(",") if s}
    ckpts = _list_checkpoints.remote()
    for c in ckpts:
        if c.rsplit("-", 1)[-1] in _skip:
            print(f"[main] skipping {os.path.basename(c)} (V6DPO2_SKIP_STEPS)")
            continue
        specs[os.path.basename(c)] = c   # final adapter == last checkpoint, so no separate entry
    print(f"[main] evaluating adapters: {list(specs)}")

    if eval_only:
        # Detach-safe GPU eval: run eval_valid (which persists per-adapter to the
        # volume) and stop. Recover scoring/select/push locally with --collect-only.
        print("[main] --eval-only: GPU eval + persist to volume only (run under "
              "`modal run --detach` so a client disconnect can't kill it)")
        eval_valid.remote(specs, limit=eval_limit, max_new=eval_max_new)
        print("[main] eval-only done; generations persisted to volume "
              f"/{RUN_NAME}/_dev/. Now run: --skip-train --collect-only")
        return

    if collect_only:
        print("[main] --collect-only: loading persisted dev generations from the volume "
              "(no GPU eval)")
        outputs = _load_dev_from_volume(specs)
        missing = [nm for nm in specs if nm not in outputs]
        if missing:
            print(f"[main] collect-only: no persisted generations for {missing} "
                  "(they will be excluded from selection)")
    else:
        outputs = eval_valid.remote(specs, limit=eval_limit, max_new=eval_max_new)
    valid_rows = [json.loads(x) for x in open(LOCAL_VALID, encoding="utf-8") if x.strip()]
    if eval_limit:
        valid_rows = valid_rows[:eval_limit]

    scores = {nm: _score(outs, valid_rows) for nm, outs in outputs.items()}
    print("\n=== valid_v6 dev metrics (selection) ===")
    for nm in scores:
        s = scores[nm]
        print(f"  {nm:26} match={s['tier_policy_match']:.4f} sound={s['move_sound']:.4f} "
              f"named={s['named_rate']:.3f} fmt={s['format_rate']:.3f} per_tier={s['per_tier']}")

    v4 = scores.get("v4", {})
    cand = {nm: s for nm, s in scores.items() if nm != "v4"}
    qualifying = {nm: s for nm, s in cand.items() if _no_regression(s, v4)}
    pool = qualifying or cand
    best = max(pool, key=lambda nm: (pool[nm]["tier_policy_match"], pool[nm]["move_sound"]))
    gate_ok = bool(qualifying) and best in qualifying
    print(f"\n[select] best={best} match={cand[best]['tier_policy_match']:.4f} "
          f"(v4={v4.get('tier_policy_match', 0):.4f}, "
          f"delta={cand[best]['tier_policy_match']-v4.get('tier_policy_match',0):+.4f})")
    print(f"[select] strict no-regression vs v4 satisfied: {gate_ok} "
          f"(qualifying checkpoints: {sorted(qualifying)})")
    if not gate_ok:
        print("[select] WARNING: no checkpoint met the strict no-regression floor; "
              "best-overall selected and FLAGGED (see report).")

    # Persist dev scores + selection for the report / results doc.
    if LOCAL_DEV_SCORES is not None:
        LOCAL_DEV_SCORES.write_text(json.dumps(
            {"scores": scores, "selected": best, "gate_ok": gate_ok,
             "qualifying": sorted(qualifying), "v4": v4,
             "delta_vs_v4": cand[best]["tier_policy_match"] - v4.get("tier_policy_match", 0)},
            indent=2), encoding="utf-8")
        print(f"[select] dev scores -> {LOCAL_DEV_SCORES}")

    if not skip_push:
        best_dir = specs[best]
        if best_dir != ADAPTER_DIR:
            print(promote.remote(best_dir))
        LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = LOCAL_OUT_DIR.parent / "_v6dpo2_dl"
        shutil.rmtree(tmp, ignore_errors=True)
        _volume_get(f"/{RUN_NAME}/adapter", tmp)
        got = tmp / "adapter"
        src = got if got.exists() else tmp
        shutil.rmtree(LOCAL_OUT_DIR, ignore_errors=True)
        shutil.move(str(src), str(LOCAL_OUT_DIR))
        shutil.rmtree(tmp, ignore_errors=True)
        _write_readme(LOCAL_OUT_DIR, best, cand[best])
        url = _push_hf(LOCAL_OUT_DIR)
        print(f"[push] adapter -> {url}  (volume: {VOLUME_NAME}:/{RUN_NAME}/adapter)")

    print("\n=== SUMMARY ===\n" + json.dumps(
        {"selected": best, "gate_ok": gate_ok,
         "dev_match": cand[best]["tier_policy_match"], "dev_per_tier": cand[best]["per_tier"],
         "delta_vs_v4": cand[best]["tier_policy_match"] - v4.get("tier_policy_match", 0)},
        indent=2))
