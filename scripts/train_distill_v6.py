#!/usr/bin/env python3
"""AMBITIOUS v6 ENGINE-DISTILLATION — Qwen3-32B QLoRA SFT on Modal (Unsloth).

Thesis-upgrade under test
-------------------------
v4/v6 SFT is handed a GROUNDED prompt (Stockfish sound-pool + evals + per-tier
Maia). The tier move is then grounded *execution* — read off the engine block.
This run STRIPS the engine/Maia block and trains the model to produce the
tier-appropriate move from the RAW BOARD + tier ALONE, so the tier-selection rule
must live in the WEIGHTS. The reformat + eval live in ``distill_v6_format.py``
(shared verbatim by the local sanity check).

Method
------
* Base init: ``unsloth/Qwen3-32B-unsloth-bnb-4bit`` (clean causal attribution —
  the base-vs-distill delta is exactly what THIS distillation added; v4-init would
  confound it with v4's prior grounded training).
* Unsloth QLoRA SFT, ``train_on_responses_only`` (mask the prompt).
* v6 sampling weights realized by deterministic fractional oversampling.
* Checkpoint every epoch; evaluate EACH on ``valid_v6`` in the SAME no-grounding
  format by **tier-policy exact match**; select the best checkpoint.
* Also evaluates the UNTUNED BASE in the same no-grounding format — the WIN
  CONDITION is distill-no-grounding > base-no-grounding.

Run (kim-lam tokens unset; NEVER kim-lam; workspace != the concurrent DPO run)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    MODAL_PROFILE=chess-instructor-4 modal run scripts/train_distill_v6.py::smoke
    MODAL_PROFILE=chess-instructor-4 modal run --detach scripts/train_distill_v6.py::train
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# --------------------------------------------------------------------------- #
# Names / layout
# --------------------------------------------------------------------------- #
APP_NAME = "chess-coach-32b-v6-distill"
VOLUME_NAME = "chess-data"
VOL = "/vol"
HF_DATASET_REPO = "khoilamalphaai/chess-coach-v6"
HF_ADAPTER_REPO = "khoilamalphaai/chess-coach-32b-v6-distill"
BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"

ROOT = Path(__file__).resolve().parents[1]
SEED = 3407

# Qwen3 ChatML response boundary (train_on_responses_only masks up to here).
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Unsloth pulls a compatible torch / transformers / trl / peft / bitsandbytes
    # / xformers stack. Keep it simple + current (the smoke test validates it).
    .pip_install("unsloth", "unsloth_zoo")
    .pip_install("hf_transfer", "huggingface_hub", "datasets", "python-chess")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    # Single source of truth for the reformat + eval extractor.
    .add_local_file(str(ROOT / "scripts" / "distill_v6_format.py"),
                    "/root/distill_v6_format.py", copy=True)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

# HF token (+ any other keys) from the repo .env, injected as container env.
hf_secret = modal.Secret.from_dotenv(str(ROOT))

HF_HOME = f"{VOL}/hf"          # persist the 32B 4-bit weights across runs
RUN_ROOT = f"{VOL}/distill_v6"


# --------------------------------------------------------------------------- #
# Robust constructors (survive TRL/Unsloth API drift)
# --------------------------------------------------------------------------- #
def _construct(cls, **kw):
    """Instantiate ``cls`` dropping any kwargs it does not accept."""
    while True:
        try:
            return cls(**kw)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if m and m.group(1) in kw:
                kw.pop(m.group(1))
                continue
            raise


def _fmt_train_text(tok, msgs: List[dict]) -> str:
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def _fmt_eval_prompt(tok, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------------------- #
# Batched greedy generation + tier-policy eval
# --------------------------------------------------------------------------- #
def _generate(model, tok, prompts: List[str], max_new_tokens: int, bs: int) -> List[str]:
    import torch

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs: List[str] = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1280, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return outs


def _evaluate(model, tok, F, eval_rows: List[dict], max_new_tokens: int, bs: int,
              label: str = "") -> Dict[str, Any]:
    """eval_rows: [{system,user,tier,fen,canonical_uci,student_uci}] -> tier-policy."""
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    prompts = [_fmt_eval_prompt(tok, r["system"], r["user"]) for r in eval_rows]
    texts = _generate(model, tok, prompts, max_new_tokens, bs)
    gen_rows = [
        {"tier": r["tier"], "fen": r["fen"], "canonical_uci": r["canonical_uci"],
         "student_uci": r["student_uci"], "output": t}
        for r, t in zip(eval_rows, texts)
    ]
    score = F.score_generations(gen_rows)
    for i in range(min(2, len(texts))):
        _san, pred = F.extract_recommended_move(
            F.strip_think(texts[i]), gen_rows[i]["fen"], gen_rows[i]["student_uci"])
        print(f"[eval:{label}] canonical={gen_rows[i]['canonical_uci']} pred={pred} "
              f"out={texts[i][:240]!r}", flush=True)
    score["_samples"] = [
        {"tier": gen_rows[i]["tier"], "canonical": gen_rows[i]["canonical_uci"],
         "out": texts[i][:200]}
        for i in range(min(3, len(texts)))
    ]
    return score


# --------------------------------------------------------------------------- #
# The training implementation (called by train / smoke)
# --------------------------------------------------------------------------- #
def _train_impl(
    *, epochs: float, r: int, lora_alpha: int, lr: float, bsz: int, grad_accum: int,
    max_seq: int, eval_max_new: int, eval_bs: int, max_steps: int, n_train_cap: int,
    n_valid_cap: int, push: bool, run_tag: str,
) -> Dict[str, Any]:
    import sys as _sys

    os.environ["HF_HOME"] = HF_HOME
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.makedirs(HF_HOME, exist_ok=True)
    _sys.path.insert(0, "/root")
    import distill_v6_format as F  # single source of truth
    # Unsloth MUST be imported before trl/transformers/peft so its patches apply
    # (otherwise SFTTrainer keeps the '<EOS_TOKEN>' sentinel + rejects tokenizer=).
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    run_dir = f"{RUN_ROOT}/{run_tag}"
    ckpt_dir = f"{run_dir}/ckpts"
    best_dir = f"{run_dir}/best"
    os.makedirs(run_dir, exist_ok=True)
    print(f"=== distill_v6 run {run_tag} — base={BASE_MODEL} epochs={epochs} "
          f"r={r} lr={lr} eff_batch={bsz * grad_accum} max_seq={max_seq} "
          f"max_steps={max_steps} ===", flush=True)

    def _push_adapter(note: str) -> Optional[str]:
        """Best-effort push of the current best adapter to HF (drain-resilience)."""
        if not push:
            return None
        try:
            from huggingface_hub import HfApi
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            api = HfApi(token=token)
            api.create_repo(HF_ADAPTER_REPO, private=True, exist_ok=True)
            api.upload_folder(folder_path=best_dir, repo_id=HF_ADAPTER_REPO,
                              repo_type="model", commit_message=f"distill_v6 {note}")
            print(f"[ship] pushed adapter ({note}) -> https://huggingface.co/{HF_ADAPTER_REPO}",
                  flush=True)
            return HF_ADAPTER_REPO
        except Exception as e:  # noqa: BLE001
            print(f"[ship] HF push FAILED ({note}; adapter safe on volume {best_dir}): {e!r}",
                  flush=True)
            return None

    # 1) data: pull raw v6 from HF, reformat to the no-grounding distill format ----
    train_path = hf_hub_download(HF_DATASET_REPO, "train_v6.jsonl", repo_type="dataset",
                                 local_dir=f"{VOL}/_v6dl")
    valid_path = hf_hub_download(HF_DATASET_REPO, "valid_v6.jsonl", repo_type="dataset",
                                 local_dir=f"{VOL}/_v6dl")
    train_rows = F.reformat_rows(F.iter_jsonl(Path(train_path)))
    valid_rows = F.reformat_rows(F.iter_jsonl(Path(valid_path)))
    if n_train_cap:
        train_rows = train_rows[:n_train_cap]
    if n_valid_cap:
        valid_rows = valid_rows[:n_valid_cap]
    train_exp = F.weighted_expand(train_rows, seed=SEED)
    print(f"[data] train unique={len(train_rows)} weighted-expanded={len(train_exp)} "
          f"valid={len(valid_rows)}", flush=True)

    eval_rows = [
        {"system": v["messages"][0]["content"], "user": v["messages"][1]["content"],
         "tier": v["meta"]["tier"], "fen": v["meta"]["fen"],
         "canonical_uci": v["meta"]["canonical_uci"], "student_uci": v["meta"]["student_uci"]}
        for v in valid_rows
    ]

    # 2) base model (4-bit) ----------------------------------------------------- #
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=max_seq, dtype=None, load_in_4bit=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 3) BASE eval (no-grounding) — the win-condition baseline ------------------ #
    print("[eval] BASE (untuned, no-grounding) ...", flush=True)
    base_score = _evaluate(model, tok, F, eval_rows, eval_max_new, eval_bs, label="BASE")
    print(f"[eval] BASE tier-policy={base_score['tier_policy_match']:.4f} "
          f"per_tier={ {k: round(v,4) for k,v in base_score['per_tier'].items()} } "
          f"parse={base_score['parse_rate']:.3f}", flush=True)

    # 4) LoRA ------------------------------------------------------------------- #
    FastLanguageModel.for_training(model)
    model = FastLanguageModel.get_peft_model(
        model, r=r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED, use_rslora=False,
    )

    # 5) dataset (manual-safe chat text; response-only masking) ----------------- #
    texts = [_fmt_train_text(tok, r_["messages"]) for r_ in train_exp]
    _lens = sorted(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts[:600])
    print(f"[data] train text tokens: p50={_lens[len(_lens)//2]} p99={_lens[int(len(_lens)*0.99)]} "
          f"p100={_lens[-1]} (max_seq={max_seq}) — targets truncated if p100>max_seq", flush=True)
    ds = Dataset.from_dict({"text": texts})

    cfg = _construct(
        SFTConfig,
        dataset_text_field="text", per_device_train_batch_size=bsz,
        gradient_accumulation_steps=grad_accum, warmup_ratio=0.03,
        num_train_epochs=epochs, max_steps=(max_steps or -1), learning_rate=lr,
        logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="cosine", seed=SEED, output_dir=ckpt_dir,
        report_to="none", save_strategy="epoch", save_total_limit=2, bf16=True,
        max_seq_length=max_seq, dataset_num_proc=2, packing=False,
    )
    try:  # Unsloth-patched TRL accepts tokenizer=; vanilla TRL wants processing_class=
        trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)
    except TypeError:
        trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=cfg)
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART,
    )

    # 6) per-epoch checkpoint eval -> select best by tier-policy ---------------- #
    per_ckpt: List[Dict[str, Any]] = []
    best = {"tier_policy_match": -1.0, "epoch": None}

    class EvalCB(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            ep = round(state.epoch or 0, 2)
            print(f"[eval] checkpoint epoch={ep} step={state.global_step} ...", flush=True)
            sc = _evaluate(model, tok, F, eval_rows, eval_max_new, eval_bs, label=f"ep{ep}")
            rec = {"epoch": ep, "step": int(state.global_step),
                   "tier_policy_match": sc["tier_policy_match"],
                   "per_tier": sc["per_tier"], "parse_rate": sc["parse_rate"]}
            per_ckpt.append(rec)
            print(f"[eval] epoch={ep} tier-policy={sc['tier_policy_match']:.4f} "
                  f"per_tier={ {k: round(v,4) for k,v in sc['per_tier'].items()} } "
                  f"parse={sc['parse_rate']:.3f}", flush=True)
            if sc["tier_policy_match"] > best["tier_policy_match"]:
                best.update({"tier_policy_match": sc["tier_policy_match"], "epoch": ep,
                             "per_tier": sc["per_tier"], "parse_rate": sc["parse_rate"]})
                model.save_pretrained(best_dir)
                tok.save_pretrained(best_dir)
                print(f"[eval] NEW BEST epoch={ep} -> saved adapter to {best_dir}", flush=True)
                volume.commit()
                _push_adapter(f"epoch {ep} tier-policy={sc['tier_policy_match']:.4f}")
            else:
                volume.commit()
            FastLanguageModel.for_training(model)

    trainer.add_callback(EvalCB())

    # 7) train ------------------------------------------------------------------ #
    print("[train] starting ...", flush=True)
    stats = trainer.train()
    train_runtime = getattr(stats, "metrics", {}).get("train_runtime") if stats else None
    global_step = int(trainer.state.global_step)

    # Fallback: if no epoch ckpt beat -1 (e.g. max_steps<1 epoch), eval + save now.
    if best["epoch"] is None:
        sc = _evaluate(model, tok, F, eval_rows, eval_max_new, eval_bs, label="final")
        per_ckpt.append({"epoch": round(trainer.state.epoch or 0, 2),
                         "step": global_step, "tier_policy_match": sc["tier_policy_match"],
                         "per_tier": sc["per_tier"], "parse_rate": sc["parse_rate"]})
        best.update({"tier_policy_match": sc["tier_policy_match"],
                     "epoch": round(trainer.state.epoch or 0, 2),
                     "per_tier": sc["per_tier"], "parse_rate": sc["parse_rate"]})
        model.save_pretrained(best_dir)
        tok.save_pretrained(best_dir)
        volume.commit()

    # 8) publish best adapter to HF -------------------------------------------- #
    pushed = _push_adapter(f"FINAL best epoch={best['epoch']} "
                           f"tier-policy={best['tier_policy_match']:.4f}")

    metrics = {
        "run_tag": run_tag, "base_model": BASE_MODEL, "init": "base",
        "format": "no-grounding (FEN+ASCII board+tier+student move; Maia+Stockfish stripped)",
        "hyper": {"epochs": epochs, "max_steps": max_steps, "r": r,
                  "lora_alpha": lora_alpha, "lr": lr, "eff_batch": bsz * grad_accum,
                  "max_seq": max_seq},
        "budget": {"global_step": global_step,
                   "train_rows_unique": len(train_rows),
                   "train_rows_weighted": len(train_exp),
                   "eff_batch": bsz * grad_accum,
                   "optimizer_steps": global_step,
                   "train_runtime_s": train_runtime},
        "base_no_grounding": {"tier_policy_match": base_score["tier_policy_match"],
                              "per_tier": base_score["per_tier"],
                              "parse_rate": base_score["parse_rate"],
                              "n": base_score["n"], "samples": base_score.get("_samples")},
        "per_checkpoint": per_ckpt,
        "best_distill_no_grounding": best,
        "delta_vs_base": (best["tier_policy_match"] - base_score["tier_policy_match"]),
        "adapter_volume_path": best_dir,
        "adapter_hf": pushed,
        "wall_s": round(time.time() - t0, 1),
    }
    with open(f"{run_dir}/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    volume.commit()
    print("FINAL_METRICS " + json.dumps(metrics), flush=True)
    return metrics


# --------------------------------------------------------------------------- #
# Modal functions
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu="H100", volumes={VOL: volume}, secrets=[hf_secret],
              timeout=6 * 3600)
def train(epochs: float = 3.0, r: int = 16, lora_alpha: int = 16, lr: float = 2e-4,
          bsz: int = 8, grad_accum: int = 2, max_seq: int = 768,
          eval_max_new: int = 128, eval_bs: int = 48, max_steps: int = 0,
          push: bool = True, run_tag: str = "") -> Dict[str, Any]:
    run_tag = run_tag or time.strftime("run_%Y%m%d_%H%M%S")
    return _train_impl(
        epochs=epochs, r=r, lora_alpha=lora_alpha, lr=lr, bsz=bsz, grad_accum=grad_accum,
        max_seq=max_seq, eval_max_new=eval_max_new, eval_bs=eval_bs, max_steps=max_steps,
        n_train_cap=0, n_valid_cap=0, push=push, run_tag=run_tag,
    )


@app.function(image=image, gpu="A100-80GB", volumes={VOL: volume}, secrets=[hf_secret],
              timeout=2 * 3600)
def smoke() -> Dict[str, Any]:
    """Validate the whole stack cheaply: load 32B 4-bit, 4 steps, eval 18 valid rows."""
    return _train_impl(
        epochs=1.0, r=16, lora_alpha=16, lr=2e-4, bsz=2, grad_accum=2, max_seq=1024,
        eval_max_new=256, eval_bs=6, max_steps=4, n_train_cap=64, n_valid_cap=18,
        push=False, run_tag="smoke",
    )


@app.local_entrypoint()
def main(mode: str = "train", epochs: float = 4.0, max_steps: int = 0,
         push: bool = True, run_tag: str = ""):
    if mode == "smoke":
        out = smoke.remote()
    else:
        out = train.remote(epochs=epochs, max_steps=max_steps, push=push, run_tag=run_tag)
    print(json.dumps(out, indent=2))
