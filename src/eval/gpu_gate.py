"""GPU generation backend for the HONEST eval — the Mac-independent port.

The shipped honest eval (:mod:`src.eval.honest.gated`) generates coaching with a
LOCAL MLX model (``MLXSamplingCoach``) and drives the shared faithfulness gate
(:func:`src.teacher.coach_gate.run_gate`) one scenario at a time. That ties the
eval to the Mac. This module provides the SAME behaviour on a Modal GPU:

* :class:`GPUCoach` — an Unsloth/transformers chat model (the untuned 4B base, or
  the base + our LoRA adapter) that matches the server's sampling knobs
  (``temp=0.7, top_p=0.8, top_k=20``) so the gate's re-samples genuinely explore.
  Exposes ``run(system, user) -> text`` (drop-in for the honest-eval ``run_fn``,
  so :func:`src.eval.honest.promptopt.optimize` is reused verbatim) plus a
  ``generate_texts(prompts, seed) -> list[str]`` batch primitive.
* :func:`batched_gated_generate` — a faithful, BATCHED re-expression of
  :func:`src.teacher.coach_gate.run_gate` over many scenarios: it reuses every
  PURE gate helper (``verify_text_ext`` for the verify check, ``extract_recommended``,
  ``split_coaching``, ``compose``, ``pick_fallback_move``, ``verified_coaching``)
  unchanged, and only batches the model calls per attempt-round so a full-val eval
  fits the cheap GPU budget. Rows are written in the byte-identical
  :func:`src.eval.honest.gated.gated_row` schema, resumable by ``scenario_id``.

The ONLY difference from ``run_gate`` is generation batching + per-attempt (rather
than per-call-RNG-advance) seeding; the rubric/gates/council/report all read the
resulting rows unchanged. Heavy deps (torch/transformers/unsloth) are imported
lazily so this module imports cleanly on the Mac for ``modal`` build.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess

# All PURE (python-chess only) — safe to import locally and on the GPU worker.
from src.eval.benchmark.prompts import build_user_prompt, load_system_prompt
from src.eval.evaluate import _strip_think
from src.engine.faithfulness_ext import verify_text_ext
from src.teacher.coach_gate import (
    compose,
    extract_recommended,
    pick_fallback_move,
    split_coaching,
    verified_coaching,
)

log = logging.getLogger("eval.gpu_gate")

# Server sampling knobs (identical to MLXSamplingCoach) so gate re-samples explore.
TEMP = 0.7
TOP_P = 0.8
TOP_K = 20
MAX_NEW_TOKENS = 640


def _seed_int(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


class GPUCoach:
    """A 4B chat model on a GPU (untuned base, or base + LoRA adapter).

    ``adapter_dir`` (a saved PEFT adapter directory, e.g. the iter-N adapter on the
    Modal volume) is loaded on top of the 4-bit base via Unsloth — the exact
    mechanism the trainer + ``eval_modal_v3`` use, so ``ours_4b`` is the trained
    model. When ``adapter_dir`` is ``None`` the untuned base is loaded (``base_4b``
    / ``pbase_4b``). Generation samples with the server's knobs; ``<think>`` is
    stripped to match the shipped coach.
    """

    def __init__(
        self,
        base_model: str,
        adapter_dir: Optional[str] = None,
        *,
        max_seq_len: int = 2048,
        max_new_tokens: int = MAX_NEW_TOKENS,
        load_in_4bit: bool = True,
    ) -> None:
        from unsloth import FastLanguageModel  # heavy; lazy

        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens

        # Unsloth loads base+adapter when given an adapter dir (adapter_config.json
        # points at the base); otherwise it loads the base 4-bit weights.
        load_name = adapter_dir or base_model
        t0 = time.time()
        model, tok = FastLanguageModel.from_pretrained(
            model_name=load_name, max_seq_length=max_seq_len,
            load_in_4bit=load_in_4bit, dtype=None,
        )
        FastLanguageModel.for_inference(model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self.model = model
        self.tok = tok
        log.info("loaded GPU model %s%s in %.1fs", base_model,
                 f" +adapter {adapter_dir}" if adapter_dir else "", time.time() - t0)

    # -- prompt rendering ---------------------------------------------------- #
    def _render(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:  # tokenizer without enable_thinking kwarg
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    # -- batched generation primitive --------------------------------------- #
    def generate_texts(self, prompts: Sequence[str], *, seed: int = 0) -> List[str]:
        """Sample one completion per rendered ``prompt`` (batched)."""
        import torch
        from transformers import set_seed

        if not prompts:
            return []
        set_seed(int(seed) & 0x7FFFFFFF)
        enc = self.tok(list(prompts), return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_seq_len).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=TEMP, top_p=TOP_P, top_k=TOP_K,
                pad_token_id=self.tok.pad_token_id,
            )
        in_len = enc["input_ids"].shape[1]
        texts = self.tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
        return [_strip_think(t) for t in texts]

    # -- single-prompt run_fn (drop-in for honest-eval optimize) ------------ #
    def seed(self, seed: int) -> None:  # matches MLXSamplingCoach.seed
        self._next_seed = int(seed) & 0x7FFFFFFF

    def run(self, system: str, user: str) -> str:
        seed = getattr(self, "_next_seed", 0)
        # advance the RNG per call so the gate's re-samples differ (like MLX).
        self._next_seed = _seed_int(seed, "next")
        return self.generate_texts([self._render(system, user)], seed=seed)[0]


# --------------------------------------------------------------------------- #
# Batched, faithful re-expression of coach_gate.run_gate over many scenarios
# --------------------------------------------------------------------------- #


def _finalize_clean(reply: str, board: chess.Board, pool: Sequence[Any],
                    student_uci: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Exactly coach_gate.run_gate's clean-path composition (pure helpers)."""
    rec_san, rec_uci = extract_recommended(reply, board, pool, student_uci)
    body, takeaway = split_coaching(reply)
    if (rec_san is None or rec_uci is None) and pool:
        rec_san, rec_uci = pool[0]["san"], pool[0]["uci"]
    shipped = compose(body, takeaway) or (reply or "").strip()
    return shipped, rec_san, rec_uci


def _finalize_fallback(board: chess.Board, pool: Sequence[Any], student_uci: str
                       ) -> Tuple[str, Optional[str], Optional[str]]:
    """Exactly coach_gate.run_gate's verified-fallback path (pure helpers)."""
    fb_move = pick_fallback_move(board, pool, student_uci)
    if fb_move is None and pool:
        fb_move = chess.Move.from_uci(pool[0]["uci"])
    if fb_move is None:
        return "", None, None
    body, takeaway = verified_coaching(board, fb_move)
    return compose(body, takeaway), board.san(fb_move), fb_move.uci()


def _done_ids(path: Path) -> set:
    done: set = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["scenario_id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


def batched_gated_generate(
    scenarios: Sequence[Dict[str, Any]],
    coach: GPUCoach,
    model_key: str,
    out_path: Path,
    *,
    system_prompt: Optional[str] = None,
    max_attempts: int = 6,
    gate_on: bool = True,
    batch_size: int = 16,
    commit_cb=None,
) -> Dict[str, int]:
    """Gated-generate every scenario to ``out_path`` in the honest-eval row schema.

    Faithful to :func:`src.teacher.coach_gate.run_gate`: for each scenario, resample
    the whole answer while ``verify_text_ext(candidate, fen).ok`` is False, up to
    ``max_attempts``, keeping the first clean draft; else emit the verified,
    engine-derived fallback. Generation is BATCHED across the still-pending
    scenarios per attempt-round (the only change from the shipped one-at-a-time
    loop). Resumable by ``scenario_id``.
    """
    system = system_prompt if system_prompt is not None else load_system_prompt()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)

    # State per pending scenario.
    pending: List[Dict[str, Any]] = []
    for scn in scenarios:
        if scn["id"] in done:
            continue
        board = chess.Board(scn["fen"])
        pending.append({
            "scn": scn,
            "board": board,
            "fen_norm": board.fen(),
            "user": build_user_prompt(scn, "grounded"),
            "rendered": coach._render(system, build_user_prompt(scn, "grounded")),
            "attempts": 0,
        })
    log.info("%s: %d pending of %d (%d done)", model_key, len(pending),
             len(scenarios), len(done))
    if not pending:
        return {"ok": 0, "fail": 0, "skipped": len(done)}

    finalized: List[Tuple[Dict[str, Any], str, Optional[str], Optional[str], int, bool]] = []
    t0 = time.time()

    n_rounds = max(1, max_attempts) if gate_on else 1
    for attempt in range(n_rounds):
        if not pending:
            break
        still: List[Dict[str, Any]] = []
        for i in range(0, len(pending), batch_size):
            chunk = pending[i:i + batch_size]
            seed = _seed_int(model_key, attempt, i)
            texts = coach.generate_texts([c["rendered"] for c in chunk], seed=seed)
            for c, cand in zip(chunk, texts):
                c["attempts"] += 1
                clean = True if not gate_on else verify_text_ext(cand, c["fen_norm"]).ok
                if clean:
                    scn = c["scn"]
                    text, rec_san, rec_uci = _finalize_clean(
                        cand, c["board"], scn["sound_pool"], scn["student_move"].get("uci") or "")
                    finalized.append((scn, text, rec_san, rec_uci, c["attempts"], False))
                else:
                    still.append(c)
        pending = still
        log.info("  %s attempt %d/%d: %d clean so far, %d still pending (%.0fs)",
                 model_key, attempt + 1, n_rounds, len(finalized), len(pending),
                 time.time() - t0)

    # Anything that never verified -> the deterministic verified fallback.
    for c in pending:
        scn = c["scn"]
        text, rec_san, rec_uci = _finalize_fallback(
            c["board"], scn["sound_pool"], scn["student_move"].get("uci") or "")
        finalized.append((scn, text, rec_san, rec_uci, c["attempts"], True))

    ok = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for scn, text, rec_san, rec_uci, attempts, fallback in finalized:
            fh.write(json.dumps({
                "scenario_id": scn["id"],
                "model": model_key,
                "condition": "gated",
                "tier": scn["tier"],
                "phase": scn["phase"],
                "severity": scn["severity"],
                "pos_id": scn.get("pos_id"),
                "output": text,
                "rec_uci": rec_uci,
                "rec_san": rec_san,
                "attempts": attempts,
                "verified_fallback": fallback,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "ts": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
            ok += 1
        fh.flush()
    if commit_cb is not None:
        try:
            commit_cb()
        except Exception as exc:  # noqa: BLE001
            log.warning("commit_cb failed: %s", exc)
    log.info("%s: wrote %d rows in %.0fs", model_key, ok, time.time() - t0)
    return {"ok": ok, "fail": 0, "skipped": len(done)}
