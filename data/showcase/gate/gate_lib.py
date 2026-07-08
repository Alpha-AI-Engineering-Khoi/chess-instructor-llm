"""Shared gating core — mirrors src/api/server.py's verify-and-regenerate gate.

Everything here REUSES the trusted machinery so a gated cell is produced exactly
the way the live platform would produce it:

* prompt        -> src.eval.benchmark.prompts.build_user_prompt(scn, "grounded")
                   + load_system_prompt()  (byte-identical to the showcase gens)
* verifier      -> src.engine.faithfulness_ext.verify_text_ext  (DO NOT edit)
* local backend -> a Coach mirroring server.Coach (temp 0.7 / top_p 0.8 / top_k 20,
                   <think> stripped) so re-samples actually differ from the greedy
                   attempt #1 — exactly what the live gate does.
* frontier/open -> src.eval.benchmark.backends.TFYChat (no temperature; retries).
* fallback      -> a faithful copy of server._verified_coaching / _finalize_verified
                   (deterministic, engine-derived, true by construction) used only
                   when no re-sample verifies within the budget.

No file here is edited that the task marks protected; server.py is *mirrored*,
not imported, to avoid the FastAPI app side-effects.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
PIPE = ROOT / "data" / "showcase" / "pipeline"
for p in (str(ROOT), str(PIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import chess  # noqa: E402

from common import (  # noqa: E402
    FIELD, LOCAL_KEYS, SPLIT_DIRS, model_meta, read_jsonl, resolved_ident, usd_for,
)
from src.engine.faithfulness_ext import verify_text_ext  # noqa: E402
from src.engine.position_facts import PIECE_NAME, move_facts  # noqa: E402
from src.eval.benchmark.prompts import build_user_prompt, load_system_prompt  # noqa: E402

TIERS = ("beginner", "intermediate", "advanced")

# Local re-sample sampling (mirror server.Coach exactly).
LOCAL_TEMP = 0.7
LOCAL_TOP_P = 0.8
LOCAL_TOP_K = 20
LOCAL_MAX_TOKENS = 400          # matches the showcase local-gen length
TFY_MAX_TOKENS = 1500           # caps runaway reasoning; coaching is short
MAX_TOTAL_ATTEMPTS = 6          # attempt #1 = the pre-existing raw coaching


# --------------------------------------------------------------------------- #
# Scenario index + model-name -> key
# --------------------------------------------------------------------------- #
def build_scn_index() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Global ``(pos_id, tier) -> scenario`` across all three splits."""
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for split_dir in SPLIT_DIRS.values():
        for s in read_jsonl(split_dir / "scenarios.jsonl"):
            idx[(s.get("pos_id", s["id"]), s["tier"])] = s
    return idx


def name_to_key_map() -> Dict[str, str]:
    return {model_meta(k)["name"]: k for k in FIELD}


# --------------------------------------------------------------------------- #
# Prompt (cached per scenario) + verify
# --------------------------------------------------------------------------- #
class PromptCache:
    def __init__(self) -> None:
        self.system = load_system_prompt()
        self._user: Dict[str, str] = {}

    def user_for(self, scn: Dict[str, Any]) -> str:
        sid = scn["id"]
        if sid not in self._user:
            self._user[sid] = build_user_prompt(scn, "grounded")
        return self._user[sid]


def is_clean(text: str, fen: str, rec_uci: Optional[str]) -> bool:
    return verify_text_ext(text or "", fen, recommended_uci=rec_uci).ok


# --------------------------------------------------------------------------- #
# Local MLX coach (mirror of server.Coach — stochastic re-sampling)
# --------------------------------------------------------------------------- #
def _strip_think(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text and "<think>" not in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


class LocalCoach:
    """A local MLX chat model (loads once, samples at temp 0.7 like server.Coach)."""

    def __init__(self, model_path: str, *, max_tokens: int = LOCAL_MAX_TOKENS) -> None:
        from mlx_lm import generate, load

        self.model_path = model_path
        self.max_tokens = max_tokens
        self._generate = generate
        t0 = time.time()
        self.model, self.tokenizer = load(model_path)
        print(f"[local] loaded {model_path!r} in {time.time()-t0:.1f}s", file=sys.stderr)
        try:
            from mlx_lm.sample_utils import make_sampler

            self._sampler = make_sampler(temp=LOCAL_TEMP, top_p=LOCAL_TOP_P, top_k=LOCAL_TOP_K)
        except Exception:  # noqa: BLE001
            self._sampler = None
        self._lock = threading.Lock()

    def _template(self, system: str, user: str) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def complete(self, system: str, user: str) -> Tuple[str, Dict[str, int]]:
        prompt = self._template(system, user)
        kwargs: Dict[str, Any] = {"max_tokens": self.max_tokens, "verbose": False}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        with self._lock:
            raw = self._generate(self.model, self.tokenizer, prompt=prompt, **kwargs)
        return _strip_think(raw).strip(), {"prompt_tokens": 0, "completion_tokens": 0}


# --------------------------------------------------------------------------- #
# TFY backend factory (frontier + open, via the gateway)
# --------------------------------------------------------------------------- #
def make_tfy_backends(keys: List[str], *, timeout: float, min_interval: float,
                      max_retries: int):
    """Return ``{key: TFYChat}`` sharing one client + rate limiter."""
    from dotenv import load_dotenv

    from src.eval.benchmark import config as bcfg
    from src.eval.benchmark.backends import RateLimiter, TFYChat, make_tfy_client

    load_dotenv(ROOT / ".env")
    client = make_tfy_client(timeout)
    limiter = RateLimiter(min_interval)
    out: Dict[str, Any] = {}
    for k in keys:
        m = bcfg.MODELS[k]
        out[k] = TFYChat(client, model_id=m.ident, max_tokens=TFY_MAX_TOKENS,
                         max_retries=max_retries, limiter=limiter,
                         reasoning_effort=m.reasoning_effort)
    return out


# --------------------------------------------------------------------------- #
# Verified, engine-derived fallback (faithful copy of server.py)
# --------------------------------------------------------------------------- #
def _pick_fallback_move(board: chess.Board, pool: List[Dict[str, Any]],
                        student_uci: str) -> Optional[chess.Move]:
    ordered = [m for m in pool if m.get("uci") and m["uci"] != student_uci]
    ordered += [m for m in pool if m.get("uci") and m["uci"] == student_uci]
    for m in ordered:
        try:
            mv = chess.Move.from_uci(m["uci"])
        except ValueError:
            continue
        if mv in board.legal_moves:
            return mv
    return None


def _finalize_verified(board: chess.Board, san: str, body: str, takeaway: str,
                       rec_uci: Optional[str]) -> Tuple[str, str]:
    """Assert deterministic text is faithful; else swap a claim-free template."""
    if verify_text_ext(f"{body} {takeaway}", board.fen(), recommended_uci=rec_uci).ok:
        return body, takeaway
    body = (
        f"I'd play {san}. It's a sound, engine-approved move that keeps your "
        "position solid and your king safe."
    )
    takeaway = "When unsure, choose a safe developing move and don't leave a piece undefended."
    return body, takeaway


def _verified_coaching(board: chess.Board, move: chess.Move) -> Tuple[str, str]:
    """Deterministic ``(coaching, takeaway)`` built ONLY from verified move facts."""
    f = move_facts(board, move)
    san = f["san"]

    if f["castle"]:
        body = (f"I'd play {san}. Castling gets your king to safety and brings a rook "
                "toward the center where it can help.")
        takeaway = "Castle early — get your king safe, then start making plans."
        return _finalize_verified(board, san, body, takeaway, move.uci())

    if f["is_capture"]:
        if board.is_en_passant(move):
            lead = "captures a pawn en passant"
        elif f["captured"]:
            lead = f"captures the {f['captured']} on {f['to']}"
        else:
            lead = f"makes a capture on {f['to']}"
    elif f["develops"]:
        lead = f"develops the {f['piece']}"
    else:
        lead = f"brings the {f['piece']} to {f['to']}"

    tail: List[str] = []
    attacks = [(s, n) for s, n in f["attacks"] if n != "king"]
    if attacks:
        tgts = ", ".join(f"the {n} on {s}" for s, n in attacks[:2])
        tail.append(f"and pressures {tgts}")
    if f["defends"]:
        tgts = ", ".join(f"the {n} on {s}" for s, n in f["defends"][:1])
        tail.append(f"while covering {tgts}")
    if f["is_check"]:
        tail.append("and gives check")

    sentence = f"It {lead}"
    if tail:
        sentence += " " + " ".join(tail)
    body = f"I'd play {san}. {sentence}."

    if f["is_check"]:
        takeaway = "A check with a point forces your opponent to react on your terms."
    elif f["is_capture"]:
        takeaway = "Look for safe captures that win material or trade in your favor."
    elif f["develops"]:
        takeaway = "Develop your pieces toward the center before you attack."
    elif f["attacks"]:
        takeaway = "Put your pieces on squares where they do the most work."
    else:
        takeaway = "Prefer purposeful moves that improve a piece and keep your king safe."
    return _finalize_verified(board, san, body, takeaway, move.uci())


def verified_fallback(board: chess.Board, cell_move_uci: Optional[str],
                      pool: List[Dict[str, Any]], student_uci: str) -> str:
    """Full engine-derived coaching string for the cell's move (true by construction)."""
    mv: Optional[chess.Move] = None
    if cell_move_uci:
        try:
            cand = chess.Move.from_uci(cell_move_uci)
            if cand in board.legal_moves:
                mv = cand
        except ValueError:
            mv = None
    if mv is None:
        mv = _pick_fallback_move(board, pool, student_uci)
    if mv is None:
        # last resort: any legal move
        mv = next(iter(board.legal_moves), None)
    if mv is None:
        return ""
    body, takeaway = _verified_coaching(board, mv)
    return f"{body} Takeaway: {takeaway}"


# --------------------------------------------------------------------------- #
# Gate one cell
# --------------------------------------------------------------------------- #
def gate_cell(
    *,
    coaching: str,
    fen: str,
    move_uci: Optional[str],
    scn: Dict[str, Any],
    backend,           # object with .complete(system, user) -> (text, usage)
    key: str,
    prompts: PromptCache,
    raw_ok: Optional[bool] = None,
    allow_regen: bool = True,
) -> Dict[str, Any]:
    """Run the verify-and-regenerate gate for one cell.

    Returns a record with the gated coaching + provenance flags + cost. Attempt #1
    is the pre-existing ``coaching``; up to 5 re-samples follow; then the verified
    engine-derived fallback (which is guaranteed faithful). ``raw_ok`` may be
    supplied when the caller already verified attempt #1 (avoids a re-check).
    """
    board = chess.Board(fen)
    student_uci = (scn.get("student_move") or {}).get("uci") or ""
    pool = scn.get("sound_pool", [])

    if raw_ok is None:
        raw_ok = is_clean(coaching, fen, move_uci)
    prompt_tokens = 0
    completion_tokens = 0
    regens = 0

    if raw_ok:
        gated = coaching
        attempts = 1
        fallback = False
    elif not allow_regen:
        # Provider is unavailable (e.g. Bedrock outage): the raw is the only
        # generation attempt; fall straight back to engine-derived truth.
        gated = verified_fallback(board, move_uci, pool, student_uci)
        attempts = 1
        fallback = True
    else:
        gated = None
        attempts = 1
        # up to (MAX_TOTAL_ATTEMPTS - 1) stochastic re-samples
        system = prompts.system
        user = prompts.user_for(scn)
        for attempts in range(2, MAX_TOTAL_ATTEMPTS + 1):
            try:
                cand, usage = backend.complete(system, user)
            except Exception as exc:  # noqa: BLE001 - a failed call just costs an attempt
                print(f"  ! regen {key} {scn['id']}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                cand, usage = "", {"prompt_tokens": 0, "completion_tokens": 0}
            regens += 1
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            if cand and is_clean(cand, fen, move_uci):
                gated = cand
                break
        if gated is None:
            gated = verified_fallback(board, move_uci, pool, student_uci)
            fallback = True
            attempts = MAX_TOTAL_ATTEMPTS
        else:
            fallback = False

    post_ok = is_clean(gated, fen, move_uci)
    usd = usd_for(key, prompt_tokens, completion_tokens) if key in _paid_keys() else 0.0
    return {
        "raw_fabricated": (not raw_ok),
        "gate_attempts": attempts,
        "verified_fallback": fallback,
        "fabricated": (not post_ok),          # POST-gate residual (should be ~0)
        "coaching": gated,
        "regens": regens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usd": round(usd, 6),
    }


def _paid_keys() -> set:
    return set(FIELD) - set(LOCAL_KEYS)
