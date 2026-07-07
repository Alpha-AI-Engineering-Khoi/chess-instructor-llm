"""Verifier-gate eval — RAW vs GATED faithfulness, using the REAL production gate.

Research question
-----------------
Does the production **verify-and-regenerate faithfulness gate** (the loop in
``src/api/server.py`` backed by the deterministic checker in
``src/engine/faithfulness.py``) actually drive the **user-visible** fabrication a
learner sees to ~0 — and at what honest cost (how often does it have to fall back
to the engine-derived explanation instead of the model passing on its own)?

Two conditions per position (measured on the SAME generations)
--------------------------------------------------------------
* **RAW** — one generation, gate OFF. This is exactly what
  ``src/api/server.py`` returns when ``COACH_FAITHFULNESS_GATE=0``: it splits the
  single reply into ``coaching`` + ``Takeaway:`` (``_split_coaching``) and serves
  that. Fabrication is scored on that **user-visible** text with ``verify_text``.
* **GATED** — the real verify-and-regenerate loop. Re-sample the whole answer up
  to ``COACH_MAX_ATTEMPTS`` times, keeping the FIRST reply whose *full* text
  passes ``verify_text`` (short-circuit, never strip sentences). If none pass
  within budget, emit the deterministic, engine-derived explanation
  (``_verified_coaching``) that is true by construction. Fabrication is scored on
  the FINAL **user-visible** text.

Because attempt 1 of the gated loop *is* the raw generation, GATED is literally
"RAW + the gate" on the same sampling — the cleanest possible before/after.

Isolation
---------
This module only *imports* the production gate/fallback/parse helpers and the
deterministic verifier; it never edits ``src/api/server.py``,
``src/engine/faithfulness.py`` or the benchmark. Importing ``src.api.server``
does NOT load the MLX model or start the HTTP server (both are lazy / guarded by
``__main__``), so the live servers are undisturbed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import chess

from src.engine.faithfulness import verify_text
from src.experiments.rich_grounding import (
    SYSTEM_PROMPT,
    render_baseline_user,
    scenario_to_teacher_input,
)

# The REAL production gate helpers (imported verbatim, never modified). Importing
# the module runs only cheap module-level setup (reads prompts / .env, builds the
# FastAPI app object); it does NOT load the MLX coach or run uvicorn.
from src.api import server

__all__ = [
    "SYSTEM_PROMPT",
    "PRODUCTION_USER_RENDERER",
    "MAX_ATTEMPTS",
    "user_visible_text",
    "is_fabricated",
    "run_gate",
]

#: The user message the product actually serves today (prose VERIFIED-FACTS block
#: + ascii-board prompt) — i.e. what ``src/api/server.py`` assembles into
#: ``user_prompt``. RAW and GATED both use this identical grounding.
PRODUCTION_USER_RENDERER: Callable[[Dict[str, Any]], str] = render_baseline_user

#: The production attempt budget (``COACH_MAX_ATTEMPTS``, default 4).
MAX_ATTEMPTS: int = server.MAX_COACH_ATTEMPTS


def user_visible_text(reply: str) -> str:
    """The exact string a learner sees: ``coaching`` + ``Takeaway:`` line.

    Uses the production ``_split_coaching`` so this matches what the API returns
    (body + takeaway), which is what "user-visible fabrication" is scored on.
    """
    body, takeaway = server._split_coaching(reply or "")
    return f"{body} {takeaway}".strip()


def is_fabricated(text: str, fen: str) -> Tuple[bool, List[Dict[str, str]]]:
    """(fabricated?, violations) for ``text`` on ``fen`` via the production checker."""
    v = verify_text(text or "", fen)
    return (not v.ok), [{"sentence": vi.sentence, "reason": vi.reason} for vi in v.violations[:5]]


def run_gate(
    scn: Dict[str, Any],
    generate_fn: Callable[[int], str],
    *,
    max_attempts: int = MAX_ATTEMPTS,
) -> Dict[str, Any]:
    """Replicate ``src/api/server.py``'s verify-and-regenerate loop EXACTLY.

    Parameters
    ----------
    scn:
        A ``data/benchmark_v2/scenarios.jsonl`` row (grounding reused so no engine
        needs to run live).
    generate_fn:
        ``generate_fn(attempt_index) -> reply`` produces a *fresh* model reply for
        the given 1-based attempt. It is called lazily and short-circuited exactly
        as production does (so RAW = attempt 1; extra samples happen only if the
        gate needs them), keeping local/API cost minimal.
    max_attempts:
        The attempt budget (defaults to the production ``COACH_MAX_ATTEMPTS``).

    Returns a dict carrying both the RAW (gate-off) and the FINAL GATED
    user-visible outputs, their fabrication verdicts, and the gate bookkeeping
    (attempts used, whether the model passed within budget, whether the verified
    fallback was used).
    """
    fen = scn["fen"]
    board = chess.Board(fen)
    fen_norm = board.fen()  # production checks against board.fen()
    pool = list(scenario_to_teacher_input(scn)["sound_pool"])  # best-first SoundMove dicts
    student_uci = (scn["student_move"].get("uci") or "") if scn.get("student_move") else ""

    # ---- the real gate loop (server.py lines ~719-735) ------------------- #
    verified_reply: Optional[str] = None
    raw_first: Optional[str] = None
    per_attempt: List[Dict[str, Any]] = []
    attempts = 0
    for attempts in range(1, max_attempts + 1):
        candidate = generate_fn(attempts)
        if attempts == 1:
            raw_first = candidate
        v = verify_text(candidate, fen_norm)  # gate checks the FULL reply
        per_attempt.append({"attempt": attempts, "full_ok": v.ok, "n_violations": len(v.violations)})
        if v.ok:
            verified_reply = candidate
            break

    # ---- turn the (verified) reply into the served output (server.py ~740) #
    if verified_reply is not None:
        rec_san, rec_uci = server._extract_recommended(verified_reply, board, pool, student_uci)
        body, takeaway = server._split_coaching(verified_reply)
        if rec_san is None or rec_uci is None:  # pool is non-empty; belt-and-suspenders
            rec_san, rec_uci = pool[0]["san"], pool[0]["uci"]
        used_fallback = False
        final_source = "model"
    else:
        # No attempt verified within budget -> the deterministic, engine-derived,
        # true-by-construction explanation (the production verified fallback).
        used_fallback = True
        fb_move = server._pick_fallback_move(board, pool, student_uci) or chess.Move.from_uci(
            pool[0]["uci"]
        )
        body, takeaway = server._verified_coaching(board, fb_move)
        rec_san, rec_uci = board.san(fb_move), fb_move.uci()
        final_source = "fallback"

    gated_visible = f"{body} {takeaway}".strip()

    # RAW (gate OFF) is exactly what production serves with COACH_FAITHFULNESS_GATE=0:
    # _split_coaching(attempt-1). Score fabrication on that user-visible text.
    raw_visible = user_visible_text(raw_first or "")

    raw_fab, raw_viol = is_fabricated(raw_visible, fen_norm)
    raw_full_fab, _ = is_fabricated(raw_first or "", fen_norm)  # full-reply fab (context)
    gated_fab, gated_viol = is_fabricated(gated_visible, fen_norm)

    return {
        "fen": fen_norm,
        "max_attempts": max_attempts,
        "attempts_used": attempts,
        "passed_within_budget": verified_reply is not None,
        "used_fallback": used_fallback,
        "final_source": final_source,
        # RAW (gate off) — what the student would see with the gate disabled
        "raw_output": raw_first,
        "raw_visible": raw_visible,
        "raw_fabricated": raw_fab,
        "raw_full_fabricated": raw_full_fab,
        "raw_violations": raw_viol,
        # GATED (gate on) — the FINAL user-visible output the student sees
        "gated_body": body,
        "gated_takeaway": takeaway,
        "gated_visible": gated_visible,
        "gated_fabricated": gated_fab,
        "gated_violations": gated_viol,
        "rec_san": rec_san,
        "rec_uci": rec_uci,
        "per_attempt": per_attempt,
    }
