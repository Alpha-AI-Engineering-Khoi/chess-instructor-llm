"""The shipped faithfulness gate + verified fallback, as ONE reusable unit.

This is the single source of truth for the coach's *gated* generation pipeline —
the VERIFY-AND-REGENERATE loop plus the deterministic, engine-derived fallback.
It was previously inlined in :mod:`src.api.server`; extracting it here lets the
HONEST base-vs-tuned evaluation (:mod:`src.eval.honest`) run the base and the
tuned model through **byte-identical** gate + fallback code that ships in the
live coach, so the only variable between the two is the model weights.

Everything here is pure (python-chess + the deterministic verifiers only): no
FastAPI, no model, no engine process. A caller supplies a ``run_fn(system, user)
-> text`` that produces one coaching draft; :func:`run_gate` does the rest:

1. Ask ``run_fn`` for a draft, check every board claim with
   :func:`src.engine.faithfulness_ext.verify_text_ext`; if any is false, RE-SAMPLE
   the whole answer (never strip sentences) up to ``max_attempts`` times, keeping
   the first draft that verifies clean.
2. If no draft verifies within the budget, emit a deterministic explanation of a
   sound move built only from :func:`src.engine.position_facts.move_facts` — true
   by construction, so the student still gets a real (if plainer) explanation.

:mod:`src.api.server` imports the helpers below (and re-exports the historical
``_``-prefixed names it and :mod:`src.demo.app` depend on), so the live pipeline
and the eval cannot silently diverge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import chess

from src.engine.faithfulness_ext import verify_text_ext
from src.engine.position_facts import move_facts

__all__ = [
    "extract_recommended",
    "split_coaching",
    "pick_fallback_move",
    "verified_coaching",
    "finalize_verified",
    "GateResult",
    "run_gate",
]

# --------------------------------------------------------------------------- #
# Parsing (identical patterns to the shipped server)
# --------------------------------------------------------------------------- #

#: SAN token (incl. castling / promotion / check markers).
_SAN_RE = re.compile(r"(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)")

#: Phrases that typically precede the coach's recommended move.
_CUE_RE = re.compile(
    r"(?:i['\u2019]?d\s+play|i\s+would\s+play|i['\u2019]?ll\s+play|i\s+play|"
    r"recommend(?:ed)?(?:\s+move)?(?:\s+is)?|best\s+move\s+is|go\s+with|"
    r"choose|consider|play)\s*[:\-]?\s*",
    re.IGNORECASE,
)

#: Splits the coaching body from the trailing "Takeaway:" line.
_TAKEAWAY_RE = re.compile(r"\b(?:key\s+)?take[-\s]?away\s*:\s*", re.IGNORECASE)

#: A markdown horizontal rule on its own line (base models sometimes emit these).
_HR_LINE_RE = re.compile(r"(?m)^[ \t]*[-*_]{3,}[ \t]*$")


def extract_recommended(
    text: str, board: chess.Board, pool: Sequence[Any], student_uci: str
) -> Tuple[Optional[str], Optional[str]]:
    """Extract the recommended move (SAN, UCI) from the coach's free text.

    The recommendation is always a *sound* move that is NOT the student's own
    move. Strategy: a sound move right after a cue phrase ("I'd play ..."), then
    the first sound move named anywhere in the prose, and finally the engine's
    best sound move. We never return the student's move (a coach that phrases the
    pick as "develop the knight to f3" rather than "Nf3" must not be mis-read as
    recommending the mistake the student just played).
    """
    pool_ucis = {m["uci"] for m in pool}

    def _try(token: str) -> Optional[Tuple[str, str]]:
        try:
            move = board.parse_san(token)
        except ValueError:
            return None
        return board.san(move), move.uci()

    # 1) Cue phrase -> a move that is not the student's.
    for cue in _CUE_RE.finditer(text):
        window = text[cue.end() : cue.end() + 16]
        match = _SAN_RE.search(window)
        if match:
            parsed = _try(match.group(1))
            if parsed and parsed[1] != student_uci and parsed[1] in pool_ucis:
                return parsed

    # 2) First sound move named anywhere in the prose (never the student's).
    for match in _SAN_RE.finditer(text):
        parsed = _try(match.group(1))
        if parsed and parsed[1] != student_uci and parsed[1] in pool_ucis:
            return parsed

    # 3) Fallback: the engine's best sound move (guaranteed != the mistake).
    if pool:
        return pool[0]["san"], pool[0]["uci"]
    return None, None


def split_coaching(text: str) -> Tuple[str, str]:
    """Split the reply into (coaching_body, takeaway).

    Splits at the FIRST "Takeaway:" marker: the body is everything before it and
    the takeaway is the single line after it. Anything past that (small models
    sometimes repeat the whole answer) is dropped, and stray markdown rules are
    removed, so the UI never shows duplicated text or a "Takeaway:" inside the
    body.
    """
    text = (text or "").strip()
    match = _TAKEAWAY_RE.search(text)
    if not match:
        body, takeaway = text, ""
    else:
        body = text[: match.start()].strip()
        rest = text[match.end() :].strip()
        takeaway = rest.split("\n", 1)[0].strip()
        if not body:
            body = text
    body = _HR_LINE_RE.sub("", body).strip()
    return body, takeaway


# --------------------------------------------------------------------------- #
# Verified fallback (guaranteed-truthful coaching, no LLM)
# --------------------------------------------------------------------------- #


def pick_fallback_move(
    board: chess.Board, pool: Sequence[Any], student_uci: str
) -> Optional[chess.Move]:
    """A sound move for the verified fallback — prefer one that isn't the student's."""
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


def finalize_verified(
    board: chess.Board, san: str, body: str, takeaway: str
) -> Tuple[str, str]:
    """Assert the deterministic text is faithful; if an edge case slipped a false
    claim through, swap in a claim-free template wholesale (never strips a line)."""
    if verify_text_ext(f"{body} {takeaway}", board.fen()).ok:
        return body, takeaway
    body = (
        f"I'd play {san}. It's a sound, engine-approved move that keeps your "
        "position solid and your king safe."
    )
    takeaway = "When unsure, choose a safe developing move and don't leave a piece undefended."
    return body, takeaway


def verified_coaching(board: chess.Board, move: chess.Move) -> Tuple[str, str]:
    """Deterministic ``(coaching, takeaway)`` built ONLY from verified move facts.

    Truthful by construction: every concrete claim is derived from
    :func:`move_facts` (computed from the board with python-chess) and phrased so
    it also holds on the CURRENT position, so it passes the verifier untouched.
    Used only when the model cannot produce a faithful explanation within the
    attempt budget — the student still gets a guaranteed-true explanation of a
    sound move instead of a fabricated one.
    """
    f = move_facts(board, move)
    san = f["san"]

    if f["castle"]:
        body = (
            f"I'd play {san}. Castling gets your king to safety and brings a rook "
            "toward the center where it can help."
        )
        takeaway = "Castle early — get your king safe, then start making plans."
        return finalize_verified(board, san, body, takeaway)

    # What the piece itself does (each phrase is true on the current board).
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
    # The king is covered by "gives check"; don't also list it under "pressures".
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
    return finalize_verified(board, san, body, takeaway)


# --------------------------------------------------------------------------- #
# The gate (verify-and-regenerate) + composition
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    """Everything the caller needs after one gated coaching generation.

    ``text`` is the shipped, user-visible coaching (body + a ``Takeaway:`` line)
    — the string that should be scored/judged, since it is what the student sees.
    ``raw`` is the first clean model draft (``None`` when the verified fallback
    was used). ``attempts`` counts model calls (1 = clean first try);
    ``verified_fallback`` is True when every attempt failed and the deterministic
    engine-derived reply was substituted.
    """

    text: str
    body: str
    takeaway: str
    rec_san: Optional[str]
    rec_uci: Optional[str]
    attempts: int
    verified_fallback: bool
    raw: Optional[str]


def compose(body: str, takeaway: str) -> str:
    """Recombine (body, takeaway) into the single shipped coaching string."""
    body = (body or "").strip()
    takeaway = (takeaway or "").strip()
    if takeaway:
        return f"{body}\nTakeaway: {takeaway}".strip()
    return body


def run_gate(
    run_fn: Callable[[str, str], str],
    system: str,
    user: str,
    fen: str,
    pool: Sequence[Any],
    student_uci: str,
    *,
    max_attempts: int = 6,
    gate_on: bool = True,
) -> GateResult:
    """Run the VERIFY-AND-REGENERATE gate over ``run_fn`` and return a GateResult.

    This is the exact loop the live coach ships (see :mod:`src.api.server`):
    resample the whole answer while any board claim is false, keep the first
    clean draft, and fall back to :func:`verified_coaching` if none verifies. The
    faithfulness check is ``verify_text_ext(candidate, fen).ok`` — the same call,
    with the same (current-board) strictness, the server uses in its gate loop.
    """
    board = chess.Board(fen)
    fen_norm = board.fen()

    attempts = 0
    verified_reply: Optional[str] = None
    if gate_on:
        for _ in range(max(1, max_attempts)):
            attempts += 1
            candidate = run_fn(system, user)
            if verify_text_ext(candidate, fen_norm).ok:
                verified_reply = candidate
                break
    else:
        attempts = 1
        verified_reply = run_fn(system, user)

    verified_fallback = False
    if verified_reply is not None:
        rec_san, rec_uci = extract_recommended(verified_reply, board, pool, student_uci)
        body, takeaway = split_coaching(verified_reply)
        if (rec_san is None or rec_uci is None) and pool:
            rec_san, rec_uci = pool[0]["san"], pool[0]["uci"]
        shipped = compose(body, takeaway) or (verified_reply or "").strip()
        return GateResult(
            text=shipped, body=body, takeaway=takeaway, rec_san=rec_san,
            rec_uci=rec_uci, attempts=attempts, verified_fallback=False,
            raw=verified_reply,
        )

    verified_fallback = True
    fb_move = pick_fallback_move(board, pool, student_uci)
    if fb_move is None and pool:
        fb_move = chess.Move.from_uci(pool[0]["uci"])
    if fb_move is None:  # empty pool (should never happen for a coachable position)
        return GateResult(
            text="", body="", takeaway="", rec_san=None, rec_uci=None,
            attempts=attempts, verified_fallback=True, raw=None,
        )
    body, takeaway = verified_coaching(board, fb_move)
    return GateResult(
        text=compose(body, takeaway), body=body, takeaway=takeaway,
        rec_san=board.san(fb_move), rec_uci=fb_move.uci(),
        attempts=attempts, verified_fallback=verified_fallback, raw=None,
    )
