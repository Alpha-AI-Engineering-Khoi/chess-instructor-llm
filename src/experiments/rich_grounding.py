"""Rich / structured grounding — an A/B experiment renderer (NEW, isolated).

Research question
-----------------
Does giving the coach a **COMPLETE, explicit board state** (every occupied
square with piece+color, castling rights, en-passant square, side-to-move,
move number) plus the Stockfish sound pool (evals + short PV lines) and the
Maia human-likelihoods **as structured data** reduce fabrication vs. our
current *prose* grounding (``render_pool_facts`` + the ascii-board prompt)?

This module defines the two user-message renderers compared in the experiment
and a thin scoring helper. It is deliberately self-contained and **imports (but
never edits)** the production grounding + verifier so the benchmark worker that
reads ``position_facts.py`` / ``faithfulness.py`` / ``server.py`` /
``src/eval/benchmark/`` is untouched.

* **Condition A (baseline)** — exactly what the product serves today
  (``src/api/server.py``): ``render_pool_facts`` (prose piece list / loose
  pieces / what each candidate move does) followed by ``render_user_prompt``
  (ascii board + sound pool with internal evals + Maia + the task line).
* **Condition B (rich)** — a fully explicit, structured board state (every
  occupied square + castling + en passant + side-to-move + move number),
  followed by the *same* engine sound pool (san/uci/eval/short PV) and Maia
  likelihoods rendered as explicit tables, and the *same* move-recommendation
  task. Same system prompt, same decode.

Only the grounding block changes between A and B; system prompt, engine facts,
Maia signal, student-move context and the task line are identical.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chess

from config import settings
from config.schema import TeacherInput, render_user_prompt

# Production grounding + verifier + scorers — IMPORTED, never modified.
from src.engine.faithfulness import verify_text
from src.engine.position_facts import (
    PIECE_NAME,
    color_name,
    hanging_pieces,
    piece_map,
    render_pool_facts,
)
from src.eval.evaluate import (
    extract_recommended_move,
    find_engine_speak,
    longest_narrated_line,
)

# --------------------------------------------------------------------------- #
# System prompt — reused VERBATIM from src/api/server.py (read for reference).
# coach_system.md + the grounding suffix + the output-format suffix. Both
# conditions get this identical system prompt so the ONLY variable is the
# grounding block in the user message.
# --------------------------------------------------------------------------- #

_COACH_SYSTEM: str = (settings.PROMPTS / "coach_system.md").read_text(encoding="utf-8").strip()

_GROUNDING: str = (
    "\n\nYou will be given a VERIFIED FACTS block listing the exact pieces on the "
    "board, which pieces are loose, and what each candidate move concretely does. "
    "Ground EVERY concrete claim — pieces, squares, captures, threats — in that "
    "block. Never mention a piece, square, or capture that is not in the facts. If "
    "you are unsure a detail is true, leave it out and speak about the plan instead."
)
_FORMAT_SUFFIX: str = (
    "\n\nWrite your reply as plain prose for the student: two to four short "
    "sentences of coaching, then a final separate line that begins exactly with "
    '"Takeaway:" stating one transferable idea in a single sentence. Do not use '
    "markdown, headings, or bullet points."
)

#: The exact system prompt the production coach uses.
SYSTEM_PROMPT: str = _COACH_SYSTEM + _GROUNDING + _FORMAT_SUFFIX


# --------------------------------------------------------------------------- #
# Scenario -> shared TeacherInput contract
# --------------------------------------------------------------------------- #


def scenario_to_teacher_input(scn: Dict[str, Any]) -> TeacherInput:
    """Rebuild the shared :class:`TeacherInput` contract from a benchmark scenario."""
    sm = scn["student_move"]
    return {
        "tier": scn["tier"],
        "fen": scn["fen"],
        "move_history_san": None,
        "student_move": {
            "san": sm["san"],
            "uci": sm["uci"],
            "cp_loss": int(sm["cp_loss"]),
            "severity": sm["severity"],
        },
        "sound_pool": [
            {"uci": m["uci"], "san": m["san"], "cp": int(m["cp"]), "pv": list(m.get("pv") or [])}
            for m in scn["sound_pool"]
        ],
        "maia_human_moves": [
            {"uci": m["uci"], "san": m["san"], "policy": float(m["policy"])}
            for m in scn.get("maia", [])
        ],
    }


# --------------------------------------------------------------------------- #
# Condition A — the current PROSE grounding (identical to src/api/server.py)
# --------------------------------------------------------------------------- #


def render_baseline_user(scn: Dict[str, Any]) -> str:
    """Condition A user message: exactly what the product serves today.

    ``render_pool_facts`` (prose facts) + ``render_user_prompt`` (ascii board +
    sound pool with internal evals + Maia + the task line) — i.e. the two lines
    ``src/api/server.py`` assembles into ``user_prompt``.
    """
    ti = scenario_to_teacher_input(scn)
    facts = render_pool_facts(scn["fen"], list(ti["sound_pool"]))
    return f"{facts}\n\n{render_user_prompt(ti)}"


# --------------------------------------------------------------------------- #
# Condition B — the RICH / STRUCTURED grounding (new)
# --------------------------------------------------------------------------- #


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def render_rich_facts(
    fen: str,
    sound_pool: List[Dict[str, Any]],
    maia_moves: List[Dict[str, Any]],
    student_move: Dict[str, Any],
    tier: str,
    *,
    pv_plies: int = 3,
    maia_top: int = 6,
) -> str:
    """A fully explicit, structured VERIFIED-FACTS block (Condition B).

    Every occupied square is enumerated with its piece+color; castling rights,
    en-passant target, side-to-move and move number are stated outright; then the
    engine sound pool (san/uci/eval/short PV) and Maia likelihoods are given as
    explicit tables, followed by the same move-recommendation task. All of it is
    computed from the board / engine analysis — never guessed.
    """
    board = chess.Board(fen)
    stm = "White" if board.turn == chess.WHITE else "Black"
    L: List[str] = [
        "VERIFIED FACTS — use ONLY these. Never mention a piece, square, capture, "
        "or threat that is not derivable from this data.",
        "",
        "COMPLETE BOARD STATE (every occupied square is listed explicitly):",
        f"- Side to move: {stm}",
        f"- Move number: {board.fullmove_number}",
        (
            "- Castling rights available: "
            f"White kingside={_yn(board.has_kingside_castling_rights(chess.WHITE))}, "
            f"White queenside={_yn(board.has_queenside_castling_rights(chess.WHITE))}, "
            f"Black kingside={_yn(board.has_kingside_castling_rights(chess.BLACK))}, "
            f"Black queenside={_yn(board.has_queenside_castling_rights(chess.BLACK))}"
        ),
        "- En passant target square: "
        + (chess.square_name(board.ep_square) if board.ep_square is not None else "none"),
    ]

    for color in (chess.WHITE, chess.BLACK):
        pcs = piece_map(board, color)  # king-first, then by descending value
        L.append(f"- {color_name(color)} pieces ({len(pcs)} on board):")
        for sq, piece in pcs:
            L.append(f"    * {PIECE_NAME[piece.piece_type]} on {chess.square_name(sq)}")

    loose = hanging_pieces(board, board.turn) + hanging_pieces(board, not board.turn)
    if loose:
        toks = ", ".join(
            f"{color_name(p.color)} {PIECE_NAME[p.piece_type]} on {chess.square_name(sq)}"
            for sq, p in loose
        )
        L.append(f"- Undefended / unfavorably attacked pieces: {toks}")
    else:
        L.append("- Undefended / unfavorably attacked pieces: none")

    L.append("")
    L.append(
        f"STUDENT MOVE PLAYED: {student_move['san']} ({student_move['uci']}) "
        f"— severity: {student_move['severity']}; it loses about "
        f"{int(student_move['cp_loss'])} centipawns."
    )

    L.append("")
    L.append(
        "ENGINE-SOUND CANDIDATE MOVES [internal reference — never quote these "
        "numbers to the student]:"
    )
    for i, m in enumerate(sound_pool, start=1):
        pv = " ".join(list(m.get("pv") or [])[:pv_plies])
        L.append(f"  {i}. {m['san']} ({m['uci']})  eval {int(m['cp'])}cp  line: {pv}")

    L.append("")
    L.append("HUMAN-LIKELIHOOD AT THIS TIER (Maia — how often a human at this level plays each):")
    if maia_moves:
        for m in maia_moves[:maia_top]:
            L.append(f"  - {m['san']} ({m['uci']}): {round(float(m['policy']) * 100)}%")
    else:
        L.append("  - (unavailable)")

    t = settings.TIERS[tier]
    L.append("")
    L.append(
        f"TASK: Recommend exactly ONE move from the sound candidate list — the most "
        f"instructive for a {tier} player — and coach them. Ply cap: {t['ply_cap']}."
    )
    return "\n".join(L)


def render_rich_user(scn: Dict[str, Any]) -> str:
    """Condition B user message built from a benchmark scenario."""
    ti = scenario_to_teacher_input(scn)
    return render_rich_facts(
        scn["fen"],
        list(ti["sound_pool"]),
        list(ti["maia_human_moves"]),
        ti["student_move"],
        ti["tier"],
    )


USER_RENDERERS = {
    "A_baseline": render_baseline_user,
    "B_rich": render_rich_user,
}


# --------------------------------------------------------------------------- #
# Scoring — reuses the SAME deterministic functions as the benchmark's
# objective scorer (src/eval/benchmark/objective.py), so fabrication +
# soundness are measured identically to data/benchmark_v2.
# --------------------------------------------------------------------------- #


def score_output(output: str, scn: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic checks for one coaching output against its scenario."""
    fen = scn["fen"]
    tier = scn["tier"]
    student_uci = scn["student_move"]["uci"]
    sound_uci = set(scn["sound_uci"])
    ply_cap = settings.TIERS[tier]["ply_cap"]

    rec_san, rec_uci = extract_recommended_move(output, fen, student_uci)
    speak_hits = find_engine_speak(output)
    verdict = verify_text(output, fen)

    return {
        "rec_san": rec_san,
        "rec_uci": rec_uci,
        "produced_nonempty": bool((output or "").strip()),
        "move_parseable": rec_uci is not None,
        "move_sound": rec_uci is not None and rec_uci in sound_uci,
        "no_engine_speak": len(speak_hits) == 0,
        "ply_cap_ok": longest_narrated_line(output or "") <= ply_cap,
        "engine_speak_hits": speak_hits,
        "n_violations": len(verdict.violations),
        "fabricated": len(verdict.violations) >= 1,
        "violations": [
            {"sentence": v.sentence, "reason": v.reason} for v in verdict.violations[:5]
        ],
    }


def prompt_char_lengths(scn: Dict[str, Any]) -> Tuple[int, int]:
    """(len(A_user), len(B_user)) in characters — a cheap size proxy for cost."""
    return len(render_baseline_user(scn)), len(render_rich_user(scn))
