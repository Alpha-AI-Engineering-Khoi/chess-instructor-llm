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
    "pick_recommendation",
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

#: EXPLICIT, first-person / imperative recommendation cues. A move named right
#: after one of these is the coach's OWN pick — even when it equals the student's
#: move (a coach can endorse a good move: "I'd play Kd6" when the student played
#: Kd6). Deliberately NOT bare "play"/"consider": those also match "you played"
#: (the mistake being restated) and "if White plays" (the opponent's reply).
_ENDORSE_CUE_RE = re.compile(
    r"(?:i['\u2019]?d\s+play|i\s+would\s+play|i['\u2019]?ll\s+play|i\s+play|"
    r"i\s+recommend|recommend(?:ed)?(?:\s+move)?(?:\s+is)?|"
    r"(?:the\s+)?move\s*(?:is|:)|/move\s*:|best\s+move\s+is|go\s+with|choose|"
    r"leads?\s+you\s+to|points?\s+you\s+to|improvement\s+is|better\s+is)"
    r"\s*[:\-]?\s*",
    re.IGNORECASE,
)

#: Framing that marks the FOLLOWING move as one to AVOID / NOT the pick
#: ("rather than ...b2+", "instead of Rh8", "a forcing move like ...b2+").
_AVOID_CUE_RE = re.compile(
    r"rather than|instead of|such as|\bavoid\w*|\blike\b|\bnot\b|\bnever\b|"
    r"n['\u2019]?t\b|don['\u2019]?t|do not|rush\w*\s+into|forcing[- ]?looking",
    re.IGNORECASE,
)

#: Hypothetical / opponent-reply / continuation framing (also NOT the pick):
#: "if White plays Ke4", "after d2+ ...f3", "you have Bc4", "for example ...".
_HYPO_CUE_RE = re.compile(
    r"\bif\b|\bafter\b|\bthen\b|for\s+example|follow[- ]?up|followed\s+by|"
    r"you\s+have|you['\u2019]?ll\s+have|\breplies\b|\brespond|\bnext\b|continu|"
    r"(?:white|black)\s+(?:plays|has|goes|replies)",
    re.IGNORECASE,
)

#: A move token that is really a SQUARE reference, not a recommendation:
#: "the rook goes to h3", "the pawn on d6", "the g4-pawn".
_COORD_BEFORE_RE = re.compile(r"\b(?:on|to)\b\W*$", re.IGNORECASE)
_COORD_AFTER_RE = re.compile(r"^[- ]?(?:pawn|square|file|rank)s?\b", re.IGNORECASE)

#: A move immediately CONCEDED then dismissed ("Kd5 was already active, but ...").
_CONCESSION_AFTER_RE = re.compile(
    r"^\W{0,3}(?:was|is|were|are)\s+(?:already\s+|also\s+)?"
    r"(?:playable|possible|fine|active|ok|okay|tempting|reasonable)",
    re.IGNORECASE,
)

#: A sentence boundary: a terminator followed by whitespace, or a newline. NOT a
#: comma, so an avoided LIST keeps its framing across commas ("a quiet rook move
#: like Rh8, Ra8, or even Re8+" marks ALL of Rh8/Ra8/Re8+ as moves to avoid). A
#: genuine pick that follows a list/avoid cue is recovered instead by an EXPLICIT
#: endorsement cue right before it ("...such as Nc3, the move is Nf3"), which is
#: localized in :func:`_endorsed_indices`. Deliberately does NOT fire on the dots
#: of a chess ellipsis ("...b2+") or a move number ("12."), so avoid/hypothetical
#: framing before a "...move" is kept.
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?:;](?=\s)|\n")

#: Splits the coaching body from the trailing "Takeaway:" line.
_TAKEAWAY_RE = re.compile(r"\b(?:key\s+)?take[-\s]?away\s*:\s*", re.IGNORECASE)

#: A markdown horizontal rule on its own line (base models sometimes emit these).
_HR_LINE_RE = re.compile(r"(?m)^[ \t]*[-*_]{3,}[ \t]*$")


def _clause_before(text: str, start: int, span: int = 90) -> str:
    """The clause immediately preceding ``start`` — everything after the last real
    sentence boundary within ``span`` chars. Used to read a move's framing without
    being fooled by the dots of a chess ellipsis ("...b2+")."""
    pre = text[max(0, start - span) : start]
    last = -1
    for m in _CLAUSE_BOUNDARY_RE.finditer(pre):
        last = m.end()
    return pre[last:] if last != -1 else pre


def _san_candidates(
    board: chess.Board, text: str
) -> List[Tuple[int, int, str, str]]:
    """Every legally-parseable SAN token in ``text`` as (start, end, san, uci)."""
    out: List[Tuple[int, int, str, str]] = []
    for m in _SAN_RE.finditer(text):
        try:
            mv = board.parse_san(m.group(1))
        except ValueError:
            continue
        out.append((m.start(), m.end(), board.san(mv), mv.uci()))
    return out


def _is_avoid_framed(text: str, start: int, end: int) -> bool:
    """True if the SAN at ``[start, end)`` is framed as a move to AVOID / not the
    coach's pick: an avoid cue ("rather than", "like", "instead of"), a
    hypothetical/continuation ("if White plays", "after ... "), a bare square
    reference ("the rook goes to h3", "the g4-pawn"), or a dismissed concession."""
    pre = _clause_before(text, start)
    if _AVOID_CUE_RE.search(pre) or _HYPO_CUE_RE.search(pre):
        return True
    tok = text[start:end]
    if tok[:1] in "abcdefgh" and "x" not in tok:  # bare pawn move -> maybe a square ref
        if _COORD_BEFORE_RE.search(text[max(0, start - 6) : start]):
            return True
        if _COORD_AFTER_RE.search(text[end : end + 8]):
            return True
    if _CONCESSION_AFTER_RE.search(text[end : end + 26]):
        return True
    return False


def _endorsed_indices(
    text: str, cands: Sequence[Tuple[int, int, str, str]]
) -> set:
    """Indices of candidates the coach explicitly ENDORSES as its own pick — named
    right after an endorsement cue ("I'd play X", "the move: X"), or in the
    imperative "Play X again" pattern. The endorsement is LOCALIZED: the move must
    sit right after the cue with no avoid/hypothetical framing in between, so
    "...such as Nc3, the move is Nf3" endorses Nf3 (the cue is right before it)
    without being fooled by the earlier "such as". Because it is this tightly
    scoped, an endorsed move is trusted even when the wider sentence opened with a
    list/avoid cue — that is how a genuine pick after an avoided list is recovered.
    May be the student's own move (a coach can endorse a move it agrees with)."""
    endorsed: set = set()
    for cue in _ENDORSE_CUE_RE.finditer(text):
        lo, hi = cue.end(), cue.end() + 16
        for i, (s, _e, _san, _uci) in enumerate(cands):
            if lo <= s <= hi:
                between = text[cue.end() : s]
                if not (_AVOID_CUE_RE.search(between) or _HYPO_CUE_RE.search(between)):
                    endorsed.add(i)
                break
    for i, (s, e, _san, _uci) in enumerate(cands):
        if "again" in text[e : e + 10].lower() and "play" in text[max(0, s - 8) : s].lower():
            endorsed.add(i)
    return endorsed


def pick_recommendation(
    text: str,
    board: chess.Board,
    student_uci: str,
    accept: Callable[[str], bool],
) -> Optional[Tuple[str, str]]:
    """Return (SAN, UCI) of the coach's ACTUAL recommended move, or ``None``.

    ``accept(uci) -> bool`` is the caller's move filter (in the Stockfish sound
    pool for the shipped coach; any legal move for the honest metrics extractor).
    In order of preference the recommendation is:

    1. a move named right after an EXPLICIT endorsement cue ("I'd play X",
       "the move: X", "Play X again") — even when it equals the student's own
       move, because the coach can endorse a good move it agrees with;
    2. otherwise the first non-student move that is NOT framed as one to avoid;
    3. otherwise the student's own move if it is stated approvingly (non-avoid).

    Moves framed as things to AVOID ("rather than ...b2+", "instead of Rh8",
    "a forcing move like ...b2+"), opponent replies / continuations ("if White
    plays Ke4", "after d2+ ...f3"), and bare square references ("the rook goes to
    h3") are never chosen. That avoid-framing class is the bug this fixes: the old
    scanner grabbed the first non-student legal SAN, which — when the coach agreed
    with the student's move — was often a move the coach told the student NOT to
    play, corrupting the shown move and faking a tier "fork".
    """
    cands = [t for t in _san_candidates(board, text) if accept(t[3])]
    if not cands:
        return None
    avoid = [_is_avoid_framed(text, s, e) for (s, e, _san, _uci) in cands]
    endorsed = _endorsed_indices(text, cands)

    for i, (_s, _e, san, uci) in enumerate(cands):  # 1) explicitly endorsed pick
        if i in endorsed:  # endorsement is already localized (avoid-free span)
            return san, uci
    for i, (_s, _e, san, uci) in enumerate(cands):  # 2) first clean alternative
        if uci != student_uci and not avoid[i]:
            return san, uci
    for i, (_s, _e, san, uci) in enumerate(cands):  # 3) approvingly-stated student move
        if uci == student_uci and not avoid[i]:
            return san, uci
    return None


def extract_recommended(
    text: str, board: chess.Board, pool: Sequence[Any], student_uci: str
) -> Tuple[Optional[str], Optional[str]]:
    """Extract the coach's recommended *sound* move as ``(SAN, UCI)``.

    Uses :func:`pick_recommendation` restricted to the Stockfish sound pool, then
    falls back to the engine-best sound move if the coach named no sound pick.
    Unlike the previous scanner this can return the student's own move when the
    coach explicitly endorses it (e.g. the student already played the best move),
    and it never returns a move the coach framed as one to avoid.
    """
    pool_ucis = {m["uci"] for m in pool}
    picked = pick_recommendation(
        text, board, student_uci, accept=lambda u: u in pool_ucis
    )
    if picked is not None:
        return picked
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
