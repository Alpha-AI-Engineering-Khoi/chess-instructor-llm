"""Widened deterministic truth checker — the second, structural truth layer.

:mod:`src.engine.faithfulness` catches *location / existence* claims ("a knight
on c5", "White has no queen") with regex + :func:`position_facts.piece_at_claim_ok`.
A live-backend audit proved that is not enough: **relational** ("Nc5 attacks the
bishop on e5") and **move-consequence** ("develops the knight from b1 to f3")
claims are false yet sail through, because nothing checks them against the board.

This module adds those checks. :func:`verify_text_ext` *reuses* the original
:func:`verify_text` for the location class and layers on relational,
move-consequence, turn/rights, material and hanging checks.

**Calibration (precision over recall).** A live/v4 audit showed the first cut of
this checker over-fired (~90% false positives on real coaching), because coaching
overwhelmingly describes the position **after** the recommended move ("then the
knight covers e5", "your rook now eyes the file") while the checks read the
**current** (pre-move) board. This module fixes that:

* **Both-board evaluation.** When ``recommended_uci`` is given, every board-state
  claim (relational, hanging/loose, at-location, turn, rights, material) is
  evaluated on BOTH the current board and the board *after the recommended move*.
  A claim proven true on EITHER board is never flagged; a claim is flagged only
  when it is demonstrably false on the current board AND (if a move is given) the
  post-move board, and true on neither. When it can be resolved on neither, the
  check ABSTAINS.
* **Unambiguous attribution.** Move-consequence verbs/targets are bound to a move
  only inside the same clause (no cross-clause "…threatens to take the pawn on
  f7" bleed); hanging/undefended keywords bind to the *nearest* piece, never a
  king; multi-move lines ("follow up with …", "after … captures back") abstain.

Every check flags a sentence *only* when a concrete board computation proves the
claim false on the relevant board(s); ambiguity abstains. Output is the same
``VerifyResult(clean, violations=[Violation(sentence, reason), ...])`` shape as
:func:`verify_text`, so it drops straight into the gate.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import chess

# Reuse the base verifier's public + internal shapes so the widened result is a
# drop-in replacement (same dataclasses, same sentence splitter, same colour map).
from src.engine.faithfulness import (
    Violation,
    VerifyResult,
    verify_text,
    _SENTENCE_SPLIT,
    _color_from_word,
)
from src.engine.position_facts import (
    PIECE_NAME,
    PIECE_VALUE,
    is_hanging,
    move_facts,
    piece_at_claim_ok,
)

__all__ = ["verify_text_ext", "VerifyResult", "Violation"]

# --------------------------------------------------------------------------- #
# Three-valued per-board verdict + the both-board combinator
# --------------------------------------------------------------------------- #
#
# Each board-state check returns ``(verdict, reason)`` for ONE board:
#   _ABSTAIN — could not resolve the claim on this board (no opinion).
#   _TRUE    — the claim was verified TRUE on this board.
#   _FALSE   — the claim was proven FALSE on this board (``reason`` set).
#
# The combinator implements the calibration contract: a claim true on EITHER
# board is never flagged; a claim is flagged only when false on some board and
# true on none. When every board abstains, the checker abstains.

_ABSTAIN = 0
_TRUE = 1
_FALSE = 2

_Verdict = Tuple[int, Optional[str]]


def _combine(verdicts: List[_Verdict]) -> Optional[str]:
    """First flag reason iff false-on-some and true-on-none, else ``None``."""
    if any(v == _TRUE for v, _ in verdicts):
        return None
    for v, r in verdicts:
        if v == _FALSE:
            return r
    return None


def _verdicts(fn, sentence: str, boards: List[chess.Board], stm: chess.Color) -> Optional[str]:
    """Run a per-board verdict fn on every board and combine (never raises).

    ``stm`` is the *mover's* perspective — the side to move on the CURRENT board.
    It is passed to every board so first/second-person pronouns ("your", "their")
    keep the same meaning after the recommended move is played (on the post-move
    board ``board.turn`` has flipped, which would otherwise invert "your").
    """
    out: List[_Verdict] = []
    for b in boards:
        try:
            out.append(fn(sentence, b, stm))
        except Exception:  # noqa: BLE001 - a checker bug must never gate wrongly
            out.append((_ABSTAIN, None))
    return _combine(out)


# --------------------------------------------------------------------------- #
# Small vocab helpers
# --------------------------------------------------------------------------- #

_SQ = r"[a-h][1-8]"
#: Full piece words (+ plural, + generic "piece"). Single letters are handled
#: only inside SAN tokens, never as a free-standing piece word, so a stray file
#: letter like "b" can never be misread as a bishop.
_PIECE = r"(?:pawns?|knights?|bishops?|rooks?|queens?|kings?|pieces?)"

_WORD_TO_TYPE = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}
_LETTER_TO_TYPE = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}


def _piece_type(word: str) -> Optional[int]:
    """Piece type for a word, or ``None`` for the generic "piece"/"unit"."""
    w = word.strip().lower()
    if w.endswith("s") and w[:-1] in _WORD_TO_TYPE:
        w = w[:-1]
    return _WORD_TO_TYPE.get(w)  # None => generic (any piece)


def _side_from_word(word: Optional[str], stm: chess.Color) -> Optional[chess.Color]:
    """Colour for a subject word incl. 1st/2nd/3rd person; ``None`` if ambiguous."""
    if not word:
        return None
    w = word.lower().strip().rstrip(".").rstrip("'s").strip()
    if w == "white":
        return chess.WHITE
    if w == "black":
        return chess.BLACK
    if w in ("your", "you", "our", "we", "us", "yours", "ours"):
        return stm
    if w in ("their", "they", "them", "opponent", "opponents", "opponent's",
             "the opponent", "enemy", "theirs"):
        return not stm
    return None  # his/her/it — deliberately ambiguous


def _count(board: chess.Board, color: chess.Color, ptype: int) -> int:
    return sum(1 for p in board.piece_map().values()
               if p.color == color and p.piece_type == ptype)


def _material(board: chess.Board, color: chess.Color) -> int:
    """Total non-king material for ``color`` (pawn-value units)."""
    return sum(PIECE_VALUE[p.piece_type] for p in board.piece_map().values()
               if p.color == color and p.piece_type != chess.KING)


# --------------------------------------------------------------------------- #
# Clause segmentation + line-shape cues (used to bind claims unambiguously)
# --------------------------------------------------------------------------- #

#: Clause boundaries. A move-consequence verb/target and its move must sit in the
#: same clause, so "…attacks g7 and threatens to take the pawn on f7" does not
#: bind the capture "take"/"f7" to a move mentioned in an earlier clause.
_CLAUSE_DELIM = re.compile(
    r"[,;:]|—|–|\s--\s|\s-\s|"
    r"\b(?:and|but|so|because|while|then|after|before|however|yet|although|"
    r"though|which|whereas|plus|also|meanwhile|instead|since|when|where)\b",
    re.IGNORECASE,
)

#: A multi-ply line whose described position is beyond the single post-move board
#: (a reply + a follow-up). The deterministic checker cannot resolve these to a
#: concrete board, so relational / hanging / move-consequence ABSTAIN and leave
#: them to the context-aware LLM judge.
_MULTI_MOVE_CUE = re.compile(
    r"\b(?:follow(?:s|ed)?[\s-]?up\s+with|followed\s+by|continue[sd]?\s+with|"
    r"captures?\s+back|recaptur\w+|takes?\s+back|wins?\s+it\s+back|"
    r"after\s+\w+\s+(?:recaptur\w+|takes\s+back|plays|responds|captures))\b",
    re.IGNORECASE,
)


def _clause_bounds(sentence: str, pos: int) -> Tuple[int, int]:
    """``(start, end)`` of the clause containing character ``pos``."""
    start, end = 0, len(sentence)
    for m in _CLAUSE_DELIM.finditer(sentence):
        if m.end() <= pos:
            start = m.end()
        elif m.start() >= pos:
            end = m.start()
            break
    return start, end


def _cut_at_conjunction(text: str) -> str:
    """Truncate ``text`` at the first conjunction / punctuation boundary."""
    m = re.search(r"[,;:]|\b(?:and|but|so|then|while|because|which|also|plus)\b",
                  text, re.IGNORECASE)
    return text[:m.start()] if m else text


def _is_multi_move_line(sentence: str) -> bool:
    return _MULTI_MOVE_CUE.search(sentence) is not None


# --------------------------------------------------------------------------- #
# Pin geometry (relative + absolute)
# --------------------------------------------------------------------------- #


def _pins(board: chess.Board, subj_sq: chess.Square, tsq: chess.Square) -> bool:
    """True iff a slider on ``subj_sq`` pins the enemy piece on ``tsq``.

    Covers absolute pins (a king behind the target) and relative pins (a more
    valuable friendly piece behind it), by walking the ray from subject through
    target: the target must be the first piece hit, and the next piece along the
    ray must be a same-colour, more-valuable piece (or the king).
    """
    sp = board.piece_at(subj_sq)
    tp = board.piece_at(tsq)
    if sp is None or tp is None or sp.color == tp.color:
        return False
    if sp.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return False

    sf, sr = chess.square_file(subj_sq), chess.square_rank(subj_sq)
    tf, tr = chess.square_file(tsq), chess.square_rank(tsq)
    df, dr = tf - sf, tr - sr
    if df == 0 and dr == 0:
        return False
    diagonal = abs(df) == abs(dr)
    orthogonal = df == 0 or dr == 0
    if not (diagonal or orthogonal):
        return False
    if diagonal and sp.piece_type == chess.ROOK:
        return False
    if orthogonal and sp.piece_type == chess.BISHOP:
        return False

    step_f = (df > 0) - (df < 0)
    step_r = (dr > 0) - (dr < 0)

    # Squares strictly between subject and target must be empty (line of sight).
    f, r = sf + step_f, sr + step_r
    while (f, r) != (tf, tr):
        if board.piece_at(chess.square(f, r)) is not None:
            return False
        f, r = f + step_f, r + step_r

    # Continue past the target: the next occupied square is what it is pinned to.
    f, r = tf + step_f, tr + step_r
    while 0 <= f <= 7 and 0 <= r <= 7:
        behind = board.piece_at(chess.square(f, r))
        if behind is not None:
            if behind.color != tp.color:
                return False
            return (behind.piece_type == chess.KING
                    or PIECE_VALUE[behind.piece_type] > PIECE_VALUE[tp.piece_type])
        f, r = f + step_f, r + step_r
    return False


# --------------------------------------------------------------------------- #
# 1) "at <sq>" location existence — the gap the base verifier leaves open
#    (base handles "on <sq>"; it never checks "at <sq>").
# --------------------------------------------------------------------------- #

_AT_SQUARE = re.compile(
    rf"\b(white|black|your|our|the opponent'?s?|opponent'?s?|their|his|her|its)?\s*"
    rf"({'|'.join(_WORD_TO_TYPE)})\s+at\s+(?:the\s+)?({_SQ})\b",
    re.IGNORECASE,
)


def _at_location_verdict(sentence: str, board: chess.Board, stm: chess.Color) -> _Verdict:
    verified_true = False
    for m in _AT_SQUARE.finditer(sentence):
        color_word, piece_word, sq = m.group(1), m.group(2), m.group(3)
        c = _color_from_word(color_word, stm)
        cw = None if c is None else ("white" if c == chess.WHITE else "black")
        if not piece_at_claim_ok(board, sq, piece_word, cw):
            return (_FALSE, f"no {piece_word} on {sq}")
        verified_true = True
    return (_TRUE, None) if verified_true else (_ABSTAIN, None)


# --------------------------------------------------------------------------- #
# 2) Relational claims
# --------------------------------------------------------------------------- #

_ATTACK = {"attack", "attacks", "attacking", "hit", "hits", "hitting",
           "threaten", "threatens", "threatening", "target", "targets", "targeting"}
_DEFEND = {"defend", "defends", "defending", "guard", "guards", "guarding",
           "protect", "protects", "protecting", "support", "supports", "supporting",
           "cover", "covers", "covering"}
_PIN = {"pin", "pins", "pinning"}

_SAN_CORE = r"(?:O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)"

# subject  +  verb  +  target(on/at sq | sq-piece)
_REL = re.compile(
    rf"(?P<subject>"
    rf"{_SAN_CORE}"
    rf"|(?:the\s+)?{_PIECE}\s+(?:on|at)\s+(?:the\s+)?{_SQ}"
    rf"|(?:the\s+)?{_SQ}[-\s]{_PIECE}"
    rf"|(?:white'?s?|black'?s?|your|our|their|the\s+opponent'?s?|opponent'?s?|enemy|his|her|its)\s+{_PIECE}"
    rf"|the\s+{_PIECE}"
    rf")"
    rf"\s+(?P<verb>attacks?|attacking|hits?|hitting|threatens?|threatening|targets?|targeting"
    rf"|defends?|defending|guards?|guarding|protects?|protecting|supports?|supporting|covers?|covering"
    rf"|pins?|pinning)\s+"
    rf"(?:(?:the|an?|its|his|her|their|your|our|white'?s?|black'?s?|opponent'?s?|enemy|opposing"
    rf"|undefended|unprotected|hanging|loose|weak|isolated|backward|passed)\s+)*"
    rf"(?:(?P<tp>{_PIECE})\s+(?:on|at)\s+(?:the\s+)?(?P<tsq>{_SQ})"
    rf"|(?P<tsq2>{_SQ})[-\s](?P<tp2>{_PIECE}))",
    re.IGNORECASE,
)

_SUBJ_PIECE_ON = re.compile(rf"(?:the\s+)?({'|'.join(_WORD_TO_TYPE)})\s+(?:on|at)\s+(?:the\s+)?({_SQ})", re.IGNORECASE)
_SUBJ_SQ_PIECE = re.compile(rf"(?:the\s+)?({_SQ})[-\s]({'|'.join(_WORD_TO_TYPE)})", re.IGNORECASE)
_SUBJ_COLOR_PIECE = re.compile(
    rf"(white'?s?|black'?s?|your|our|their|the\s+opponent'?s?|opponent'?s?|enemy|his|her|its)\s+({'|'.join(_WORD_TO_TYPE)})",
    re.IGNORECASE,
)
_SUBJ_BARE = re.compile(rf"the\s+({'|'.join(_WORD_TO_TYPE)})\b", re.IGNORECASE)


def _resolve_subject(
    subject: str, board: chess.Board, verb_kind: str,
    target_color: Optional[chess.Color], stm: chess.Color,
) -> Optional[Tuple[chess.Square, chess.Board]]:
    """Return ``(subject_square, board_to_evaluate_on)`` or ``None`` if unresolved.

    A SAN subject ("Nc5") is resolved by *playing* the move on a copy, so attack
    geometry is evaluated on the resulting position (mirroring how
    ``move_facts`` reads ``tmp.attacks(to_square)``). An explicit-square subject
    is evaluated on the passed board. A bare/coloured subject is resolved only
    when exactly one candidate piece exists — otherwise the check abstains.
    """
    s = subject.strip()

    # (a) SAN move token: play it, evaluate on the resulting board.
    m = re.fullmatch(_SAN_CORE, s, re.IGNORECASE)
    if m:
        token = s.rstrip("+#")
        try:
            mv = board.parse_san(token)
        except (ValueError, AssertionError):
            mv = None
        if mv is not None and mv in board.legal_moves:
            tmp = board.copy(stack=False)
            tmp.push(mv)
            return mv.to_square, tmp
        # Fallback: a piece of that letter already sits on the destination square.
        dm = re.search(rf"([KQRBN])?.*?({_SQ})$", s)
        if dm and dm.group(1):
            sq = chess.parse_square(dm.group(2))
            p = board.piece_at(sq)
            if p is not None and p.piece_type == _LETTER_TO_TYPE[dm.group(1).upper()]:
                return sq, board
        return None

    # (b) "<piece> on|at <sq>"
    m = _SUBJ_PIECE_ON.fullmatch(s)
    if m:
        sq = chess.parse_square(m.group(2))
        p = board.piece_at(sq)
        if p is not None and p.piece_type == _piece_type(m.group(1)):
            return sq, board
        return None

    # (c) "<sq> <piece>"  ("the c5 knight")
    m = _SUBJ_SQ_PIECE.fullmatch(s)
    if m:
        sq = chess.parse_square(m.group(1))
        p = board.piece_at(sq)
        if p is not None and p.piece_type == _piece_type(m.group(2)):
            return sq, board
        return None

    # (d) coloured or bare piece — resolve only if unique.
    color_word: Optional[str] = None
    piece_word: Optional[str] = None
    mc = _SUBJ_COLOR_PIECE.fullmatch(s)
    if mc:
        color_word, piece_word = mc.group(1), mc.group(2)
    else:
        mb = _SUBJ_BARE.fullmatch(s)
        if mb:
            piece_word = mb.group(1)
    if piece_word is None:
        return None
    ptype = _piece_type(piece_word)
    if ptype is None:
        return None
    color = _side_from_word(color_word, stm)
    if color is None and target_color is not None:
        # Infer: you attack enemies, you defend/pin (against) — attack & pin
        # subjects are the enemy of the target; defence is same-coloured.
        color = target_color if verb_kind == "defend" else (not target_color)
    candidates = [sq for sq, p in board.piece_map().items()
                  if p.piece_type == ptype and (color is None or p.color == color)]
    if len(candidates) == 1:
        return candidates[0], board
    return None


def _relational_verdict(sentence: str, board: chess.Board, stm: chess.Color) -> _Verdict:
    verified_true = False
    for m in _REL.finditer(sentence):
        verb = m.group("verb").lower()
        verb_kind = ("attack" if verb in _ATTACK
                     else "defend" if verb in _DEFEND
                     else "pin")
        tp_word = m.group("tp") or m.group("tp2")
        tsq_name = m.group("tsq") or m.group("tsq2")
        tsq = chess.parse_square(tsq_name)

        # Resolve the subject first (need its board for the target existence check
        # when the subject is a move that changes the position).
        target_piece_now = board.piece_at(tsq)
        target_color = target_piece_now.color if target_piece_now else None
        resolved = _resolve_subject(m.group("subject"), board, verb_kind, target_color, stm)
        if resolved is None:
            continue  # subject unresolved on this board — abstain for this claim
        subj_sq, ev = resolved

        # Verify the target actually exists (with the claimed type) on the board
        # we will evaluate against. If not, abstain — base/at-checks own existence.
        tpiece = ev.piece_at(tsq)
        want_type = _piece_type(tp_word)
        if tpiece is None or (want_type is not None and tpiece.piece_type != want_type):
            continue

        attacks = ev.attacks(subj_sq)
        subj_piece = ev.piece_at(subj_sq)
        if subj_piece is None:
            continue
        tname = PIECE_NAME[tpiece.piece_type]

        if verb_kind == "attack":
            if tsq not in attacks:
                return (_FALSE, f"does not attack the {tname} on {tsq_name}")
            verified_true = True
        elif verb_kind == "defend":
            if tsq not in attacks:
                return (_FALSE, f"does not defend the {tname} on {tsq_name}")
            if subj_piece.color != tpiece.color:
                return (_FALSE, f"cannot defend the enemy {tname} on {tsq_name}")
            verified_true = True
        else:  # pin
            if not _pins(ev, subj_sq, tsq):
                return (_FALSE, f"does not pin the {tname} on {tsq_name}")
            verified_true = True
    return (_TRUE, None) if verified_true else (_ABSTAIN, None)


# --------------------------------------------------------------------------- #
# 3) Move-consequence claims (against the actual move via move_facts)
# --------------------------------------------------------------------------- #

_SAN_TOKEN = re.compile(
    r"(?<![\w-])(O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|[a-h][1-8](?:=[QRBN])?[+#]?)(?![\w-])"
)
_CAPTURE_VERB = re.compile(
    r"\b(captures?|capturing|captured|takes?|taking|took|wins?|winning|grabs?|snags?)\b",
    re.IGNORECASE,
)
_CHECK_PHRASE = re.compile(
    r"\b(gives?\s+check|delivers?\s+check|is\s+a?\s*check|with\s+check|checks?\s+the\s+king)\b",
    re.IGNORECASE,
)
_PROMO_PHRASE = re.compile(r"\b(promot\w+|queens?\s+the\s+pawn|underpromot\w+)\b", re.IGNORECASE)
_EP_PHRASE = re.compile(r"\ben\s*passant\b", re.IGNORECASE)
_CASTLE_PHRASE = re.compile(r"\b(castl\w+)\b", re.IGNORECASE)
_TARGET_ON = re.compile(rf"(?:the\s+)?(?:\w+\s+){{0,2}}?({'|'.join(_WORD_TO_TYPE)})\s+(?:on|at)\s+(?:the\s+)?({_SQ})", re.IGNORECASE)
_FROM_SQ = re.compile(rf"\bfrom\s+(?:the\s+)?({_SQ})\b", re.IGNORECASE)
_TO_SQ = re.compile(rf"\bto\s+(?:the\s+)?({_SQ})\b", re.IGNORECASE)


#: Words that turn a following ``[a-h][1-8]`` into a *square reference* ("to f3",
#: "on e5") rather than a bare pawn move ("play e4"). Bare-square SAN tokens after
#: these are skipped so trajectory prose is not mistaken for a move name.
_SQUARE_REF_PREPS = {"to", "from", "on", "at", "onto", "via", "square",
                     "reaches", "reach", "toward", "towards", "into"}


def _legal_sans(sentence: str, board: chess.Board) -> List[Tuple[int, chess.Move]]:
    """(position, move) for every token in ``sentence`` that is a legal SAN.

    Bare pawn-destination squares that are really square references ("to f3",
    "on e5") are excluded — only piece-letter / capture / castling SANs, or a
    bare pawn push not preceded by a locative preposition, count as a move.
    """
    out: List[Tuple[int, chess.Move]] = []
    for m in _SAN_TOKEN.finditer(sentence):
        token = m.group(1)
        bare_square = re.fullmatch(r"[a-h][1-8]", token) is not None
        if bare_square:
            pre = re.search(r"([A-Za-z]+)\W*$", sentence[:m.start()])
            if pre and pre.group(1).lower() in _SQUARE_REF_PREPS:
                continue
        try:
            mv = board.parse_san(token.rstrip("+#!?"))
        except (ValueError, AssertionError):
            continue
        if mv in board.legal_moves:
            out.append((m.start(), mv))
    return out


def _bound_move(
    sentence: str, all_sans: List[Tuple[int, chess.Move]],
    rec_move: Optional[chess.Move], trigger_pos: int,
) -> Optional[chess.Move]:
    """The move a consequence verb/phrase at ``trigger_pos`` refers to, or None.

    Attribution is CLAUSE-LOCAL: only moves named in the same clause as the
    trigger are candidates, so a verb never binds to a move in a different clause
    ("…threatens to take the pawn on f7" does not bind to a move from an earlier
    clause). Prefer the recommended move if it is named in the clause; otherwise
    accept only when exactly one distinct legal move is named there. Anything
    else is ambiguous and abstains.
    """
    cs, ce = _clause_bounds(sentence, trigger_pos)
    in_clause = [(p, mv) for p, mv in all_sans if cs <= p < ce]
    if not in_clause:
        return None
    if rec_move is not None and any(mv == rec_move for _, mv in in_clause):
        return rec_move
    distinct = {mv.uci() for _, mv in in_clause}
    if len(distinct) == 1:
        return in_clause[0][1]
    return None


def _move_consequence_reason(
    sentence: str, board: chess.Board, rec_move: Optional[chess.Move]
) -> Optional[str]:
    all_sans = _legal_sans(sentence, board)

    def facts_for(mv: chess.Move):
        return move_facts(board, mv)

    # -- captures ---------------------------------------------------------- #
    for cm in _CAPTURE_VERB.finditer(sentence):
        mv = _bound_move(sentence, all_sans, rec_move, cm.start())
        if mv is None:
            continue
        f = facts_for(mv)
        san = f["san"]
        _, ce = _clause_bounds(sentence, cm.start())
        tail = _cut_at_conjunction(sentence[cm.end():ce])
        # The captured object must IMMEDIATELY follow the verb (anchored), so
        # figurative "takes action / takes control / takes the initiative … the
        # knight on c6" is not misread as a capture of a downstream piece.
        tm = _TARGET_ON.match(tail.lstrip())
        if not f["is_capture"]:
            # Only fire when an actual capture object is asserted in this clause,
            # not for figurative "wins the game / wins a tempo".
            if tm is not None or re.match(r"\s+(?:on|at)\s+" + _SQ, tail, re.IGNORECASE):
                return f"{san} does not capture anything"
            continue
        if tm is not None:
            claimed_type = _piece_type(tm.group(1))
            claimed_sq = tm.group(2)
            if (claimed_type is not None and f["captured"]
                    and _WORD_TO_TYPE.get(f["captured"]) != claimed_type):
                return f"{san} captures a {f['captured']}, not a {tm.group(1).lower()}"
            if (not board.is_en_passant(mv) and claimed_sq
                    and claimed_sq != f["to"]):
                return f"{san} captures on {f['to']}, not {claimed_sq}"

    # -- gives check ------------------------------------------------------- #
    for chk in _CHECK_PHRASE.finditer(sentence):
        mv = _bound_move(sentence, all_sans, rec_move, chk.start())
        if mv is None:
            continue
        f = facts_for(mv)
        if not f["is_check"]:
            return f"{f['san']} does not give check"

    # -- promotion --------------------------------------------------------- #
    for pr in _PROMO_PHRASE.finditer(sentence):
        mv = _bound_move(sentence, all_sans, rec_move, pr.start())
        if mv is None:
            continue
        f = facts_for(mv)
        if not f["promotion"]:
            return f"{f['san']} does not promote"

    # -- en passant -------------------------------------------------------- #
    for ep in _EP_PHRASE.finditer(sentence):
        mv = _bound_move(sentence, all_sans, rec_move, ep.start())
        if mv is None:
            continue
        if not board.is_en_passant(mv):
            return f"{board.san(mv)} is not an en-passant capture"

    # -- castling ---------------------------------------------------------- #
    for cs in _CASTLE_PHRASE.finditer(sentence):
        mv = _bound_move(sentence, all_sans, rec_move, cs.start())
        if mv is None:
            continue
        f = facts_for(mv)
        if f["castle"] is None:
            return f"{f['san']} is not a castling move"
        side = None
        window = sentence[cs.start():cs.start() + 24].lower()
        if "kingside" in window or "short" in window:
            side = "kingside"
        elif "queenside" in window or "long" in window:
            side = "queenside"
        if side is not None and f["castle"] != side:
            return f"{f['san']} castles {f['castle']}, not {side}"

    # -- from / to origin -------------------------------------------------- #
    reason = _from_to_reason(sentence, board, all_sans, rec_move)
    if reason:
        return reason
    return None


def _from_to_reason(
    sentence: str, board: chess.Board, sans: List[Tuple[int, chess.Move]],
    rec_move: Optional[chess.Move],
) -> Optional[str]:
    """Origin/destination check, restricted to the unambiguous no-SAN case.

    A full SAN already encodes its own destination, and a stray "to <sq>" in a
    sentence usually belongs to a *different* piece ("exf7+ clears a path for the
    bishop to swing to h7"). So we only check origin when the move is referenced
    WITHOUT a SAN, by piece + destination — the "develops the knight from b1 to
    f3" shape, resolved against the recommended move.
    """
    froms = [chess.parse_square(x) for x in _FROM_SQ.findall(sentence)]
    tos = [chess.parse_square(x) for x in _TO_SQ.findall(sentence)]
    if not froms and not tos:
        return None

    # Case A: the RECOMMENDED move is named by SAN (and is the only SAN) — check
    # ONLY its ORIGIN against an explicit "from <sq>". A piece-letter SAN ("Ng4")
    # does not encode its origin, so "develops the knight from f6 to g4" (when it
    # is really on h2) is a checkable lie. Gating to the recommended move avoids
    # binding a stray "from <sq>" to a non-recommended/hypothetical SAN ("…when
    # White tries something like Qa1"). We deliberately do NOT check "to <sq>": a
    # SAN already encodes its destination, and a stray "to <sq>" usually names
    # another piece's square ("exf7 … the bishop swings to h7").
    if len(sans) == 1 and froms and rec_move is not None and sans[0][1] == rec_move:
        san = board.san(rec_move)
        for fsq in froms:
            if fsq not in (rec_move.from_square, rec_move.to_square):
                return (f"{san} moves from {chess.square_name(rec_move.from_square)}, "
                        f"not {chess.square_name(fsq)}")

    if not sans and rec_move is not None:
        piece = board.piece_at(rec_move.from_square)
        if piece is None:
            return None
        piece_word = PIECE_NAME[piece.piece_type]
        mentions_piece = re.search(rf"\b{piece_word}s?\b", sentence, re.IGNORECASE)
        # Require the destination via an explicit "to <dest>" (the "from ORIGIN to
        # DEST" shape). A bare "from <dest>" is a post-move locative ("from f3,
        # the queen …" = now on f3), NOT an origin claim, so skip any such square.
        mentions_dest = rec_move.to_square in tos
        if mentions_piece and mentions_dest:
            for fsq in froms:
                if fsq in (rec_move.from_square, rec_move.to_square):
                    continue
                return (f"the {piece_word} moves from "
                        f"{chess.square_name(rec_move.from_square)}, not {chess.square_name(fsq)}")
    return None


# --------------------------------------------------------------------------- #
# 4) Turn / castling rights / en-passant availability
# --------------------------------------------------------------------------- #

_SIDE_TO_MOVE = re.compile(
    r"\b(?:it'?s\s+)?(white|black)(?:'s)?\s+(?:is\s+)?(?:to\s+(?:move|play)|turn|on\s+the\s+move|moves\s+next)\b",
    re.IGNORECASE,
)
_TO_MOVE_SIDE = re.compile(r"\bto\s+(?:move|play)\s*[:,]?\s*(white|black)\b", re.IGNORECASE)


def _turn_verdict(sentence: str, board: chess.Board, stm: chess.Color) -> _Verdict:
    verified_true = False
    for rx in (_SIDE_TO_MOVE, _TO_MOVE_SIDE):
        for m in rx.finditer(sentence):
            side = m.group(1).lower()
            claimed = chess.WHITE if side == "white" else chess.BLACK
            if claimed != board.turn:
                real = "White" if board.turn == chess.WHITE else "Black"
                return (_FALSE, f"it is {real} to move, not {side.capitalize()}")
            verified_true = True
    return (_TRUE, None) if verified_true else (_ABSTAIN, None)


_CASTLE_CLAIM = re.compile(
    r"\b(white|black|you|your|we|our|they|their|opponent'?s?|the\s+opponent)?\s*"
    r"(?:can|could|is\s+able\s+to|are\s+able\s+to|may|should|will|has\s+the\s+right\s+to|is\s+free\s+to)\s+"
    r"(?:still\s+)?castles?\s*(kingside|queenside|long|short)?",
    re.IGNORECASE,
)


def _castle_verdict(sentence: str, board: chess.Board, stm: chess.Color) -> _Verdict:
    verified_true = False
    for m in _CASTLE_CLAIM.finditer(sentence):
        color = _side_from_word(m.group(1), stm)
        if color is None:
            continue  # unattributed "can castle" — abstain (could be either side)
        side = (m.group(2) or "").lower()
        who = "White" if color == chess.WHITE else "Black"
        if side in ("kingside", "short"):
            if not board.has_kingside_castling_rights(color):
                return (_FALSE, f"{who} has no kingside castling rights")
            verified_true = True
        elif side in ("queenside", "long"):
            if not board.has_queenside_castling_rights(color):
                return (_FALSE, f"{who} has no queenside castling rights")
            verified_true = True
        else:
            if not (board.has_kingside_castling_rights(color)
                    or board.has_queenside_castling_rights(color)):
                return (_FALSE, f"{who} has no castling rights")
            verified_true = True
    return (_TRUE, None) if verified_true else (_ABSTAIN, None)


_EP_AVAIL = re.compile(
    r"\b(?:(no|not|isn'?t|cannot|can'?t|there\s+is\s+no)\s+[\w\s]{0,20}?)?en\s*passant\b",
    re.IGNORECASE,
)


def _ep_verdict(sentence: str, board: chess.Board, stm: chess.Color) -> _Verdict:
    verified_true = False
    for m in _EP_AVAIL.finditer(sentence):
        if m.group(1):  # negated ("no en passant") — not an availability claim
            continue
        if board.ep_square is None:
            return (_FALSE, "no en-passant capture is available")
        verified_true = True
    return (_TRUE, None) if verified_true else (_ABSTAIN, None)


# --------------------------------------------------------------------------- #
# 5) Material counts + up/down direction
# --------------------------------------------------------------------------- #

_COUNT_WORDS = {"two": 2, "both": 2, "double": 2, "doubled": 2, "three": 3, "pair": 2}
_COUNT_CLAIM = re.compile(
    r"\b(white|black|your|our|their|the\s+opponent'?s?|opponent'?s?|his|her|you|we|they)?\s*"
    r"(?:has|have|with|possess(?:es)?|keeps?|owns?)?\s*"
    r"(two|both|three|double|doubled|a\s+pair\s+of)\s+"
    r"(rooks?|bishops?|knights?|pawns?|queens?)",
    re.IGNORECASE,
)
_PAIR_CLAIM = re.compile(
    r"\b(white|black|your|our|their|the\s+opponent'?s?|opponent'?s?|his|her)?\s*"
    r"(rook|bishop|knight|queen)\s+pair\b",
    re.IGNORECASE,
)
_UPDOWN_CLAIM = re.compile(
    r"\b(white|black|you|your|we|our|they|their|opponent'?s?|the\s+opponent)?\s*"
    r"(?:is|are|'re|'s|has|have|being)?\s*"
    r"(up|ahead|down|behind)\s+(?:by\s+)?(?:a|an|one|two|the\s+)?\s*"
    r"(pawns?|pieces?|knights?|bishops?|rooks?|queens?|exchange)\b",
    re.IGNORECASE,
)


def _material_verdict(sentence: str, board: chess.Board, stm: chess.Color) -> _Verdict:
    verified_true = False
    for m in _COUNT_CLAIM.finditer(sentence):
        color = _side_from_word(m.group(1), stm)
        if color is None:
            continue
        num_word = re.sub(r"a\s+pair\s+of", "pair", m.group(2).lower())
        want = _COUNT_WORDS.get(num_word.split()[0], None)
        if want is None:
            continue
        ptype = _piece_type(m.group(3))
        if ptype is None:
            continue
        have = _count(board, color, ptype)
        who = "White" if color == chess.WHITE else "Black"
        if have < want:
            return (_FALSE, f"{who} has {have} {PIECE_NAME[ptype]}(s), not {want}")
        verified_true = True

    for m in _PAIR_CLAIM.finditer(sentence):
        color = _side_from_word(m.group(1), stm)
        if color is None:
            continue
        ptype = _piece_type(m.group(2))
        if ptype is None:
            continue
        have = _count(board, color, ptype)
        who = "White" if color == chess.WHITE else "Black"
        if have < 2:
            return (_FALSE, f"{who} does not have the {PIECE_NAME[ptype]} pair ({have})")
        verified_true = True

    for m in _UPDOWN_CLAIM.finditer(sentence):
        color = _side_from_word(m.group(1), stm)
        if color is None:
            continue
        direction = m.group(2).lower()
        diff = _material(board, color) - _material(board, not color)
        who = "White" if color == chess.WHITE else "Black"
        if direction in ("up", "ahead"):
            if diff <= 0:
                state = "even" if diff == 0 else "behind"
                return (_FALSE, f"{who} is not up material ({state} on material)")
            verified_true = True
        if direction in ("down", "behind"):
            if diff >= 0:
                state = "even" if diff == 0 else "ahead"
                return (_FALSE, f"{who} is not down material ({state} on material)")
            verified_true = True
    return (_TRUE, None) if verified_true else (_ABSTAIN, None)


# --------------------------------------------------------------------------- #
# 6) Hanging / loose / undefended
# --------------------------------------------------------------------------- #

_HANG_KEY = re.compile(r"\b(hanging|hangs|loose|en\s*prise|undefended|unprotected)\b", re.IGNORECASE)
_PIECE_ON_ANY = re.compile(rf"({'|'.join(_WORD_TO_TYPE)})\s+(?:on|at)\s+(?:the\s+)?({_SQ})", re.IGNORECASE)
_SQ_PIECE_ANY = re.compile(rf"({_SQ})[-\s]({'|'.join(_WORD_TO_TYPE)})", re.IGNORECASE)


def _piece_refs(hay: str) -> List[Tuple[int, str, str]]:
    """All ``(start, square_name, piece_word)`` piece references in ``hay``."""
    refs: List[Tuple[int, str, str]] = []
    for rx, pw_i, sq_i in ((_PIECE_ON_ANY, 1, 2), (_SQ_PIECE_ANY, 2, 1)):
        for m in rx.finditer(hay):
            refs.append((m.start(), m.group(sq_i), m.group(pw_i)))
    refs.sort()
    return refs


def _nearest_piece_ref(
    sentence: str, kw_start: int, kw_end: int
) -> Optional[Tuple[str, str]]:
    """The ``(square_name, piece_word)`` the hanging/undefended keyword is about.

    Attribution is clause-local and prefers the SUBJECT piece *before* the
    keyword ("the rook on a1 is hanging"; "the knight on e4 is hanging to the
    pawn on f5" → the knight, not the attacking pawn). Only when nothing precedes
    the keyword do we fall back to the following piece — the attributive shape
    ("the undefended pawn on b2 …"). Both fixes the old misattribution that bound
    the keyword to the *last* piece anywhere in a fixed window.
    """
    cs, ce = _clause_bounds(sentence, kw_start)
    before = _piece_refs(sentence[cs:kw_start])
    if before:  # nearest preceding = subject of the predicate
        _, sq_name, pw = before[-1]
        return (sq_name, pw)
    after = _piece_refs(sentence[kw_end:ce])
    if after:  # attributive: "the undefended <piece> on <sq>"
        _, sq_name, pw = after[0]
        return (sq_name, pw)
    return None


def _hanging_verdict(sentence: str, board: chess.Board, stm: chess.Color) -> _Verdict:
    verified_true = False
    for key in _HANG_KEY.finditer(sentence):
        kw = key.group(1).lower().replace(" ", "")
        disp = "en prise" if kw == "enprise" else ("hanging" if kw == "hangs" else kw)
        ref = _nearest_piece_ref(sentence, key.start(), key.end())
        if ref is None:
            continue
        sq_name, pw = ref
        try:
            sq = chess.parse_square(sq_name)
        except ValueError:
            continue
        piece = board.piece_at(sq)
        want = _piece_type(pw)
        if piece is None or (want is not None and piece.piece_type != want):
            continue  # not on this board (with the claimed type) — abstain
        if piece.piece_type == chess.KING:
            continue  # a king is never "hanging / undefended" — abstain
        if kw in ("undefended", "unprotected"):
            if board.attackers(piece.color, sq):
                return (_FALSE, f"the {pw} on {sq_name} is defended, not {disp}")
            verified_true = True
        else:  # hanging / loose / en prise
            if not is_hanging(board, sq):
                return (_FALSE, f"the {pw} on {sq_name} is not {disp}")
            verified_true = True
    return (_TRUE, None) if verified_true else (_ABSTAIN, None)


# --------------------------------------------------------------------------- #
# Sentence dispatcher + public entry point
# --------------------------------------------------------------------------- #

def _ext_sentence_reason(
    sentence: str, board: chess.Board, rec_move: Optional[chess.Move]
) -> Optional[str]:
    """First demonstrably-false extended claim in ``sentence`` (or ``None``).

    Board-state claims are evaluated on the current board and — when a
    recommended move is supplied — the board after that move; a claim true on
    either board is never flagged. Relational / hanging / move-consequence checks
    abstain on multi-move lines (which resolve to no single board), leaving those
    to the LLM judge.
    """
    stm = board.turn  # the mover's perspective, kept stable across both boards
    boards = [board]
    if rec_move is not None:
        try:
            post = board.copy(stack=False)
            post.push(rec_move)
            boards.append(post)
        except Exception:  # noqa: BLE001 - never gate on a copy/push failure
            pass

    reason = _verdicts(_at_location_verdict, sentence, boards, stm)
    if reason:
        return reason

    multi = _is_multi_move_line(sentence)
    if not multi:
        reason = _verdicts(_relational_verdict, sentence, boards, stm)
        if reason:
            return reason
        try:
            reason = _move_consequence_reason(sentence, board, rec_move)
        except Exception:  # noqa: BLE001
            reason = None
        if reason:
            return reason

    for fn in (_turn_verdict, _castle_verdict, _ep_verdict, _material_verdict):
        reason = _verdicts(fn, sentence, boards, stm)
        if reason:
            return reason

    if not multi:
        reason = _verdicts(_hanging_verdict, sentence, boards, stm)
        if reason:
            return reason
    return None


def verify_text_ext(
    text: str, fen: str, recommended_uci: Optional[str] = None
) -> VerifyResult:
    """Location-class checks (reused) **plus** the widened structural checks.

    Runs :func:`verify_text` for piece location/existence (unchanged, on the
    current board), then adds relational, move-consequence, turn/rights, material
    and hanging checks. Returns the same ``VerifyResult`` shape: ``clean`` is the
    text with every flagged sentence removed; ``violations`` is one
    :class:`Violation` per dropped sentence.

    ``recommended_uci`` (the move the coach recommends) is used two ways: it
    unlocks move-consequence claims phrased without SAN ("develops the knight
    from b1 to f3"), and it lets every board-state claim be checked against the
    position AFTER the move — so a truthful post-move description ("then the
    knight covers e5") is not mistaken for a false statement about the current
    board.
    """
    base = verify_text(text, fen)
    try:
        board = chess.Board(fen)
    except ValueError:
        return base  # unparseable FEN — base already returned clean=text

    rec_move: Optional[chess.Move] = None
    if recommended_uci:
        try:
            cand = chess.Move.from_uci(recommended_uci)
            if cand in board.legal_moves:
                rec_move = cand
        except ValueError:
            rec_move = None

    base_flagged = {v.sentence for v in base.violations}
    violations: List[Violation] = list(base.violations)
    kept: List[str] = []
    for s in (s for s in _SENTENCE_SPLIT.split(text or "") if s.strip()):
        ss = s.strip()
        if ss in base_flagged:
            continue  # base already dropped this sentence
        reason = _ext_sentence_reason(ss, board, rec_move)
        if reason is None:
            kept.append(ss)
        else:
            violations.append(Violation(sentence=ss, reason=reason))

    return VerifyResult(clean=" ".join(kept).strip(), violations=violations)
