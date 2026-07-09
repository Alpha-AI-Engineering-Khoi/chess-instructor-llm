"""The shared faithfulness gate (:mod:`src.teacher.coach_gate`) — behaviour tests.

These lock in the honesty contract the whole base-vs-tuned comparison rests on:
the gate keeps a clean draft, RE-SAMPLES past a fabricated one, and falls back to
a deterministic, verifiably-true explanation when every draft fails — and that
fallback passes the extended verifier. Because the eval and the shipped server
both call this exact code, the two cannot diverge.
"""

from __future__ import annotations

import chess

from src.engine.faithfulness_ext import verify_text_ext
from src.eval.evaluate import extract_recommended_move
from src.teacher.coach_gate import (
    GateResult,
    extract_recommended,
    pick_recommendation,
    run_gate,
    verified_coaching,
)

# 1.e4 e5, White to move. e5 holds a black PAWN (never a rook) -> a "rook on e5"
# claim is demonstrably false on the current board (and after any White move).
FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
POOL = [
    {"uci": "g1f3", "san": "Nf3", "cp": 20, "pv": []},
    {"uci": "b1c3", "san": "Nc3", "cp": 15, "pv": []},
]

FABRICATED = ("I'd play Nf3. There is a black rook on e5 that you can win. "
              "Takeaway: grab free material.")
CLEAN = ("I'd play Nf3. It develops a knight toward the center and prepares to "
         "castle. Takeaway: develop your pieces early.")


def test_fabricated_draft_is_flagged_and_clean_is_not():
    # Sanity: the gate's own check must reject the fabricated draft and pass the clean one.
    assert not verify_text_ext(FABRICATED, FEN).ok
    assert verify_text_ext(CLEAN, FEN).ok


def test_clean_first_try_is_kept():
    res = run_gate(lambda s, u: CLEAN, "sys", "user", FEN, POOL, "e2e4",
                   max_attempts=6, gate_on=True)
    assert isinstance(res, GateResult)
    assert res.attempts == 1
    assert res.verified_fallback is False
    assert res.rec_uci == "g1f3"
    assert "Nf3" in res.text


def test_regenerates_past_fabrication():
    drafts = [FABRICATED, FABRICATED, CLEAN]
    calls = {"n": 0}

    def run_fn(_s, _u):
        i = min(calls["n"], len(drafts) - 1)
        calls["n"] += 1
        return drafts[i]

    res = run_gate(run_fn, "sys", "user", FEN, POOL, "e2e4", max_attempts=6, gate_on=True)
    assert res.attempts == 3
    assert res.verified_fallback is False
    assert "rook on e5" not in res.text.lower()


def test_falls_back_to_verified_when_all_drafts_fabricate():
    res = run_gate(lambda s, u: FABRICATED, "sys", "user", FEN, POOL, "e2e4",
                   max_attempts=3, gate_on=True)
    assert res.attempts == 3
    assert res.verified_fallback is True
    # The fallback text is truthful by construction.
    assert verify_text_ext(res.text, FEN).ok
    assert res.rec_uci in {m["uci"] for m in POOL}


def test_gate_off_keeps_single_draft_even_if_fabricated():
    res = run_gate(lambda s, u: FABRICATED, "sys", "user", FEN, POOL, "e2e4",
                   max_attempts=6, gate_on=False)
    assert res.attempts == 1
    assert res.verified_fallback is False
    assert "e5" in res.text  # the (ungated) draft is returned verbatim-ish


def test_verified_coaching_is_faithful():
    board = chess.Board(FEN)
    for uci in ("g1f3", "b1c3", "f1c4"):
        body, takeaway = verified_coaching(board, chess.Move.from_uci(uci))
        assert verify_text_ext(f"{body} {takeaway}", FEN).ok
        assert body.startswith("I'd play")


# --------------------------------------------------------------------------- #
# Recommended-move extraction — avoid-framing awareness (the moat-metric bug)
# --------------------------------------------------------------------------- #
# A rook-and-pawn endgame where the student ALREADY played the best move (Kd6)
# and the coach agrees, while also naming a move to AVOID ("...b2+", a legal,
# sound discovered check). The old scanner skipped the student's own move and
# grabbed the FIRST other legal SAN — here b2+, a move the coach explicitly told
# the student NOT to play — corrupting the shown move and faking a tier "fork".
_ENDGAME_FEN = "8/8/4k2p/1R6/7P/rp3KP1/8/8 b - - 5 46"
_ENDGAME_POOL = [
    {"uci": "e6d6", "san": "Kd6", "cp": 0, "pv": []},   # student's move == best
    {"uci": "b3b2", "san": "b2+", "cp": -30, "pv": []},  # sound, but coached AGAINST
    {"uci": "a3a2", "san": "Ra2", "cp": -40, "pv": []},
]
_ADVANCED_ENDORSE_STUDENT = (
    "I'd play Kd6. Play Kd6 again — you found the right kind of endgame move. "
    "Improving the king matters more than forcing checks too early. "
    "Run a checklist before choosing a forcing-looking move like ...b2+. "
    "That routine leads you to Kd6 rather than rushing into checks."
)


def test_avoid_framed_move_is_not_recommended_shipped():
    # The shipped coach's extractor must return Kd6 (the endorsed pick), never b2+.
    board = chess.Board(_ENDGAME_FEN)
    assert extract_recommended(
        _ADVANCED_ENDORSE_STUDENT, board, _ENDGAME_POOL, "e6d6"
    ) == ("Kd6", "e6d6")


def test_avoid_framed_move_is_not_recommended_metrics():
    # The metrics/showcase extractor (any-legal) must agree — this is what feeds
    # tier-fit / distinct-moves and the showcase move; b2+ here was the artifact.
    assert extract_recommended_move(_ADVANCED_ENDORSE_STUDENT, _ENDGAME_FEN, "e6d6") == (
        "Kd6",
        "e6d6",
    )


def test_endorsed_move_may_equal_student_move():
    # When the student already played the best move and the coach endorses it,
    # the recommendation IS the student's move (previously it was never returned).
    text = "I'd play Nf3. That was your move and it's the right developing idea."
    assert extract_recommended_move(text, FEN, "g1f3") == ("Nf3", "g1f3")


def test_restated_mistake_is_not_recommended():
    # Coach concedes the student's move then gives the real pick: return the pick.
    text = "Your Nc3 is playable, but I'd play Nf3 instead."
    assert extract_recommended_move(text, FEN, "b1c3") == ("Nf3", "g1f3")


def test_rather_than_frames_following_move_as_avoid():
    board = chess.Board(FEN)
    got = pick_recommendation(
        "I'd play Nf3 rather than a slow move like Nc3.",
        board, "e2e4", accept=lambda _u: True,
    )
    assert got == ("Nf3", "g1f3")  # Nf3 endorsed; Nc3 after "rather than ... like" is avoided


def test_endorsement_overrides_a_preceding_avoided_list():
    # An avoided LIST keeps its framing across commas ("such as Nc3, Nb1"), but an
    # explicit endorsement cue right before the pick still recovers it.
    board = chess.Board(FEN)
    got = pick_recommendation(
        "Instead of a slow move such as Nc3, Na3, the move is Nf3.",
        board, "e2e4", accept=lambda _u: True,
    )
    assert got == ("Nf3", "g1f3")


def test_avoided_list_across_commas_is_all_skipped():
    # The regression that reappeared as a fake tier "fork": a comma-separated list
    # of moves to AVOID after one cue must ALL be treated as avoid, not just the
    # first. Here the endorsed/played move is d2+; Rh8, Ra8, Re8+ are the avoided
    # list. (Rook endgame, Black to move; d2+ is a sound discovered check.)
    fen = "2r5/8/1RP5/8/1N3p2/2kp4/8/4K3 b - - 5 48"
    board = chess.Board(fen)
    text = (
        "I'd play d2+. Instead of making a quiet rook move like Rh8, Ra8, or even "
        "Re8+, you push the passed pawn with tempo."
    )
    got = pick_recommendation(text, board, "d3d2", accept=lambda _u: True)
    assert got == ("d2+", "d3d2")  # NOT Ra8/Re8+ (each is in the avoided list)


def test_hypothetical_opponent_reply_is_not_recommended():
    # "if Black plays" / "you have ... next" introduce opponent replies / lines.
    text = "I'd play Nf3. If Black plays d5, you have Bc4 as a follow-up."
    assert extract_recommended_move(text, FEN, "b1c3") == ("Nf3", "g1f3")


def test_bare_square_reference_is_not_a_move():
    # "the d4 square" is a coordinate reference, not a recommendation of ...d4.
    board = chess.Board(FEN)
    got = pick_recommendation(
        "Control the d4 square first. The plan Nf3 develops a piece.",
        board, "e2e4", accept=lambda _u: True,
    )
    assert got == ("Nf3", "g1f3")


def test_pool_restriction_still_falls_back_to_engine_best():
    # If the only named move is unsound (not in pool), the shipped coach still
    # returns the engine-best sound move rather than an out-of-pool pick.
    board = chess.Board(FEN)
    san, uci = extract_recommended("I'd play Qh5, going for a cheap attack.", board, POOL, "e2e4")
    assert uci == POOL[0]["uci"]  # Qh5 not in pool -> engine best (Nf3)
