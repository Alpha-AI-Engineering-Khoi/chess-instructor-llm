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
from src.teacher.coach_gate import GateResult, run_gate, verified_coaching

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
