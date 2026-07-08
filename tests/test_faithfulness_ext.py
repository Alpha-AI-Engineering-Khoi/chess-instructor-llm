"""Real tests for the widened structural checker + the LLM-judge aggregation.

Every FEN is hand-built and the expected board fact is stated in a comment. The
guiding invariant of :mod:`src.engine.faithfulness_ext` is **high precision**: a
true or ambiguous sentence must never be flagged; a demonstrably false one must
be. These tests assert both directions for each claim class.

Run:  pytest tests/test_faithfulness_ext.py -v
The LLM-judge tests use a mock client (offline). A live smoke test that hits the
TrueFoundry gateway is included but skipped unless ``RUN_LIVE_JUDGE=1`` and
``TFY_API_KEY`` are set.
"""

import os

import chess
import pytest

from src.engine.faithfulness_ext import verify_text_ext
from src.eval.truthfulness.judge import (
    JudgeClient,
    TruthfulnessJudge,
    aggregate,
    parse_judge_reply,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def reasons(text, fen, uci=None):
    return [v.reason for v in verify_text_ext(text, fen, uci).violations]


def flagged(text, fen, uci=None):
    """True iff at least one sentence in ``text`` was flagged as false."""
    return not verify_text_ext(text, fen, uci).ok


# --------------------------------------------------------------------------- #
# Relational: attacks / defends / pins  (TRUE not flagged, FALSE flagged)
# --------------------------------------------------------------------------- #

def test_relational_attack_true_not_flagged():
    # White bishop b2, black knight f6, on the open a1-h8 diagonal -> bishop
    # really does attack f6.
    fen = "4k3/8/5n2/8/8/8/1B6/4K3 w - - 0 1"
    res = verify_text_ext("The bishop on b2 attacks the knight on f6.", fen)
    assert res.ok, res.violations
    assert "attacks the knight" in res.clean


def test_relational_attack_false_flagged():
    # White knight c5, black bishop e5. A knight on c5 attacks
    # a4,a6,b3,b7,d3,d7,e4,e6 — NOT e5. So the claim is false.
    fen = "4k3/8/8/2N1b3/8/8/8/4K3 w - - 0 1"
    r = reasons("The knight on c5 attacks the bishop on e5.", fen)
    assert r == ["does not attack the bishop on e5"], r


def test_relational_defend_false_flagged():
    # White rook a1, white pawn h2. The rook does not defend h2.
    fen = "4k3/8/8/8/8/8/7P/R3K3 w - - 0 1"
    r = reasons("The rook on a1 defends the pawn on h2.", fen)
    assert r and "does not defend" in r[0], r


def test_relational_pin_true_not_flagged():
    # White bishop g5, black knight f6, black queen e7 behind it on the
    # g5-e7 diagonal -> a real (relative) pin of the knight.
    fen = "4k3/4q3/5n2/6B1/8/8/8/4K3 w - - 0 1"
    assert not flagged("The bishop on g5 pins the knight on f6.", fen)


def test_relational_pin_false_flagged():
    # Same bishop/knight but nothing valuable behind f6 -> not a pin.
    fen = "4k3/8/5n2/6B1/8/8/8/4K3 w - - 0 1"
    r = reasons("The bishop on g5 pins the knight on f6.", fen)
    assert r == ["does not pin the knight on f6"], r


# --------------------------------------------------------------------------- #
# Move-consequence: capture / check / from-square
# --------------------------------------------------------------------------- #

def test_false_capture_flagged():
    # Black bishop really on e5 (so the base location check passes), white
    # knight g1. Nf3 is a quiet move, not a capture.
    fen = "4k3/8/8/4b3/8/8/8/4K1N1 w - - 0 1"
    r = reasons("Playing Nf3 captures the bishop on e5.", fen)
    assert r == ["Nf3 does not capture anything"], r


def test_true_capture_not_flagged():
    # White knight f3, black bishop e5: Nxe5 really captures the bishop.
    fen = "4k3/8/8/4b3/8/5N2/8/4K3 w - - 0 1"
    assert not flagged("Nxe5 captures the bishop on e5.", fen)


def test_false_check_flagged():
    # From the start position Nf3 does not give check.
    fen = chess.STARTING_FEN
    r = reasons("Nf3 gives check.", fen)
    assert r == ["Nf3 does not give check"], r


def test_from_square_false_flagged():
    # Recommended move is Ng1-f3; claiming it comes from b1 is false.
    fen = chess.STARTING_FEN
    r = reasons("This develops the knight from b1 to f3.", fen, "g1f3")
    assert r and "from g1, not b1" in r[0], r


def test_from_square_true_not_flagged():
    # Same move, correct origin g1 -> not flagged (notation robustness: "from g1").
    fen = chess.STARTING_FEN
    assert not flagged("This develops the knight from g1 to f3.", fen, "g1f3")


# --------------------------------------------------------------------------- #
# Turn / rights
# --------------------------------------------------------------------------- #

def test_wrong_side_to_move_flagged():
    fen = chess.STARTING_FEN  # White to move
    r = reasons("It is Black to move here.", fen)
    assert r == ["it is White to move, not Black"], r


def test_correct_side_to_move_not_flagged():
    fen = chess.STARTING_FEN
    assert not flagged("It is White to move here.", fen)


def test_castling_rights_false_flagged():
    # No castling rights in the FEN ("-"), so "White can castle kingside" is false.
    fen = "r3k2r/8/8/8/8/8/8/4K3 w - - 0 1"
    r = reasons("White can castle kingside.", fen)
    assert r == ["White has no kingside castling rights"], r


# --------------------------------------------------------------------------- #
# Material / counts  (TRUE not flagged, FALSE flagged)
# --------------------------------------------------------------------------- #

def test_false_material_count_flagged():
    # Only one white rook on the board; "both rooks" is false.
    fen = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    r = reasons("White has both rooks ready.", fen)
    assert r == ["White has 1 rook(s), not 2"], r


def test_true_material_count_not_flagged():
    # Two white bishops (c1, f1).
    fen = "4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1"
    assert not flagged("White has two bishops.", fen)


def test_up_material_direction_false_flagged():
    # Black is a whole rook up; "White is up a pawn" is false in direction.
    fen = "r3k3/8/8/8/8/8/8/4K3 b - - 0 1"
    r = reasons("White is up a pawn.", fen)
    assert r and "not up material" in r[0], r


# --------------------------------------------------------------------------- #
# Hanging / undefended  (hanging=TRUE not flagged, not-hanging flagged)
# --------------------------------------------------------------------------- #

def test_not_hanging_flagged():
    # a1 rook is defended by the a2 rook and not attacked by Black -> not hanging.
    fen = "4k3/8/8/8/8/8/R7/R3K3 w - - 0 1"
    r = reasons("The rook on a1 is hanging.", fen)
    assert r == ["the rook on a1 is not hanging"], r


def test_hanging_true_not_flagged():
    # Black rook d8 attacks the undefended white knight d4 down the open d-file
    # -> it really is hanging, so the claim must NOT be flagged.
    fen = "3rk3/8/8/8/3N4/8/8/4K3 w - - 0 1"
    assert not flagged("The knight on d4 is hanging.", fen)


def test_undefended_false_flagged():
    # Knight d4 is defended by the rook on d2 -> "undefended" is false.
    fen = "4k3/8/8/8/3N4/8/3R4/4K3 w - - 0 1"
    r = reasons("The knight on d4 is undefended.", fen)
    assert r == ["the knight on d4 is defended, not undefended"], r


# --------------------------------------------------------------------------- #
# Paraphrase / notation robustness: "at <sq>", piece-letter SAN, "from g1"
# --------------------------------------------------------------------------- #

def test_paraphrase_at_square_true_not_flagged():
    # "attacks the knight at f6" (uses "at", not "on") — a true claim.
    fen = "4k3/8/5n2/8/8/8/1B6/4K3 w - - 0 1"
    assert not flagged("The bishop on b2 attacks the knight at f6.", fen)


def test_paraphrase_at_square_false_flagged():
    # Same but a white pawn on d4 blocks the b2->f6 diagonal, so it is false.
    fen = "4k3/8/5n2/8/3P4/8/1B6/4K3 w - - 0 1"
    r = reasons("The bishop on b2 attacks the knight at f6.", fen)
    assert r == ["does not attack the knight on f6"], r


def test_piece_letter_san_subject_false_flagged():
    # SAN subject with a piece letter: Nc5 (from d3). A knight on c5 does not
    # attack e5, so this is false.
    fen = "4k3/8/8/4b3/8/3N4/8/4K3 w - - 0 1"
    r = reasons("Nc5 attacks the bishop on e5.", fen)
    assert r == ["does not attack the bishop on e5"], r


def test_piece_letter_san_subject_true_not_flagged():
    # Nc5 (from d3); a knight on c5 does attack e4 (a real black knight is there).
    fen = "4k3/8/8/8/4n3/3N4/8/4K3 w - - 0 1"
    assert not flagged("Nc5 attacks the knight on e4.", fen)


# --------------------------------------------------------------------------- #
# Post-move calibration: claims about the position AFTER the recommended move
# are checked against the post-move board too, so a true post-move description
# is not mistaken for a false statement about the current board — but a claim
# false on BOTH boards (and a genuine current-board lie) is still flagged.
# --------------------------------------------------------------------------- #

def test_post_move_true_claim_not_flagged():
    # White knight g1; the recommended move is Nf3 (g1f3). On the CURRENT board a
    # knight on g1 does NOT attack e5, but AFTER Nf3 the knight on f3 does attack
    # the black bishop on e5 — so this post-move description is true and must not
    # be flagged (the over-fire bug this calibration fixes).
    fen = "4k3/8/8/4b3/8/8/8/4K1N1 w - - 0 1"
    res = verify_text_ext("After Nf3, the knight attacks the bishop on e5.", fen, "g1f3")
    assert res.ok, res.violations


def test_post_move_false_claim_flagged():
    # Same move Nf3, but the bishop is on d5: a knight on f3 does not attack d5
    # (and neither does the knight on g1), so the post-move-framed claim is false
    # on BOTH boards and must be flagged.
    fen = "4k3/8/8/3b4/8/8/8/4K1N1 w - - 0 1"
    r = reasons("After Nf3, the knight attacks the bishop on d5.", fen, "g1f3")
    assert r == ["does not attack the bishop on d5"], r


def test_current_board_lie_still_flagged_with_move():
    # A genuine current-board lie: a knight on c5 does not attack e5. The
    # recommended move (Kd1) does not touch the knight or bishop, so the claim is
    # false on both boards — passing a recommended move must NOT rescue a real lie.
    fen = "4k3/8/8/2N1b3/8/8/8/4K3 w - - 0 1"
    r = reasons("The knight on c5 attacks the bishop on e5.", fen, "e1d1")
    assert r == ["does not attack the bishop on e5"], r


# --------------------------------------------------------------------------- #
# Precision guards + reuse of the base location verifier
# --------------------------------------------------------------------------- #

def test_strategic_sentence_not_flagged():
    # Pure strategy, no checkable board fact -> must pass untouched.
    fen = chess.STARTING_FEN
    text = "Control the center and develop your pieces before starting an attack."
    res = verify_text_ext(text, fen)
    assert res.ok
    assert res.clean == text


def test_base_location_check_still_applies():
    # verify_text_ext must still catch the location class handled by verify_text.
    fen = chess.STARTING_FEN  # no queen on a1
    r = reasons("The queen on a1 is a monster.", fen)
    assert r == ["no queen on a1"], r


def test_invalid_fen_returns_text_unchanged():
    res = verify_text_ext("Anything at all.", "not a real fen")
    assert res.clean == "Anything at all."
    assert res.ok


# --------------------------------------------------------------------------- #
# LLM-judge aggregation (offline, mocked clients)
# --------------------------------------------------------------------------- #

class MockChat:
    """Stands in for backends.TFYChat: complete(system, user) -> (text, usage)."""

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user):
        return self.reply, {"prompt_tokens": 7, "completion_tokens": 3}


FEN0 = chess.STARTING_FEN
FACTS0 = "Side to move: White. Material is equal. Nf3 develops a knight; not a capture."


def test_judge_flags_when_any_judge_objects():
    panel = [
        JudgeClient("gpt", MockChat('{"truthful": false, "flagged": '
                                    '[{"claim":"wins the queen","reason":"no such tactic"}]}')),
        JudgeClient("claude", MockChat('{"truthful": true, "flagged": []}')),
    ]
    res = TruthfulnessJudge(panel, aggregation="any").assess(
        "Nf3 wins the queen.", FEN0, "Nf3", FACTS0
    ).to_dict()
    assert res["truthful"] is False
    assert res["n_judges"] == 2
    assert {f["judge"] for f in res["flagged"]} == {"gpt"}


def test_judge_truthful_when_all_clean():
    panel = [
        JudgeClient("gpt", MockChat('{"truthful": true, "flagged": []}')),
        JudgeClient("gemini", MockChat('{"truthful": true, "flagged": []}')),
    ]
    res = TruthfulnessJudge(panel).assess("Develop your pieces.", FEN0, None, FACTS0).to_dict()
    assert res["truthful"] is True
    assert res["flagged"] == []
    assert res["agreement"] == 1.0


def test_judge_majority_vs_any_differ():
    # One of three flags: "any" -> not truthful; "majority" -> truthful.
    panel = [
        JudgeClient("gpt", MockChat('{"truthful": false, "flagged": '
                                    '[{"claim":"mate in 3","reason":"not forced"}]}')),
        JudgeClient("claude", MockChat('{"truthful": true, "flagged": []}')),
        JudgeClient("gemini", MockChat('{"truthful": true, "flagged": []}')),
    ]
    any_res = TruthfulnessJudge(panel, aggregation="any").assess("x", FEN0, None, FACTS0).to_dict()
    maj_res = TruthfulnessJudge(panel, aggregation="majority").assess("x", FEN0, None, FACTS0).to_dict()
    assert any_res["truthful"] is False
    assert maj_res["truthful"] is True
    assert any_res["agreement"] == maj_res["agreement"]  # same 2/3 split


def test_judge_survives_a_broken_judge():
    class Boom:
        def complete(self, system, user):
            raise RuntimeError("gateway 500")

    panel = [
        JudgeClient("gpt", MockChat('{"truthful": true, "flagged": []}')),
        JudgeClient("claude", Boom()),
    ]
    res = TruthfulnessJudge(panel).assess("x", FEN0, None, FACTS0).to_dict()
    assert res["n_judges"] == 1          # broken judge dropped
    assert "claude" in res["errors"]


def test_judge_requires_two_judges():
    with pytest.raises(ValueError):
        TruthfulnessJudge([JudgeClient("solo", MockChat("{}"))])


def test_parse_judge_reply_robustness():
    # Prose wrapper around JSON is tolerated.
    p = parse_judge_reply('Sure, here you go: {"truthful": false, '
                          '"flagged": [{"claim":"c","reason":"r"}]} done')
    assert p["truthful"] is False and p["flagged"][0]["claim"] == "c"
    # truthful:false with no items becomes one unspecified flag.
    assert parse_judge_reply('{"truthful": false}')["flagged"][0]["claim"] == "(unspecified)"
    # garbage -> treated as truthful (no concrete flags to act on).
    assert parse_judge_reply("not json")["truthful"] is True


def test_aggregate_empty_panel():
    # No successful judges -> truthful defaults True with n_judges 0 (caller detects).
    agg = aggregate([], mode="any")
    assert agg == {"truthful": True, "flagged": [], "n_judges": 0, "agreement": 0.0}


@pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_JUDGE") and os.environ.get("TFY_API_KEY")),
    reason="live gateway smoke test; set RUN_LIVE_JUDGE=1 and TFY_API_KEY to run",
)
def test_llm_judge_live_smoke():
    from src.eval.truthfulness.judge import assess_truthfulness

    res = assess_truthfulness(
        "Play Nf3. This forces checkmate in two and wins Black's queen on the spot.",
        chess.STARTING_FEN,
        "Nf3",
        "Side to move: White. Material is equal. Nf3 develops a knight; not a capture.",
        judge_keys=("gpt", "gemini"),
        aggregation="any",
    )
    assert res["n_judges"] >= 2
    assert res["truthful"] is False
