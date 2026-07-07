"""Build the self-contained "same-input faithfulness head-to-head" data slice.

Real data only. Pulls verbatim coaching text from ``generations.jsonl``, the
per-generation fabrication flags/violations from ``objective.jsonl``, and the
position + identical grounded VERIFIED-FACTS input from ``scenarios.jsonl``
(rebuilt with the exact same ``render_pool_facts`` the benchmark fed every model).

Nothing is paraphrased or invented. The output JSON is embedded into the Space's
index.html so the page is self-contained (no runtime).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.engine.position_facts import render_pool_facts  # noqa: E402

BENCH = ROOT / "data" / "benchmark_v2"
COND = "grounded"

# Hand-picked from the 29 clean candidates (OURS move sound + fabricates a board
# fact; all three frontier models faithful). 2 beginner / 2 intermediate / 2
# advanced, spread across phases + severities. See selection notes in the chat.
PICKS = [
    "WV60r7W7_20",   # beginner / opening / inaccuracy   — "queen on b2"
    "9M8bagRc_36",   # beginner / middlegame / blunder   — "rook on d1"
    "bWwdtmJb_18",   # intermediate / opening / inaccuracy — "pawn on c3"
    "TiMre7nm_43",   # intermediate / middlegame / inacc. — "bishop on c3" (all 5 pick Bxb6)
    "VsykaIN0_82",   # advanced / endgame / inaccuracy    — "rook on a8"
    "uHDsJlF3_24",   # advanced / opening / mistake       — "queen on g4"
]

MODEL_ORDER = ["ours", "gpt", "claude", "gemini", "base"]
MODEL_NAME = {
    "ours": "OURS-v2",
    "base": "BASE",
    "gpt": "GPT-5.5",
    "claude": "Claude Opus 4.8",
    "gemini": "Gemini 3.1 Pro",
}


def norm_ws(s: str) -> str:
    """Collapse all whitespace to single spaces (prose reads fine, matching robust)."""
    return re.sub(r"\s+", " ", s).strip()


def labeled_board(fen: str) -> str:
    """ASCII board with rank/file labels; uppercase=White, lowercase=Black."""
    board = chess.Board(fen)
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.symbol() if piece else ".")
        rows.append(f"{rank + 1}  " + " ".join(cells))
    rows.append("   " + " ".join("abcdefgh"))
    return "\n".join(rows)


def facts_summary(fen: str, sound_sans: list[str], facts_full: str) -> str:
    board = chess.Board(fen)
    side = "White" if board.turn == chess.WHITE else "Black"
    loose = "none"
    for line in facts_full.splitlines():
        low = line.lower()
        if "undefended" in low or "loose" in low:
            # keep the text after the first colon
            if ":" in line:
                loose = line.split(":", 1)[1].strip().rstrip(".")
            break
    pool = ", ".join(sound_sans[:6])
    return f"{side} to move · sound moves: {pool} · undefended: {loose}"


def main() -> int:
    scn = {}
    for line in (BENCH / "scenarios.jsonl").open():
        d = json.loads(line)
        scn[d["id"]] = d

    obj = {}
    for line in (BENCH / "objective.jsonl").open():
        d = json.loads(line)
        if d["condition"] == COND:
            obj[(d["scenario_id"], d["model"])] = d

    gen = {}
    for line in (BENCH / "generations.jsonl").open():
        d = json.loads(line)
        if d["condition"] == COND:
            gen[(d["scenario_id"], d["model"])] = d["output"]

    out = []
    problems = []
    for sid in PICKS:
        s = scn[sid]
        sound_sans = [m["san"] for m in s["sound_pool"]]
        facts_full = render_pool_facts(
            s["fen"], [{"uci": m["uci"], "san": m["san"], "cp": int(m["cp"])} for m in s["sound_pool"]]
        )
        rec = {
            "scenario_id": sid,
            "fen": s["fen"],
            "tier": s["tier"],
            "phase": s["phase"],
            "severity": s["severity"],
            "student_move_san": s["student_move"]["san"],
            "side_to_move": "White" if chess.Board(s["fen"]).turn == chess.WHITE else "Black",
            "board_ascii": labeled_board(s["fen"]),
            "facts_full": facts_full,
            "facts_summary": facts_summary(s["fen"], sound_sans, facts_full),
            "models": {},
        }
        for m in MODEL_ORDER:
            o = obj[(sid, m)]
            text = norm_ws(gen[(sid, m)])
            viols = o.get("violations") or []
            # sanity: every flagged sentence must be locatable in the (normalized) text
            for v in viols:
                if norm_ws(v["sentence"]) not in text:
                    problems.append((sid, m, v["sentence"]))
            rec["models"][m] = {
                "name": MODEL_NAME[m],
                "rec_san": o.get("rec_san"),
                "move_sound": bool(o.get("move_sound")),
                "fabricated": bool(o.get("fabricated")),
                "output": text,
                "violations": [{"sentence": norm_ws(v["sentence"]), "reason": v["reason"]} for v in viols],
            }
        out.append(rec)

    # Guard: our thesis for this section requires OURS fab + all frontier faithful.
    for rec in out:
        assert rec["models"]["ours"]["fabricated"], f"{rec['scenario_id']} ours not fabricated"
        for m in ("gpt", "claude", "gemini"):
            assert not rec["models"][m]["fabricated"], f"{rec['scenario_id']} {m} fabricated"
        assert rec["models"]["ours"]["move_sound"], f"{rec['scenario_id']} ours move not sound"

    dest = BENCH / "headtohead.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")

    print(f"wrote {dest} · positions={len(out)}")
    if problems:
        print("!! UNLOCATABLE flagged sentences (need highlight fallback):")
        for p in problems:
            print("   ", p)
    else:
        print("all flagged sentences located in normalized OURS text (inline highlight OK)")
    for rec in out:
        om = rec["models"]["ours"]
        print(
            f"  {rec['scenario_id']:14s} {rec['tier'][:4]:4s}/{rec['phase'][:3]}/{rec['severity'][:5]:5s} "
            f"ours={om['rec_san']} flags={len(om['violations'])}  "
            f"gpt={rec['models']['gpt']['rec_san']} claude={rec['models']['claude']['rec_san']} "
            f"gemini={rec['models']['gemini']['rec_san']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
