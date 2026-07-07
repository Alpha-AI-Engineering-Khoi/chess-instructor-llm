"""Re-score the objective "move" axis as MOVE SAFETY (no blunders).

The v2 benchmark scored `move_sound` = "pick is inside Stockfish's sound pool"
(cp-loss <= SOUND_TOLERANCE_CP, 150). That is a *quality* bar, not a *safety*
bar. This script recomputes a blunder-only floor:

    a pick is SAFE  unless it is a BLUNDER (cp-loss >= BLUNDER_CP, 250).

Method (self-consistent, non-LLM):
  - For every scenario that has >=1 non-sound *parseable* pick, evaluate the
    base position once with Stockfish -> best_cp_fresh (mover's POV).
  - For every distinct (fen, rec_uci) among those picks, push the move and
    evaluate the resulting position; rec_cp = -score (back to mover's POV).
  - cp_loss = best_cp_fresh - rec_cp ; blunder iff cp_loss >= BLUNDER_CP.
  - Picks already inside the sound pool (move_sound == True) are safe by
    construction (cp-loss <= 150 < 250). Unparseable / empty picks recommend no
    usable move -> counted as NOT safe (same denominator as move_sound: n=100).

Sign convention is validated against the stored student_move.cp_loss before any
re-scoring is trusted. All evals use one Stockfish process, fixed movetime.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import BLUNDER_CP, STOCKFISH_BIN  # noqa: E402

BENCH = ROOT / "data" / (sys.argv[1] if len(sys.argv) > 1 else "benchmark_v2")
MOVETIME_MS = 500
HASH_MB = 256
THREADS = 2
MATE = 100_000
MODELS = ["ours", "base", "gpt", "claude", "gemini"]
CONDS = ["ungrounded", "grounded"]


class Engine:
    def __init__(self, path: str):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._send("uci"); self._wait("uciok")
        self._send(f"setoption name Threads value {THREADS}")
        self._send(f"setoption name Hash value {HASH_MB}")
        self._send("setoption name MultiPV value 1")
        self._send("isready"); self._wait("readyok")

    def _send(self, cmd: str) -> None:
        assert self.p.stdin
        self.p.stdin.write(cmd + "\n"); self.p.stdin.flush()

    def _wait(self, token: str) -> None:
        assert self.p.stdout
        for line in self.p.stdout:
            if line.strip().startswith(token):
                return

    def eval_cp(self, fen: str, moves: list[str]) -> int:
        """Score (cp, mover-of-resulting-position POV) after playing `moves`."""
        assert self.p.stdout
        self._send("ucinewgame")
        pos = f"position fen {fen}"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        self._send(f"go movetime {MOVETIME_MS}")
        score = 0
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("info") and " score " in line:
                toks = line.split()
                i = toks.index("score")
                kind, val = toks[i + 1], int(toks[i + 2])
                score = (MATE - abs(val)) * (1 if val > 0 else -1) if kind == "mate" else val
            elif line.startswith("bestmove"):
                break
        return max(-MATE, min(MATE, score))

    def close(self) -> None:
        try:
            self._send("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def main() -> int:
    scen = {}
    with open(BENCH / "scenarios.jsonl") as f:
        for line in f:
            d = json.loads(line); scen[d["id"]] = d
    picks = [json.loads(l) for l in open(BENCH / "objective.jsonl")]

    # collect distinct work
    base_fens: set[str] = set()
    after: set[tuple[str, str]] = set()
    for r in picks:
        if r.get("move_sound"):
            continue
        rec = r.get("rec_uci")
        if r.get("move_parseable") and rec:
            s = scen[r["scenario_id"]]
            base_fens.add(s["fen"]); after.add((s["fen"], rec))

    eng = Engine(STOCKFISH_BIN)

    # ---- validate sign convention on stored student cp_loss (has ground truth)
    print("== sign-convention validation (fresh best_cp - student rec_cp vs stored cp_loss) ==")
    checked = 0
    for sid, s in scen.items():
        sm = s.get("student_move") or {}
        if "uci" not in sm or "cp_loss" not in sm:
            continue
        best = eng.eval_cp(s["fen"], [])
        rec_cp = -eng.eval_cp(s["fen"], [sm["uci"]])
        got = best - rec_cp
        print(f"  {sid:14} stored={sm['cp_loss']:>6}  computed={got:>7}  (best_fresh={best})")
        checked += 1
        if checked >= 6:
            break

    # ---- evaluate the real work
    best_cp = {fen: eng.eval_cp(fen, []) for fen in sorted(base_fens)}
    rec_after = {}
    for i, (fen, uci) in enumerate(sorted(after), 1):
        rec_after[(fen, uci)] = eng.eval_cp(fen, [uci])
    eng.close()

    def cp_loss(fen: str, uci: str) -> int:
        return best_cp[fen] - (-rec_after[(fen, uci)])

    # ---- classify + aggregate
    agg = {m: {c: {"safe100": 0, "safe_parse": 0, "parse": 0,
                   "blunder": 0, "n": 0} for c in CONDS} for m in MODELS}
    worst = []
    for r in picks:
        m, c = r["model"], r["condition"]
        a = agg[m][c]; a["n"] += 1
        rec = r.get("rec_uci"); parseable = bool(r.get("move_parseable") and rec)
        if parseable:
            a["parse"] += 1
        if r.get("move_sound"):
            a["safe100"] += 1; a["safe_parse"] += 1
            continue
        if not parseable:
            continue  # no usable move -> not safe (denominator stays 100)
        s = scen[r["scenario_id"]]
        loss = cp_loss(s["fen"], rec)
        if loss >= BLUNDER_CP:
            a["blunder"] += 1
            worst.append((loss, m, c, r["scenario_id"], r.get("rec_san"), s["tier"]))
        else:
            a["safe100"] += 1; a["safe_parse"] += 1

    def rate(x, d):
        return round(x / d, 4) if d else 0.0

    out = {"move_safe": {m: {} for m in MODELS},
           "move_safe_of_parseable": {m: {} for m in MODELS},
           "blunder_rate": {m: {} for m in MODELS}}
    print("\n== MOVE SAFETY (no blunders) — blunder cutoff cp-loss >= "
          f"{BLUNDER_CP}, movetime {MOVETIME_MS}ms ==")
    print(f"{'model/cond':20} {'safe/100':>9} {'safe%':>7} {'safe%|parse':>12} {'blunders':>9}")
    for m in MODELS:
        for c in CONDS:
            a = agg[m][c]
            out["move_safe"][m][c] = rate(a["safe100"], a["n"])
            out["move_safe_of_parseable"][m][c] = rate(a["safe_parse"], a["parse"])
            out["blunder_rate"][m][c] = rate(a["blunder"], a["n"])
            print(f"{m+'/'+c:20} {a['safe100']:>4}/{a['n']:<4} "
                  f"{rate(a['safe100'], a['n'])*100:>6.1f}% "
                  f"{rate(a['safe_parse'], a['parse'])*100:>11.1f}% "
                  f"{a['blunder']:>9}")

    print("\n== worst blunders recommended (cp-loss) ==")
    for loss, m, c, sid, san, tier in sorted(worst, reverse=True)[:15]:
        print(f"  cp-loss {loss:>7}  {m:<7} {c:<10} {tier:<12} {san}  ({sid})")

    (BENCH / "move_safety.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {BENCH / 'move_safety.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
