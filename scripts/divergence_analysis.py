#!/usr/bin/env python3
"""Honest move-SELECTION divergence analysis for the tuned chess coach.

Question under test (the user's suspicion): *does the tuned coach recommend the
same move at every tier and merely mirror Stockfish / Maia, adding no
move-selection value?*

This script answers it with data. For a balanced set of HELD-OUT positions
(none of whose board+side-to-move appears in ``data/dataset/train.jsonl`` /
``valid.jsonl``) it:

1. Computes the Stockfish sound pool (``pool[0]`` = engine best) and the
   tier-specific Maia top human move — reusing the repo's own engine modules.
2. Runs the tuned MLX coach THREE times per position (beginner / intermediate /
   advanced), building the **exact** grounded prompt the live app uses
   (``render_pool_facts`` + ``render_user_prompt`` and the same system prompt =
   ``coach_system.md`` + grounding + format suffix). It extracts the recommended
   move with the **same** logic the live API uses (``_extract_recommended``),
   but *instrumented* so we can tell a genuine, named pick apart from the API's
   ``pool[0]`` fallback (which would otherwise inflate "mirrors Stockfish").
3. Records, per position, the three tier moves + Stockfish best + Maia tops, and
   enough engine/coaching context to build a swappable library gallery later.

Decoding is **greedy (temp=0)** by default so tier differences reflect genuine
tier-conditioning, not sampling noise. ``--temp`` reproduces the live sampler.

It does NOT depend on the running web backend and does NOT touch the frontend.

Run (from repo root, pinned interpreter)::

    ~/.venvs/mlx/bin/python -m scripts.divergence_analysis --num 120
    ~/.venvs/mlx/bin/python -m scripts.divergence_analysis --num 120 --temp 0.7 \
        --out data/analysis/divergence_temp07.jsonl --noise-floor
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import schema, settings  # noqa: E402
from src.engine import maia_engine, stockfish_engine  # noqa: E402
from src.engine.position_facts import render_pool_facts  # noqa: E402

TIER_ORDER: Tuple[str, ...] = ("beginner", "intermediate", "advanced")
PHASES: Tuple[str, ...] = ("opening", "middlegame", "endgame")
#: The "mistake severities" we balance across (teachable mistakes).
CORE_SEVERITIES: Tuple[str, ...] = ("inaccuracy", "mistake", "blunder")

# --------------------------------------------------------------------------- #
# Prompt + extraction copied VERBATIM from src/api/server.py so this matches
# exactly what the live app shows, with zero web-stack dependency.
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
SYSTEM_PROMPT: str = _COACH_SYSTEM + _GROUNDING + _FORMAT_SUFFIX

# Live-app generation settings (used only when --temp > 0 reproduces the app).
GEN_MAX_TOKENS: int = 640
GEN_TOP_P: float = 0.8
GEN_TOP_K: int = 20

_SAN_RE = re.compile(r"(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)")
_CUE_RE = re.compile(
    r"(?:i['\u2019]?d\s+play|i\s+would\s+play|i['\u2019]?ll\s+play|i\s+play|"
    r"recommend(?:ed)?(?:\s+move)?(?:\s+is)?|best\s+move\s+is|go\s+with|"
    r"choose|consider|play)\s*[:\-]?\s*",
    re.IGNORECASE,
)
_TAKEAWAY_RE = re.compile(r"\b(?:key\s+)?take[-\s]?away\s*:\s*", re.IGNORECASE)
_HR_LINE_RE = re.compile(r"(?m)^[ \t]*[-*_]{3,}[ \t]*$")


def _strip_think(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text and "<think>" not in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def _split_coaching(text: str) -> Tuple[str, str]:
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


def extract_recommended_mode(
    text: str, board: chess.Board, pool: List[Dict[str, Any]], student_uci: str
) -> Tuple[Optional[str], Optional[str], str]:
    """The live API's recommended-move extraction, instrumented to return HOW the
    move was found.

    Delegates to the SHARED, avoid-framing-aware core
    (:func:`src.teacher.coach_gate.pick_recommendation`) so this analysis and the
    shipped coach can never diverge. ``mode``:

      "cue"            -> named right after an explicit endorsement cue ("I'd play X")
      "prose"          -> the coach's pick named elsewhere in the prose (non-avoid)
      "fallback_pool0" -> model named no sound, non-avoid move; API shows engine best
      "none"           -> empty pool (shouldn't happen)
    """
    from src.teacher.coach_gate import (
        _endorsed_indices,
        _is_avoid_framed,
        _san_candidates,
        pick_recommendation,
    )

    pool_ucis = {m["uci"] for m in pool}
    picked = pick_recommendation(
        text, board, student_uci, accept=lambda u: u in pool_ucis
    )
    if picked is not None:
        san, uci = picked
        cands = [t for t in _san_candidates(board, text) if t[3] in pool_ucis]
        endorsed = _endorsed_indices(text, cands)
        avoid = [_is_avoid_framed(text, s, e) for (s, e, _s, _u) in cands]
        mode = "prose"
        for i, (_s, _e, _san, u) in enumerate(cands):
            if u == uci and not avoid[i]:
                mode = "cue" if i in endorsed else "prose"
                break
        return san, uci, mode

    if pool:
        return pool[0]["san"], pool[0]["uci"], "fallback_pool0"
    return None, None, "none"


# --------------------------------------------------------------------------- #
# MLX coach backend (mirrors server.Coach; temp configurable for greedy runs)
# --------------------------------------------------------------------------- #


class Coach:
    def __init__(self, model_path: str, *, temp: float, max_tokens: int) -> None:
        from mlx_lm import generate, load

        self.model_path = model_path
        self.temp = temp
        self.max_tokens = max_tokens
        self._generate = generate
        t0 = time.time()
        self.model, self.tokenizer = load(model_path)
        print(f"  loaded MLX model {model_path!r} in {time.time() - t0:.1f}s", file=sys.stderr)
        try:
            from mlx_lm.sample_utils import make_sampler

            if temp <= 0.0:
                self._sampler = make_sampler(temp=0.0)  # greedy / argmax
            else:
                self._sampler = make_sampler(temp=temp, top_p=GEN_TOP_P, top_k=GEN_TOP_K)
        except Exception:  # noqa: BLE001
            self._sampler = None

    def _apply_template(self, system: str, user: str) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def run(self, system: str, user: str) -> str:
        prompt = self._apply_template(system, user)
        kwargs: Dict[str, Any] = {"max_tokens": self.max_tokens, "verbose": False}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        raw = self._generate(self.model, self.tokenizer, prompt=prompt, **kwargs)
        return _strip_think(raw)


# --------------------------------------------------------------------------- #
# Held-out set: reconstruct board+turn keys from train/valid ASCII boards
# --------------------------------------------------------------------------- #


def _ascii_block_to_placement(user_content: str) -> Optional[str]:
    """Parse the 8x8 ASCII board out of a training user message into a FEN field."""
    lines = user_content.splitlines()
    grid: List[str] = []
    for ln in lines:
        toks = ln.strip().split()
        if len(toks) == 8 and all(t == "." or (len(t) == 1 and t.isalpha()) for t in toks):
            grid.append("".join(toks))
        elif grid and len(grid) == 8:
            break
    if len(grid) != 8:
        return None
    fen_rows: List[str] = []
    for row in grid:
        out, empties = "", 0
        for ch in row:
            if ch == ".":
                empties += 1
            else:
                if empties:
                    out += str(empties)
                    empties = 0
                out += ch
        if empties:
            out += str(empties)
        fen_rows.append(out)
    return "/".join(fen_rows)


def _turn_from_content(user_content: str) -> Optional[str]:
    if "White to move." in user_content:
        return "w"
    if "Black to move." in user_content:
        return "b"
    return None


def pos_key(fen: str) -> str:
    parts = fen.split()
    return parts[0] + " " + parts[1]  # placement + side-to-move


def build_heldin_keys(train: Path, valid: Path) -> set[str]:
    keys: set[str] = set()
    for path in (train, valid):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                user = ""
                for msg in row.get("messages", []):
                    if msg.get("role") == "user":
                        user = msg.get("content", "")
                        break
                placement = _ascii_block_to_placement(user)
                turn = _turn_from_content(user)
                if placement and turn:
                    keys.add(placement + " " + turn)
    return keys


# --------------------------------------------------------------------------- #
# Candidate sourcing + balanced sampling
# --------------------------------------------------------------------------- #


def _phase(fen: str) -> str:
    board = fen.split(" ", 1)[0]
    pieces = sum(1 for c in board if c.isalpha())
    if pieces >= 26:
        return "opening"
    if pieces >= 12:
        return "middlegame"
    return "endgame"


def load_heldout_candidates(candidates: Path, heldin: set[str]) -> List[Dict[str, Any]]:
    """Held-out candidate positions (deduped by FEN) with severity + phase."""
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with candidates.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ti = d.get("teacher_input") or {}
            fen = ti.get("fen")
            if not fen or fen in seen:
                continue
            seen.add(fen)
            if pos_key(fen) in heldin:
                continue  # exclude anything trained on
            sm = ti.get("student_move") or {}
            out.append(
                {
                    "id": d.get("id"),
                    "source_tier": d.get("tier") or ti.get("tier"),
                    "fen": fen,
                    "student_san": sm.get("san"),
                    "student_uci": sm.get("uci"),
                    "severity": sm.get("severity"),
                    "cp_loss": sm.get("cp_loss"),
                    "phase": _phase(fen),
                }
            )
    return out


def balanced_sample(records: List[Dict[str, Any]], num: int, seed: int) -> List[Dict[str, Any]]:
    """Round-robin over (phase x severity) buckets for balance across both axes."""
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["severity"] in CORE_SEVERITIES:
            buckets[(r["phase"], r["severity"])].append(r)
    for b in buckets.values():
        rng.shuffle(b)

    order = [(ph, sev) for ph in PHASES for sev in CORE_SEVERITIES]
    picked: List[Dict[str, Any]] = []
    idx = 0
    while len(picked) < num and any(buckets[k] for k in order):
        k = order[idx % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop())
        idx += 1
        if idx > len(order) * 10000:
            break
    return picked


# --------------------------------------------------------------------------- #
# Per-position analysis
# --------------------------------------------------------------------------- #


def _sound_pool(fen: str) -> List[schema.SoundMove]:
    raw = stockfish_engine.sound_pool(
        fen,
        tolerance_cp=settings.SOUND_TOLERANCE_CP,
        multipv=settings.MULTIPV,
        movetime_ms=settings.DEFAULT_MOVETIME_MS,
    )
    return [
        {"san": m["san"], "uci": m["uci"], "cp": int(m["cp"]), "pv": m["pv"]}
        for m in raw
        if m.get("san") and m.get("uci")
    ]


def _classify_student(fen: str, board: chess.Board, student_uci: Optional[str]) -> schema.StudentMove:
    if student_uci:
        try:
            mv = chess.Move.from_uci(student_uci)
            if mv in board.legal_moves:
                cls = stockfish_engine.classify_mistake(
                    fen, mv.uci(), movetime_ms=settings.DEFAULT_MOVETIME_MS
                )
                return {
                    "san": board.san(mv),
                    "uci": mv.uci(),
                    "cp_loss": int(cls["cp_loss"]),
                    "severity": str(cls["severity"]),
                }
        except ValueError:
            pass
    return {"san": "(none provided)", "uci": "", "cp_loss": 0, "severity": "none"}


def _maia_top(fen: str, tier: str) -> Tuple[List[schema.MaiaMove], Optional[Dict[str, str]]]:
    try:
        res = maia_engine.human_moves(fen, tier, top_k=6)["moves"]
    except Exception as exc:  # noqa: BLE001
        print(f"    ! maia failed ({tier}): {exc}", file=sys.stderr)
        return [], None
    moves: List[schema.MaiaMove] = [
        {"san": m["san"], "uci": m["uci"], "policy": float(m["policy"])} for m in res
    ]
    top = {"san": moves[0]["san"], "uci": moves[0]["uci"]} if moves else None
    return moves, top


def analyze_position(
    rec: Dict[str, Any], coach: Coach, *, extra_beginner: bool
) -> Optional[Dict[str, Any]]:
    fen = rec["fen"]
    board = chess.Board(fen)
    if board.is_game_over():
        return None
    pool = _sound_pool(fen)
    if not pool:
        return None
    best = pool[0]
    student = _classify_student(fen, board, rec.get("student_uci"))

    tiers_out: Dict[str, Any] = {}
    maia_by_tier: Dict[str, Any] = {}
    for tier in TIER_ORDER:
        maia_moves, maia_top = _maia_top(fen, tier)
        maia_by_tier[tier] = {"moves": maia_moves, "top": maia_top}
        ti: schema.TeacherInput = {
            "tier": tier,
            "fen": board.fen(),
            "move_history_san": None,
            "student_move": student,
            "sound_pool": pool,
            "maia_human_moves": maia_moves,
        }
        facts = render_pool_facts(board.fen(), list(pool))
        user_prompt = f"{facts}\n\n{schema.render_user_prompt(ti)}"
        reply = coach.run(SYSTEM_PROMPT, user_prompt)
        rec_san, rec_uci, mode = extract_recommended_mode(
            reply, board, pool, student.get("uci") or ""
        )
        body, takeaway = _split_coaching(reply)
        tiers_out[tier] = {
            "rec_san": rec_san,
            "rec_uci": rec_uci,
            "mode": mode,
            "genuine": mode in ("cue", "prose"),
            "eq_pool0": rec_uci == best["uci"],
            "eq_maia_top": bool(maia_top and rec_uci == maia_top["uci"]),
            "coaching": body,
            "takeaway": takeaway,
            "raw": reply,
        }

    # Optional noise-floor: a 2nd beginner sample under the SAME prompt.
    beginner_resample = None
    if extra_beginner:
        maia_moves = maia_by_tier["beginner"]["moves"]
        ti = {
            "tier": "beginner",
            "fen": board.fen(),
            "move_history_san": None,
            "student_move": student,
            "sound_pool": pool,
            "maia_human_moves": maia_moves,
        }
        facts = render_pool_facts(board.fen(), list(pool))
        user_prompt = f"{facts}\n\n{schema.render_user_prompt(ti)}"
        reply = coach.run(SYSTEM_PROMPT, user_prompt)
        r_san, r_uci, r_mode = extract_recommended_mode(
            reply, board, pool, student.get("uci") or ""
        )
        beginner_resample = {"rec_san": r_san, "rec_uci": r_uci, "mode": r_mode}

    b_uci = tiers_out["beginner"]["rec_uci"]
    i_uci = tiers_out["intermediate"]["rec_uci"]
    a_uci = tiers_out["advanced"]["rec_uci"]
    distinct = len({b_uci, i_uci, a_uci})
    genuine_all = all(tiers_out[t]["genuine"] for t in TIER_ORDER)

    return {
        "id": rec.get("id"),
        "fen": fen,
        "phase": rec.get("phase"),
        "source_tier": rec.get("source_tier"),
        "student_move": student,
        "sound_pool": pool,
        "stockfish_best": {"san": best["san"], "uci": best["uci"], "cp": int(best["cp"])},
        "maia_by_tier": maia_by_tier,
        "beginner_move": {"san": tiers_out["beginner"]["rec_san"], "uci": b_uci,
                          "mode": tiers_out["beginner"]["mode"]},
        "intermediate_move": {"san": tiers_out["intermediate"]["rec_san"], "uci": i_uci,
                              "mode": tiers_out["intermediate"]["mode"]},
        "advanced_move": {"san": tiers_out["advanced"]["rec_san"], "uci": a_uci,
                          "mode": tiers_out["advanced"]["mode"]},
        "tiers": tiers_out,
        "beginner_resample": beginner_resample,
        "n_distinct_tier_moves": distinct,
        "tiers_all_agree": distinct == 1,
        "genuine_all_tiers": genuine_all,
        "diverges_from_sf_any_tier": any(not tiers_out[t]["eq_pool0"] for t in TIER_ORDER),
        "interesting": (distinct > 1) or any(not tiers_out[t]["eq_pool0"] for t in TIER_ORDER),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    done.add(json.loads(raw)["id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/mlx/chess-coach-v1")
    p.add_argument("--candidates", default="data/generated/candidates_v1.jsonl")
    p.add_argument("--positions", default="data/positions/positions_v1.jsonl")
    p.add_argument("--train", default="data/dataset/train.jsonl")
    p.add_argument("--valid", default="data/dataset/valid.jsonl")
    p.add_argument("--out", default="data/analysis/divergence.jsonl")
    p.add_argument("--num", type=int, default=120)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--temp", type=float, default=0.0, help="0 = greedy (default).")
    p.add_argument("--max-tokens", type=int, default=GEN_MAX_TOKENS)
    p.add_argument("--noise-floor", action="store_true",
                   help="Also draw a 2nd beginner sample per position (temp>0 only).")
    p.add_argument("--limit", type=int, default=0, help="Smoke cap on positions (0 = no cap).")
    p.add_argument("--resume", action="store_true", help="Skip ids already in --out.")
    args = p.parse_args(argv)

    def _abs(x: str) -> Path:
        pp = Path(x)
        return pp if pp.is_absolute() else _ROOT / pp

    out_path = _abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building held-out key set from train/valid ...", file=sys.stderr)
    heldin = build_heldin_keys(_abs(args.train), _abs(args.valid))
    print(f"      held-in board+turn keys: {len(heldin)}", file=sys.stderr)

    print("[2/4] Loading + filtering held-out candidates ...", file=sys.stderr)
    records = load_heldout_candidates(_abs(args.candidates), heldin)
    sev_counts = defaultdict(int)
    for r in records:
        sev_counts[r["severity"]] += 1
    print(f"      held-out unique candidate positions: {len(records)} "
          f"(by severity: {dict(sev_counts)})", file=sys.stderr)

    sample = balanced_sample(records, args.num, args.seed)
    if args.limit:
        sample = sample[: args.limit]
    dist = defaultdict(int)
    for r in sample:
        dist[(r["phase"], r["severity"])] += 1
    print(f"      sampled {len(sample)} positions; (phase,severity) dist:", file=sys.stderr)
    for k in sorted(dist):
        print(f"        {k}: {dist[k]}", file=sys.stderr)

    done = _load_done_ids(out_path) if args.resume else set()
    if args.resume and done:
        print(f"      resuming: {len(done)} already done", file=sys.stderr)

    extra_beginner = bool(args.noise_floor and args.temp > 0.0)

    print(f"[3/4] Loading coach {args.model!r} (temp={args.temp}) ...", file=sys.stderr)
    coach = Coach(_abs(args.model).as_posix(), temp=args.temp, max_tokens=args.max_tokens)

    print(f"[4/4] Analyzing {len(sample)} positions x3 tiers ...", file=sys.stderr)
    mode_open = "a" if (args.resume and done) else "w"
    t0 = time.time()
    n_done = 0
    with out_path.open(mode_open, encoding="utf-8") as out_fh:
        for i, rec in enumerate(sample, 1):
            if rec.get("id") in done:
                continue
            ts = time.time()
            try:
                row = analyze_position(rec, coach, extra_beginner=extra_beginner)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! [{i}/{len(sample)}] {rec.get('id')} FAILED: {exc}", file=sys.stderr)
                continue
            if row is None:
                print(f"  - [{i}/{len(sample)}] {rec.get('id')} skipped (no pool/terminal)",
                      file=sys.stderr)
                continue
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()
            n_done += 1
            bm = row["beginner_move"]["san"]
            im = row["intermediate_move"]["san"]
            am = row["advanced_move"]["san"]
            flag = "DIFF" if row["n_distinct_tier_moves"] > 1 else "same"
            sf = "≠SF" if row["diverges_from_sf_any_tier"] else "=SF"
            print(
                f"  + [{i}/{len(sample)}] {rec.get('id')} [{rec['phase'][:3]}/{rec['severity'][:4]}] "
                f"B:{bm} I:{im} A:{am}  {flag} {sf}  ({time.time()-ts:.1f}s)",
                file=sys.stderr,
            )
    print(f"DONE — wrote {n_done} rows to {out_path} in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/max(1,n_done):.1f}s/pos)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
