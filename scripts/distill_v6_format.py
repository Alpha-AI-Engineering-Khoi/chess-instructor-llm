#!/usr/bin/env python3
"""v6 ENGINE-DISTILLATION reformat + tier-policy eval extractor (no grounding).

This is the single source of truth for the AMBITIOUS v6 distillation experiment,
shared verbatim by the local sanity check and the Modal training job
(``scripts/train_distill_v6.py``). It depends ONLY on ``python-chess`` (no repo
imports) so it runs unchanged inside the Modal training container.

The thesis-upgrade under test
-----------------------------
v4/v6 SFT hands the model a GROUNDED prompt: the Stockfish sound-pool (with evals)
and the per-tier Maia human-likelihoods. The tier-appropriate move is then a
grounded *execution* — the model just reads the answer off the engine block. This
distillation asks the harder question: can the tier-selection rule live in the
WEIGHTS? So we STRIP the engine/Maia block entirely and keep only what a player
actually sees at the board.

Reformat
--------
INPUT  (prompt): FEN + ASCII board + side to move + the student's move + the
  target TIER label. The per-tier Maia block AND the Stockfish sound-pool/eval
  grounding are REMOVED.
TARGET (completion): the tier-appropriate move
  (``provenance.canonical_uci``/``canonical_san`` — this equals ``engine_best``
  for the advanced tier) plus a short principle. Short on purpose: the trained
  behavior is the MOVE, not prose.

Eval metric
-----------
``tier-policy exact match``: extract the recommended move from the model's reply
with the SAME strict, any-legal extractor the shipped reports use
(``pick_recommendation`` / ``extract_recommended_move``, vendored verbatim from
``src.teacher.coach_gate`` + ``src.eval.evaluate``), and compare its UCI to the
row's ``canonical_uci``. Reported per tier and as the tier mean — identical
definition to ``scripts/reproduce_v4.py``.

CLI::

    # reformat the LOCAL v6 jsonl -> data/dataset/{train,valid}_distill_v6.jsonl
    python scripts/distill_v6_format.py reformat

    # or pull the raw v6 jsonl from the private HF dataset first
    python scripts/distill_v6_format.py reformat --from-hf
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import chess

# --------------------------------------------------------------------------- #
# Tiers (mirror config.settings.TIERS so this module stays repo-import-free)
# --------------------------------------------------------------------------- #
TIERS: Dict[str, Dict[str, int]] = {
    "beginner":     {"low": 1000, "high": 1200, "ply_cap": 2},
    "intermediate": {"low": 1300, "high": 1600, "ply_cap": 4},
    "advanced":     {"low": 1700, "high": 2000, "ply_cap": 6},
}
TIER_ORDER: Tuple[str, ...] = ("beginner", "intermediate", "advanced")

SEED = 3407

# --------------------------------------------------------------------------- #
# The no-grounding prompt (position facts a player sees — NO engine, NO Maia)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT: str = (
    "You are a chess coach doing move review for a student at a stated rating tier.\n"
    "You are given the position, the student's rating tier, and the move the student "
    "played. You are NOT given any engine analysis, candidate list, or move "
    "statistics — rely on your own chess judgment.\n\n"
    "Recommend exactly ONE move that is genuinely sound AND the most INSTRUCTIVE for a "
    "student at this tier. This is often NOT the sharpest move: prefer a sound move "
    "whose idea a player at this level can understand, would plausibly find, and can "
    "reuse. A stronger tier should get a sharper, more precise move than a beginner "
    "would on the same position. End with one short, transferable takeaway.\n\n"
    "Never mention engines, evaluations, centipawns, or numbers."
)

#: Closing format instruction — matches the SHORT distillation target so the
#: deterministic move-extractor is fair to base and tuned alike.
FORMAT_INSTRUCTION: str = (
    "Format your reply as plain prose in exactly this shape:\n"
    'Start with "I\'d play <MOVE>." where <MOVE> is one move in standard algebraic '
    "notation (e.g. Nf3, exd5, O-O). Then give one short sentence of coaching in "
    'plain, encouraging language. End with one line "Takeaway: <one transferable '
    'sentence>." Do not output JSON, bullet lists, headers, or long move-number '
    "sequences."
)

_PRINCIPLE = {
    "endgame": "activate your king and keep your pieces coordinated",
    "opening": "develop toward the center and get your king safe",
    "middlegame": "improve your worst-placed piece and watch your opponent's ideas",
}


def ascii_board(fen: str) -> str:
    return str(chess.Board(fen))


def build_nogrounding_user(fen: str, tier: str, student_san: Optional[str]) -> str:
    """User message for the DISTILLATION (no-grounding) condition.

    Only minimal position facts a player sees: the rating tier, the raw FEN, the
    ASCII board, side to move, and the student's move. No sound pool, no evals, no
    Maia. This is what forces the tier rule into the weights.
    """
    board = chess.Board(fen)
    t = TIERS[tier]
    lines = [
        f"Student rating tier: {tier} ({t['low']}-{t['high']}).",
        f"FEN: {fen}",
        "Board:\n" + ascii_board(fen),
        f"{'White' if board.turn else 'Black'} to move.",
    ]
    if student_san:
        lines.append(f"The student played {student_san}.")
    lines += [
        "",
        (
            f"Recommend exactly ONE move that is genuinely sound AND the most "
            f"instructive for a {tier} player"
            + (", and coach them on why it is a better choice." if student_san else ".")
            + f" Keep any concrete line within {t['ply_cap']} plies."
        ),
        "",
        FORMAT_INSTRUCTION,
    ]
    return "\n".join(lines)


def build_distill_target(
    canonical_san: str, phase: str, student_san: Optional[str], review_action: Optional[str]
) -> str:
    """The SHORT distillation target: the tier move + a one-line principle."""
    principle = _PRINCIPLE.get(phase, "make a purposeful, safe improving move")
    if review_action == "endorse" or (student_san and student_san == canonical_san):
        lead = f"{canonical_san} is a sound, level-appropriate choice here."
    elif student_san and student_san != canonical_san:
        lead = f"{canonical_san} is a clearer and sounder try than {student_san}."
    else:
        lead = f"{canonical_san} keeps your position solid and is easy to follow at your level."
    return f"I'd play {canonical_san}. {lead} Takeaway: When you are unsure, {principle}."


# --------------------------------------------------------------------------- #
# Reformat a v6 row -> a distillation chat row (+ eval/sampling meta)
# --------------------------------------------------------------------------- #
def reformat_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prov = row.get("provenance") or {}
    fen = prov.get("fen")
    tier = prov.get("tier")
    canonical_san = prov.get("canonical_san")
    canonical_uci = prov.get("canonical_uci")
    if not (fen and tier in TIERS and canonical_san and canonical_uci):
        return None
    student = prov.get("student") or {}
    student_san = student.get("san")
    student_uci = student.get("uci") or ""
    phase = prov.get("phase") or "middlegame"
    review_action = prov.get("review_action")
    weight = float(prov.get("weight") or 1.0)

    user = build_nogrounding_user(fen, tier, student_san)
    target = build_distill_target(canonical_san, phase, student_san, review_action)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "meta": {
            "tier": tier,
            "fen": fen,
            "canonical_uci": canonical_uci,
            "student_uci": student_uci,
            "weight": weight,
            "phase": phase,
        },
    }


def reformat_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = reformat_row(r)
        if rr is not None:
            out.append(rr)
    return out


def weighted_expand(rows: Sequence[Dict[str, Any]], seed: int = SEED) -> List[Dict[str, Any]]:
    """Realize the v6 sampling ``weight`` via DETERMINISTIC fractional oversampling.

    copies = floor(w) + Bernoulli(frac(w)) so E[copies] == weight exactly. This
    up-weights high-confidence discriminating rows (w=1.5) and down-weights
    all-same/benign rows (w=0.25) with essentially no data blow-up (total copies
    ~= sum of weights ~= the unique-row count).
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for r in rows:
        w = float(r.get("meta", {}).get("weight", 1.0))
        n = int(w)
        if rng.random() < (w - n):
            n += 1
        out.extend([r] * max(0, n))
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Move extraction — VENDORED VERBATIM from src.teacher.coach_gate +
# src.eval.evaluate so the tier-policy metric is byte-identical to the shipped
# reports (reproduce_v4.py). Depends only on python-chess + re.
# --------------------------------------------------------------------------- #
_SAN_RE = re.compile(r"(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)")
_ENDORSE_CUE_RE = re.compile(
    r"(?:i['\u2019]?d\s+play|i\s+would\s+play|i['\u2019]?ll\s+play|i\s+play|"
    r"i\s+recommend|recommend(?:ed)?(?:\s+move)?(?:\s+is)?|"
    r"(?:the\s+)?move\s*(?:is|:)|/move\s*:|best\s+move\s+is|go\s+with|choose|"
    r"leads?\s+you\s+to|points?\s+you\s+to|improvement\s+is|better\s+is)"
    r"\s*[:\-]?\s*",
    re.IGNORECASE,
)
_AVOID_CUE_RE = re.compile(
    r"rather than|instead of|such as|\bavoid\w*|\blike\b|\bnot\b|\bnever\b|"
    r"n['\u2019]?t\b|don['\u2019]?t|do not|rush\w*\s+into|forcing[- ]?looking",
    re.IGNORECASE,
)
_HYPO_CUE_RE = re.compile(
    r"\bif\b|\bafter\b|\bthen\b|for\s+example|follow[- ]?up|followed\s+by|"
    r"you\s+have|you['\u2019]?ll\s+have|\breplies\b|\brespond|\bnext\b|continu|"
    r"(?:white|black)\s+(?:plays|has|goes|replies)",
    re.IGNORECASE,
)
_COORD_BEFORE_RE = re.compile(r"\b(?:on|to)\b\W*$", re.IGNORECASE)
_COORD_AFTER_RE = re.compile(r"^[- ]?(?:pawn|square|file|rank)s?\b", re.IGNORECASE)
_CONCESSION_AFTER_RE = re.compile(
    r"^\W{0,3}(?:was|is|were|are)\s+(?:already\s+|also\s+)?"
    r"(?:playable|possible|fine|active|ok|okay|tempting|reasonable)",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?:;](?=\s)|\n")
_MOVE_NUMBER = re.compile(r"^\d+\.+(.*)$")


def _clause_before(text: str, start: int, span: int = 90) -> str:
    pre = text[max(0, start - span): start]
    last = -1
    for m in _CLAUSE_BOUNDARY_RE.finditer(pre):
        last = m.end()
    return pre[last:] if last != -1 else pre


def _san_candidates(board: chess.Board, text: str) -> List[Tuple[int, int, str, str]]:
    out: List[Tuple[int, int, str, str]] = []
    for m in _SAN_RE.finditer(text):
        try:
            mv = board.parse_san(m.group(1))
        except ValueError:
            continue
        out.append((m.start(), m.end(), board.san(mv), mv.uci()))
    return out


def _is_avoid_framed(text: str, start: int, end: int) -> bool:
    pre = _clause_before(text, start)
    if _AVOID_CUE_RE.search(pre) or _HYPO_CUE_RE.search(pre):
        return True
    tok = text[start:end]
    if tok[:1] in "abcdefgh" and "x" not in tok:
        if _COORD_BEFORE_RE.search(text[max(0, start - 6): start]):
            return True
        if _COORD_AFTER_RE.search(text[end: end + 8]):
            return True
    if _CONCESSION_AFTER_RE.search(text[end: end + 26]):
        return True
    return False


def _endorsed_indices(text: str, cands: Sequence[Tuple[int, int, str, str]]) -> set:
    endorsed: set = set()
    for cue in _ENDORSE_CUE_RE.finditer(text):
        lo, hi = cue.end(), cue.end() + 16
        for i, (s, _e, _san, _uci) in enumerate(cands):
            if lo <= s <= hi:
                between = text[cue.end(): s]
                if not (_AVOID_CUE_RE.search(between) or _HYPO_CUE_RE.search(between)):
                    endorsed.add(i)
                break
    for i, (s, e, _san, _uci) in enumerate(cands):
        if "again" in text[e: e + 10].lower() and "play" in text[max(0, s - 8): s].lower():
            endorsed.add(i)
    return endorsed


def pick_recommendation(
    text: str, board: chess.Board, student_uci: str, accept: Callable[[str], bool]
) -> Optional[Tuple[str, str]]:
    cands = [t for t in _san_candidates(board, text) if accept(t[3])]
    if not cands:
        return None
    avoid = [_is_avoid_framed(text, s, e) for (s, e, _san, _uci) in cands]
    endorsed = _endorsed_indices(text, cands)
    for i, (_s, _e, san, uci) in enumerate(cands):
        if i in endorsed:
            return san, uci
    for i, (_s, _e, san, uci) in enumerate(cands):
        if uci != student_uci and not avoid[i]:
            return san, uci
    for i, (_s, _e, san, uci) in enumerate(cands):
        if uci == student_uci and not avoid[i]:
            return san, uci
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start: i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def _legal_uci(board: chess.Board, san_token: str) -> Optional[str]:
    try:
        move = board.parse_san(san_token)
    except (ValueError, AssertionError):
        return None
    return move.uci()


def extract_recommended_move(
    text: str, fen: str, student_uci: str
) -> Tuple[Optional[str], Optional[str]]:
    """(san, uci) of the coach's recommended move — any legal move accepted."""
    board = chess.Board(fen)
    obj = _extract_json_object(text)
    if obj:
        uci = obj.get("recommended_move_uci")
        san = obj.get("recommended_move_san")
        if isinstance(uci, str):
            try:
                mv = chess.Move.from_uci(uci.strip())
                if mv in board.legal_moves:
                    return board.san(mv), mv.uci()
            except ValueError:
                pass
        if isinstance(san, str):
            got = _legal_uci(board, san.strip())
            if got is not None:
                return board.san(chess.Move.from_uci(got)), got
    picked = pick_recommendation(text, board, student_uci, accept=lambda _u: True)
    return picked if picked is not None else (None, None)


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


# --------------------------------------------------------------------------- #
# Tier-policy exact-match scoring
# --------------------------------------------------------------------------- #
def score_tier_policy(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """records: [{tier, canonical_uci, pred_uci}] -> per-tier + mean exact match."""
    by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIER_ORDER}
    parse_ok = [0, 0]
    for r in records:
        tier = r["tier"]
        if tier not in by_tier:
            continue
        by_tier[tier][1] += 1
        parse_ok[1] += 1
        pred = r.get("pred_uci")
        if pred:
            parse_ok[0] += 1
        if pred and pred == r.get("canonical_uci"):
            by_tier[tier][0] += 1
    per_tier = {t: (by_tier[t][0] / by_tier[t][1]) for t in TIER_ORDER if by_tier[t][1]}
    mean = sum(per_tier.values()) / len(per_tier) if per_tier else 0.0
    return {
        "tier_policy_match": mean,
        "per_tier": per_tier,
        "per_tier_counts": {t: by_tier[t] for t in TIER_ORDER if by_tier[t][1]},
        "n": parse_ok[1],
        "parse_rate": (parse_ok[0] / parse_ok[1]) if parse_ok[1] else 0.0,
    }


def score_generations(gen_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """gen_rows: [{tier, fen, canonical_uci, student_uci, output}] -> scored."""
    recs: List[Dict[str, Any]] = []
    for g in gen_rows:
        san, uci = extract_recommended_move(
            strip_think(g.get("output", "")), g["fen"], g.get("student_uci") or ""
        )
        recs.append({"tier": g["tier"], "canonical_uci": g["canonical_uci"], "pred_uci": uci})
    return score_tier_policy(recs)


# --------------------------------------------------------------------------- #
# IO helpers + CLI
# --------------------------------------------------------------------------- #
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _download_v6_from_hf(dest: Path, repo: str = "khoilamalphaai/chess-coach-v6") -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    train = hf_hub_download(repo, "train_v6.jsonl", repo_type="dataset", local_dir=str(dest))
    valid = hf_hub_download(repo, "valid_v6.jsonl", repo_type="dataset", local_dir=str(dest))
    return Path(train), Path(valid)


def _stats(rows: Sequence[Dict[str, Any]], label: str) -> None:
    from collections import Counter

    tiers = Counter(r["meta"]["tier"] for r in rows)
    print(f"[{label}] rows={len(rows)} by_tier={dict(tiers)}")


def cmd_reformat(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    ds = root / "data" / "dataset"
    if args.from_hf:
        train_in, valid_in = _download_v6_from_hf(ds / "_v6dl")
    else:
        train_in, valid_in = ds / "train_v6.jsonl", ds / "valid_v6.jsonl"
    print(f"reading train={train_in}\n        valid={valid_in}")
    train = reformat_rows(iter_jsonl(train_in))
    valid = reformat_rows(iter_jsonl(valid_in))
    _stats(train, "train")
    _stats(valid, "valid")
    if args.expand:
        exp = weighted_expand(train)
        _stats(exp, "train(weighted-expanded)")

    train_out = ds / "train_distill_v6.jsonl"
    valid_out = ds / "valid_distill_v6.jsonl"
    write_jsonl(train, train_out)
    write_jsonl(valid, valid_out)
    print(f"wrote {train_out}")
    print(f"wrote {valid_out}")

    print("\n=== sample distill row (valid[0]) ===")
    print("--- SYSTEM ---\n" + valid[0]["messages"][0]["content"])
    print("\n--- USER ---\n" + valid[0]["messages"][1]["content"])
    print("\n--- ASSISTANT (target) ---\n" + valid[0]["messages"][2]["content"])
    print("\n--- META ---\n" + json.dumps(valid[0]["meta"]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("reformat", help="Reformat v6 jsonl into the no-grounding distill format.")
    pr.add_argument("--from-hf", action="store_true", help="download raw v6 jsonl from HF first")
    pr.add_argument("--expand", action="store_true", help="also report the weighted-expanded train size")
    pr.set_defaults(func=cmd_reformat)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
