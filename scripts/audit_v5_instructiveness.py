#!/usr/bin/env python3
"""Read-only audit of chess-coaching SFT datasets against the 6-element
instructiveness rubric + tier coherence. No engine, no network, no GPU.

Rubric elements (per the v5 curation spec):
  E1  move + concrete purpose
  E2  a transferable NAMED principle (develop-before-attack, open file / 7th,
      doubled rooks, passed pawn, prophylaxis, king safety, trade-when-ahead ...)
  E3  the board-specific reason the principle applies here (square/piece refs)
  E4  how to find it next time (a thinking checklist) -> "How to find it:" clause
  E5  tier-calibrated depth/vocabulary (measured: beginner vocab leakage + ply
      run + group-level tier differentiation)
  E6  grounded + concise + no engine-speak (engine-speak regex, length, ply run)

Usage:
  python scripts/audit_v5_instructiveness.py data/dataset/train_v4.jsonl [more.jsonl ...]
  python scripts/audit_v5_instructiveness.py            # audits v2/v3/v4 train+valid
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset"

# --------------------------------------------------------------------------- #
# Parsing the rendered user/assistant text (handles v2/v3 render_user_prompt
# and v4 build_grounded_user -- both carry the same board / tier / pool lines).
# --------------------------------------------------------------------------- #

_TIER_RE = re.compile(r"Student rating tier:\s*([a-zA-Z]+)")
_BOARD_RE = re.compile(r"Board:\n(.*?)\n(White|Black) to move\.", re.DOTALL)
_STUDENT_RE = re.compile(r"The student played (\S+) \(severity:\s*([a-z]+)")
_CPLOSS_RE = re.compile(r"loses about (-?\d+) centipawns")
_POOL_RE = re.compile(r"^\s*-\s*(\S+)\s*\(eval\s*(-?\d+)cp\)", re.MULTILINE)
_MAIA_HDR = "Human-likelihood at this tier (Maia):"
_MAIA_RE = re.compile(r"^\s*-\s*(\S+):\s*(\d+)%", re.MULTILINE)
_REC_RE = re.compile(r"^I'?d play ([^.]+?)\.")


def parse_user(user: str) -> Dict[str, Any]:
    tier = (_TIER_RE.search(user) or [None, None])[1]
    bm = _BOARD_RE.search(user)
    board = bm.group(1).strip() if bm else None
    side = bm.group(2) if bm else None
    sm = _STUDENT_RE.search(user)
    student = sm.group(1) if sm else None
    severity = sm.group(2) if sm else None
    cpl = _CPLOSS_RE.search(user)
    cp_loss = int(cpl.group(1)) if cpl else None
    pool = [(m.group(1), int(m.group(2))) for m in _POOL_RE.finditer(user)]
    # Maia list lives only after the Maia header.
    maia: List[Tuple[str, int]] = []
    if _MAIA_HDR in user:
        tail = user.split(_MAIA_HDR, 1)[1]
        # stop at the blank-line/instruction that follows the Maia block
        tail = tail.split("\nRecommend", 1)[0]
        maia = [(m.group(1), int(m.group(2))) for m in _MAIA_RE.finditer(tail)]
    return {
        "tier": tier, "board": board, "side": side, "student": student,
        "severity": severity, "cp_loss": cp_loss, "pool": pool, "maia": maia,
    }


def parse_assistant(text: str) -> Dict[str, Any]:
    rm = _REC_RE.match(text.strip())
    rec = rm.group(1).strip() if rm else None
    # split method + takeaway
    method = None
    if "How to find it:" in text:
        method = text.split("How to find it:", 1)[1]
        method = method.split("Takeaway:", 1)[0].strip()
    takeaway = None
    if "Takeaway:" in text:
        takeaway = text.split("Takeaway:", 1)[1].strip()
    body_before_takeaway = text.split("Takeaway:", 1)[0]
    return {"rec": rec, "method": method, "takeaway": takeaway,
            "full": text, "body": body_before_takeaway}


# --------------------------------------------------------------------------- #
# Element detectors
# --------------------------------------------------------------------------- #

# E2: transferable NAMED principle families (strategic/positional + endgame).
PRINCIPLE_FAMILIES: Dict[str, re.Pattern] = {
    "development": re.compile(r"\b(develop(?:ing|ment|ed)?|undeveloped|bring .{0,20}\b(?:into the game|out)|finish(?:ing)? development|get .{0,15} developed)\b", re.I),
    "king_safety": re.compile(r"\b(king safety|castl\w+|king (?:is )?(?:still )?in the cent|safeguard\w* .{0,10}king|king stuck|tuck\w* .{0,10}king|get your king|expos\w* king)\b", re.I),
    "center": re.compile(r"\b(cent(?:er|re|ral)|control the cent|central (?:control|square|pawn))\b", re.I),
    "open_file_rook": re.compile(r"\b((?:half[- ]?)?open file|rook .{0,12}(?:file|column)|file for (?:your|the) rook|contest the .{0,6}file)\b", re.I),
    "seventh_rank": re.compile(r"\b(seventh rank|7th rank|on the (?:7th|seventh)|rook .{0,10}(?:7th|seventh))\b", re.I),
    "doubled_rooks": re.compile(r"\b(doubl\w* .{0,6}rooks?|stack\w* .{0,6}rooks?|connect\w* .{0,6}rooks?|rooks? .{0,10}doubl)\b", re.I),
    "passed_pawn": re.compile(r"\b(passed pawn|passer|push .{0,12}(?:passed )?pawn|promot\w+|queen(?:ing)? .{0,8}pawn|outside passed)\b", re.I),
    "prophylaxis": re.compile(r"\b(prophyla\w+|prevent\w*|stop\w* .{0,18}(?:plan|idea|break|threat)|take away|deny\w*|restrict\w*|shut down .{0,12}(?:plan|idea))\b", re.I),
    "outpost": re.compile(r"\b(outpost|strong (?:knight )?square|permanent square|anchor)\b", re.I),
    "bishop_pair": re.compile(r"\b(bishop pair|two bishops|pair of bishops)\b", re.I),
    "good_bad_bishop": re.compile(r"\b(bad bishop|good bishop|active bishop|improve .{0,12}bishop|reroute .{0,12}bishop)\b", re.I),
    "trade_logic": re.compile(r"\b(trad\w+|exchang\w+|swap\w*|simplif\w+|off the board)\b", re.I),
    "activity_worst_piece": re.compile(r"\b(activ\w+|improve your worst|worst piece|most passive|passive piece|reroute|maneuver|manoeuvre|reposition)\b", re.I),
    "space": re.compile(r"\b(space (?:advantage|edge)?|gain space|gaining space|cramp\w*|more space)\b", re.I),
    "initiative_tempo": re.compile(r"\b(initiative|with tempo|gain(?:ing)? a tempo|develop with tempo|tempo\b)\b", re.I),
    "weakness_target": re.compile(r"\b(weak(?:ness| ?square)?|isolated pawn|isolated queen|backward pawn|doubled pawns?|create a target|fix .{0,10}(?:pawn|weak))\b", re.I),
    "endgame": re.compile(r"\b(opposition|key square|active king|rook behind|zugzwang|triangulat\w+)\b", re.I),
}

# Tactic / calculation motifs (transferable, but not the strategic list #2 asks for).
TACTIC_FAMILIES: Dict[str, re.Pattern] = {
    "fork": re.compile(r"\bfork\w*\b", re.I),
    "pin": re.compile(r"\bpin\w*\b", re.I),
    "skewer": re.compile(r"\bskewer\w*\b", re.I),
    "discovered": re.compile(r"\bdiscover\w*\b", re.I),
    "double_attack": re.compile(r"\b(double attack|two threats|two targets|attack .{0,12}(?:two|both))\b", re.I),
    "overload_deflect": re.compile(r"\b(overload\w*|deflect\w*|remove the defender|overwork\w*)\b", re.I),
    "trapped": re.compile(r"\b(trap\w*|no squares|no safe square)\b", re.I),
    "hanging_counting": re.compile(r"\b(hanging|undefended|loose piece|count .{0,12}(?:attackers|defenders)|attackers? .{0,6}(?:and|vs) .{0,6}defenders?)\b", re.I),
}

# E1: concrete purpose verbs (the move DOES something specific).
PURPOSE_RE = re.compile(
    r"\b(develops?|attacks?|defends?|controls?|opens?|improves?|creates?|"
    r"protects?|targets?|activates?|prepares?|threatens?|guards?|covers?|"
    r"supports?|blocks?|stops?|frees?|connects?|centralizes?|gains?|wins?|"
    r"trades?|removes?|pressures?|hits?|eyes?)\b", re.I)

# E3: board coordinate references (specific squares).
SQUARE_RE = re.compile(r"\b([a-h][1-8])\b")
PIECE_ON_SQ_RE = re.compile(r"\b([KQRBN])[a-h]?[1-8]?x?[a-h][1-8]\b")

# E6: engine-speak (mirrors src/teacher ENGINE_SPEAK + filter.detect_engine_speak intent).
ENGINE_SPEAK_RE = re.compile(
    r"\b(centipawn|stockfish|engine|computer|eval(?:uation)?|winning by|"
    r"mate in \d|depth \d|the machine)\b|[+\-]\d+(?:\.\d+)?\s*(?:cp|pawns?)?\b|\bcp\b",
    re.I)

# E5: vocabulary that tier_guides.md forbids for BEGINNER (advanced-only concepts).
BEGINNER_FORBIDDEN_VOCAB = re.compile(
    r"\b(prophyla\w+|outpost|zugzwang|imbalance\w*|opposition|minority attack|"
    r"isolated queen|IQP|zwischenzug|overload\w*|in-between move|prophylactic|"
    r"triangulat\w+|fortress|corresponding square|bishop pair|good bishop|bad bishop|"
    r"backward pawn|fianchetto|luft|tempo|initiative)\b", re.I)

# SAN token for ply-run counting.
SAN_TOKEN = re.compile(r"\b(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b")


def longest_san_run(text: str) -> int:
    """Longest run of consecutive SAN-looking tokens (a proxy for line depth)."""
    best = cur = 0
    # Tokenize on whitespace; a SAN "run" is consecutive tokens that are all SAN.
    for tok in re.split(r"\s+", text):
        t = tok.strip(",.:;!?()")
        if t and SAN_TOKEN.fullmatch(t):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def principle_hits(text: str, families: Dict[str, re.Pattern]) -> List[str]:
    return [name for name, pat in families.items() if pat.search(text)]


def analyze_row(user: str, assistant: str) -> Dict[str, Any]:
    u = parse_user(user)
    a = parse_assistant(assistant)
    full = a["full"]
    body = a["body"]
    tier = u["tier"]

    strat = principle_hits(full, PRINCIPLE_FAMILIES)
    tac = principle_hits(full, TACTIC_FAMILIES)
    strat_in_takeaway = principle_hits(a["takeaway"] or "", PRINCIPLE_FAMILIES)

    squares = set(SQUARE_RE.findall(body))
    words = len(re.findall(r"\S+", full))

    rec = a["rec"]
    pool_sans = [p[0] for p in u["pool"]]
    engine_best = pool_sans[0] if pool_sans else None
    maia_sans = [m[0] for m in u["maia"]]
    rec_in_pool = rec in pool_sans if (rec and pool_sans) else None
    rec_is_engine_best = (rec == engine_best) if (rec and engine_best) else None
    maia_rank = (maia_sans.index(rec) if rec in maia_sans else None) if maia_sans else None

    forbidden_vocab = []
    if tier == "beginner":
        forbidden_vocab = sorted(set(m.group(0).lower() for m in BEGINNER_FORBIDDEN_VOCAB.finditer(full)))

    return {
        "tier": tier,
        "board": u["board"],
        "side": u["side"],
        "student": u["student"],
        "severity": u["severity"],
        "rec": rec,
        "engine_best": engine_best,
        "rec_in_pool": rec_in_pool,
        "rec_is_engine_best": rec_is_engine_best,
        "maia_rank": maia_rank,          # 0 = top human move
        "maia_len": len(maia_sans),
        "pool_size": len(pool_sans),
        # elements
        "E1_purpose": bool(rec and PURPOSE_RE.search(body)),
        "E2_strategic": bool(strat),
        "E2_families": strat,
        "E2_in_takeaway": bool(strat_in_takeaway),
        "E2_or_tactic": bool(strat or tac),
        "tactic_families": tac,
        "E3_square_refs": len(squares),
        "E3_ok": len(squares) >= 2,
        "E4_howtofind": a["method"] is not None and len(a["method"]) > 15,
        "E5_forbidden_vocab": forbidden_vocab,
        "E6_engine_speak": bool(ENGINE_SPEAK_RE.search(full)),
        "E6_words": words,
        "ply_run": longest_san_run(full),
        "has_takeaway": a["takeaway"] is not None and len(a["takeaway"] or "") > 3,
    }


# --------------------------------------------------------------------------- #
# Dataset-level aggregation
# --------------------------------------------------------------------------- #

def load_rows(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = d.get("messages") or []
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if user and asst:
                rows.append({"user": user, "assistant": asst})
    return rows


def pct(n: int, d: int) -> str:
    return f"{(100.0*n/d):5.1f}%" if d else "  n/a"


def audit_file(path: Path) -> Dict[str, Any]:
    raw = load_rows(path)
    # Dedup identical (user, assistant) pairs (v4 oversamples beginner-disc rows).
    seen = set()
    uniq = []
    for r in raw:
        k = (r["user"], r["assistant"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    analyses = [analyze_row(r["user"], r["assistant"]) for r in uniq]
    n = len(analyses)
    by_tier: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in analyses:
        by_tier[a["tier"] or "?"].append(a)

    def agg(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        m = len(items)
        return {
            "n": m,
            "E1": sum(a["E1_purpose"] for a in items),
            "E2_strat": sum(a["E2_strategic"] for a in items),
            "E2_take": sum(a["E2_in_takeaway"] for a in items),
            "E2_or_tac": sum(a["E2_or_tactic"] for a in items),
            "E3": sum(a["E3_ok"] for a in items),
            "E4": sum(a["E4_howtofind"] for a in items),
            "E6_jargon": sum(a["E6_engine_speak"] for a in items),
            "takeaway": sum(a["has_takeaway"] for a in items),
            "words_med": median([a["E6_words"] for a in items]) if items else 0,
            "words_mean": mean([a["E6_words"] for a in items]) if items else 0,
            "ply_over": sum(1 for a in items if a["ply_run"] > _PLY_CAP.get(a["tier"], 99)),
            "rec_engine_best": sum(1 for a in items if a["rec_is_engine_best"]),
            "rec_engine_best_known": sum(1 for a in items if a["rec_is_engine_best"] is not None),
            "maia_top": sum(1 for a in items if a["maia_rank"] == 0),
            "maia_known": sum(1 for a in items if a["maia_rank"] is not None),
            "forbidden": sum(1 for a in items if a["E5_forbidden_vocab"]),
            "all6": sum(1 for a in items if (a["E1_purpose"] and a["E2_strategic"] and a["E3_ok"] and a["E4_howtofind"] and not a["E6_engine_speak"] and a["has_takeaway"])),
        }

    overall = agg(analyses)
    tier_aggs = {t: agg(by_tier[t]) for t in ("beginner", "intermediate", "advanced") if by_tier.get(t)}

    # Tier coherence: group by board.
    board_tier_moves: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for a in analyses:
        if a["board"] and a["tier"] and a["rec"]:
            board_tier_moves[a["board"]][a["tier"]][a["rec"]] += 1

    multi = {b: tm for b, tm in board_tier_moves.items() if len(tm) >= 2}
    all3 = {b: tm for b, tm in board_tier_moves.items() if len(tm) == 3}

    coherence = Counter()
    collapse_examples: List[Dict[str, Any]] = []
    intra_inconsistent = 0  # same board+tier -> multiple distinct rec moves
    for b, tm in board_tier_moves.items():
        for t, c in tm.items():
            if len(c) > 1:
                intra_inconsistent += 1
                break

    for b, tm in all3.items():
        B = tm["beginner"].most_common(1)[0][0]
        I = tm["intermediate"].most_common(1)[0][0]
        A = tm["advanced"].most_common(1)[0][0]
        if B == I == A:
            coherence["all_same (B=I=A)"] += 1
            tag = "all_same"
        elif B == A and B != I:
            coherence["COLLAPSE B=A != I"] += 1
            tag = "collapse_BA"
        elif B == I and I != A:
            coherence["B=I != A (human vs sharp)"] += 1
            tag = "BI"
        elif I == A and I != B:
            coherence["I=A != B (beginner differs)"] += 1
            tag = "IA"
        else:
            coherence["full gradient (B!=I!=A)"] += 1
            tag = "full"
        if tag in ("collapse_BA", "all_same") and len(collapse_examples) < 12:
            collapse_examples.append({"board": b, "B": B, "I": I, "A": A, "tag": tag})

    return {
        "path": str(path.relative_to(ROOT)),
        "raw": len(raw),
        "unique": n,
        "dup_dropped": len(raw) - n,
        "overall": overall,
        "tiers": tier_aggs,
        "boards_total": len(board_tier_moves),
        "boards_multi_tier": len(multi),
        "boards_all3": len(all3),
        "coherence": dict(coherence),
        "intra_inconsistent_boardtier": intra_inconsistent,
        "collapse_examples": collapse_examples,
    }


_PLY_CAP = {"beginner": 2, "intermediate": 4, "advanced": 6}


def print_report(rep: Dict[str, Any]) -> None:
    o = rep["overall"]
    n = rep["unique"]
    print("\n" + "=" * 78)
    print(f"FILE: {rep['path']}")
    print(f"  rows: {rep['raw']} raw, {rep['unique']} unique ({rep['dup_dropped']} dup dropped)")
    print("-" * 78)
    print("  ELEMENT COVERAGE (unique rows):")
    print(f"    E1 move+purpose       {pct(o['E1'], n)}  ({o['E1']}/{n})")
    print(f"    E2 named principle    {pct(o['E2_strat'], n)}  strategic; {pct(o['E2_or_tac'], n)} incl. tactic")
    print(f"       ...in takeaway     {pct(o['E2_take'], n)}")
    print(f"    E3 board-specific     {pct(o['E3'], n)}  (>=2 square refs)")
    print(f"    E4 how-to-find-it     {pct(o['E4'], n)}  ({o['E4']}/{n})")
    print(f"    E6 no engine-speak    {pct(n - o['E6_jargon'], n)}  ({o['E6_jargon']} leaks)")
    print(f"    takeaway present      {pct(o['takeaway'], n)}")
    print(f"    ALL 6 together        {pct(o['all6'], n)}  ({o['all6']}/{n})")
    print(f"    words: median={o['words_med']:.0f} mean={o['words_mean']:.0f}")
    print("-" * 78)
    print("  PER-TIER:")
    hdr = f"    {'tier':<13}{'n':>6}{'E1':>7}{'E2str':>7}{'E2tk':>7}{'E3':>7}{'E4':>7}{'jargon':>8}{'forbid':>7}{'engBest':>8}{'maiaTop':>8}{'plyOver':>8}{'words':>7}"
    print(hdr)
    for t in ("beginner", "intermediate", "advanced"):
        a = rep["tiers"].get(t)
        if not a:
            continue
        m = a["n"]
        eb = pct(a["rec_engine_best"], a["rec_engine_best_known"])
        mt = pct(a["maia_top"], a["maia_known"])
        print(f"    {t:<13}{m:>6}{pct(a['E1'],m):>7}{pct(a['E2_strat'],m):>7}{pct(a['E2_take'],m):>7}"
              f"{pct(a['E3'],m):>7}{pct(a['E4'],m):>7}{a['E6_jargon']:>8}{a['forbidden']:>7}"
              f"{eb:>8}{mt:>8}{a['ply_over']:>8}{a['words_med']:>7.0f}")
    print("-" * 78)
    print("  TIER COHERENCE (positions taught at multiple tiers):")
    print(f"    boards total={rep['boards_total']}  multi-tier={rep['boards_multi_tier']}  all-3-tier={rep['boards_all3']}")
    print(f"    same board+tier w/ inconsistent rec move: {rep['intra_inconsistent_boardtier']}")
    for k, v in sorted(rep["coherence"].items(), key=lambda kv: -kv[1]):
        base = rep["boards_all3"] or 1
        print(f"      {k:<28} {v:>5}  ({100.0*v/base:4.1f}% of all-3)")
    if rep["collapse_examples"]:
        print("    sample collapse / all-same boards (B / I / A recommended move):")
        for ex in rep["collapse_examples"][:8]:
            print(f"      [{ex['tag']}] B={ex['B']:<6} I={ex['I']:<6} A={ex['A']:<6}")


def main(argv: List[str]) -> int:
    if argv:
        files = [Path(x) for x in argv]
    else:
        files = [
            DATASET / "train_v2.jsonl", DATASET / "valid_v2.jsonl",
            DATASET / "train_v3.jsonl", DATASET / "valid_v3.jsonl",
            DATASET / "train_v4.jsonl", DATASET / "valid_v4.jsonl",
        ]
    reports = []
    for f in files:
        if not f.exists():
            print(f"(skip missing {f})")
            continue
        rep = audit_file(f)
        reports.append(rep)
        print_report(rep)
    out = ROOT / "data" / "analysis" / "v5_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
