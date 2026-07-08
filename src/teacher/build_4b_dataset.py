#!/usr/bin/env python3
"""Build the **4B iteration-1** chess-coaching SFT set (v5 curation, data-first).

This is the DATA lever for the small (Qwen3-4B-Instruct) autonomous training loop
(see ``RALPH_TASK.md``). It re-derives training rows from the *already
Stockfish-verified* v3 teacher labels (``data/generated/candidates_v3.jsonl``)
and applies the **FREE** v5 curation fixes from
``data/analysis/V5_AUDIT_AND_PLAN.md`` — no teacher re-spend:

1. **Lead / restatement cleaner** — strips ``THE MOVE:``, ``The move is X``,
   ``This is the move``, ``Play X.``/``Consider X.`` restatements AND dangling
   leading connectors (``— and/but/so/then/in fact``). Targets the ~15.6% leading
   artifacts -> <1% (mirrors ``scripts/audit_v5_artifacts.py`` patterns).
2. **Beginner vocab scrub** — ``tempo``/``with tempo``/``initiative`` -> plain
   beginner wording ("gain time / for free" / "the attack"). Targets beginner
   forbidden-vocab leakage -> 0.
3. **``select_tier_move`` collapse fix (B==A != I)** — when the beginner and
   advanced canonical picks are the SAME move but intermediate differs, the
   intermediate row is the pathological blend artifact: DROP it (FREE form of
   "force intermediate to match"; regenerating its coaching is an iter-2 teacher
   top-up). Removes the B=A != I collapses.
4. **Deterministic takeaway-principle GATE** — every kept row's takeaway must NAME
   a transferable principle from the controlled vocabulary (drawn from
   ``data/analysis/principle_library_v5.md`` and aligned to the audit's
   ``PRINCIPLE_FAMILIES`` detector). Rows whose takeaway names none get a
   correctness-preserving, tier-appropriate principle clause (taken from the
   row's OWN body/concepts so it stays on-topic; beginner-safe wording only).
   Lifts beginner principle-in-takeaway 42% -> >=85% with no teacher calls and no
   new board claims (principles are general advice, never a board fact).

Mix (the "distinct-moves-per-level" gradient — the moat):
* **DROP** the B=A != I intermediate collapses.
* **DOWN-WEIGHT** non-differentiating all-same (B=I=A) boards (deterministic 50%).
* **UP-WEIGHT** genuinely contrastive full-gradient (B!=I!=A) triads (x2) and
  beginner-DISCRIMINATING rows (pick != engine best, x2), so training pulls the
  model toward *different sound moves per tier* where the position calls for it.

All gates from ``build_v4_dataset`` are preserved (soundness in the Stockfish
pool, legality, no engine-speak, ply cap, format lead + takeaway, NARROW
faithfulness). Output is versioned ``train_4b_iter1.jsonl`` / ``valid_4b_iter1``.

CLI
---
    python -m src.teacher.build_4b_dataset analyze
    python -m src.teacher.build_4b_dataset build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from src.eval.benchmark.prompts import build_grounded_user, load_system_prompt  # noqa: E402
from src.engine.faithfulness import verify_text  # noqa: E402
from src.filter.filter import detect_engine_speak, longest_san_run, move_is_legal  # noqa: E402

log = logging.getLogger("teacher.build_4b")

CANDIDATES_V3 = settings.GENERATED / "candidates_v3.jsonl"
TRAIN_OUT = settings.DATASET / "train_4b_iter1.jsonl"
VALID_OUT = settings.DATASET / "valid_4b_iter1.jsonl"
MANIFEST = settings.GENERATED / "4b_iter1_manifest.json"
GATE_FIX_DUMP = settings.GENERATED / "4b_iter1_topup_needed.jsonl"

TIER_ORDER = ("beginner", "intermediate", "advanced")
SEED = 3407

_SAN = r"(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)"

# --------------------------------------------------------------------------- #
# 1) Lead / restatement / dangling-connector cleaner
# --------------------------------------------------------------------------- #
# Consume a leading restatement clause (optionally including the move token and
# its trailing punctuation) so the coaching does not echo "I'd play X. Play X.".
_LEAD_RESTATE = re.compile(
    rf"^\s*(?:"
    rf"the\s+move\s*:\s*(?:{_SAN}\b[\s.:,;!\-]*)?"
    rf"|the\s+move\s+is(?:\s+to\s+play|\s+to)?\s+(?:{_SAN}\b[\s.:,;!\-]*)?"
    rf"|this\s+is\s+the\s+move\b[\s.:,;!\-]*"
    rf"|the\s+move\s+to\s+(?:play|learn|focus)(?:\s+on)?\s+(?:{_SAN}\b[\s.:,;!\-]*)?"
    rf"|(?:i['\u2019]?d\s+play|let['\u2019]?s\s+play|we['\u2019]?d\s+play|we\s+play"
    rf"|go\s+with|play|consider)\s+{_SAN}\b[\s.:,;!\-]*"
    rf")",
    re.IGNORECASE,
)
_LEAD_DANGLING = re.compile(r"^\s*[\u2014\u2013\-,]?\s*(?:and|but|so|then|in fact)\b[\s,]*", re.I)
_LEAD_PUNCT = re.compile(r"^[\s.,;:!\u2014\u2013\-]+")

# Safety-net: the EXACT patterns scripts/audit_v5_artifacts.py flags, applied to
# the text right after "I'd play X." — guarantees the measured artifact rate ~0.
_AUDIT_DANGLING = re.compile(r"^\s*[\u2014\-\u2013]\s*(and|but|so|then|in fact)\b", re.I)
_AUDIT_RESTATE = re.compile(
    r"^\s*(THE MOVE\s*:|The move is\b|This is the move\b|Play\s+\S+\.|Consider\s+\S+\.|"
    r"Let'?s play\b|The move to (?:play|learn|focus)\b|I'?d play\b)", re.I)


def clean_lead(coaching: str) -> str:
    """Strip leading move-restatements + dangling connectors from coaching."""
    prev = None
    text = coaching.strip()
    # Iterate: a restatement may be followed by a dangling connector, or vice versa.
    for _ in range(6):
        if text == prev:
            break
        prev = text
        text = _LEAD_RESTATE.sub("", text, count=1)
        text = _LEAD_DANGLING.sub("", text, count=1)
        # Safety-net against the audit's own anchored patterns.
        m = _AUDIT_RESTATE.match(text)
        if m:
            text = text[m.end():]
        m = _AUDIT_DANGLING.match(text)
        if m:
            text = text[m.end():]
        text = _LEAD_PUNCT.sub("", text).strip()
    # Re-capitalize the first letter of the (now clean) coaching.
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


# --------------------------------------------------------------------------- #
# 2) Beginner vocab scrub (tempo / initiative -> plain wording)
# --------------------------------------------------------------------------- #
_SCRUBS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bwith tempo\b", re.I), "for free"),
    (re.compile(r"\bgain(?:ing|s)?\s+a?\s*tempo\b", re.I), "gain time"),
    (re.compile(r"\ba tempo\b", re.I), "time"),
    (re.compile(r"\btempi\b", re.I), "moves"),
    (re.compile(r"\btempo\b", re.I), "time"),
    (re.compile(r"\bthe initiative\b", re.I), "the attack"),
    (re.compile(r"\binitiative\b", re.I), "pressure"),
]


def beginner_scrub(text: str) -> str:
    for pat, repl in _SCRUBS:
        text = pat.sub(repl, text)
    return text


# --------------------------------------------------------------------------- #
# 3) Deterministic takeaway-principle gate
# --------------------------------------------------------------------------- #
# Aligned to scripts.audit_v5_instructiveness.PRINCIPLE_FAMILIES (the metric).
def _principle_families() -> Dict[str, re.Pattern]:
    # Import lazily from the audit module so detection stays byte-identical.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "audit_v5_instr", str(_ROOT / "scripts" / "audit_v5_instructiveness.py"))
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    return mod.PRINCIPLE_FAMILIES


# (clause text, beginner_safe) — each clause is written to MATCH its family regex
# in the audit, is chess-correct per principle_library_v5.md, and is NON-directional
# where a directional slogan would risk contradicting the row (e.g. trades).
PRINCIPLE_CLAUSE: Dict[str, Tuple[str, bool]] = {
    "development": ("Develop your pieces before you attack", True),
    "king_safety": ("King safety first \u2014 get your king out of the center", True),
    "center": ("Fight for the center", True),
    "open_file_rook": ("Put your rooks on open files", True),
    "seventh_rank": ("A rook on the seventh rank is powerful", True),
    "doubled_rooks": ("Double your rooks so they support each other", True),
    "passed_pawn": ("Support a passed pawn before you push it", True),
    "prophylaxis": ("Take away your opponent's idea first", True),
    "outpost": ("Plant a knight on a strong outpost", False),
    "bishop_pair": ("Use the bishop pair in open positions", False),
    "good_bad_bishop": ("Improve your bad bishop", False),
    "trade_logic": ("Weigh every trade by what it changes", True),
    "activity_worst_piece": ("Improve your worst-placed piece", True),
    "space": ("Use your space to prepare a break", True),
    "initiative_tempo": ("Keep the initiative", False),
    "weakness_target": ("Target the weak pawn", True),
    "endgame": ("An active king is a strong piece in the endgame", True),
}

# Priority when several principles apply to the body (most instructive first).
_FAM_PRIORITY = [
    "king_safety", "development", "center", "passed_pawn", "open_file_rook",
    "seventh_rank", "doubled_rooks", "outpost", "good_bad_bishop", "weakness_target",
    "prophylaxis", "activity_worst_piece", "space", "trade_logic", "bishop_pair",
    "initiative_tempo", "endgame",
]

# concepts_used token substring -> family.
_CONCEPT_TO_FAM: List[Tuple[str, str]] = [
    ("develop", "development"), ("castl", "king_safety"), ("king safety", "king_safety"),
    ("center", "center"), ("central", "center"), ("open file", "open_file_rook"),
    ("seventh", "seventh_rank"), ("7th", "seventh_rank"), ("passed", "passed_pawn"),
    ("passer", "passed_pawn"), ("promot", "passed_pawn"), ("prophyl", "prophylaxis"),
    ("prevent", "prophylaxis"), ("outpost", "outpost"), ("bishop pair", "bishop_pair"),
    ("bad bishop", "good_bad_bishop"), ("good bishop", "good_bad_bishop"),
    ("trade", "trade_logic"), ("exchange", "trade_logic"), ("simplif", "trade_logic"),
    ("initiative", "initiative_tempo"), ("tempo", "initiative_tempo"), ("space", "space"),
    ("weak", "weakness_target"), ("isolated", "weakness_target"), ("backward", "weakness_target"),
    ("worst piece", "activity_worst_piece"), ("passive", "activity_worst_piece"),
    ("activ", "activity_worst_piece"), ("endgame", "endgame"), ("opposition", "endgame"),
]


def apply_takeaway_gate(
    takeaway: str, body: str, concepts: List[str], tier: str, fams: Dict[str, re.Pattern]
) -> Tuple[str, bool]:
    """Ensure the takeaway names a principle. Returns (takeaway, augmented?)."""
    if any(p.search(takeaway) for p in fams.values()):
        return takeaway, False

    beginner = tier == "beginner"

    def ok(fam: Optional[str]) -> Optional[str]:
        if not fam:
            return None
        if beginner and not PRINCIPLE_CLAUSE[fam][1]:
            return None
        return fam

    # 1) A principle already argued in the BODY (keeps the takeaway on-topic).
    chosen: Optional[str] = None
    for fam in _FAM_PRIORITY:
        if ok(fam) and fams[fam].search(body):
            chosen = fam
            break
    # 2) From the teacher's concepts_used.
    if not chosen:
        joined = " ".join(concepts).lower()
        for sub, fam in _CONCEPT_TO_FAM:
            if sub in joined and ok(fam):
                chosen = fam
                break
    # 3) Beginner-safe default (always sound general advice).
    if not chosen:
        chosen = "development" if beginner else "activity_worst_piece"

    clause = PRINCIPLE_CLAUSE[chosen][0]
    takeaway = f"{clause}. {takeaway}".strip() if takeaway else f"{clause}."
    return takeaway, True


# --------------------------------------------------------------------------- #
# v5 render (I'd play X. <clean coaching> How to find it: <method> Takeaway: ...)
# --------------------------------------------------------------------------- #


def render_v5_target(to: Dict[str, Any], tier: str, fams: Dict[str, re.Pattern]
                     ) -> Tuple[str, bool]:
    san = str(to.get("recommended_move_san") or "").strip()
    coaching = clean_lead(str(to.get("coaching") or "").strip())
    method = str(to.get("method") or "").strip()
    takeaway = str(to.get("takeaway") or "").strip()
    concepts = [str(c) for c in (to.get("concepts_used") or [])]

    takeaway, augmented = apply_takeaway_gate(takeaway, coaching, concepts, tier, fams)

    body = f"I'd play {san}. {coaching}".strip()
    if method:
        body = f"{body} How to find it: {method}"
    target = f"{body} Takeaway: {takeaway}".strip()

    if tier == "beginner":
        target = beginner_scrub(target)
    # collapse any double spaces the edits introduced
    target = re.sub(r"  +", " ", target)
    return target, augmented


# --------------------------------------------------------------------------- #
# Gate one candidate -> (row | None, reasons, info)
# --------------------------------------------------------------------------- #


def _scenario_like(ti: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fen": ti["fen"], "tier": ti["tier"], "student_move": ti["student_move"],
        "sound_pool": ti["sound_pool"], "maia": ti.get("maia_human_moves", []),
    }


def gate_and_render(cand: Dict[str, Any], system_prompt: str, fams: Dict[str, re.Pattern]
                    ) -> Tuple[Optional[Dict[str, Any]], List[str], Dict[str, Any]]:
    reasons: List[str] = []
    ti = cand.get("teacher_input") or {}
    to = cand.get("teacher_output") or {}
    engine = cand.get("engine") or {}
    meta = cand.get("meta") or {}

    fen = ti.get("fen")
    tier = ti.get("tier") or cand.get("tier")
    rec_uci = str(to.get("recommended_move_uci") or "").strip()
    rec_san = str(to.get("recommended_move_san") or "").strip()

    info: Dict[str, Any] = {
        "tier": tier,
        "is_engine_best": bool(meta.get("pick_is_engine_best", True)),
        "base_id": meta.get("base_id") or cand.get("id"),
        "cand_id": cand.get("id"),
        "rec_uci": rec_uci,
    }

    if tier not in settings.TIERS:
        reasons.append("missing_tier")
    if not fen:
        reasons.append("missing_fen")
    if not str(to.get("coaching") or "").strip():
        reasons.append("empty_coaching")
    if not str(to.get("method") or "").strip():
        reasons.append("missing_method")
    if not str(to.get("takeaway") or "").strip():
        reasons.append("empty_takeaway")
    sound_set = {str(u).lower() for u in (engine.get("sound_ucis") or [])}
    if not rec_uci or rec_uci.lower() not in sound_set:
        reasons.append("soundness")
    if fen and (rec_uci or rec_san):
        legal, _ = move_is_legal(fen, rec_uci, rec_san)
        if not legal:
            reasons.append("illegal_move")
    if reasons:
        return None, reasons, info

    target, augmented = render_v5_target(to, tier, fams)
    info["augmented_takeaway"] = augmented
    info["target"] = target

    if detect_engine_speak(target):
        reasons.append("engine_speak")
    ply_cap = settings.TIERS[tier]["ply_cap"]
    if longest_san_run(target) > ply_cap:
        reasons.append("ply_cap")
    if not target.startswith(f"I'd play {rec_san}."):
        reasons.append("format_lead")
    if "Takeaway:" not in target:
        reasons.append("format_takeaway")
    if verify_text(target, fen).violations:
        reasons.append("faithfulness_narrow")
    if reasons:
        return None, reasons, info

    user = build_grounded_user(_scenario_like(ti))
    row = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "_meta": {
            "cand_id": info["cand_id"], "base_id": info["base_id"], "tier": tier,
            "fen": fen, "rec_uci": rec_uci, "rec_san": rec_san,
            "is_engine_best": info["is_engine_best"],
            "discriminating": (tier == "beginner" and not info["is_engine_best"]),
            "augmented_takeaway": augmented,
        },
    }
    return row, [], info


def _iter_candidates() -> List[Dict[str, Any]]:
    if not CANDIDATES_V3.exists():
        raise SystemExit(f"BLOCKED: missing {CANDIDATES_V3}")
    out: List[Dict[str, Any]] = []
    with CANDIDATES_V3.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _board_class(picks: Dict[str, str]) -> Optional[str]:
    """Classify an all-3-tier board by its (b, i, a) canonical picks."""
    if not all(t in picks for t in TIER_ORDER):
        return None
    b, i, a = picks["beginner"], picks["intermediate"], picks["advanced"]
    if b == i == a:
        return "all_same"
    if b == a and b != i:
        return "collapse_BA"
    if b == i and i != a:
        return "BI"
    if i == a and i != b:
        return "IA"
    return "full"


def _drop_allsame(base_id: str, frac: float) -> bool:
    """Deterministic down-weight: drop ~frac of all-same boards by hash."""
    h = int(hashlib.sha256(base_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


# --------------------------------------------------------------------------- #
# analyze / build
# --------------------------------------------------------------------------- #


def _gather(fams: Dict[str, re.Pattern]):
    system_prompt = load_system_prompt()
    cands = _iter_candidates()
    rows: List[Dict[str, Any]] = []
    reason_hist: Counter = Counter()
    seen_keys: set = set()
    picks_by_base: Dict[str, Dict[str, str]] = defaultdict(dict)
    for c in cands:
        row, reasons, info = gate_and_render(c, system_prompt, fams)
        if reasons:
            for r in reasons:
                reason_hist[r] += 1
            continue
        m = row["_meta"]
        dk = (m["fen"], m["tier"], m["rec_uci"])
        if dk in seen_keys:
            reason_hist["duplicate"] += 1
            continue
        seen_keys.add(dk)
        picks_by_base[m["base_id"]].setdefault(m["tier"], m["rec_uci"])
        rows.append(row)
    return rows, reason_hist, picks_by_base, len(cands)


def cmd_analyze(args: argparse.Namespace) -> int:
    fams = _principle_families()
    rows, reason_hist, picks_by_base, n_cands = _gather(fams)
    board_class = {b: _board_class(p) for b, p in picks_by_base.items()}
    cc = Counter(v for v in board_class.values() if v)
    aug = sum(1 for r in rows if r["_meta"]["augmented_takeaway"])
    tier_ct = Counter(r["_meta"]["tier"] for r in rows)
    disc = sum(1 for r in rows if r["_meta"]["discriminating"])
    print("\n=== 4B iter1 candidate analysis (from candidates_v3.jsonl) ===")
    print(f"candidates:                 {n_cands}")
    print(f"kept after gates + dedup:   {len(rows)}")
    print(f"  takeaway augmented:       {aug} ({100.0*aug/max(1,len(rows)):.1f}%)")
    print(f"  by tier:                  {dict(tier_ct)}")
    print(f"  beginner-discriminating:  {disc}")
    print("\nrejected by reason:")
    for r, n in reason_hist.most_common():
        print(f"  {r:<20} {n}")
    print("\nall-3-tier board coherence (pre-fix):")
    for k, v in cc.most_common():
        print(f"  {k:<14} {v}")
    print(f"\n-> collapse_BA intermediate rows to DROP: "
          f"{sum(1 for b, cl in board_class.items() if cl == 'collapse_BA')}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    fams = _principle_families()
    rows, reason_hist, picks_by_base, n_cands = _gather(fams)
    board_class = {b: _board_class(p) for b, p in picks_by_base.items()}

    # Split by base id so a position's triples never straddle train/valid.
    by_base: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_base[r["_meta"]["base_id"]].append(r)
    base_ids = sorted(by_base)
    rng = random.Random(SEED)
    rng.shuffle(base_ids)
    n_valid = max(1, int(len(base_ids) * args.valid_frac))
    valid_bases = set(base_ids[:n_valid])

    dropped_collapse = 0
    dropped_allsame = 0
    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    topup_needed: List[Dict[str, Any]] = []

    for bid in base_ids:
        cls = board_class.get(bid)
        is_valid = bid in valid_bases
        for r in by_base[bid]:
            m = r["_meta"]
            tier = m["tier"]
            # (3) DROP the B=A != I intermediate collapse row.
            if cls == "collapse_BA" and tier == "intermediate":
                dropped_collapse += 1
                topup_needed.append({"base_id": bid, "tier": tier, "fen": m["fen"],
                                     "reason": "collapse_BA_intermediate",
                                     "cand_id": m["cand_id"]})
                continue
            if is_valid:
                valid_rows.append(r)
                continue
            # DOWN-WEIGHT non-differentiating all-same boards (train only).
            if cls == "all_same" and _drop_allsame(bid, args.allsame_drop):
                dropped_allsame += 1
                continue
            copies = 1
            if cls == "full":                 # contrastive triad: up-weight
                copies += 1
            if m["discriminating"]:           # beginner pick != engine best
                copies += 1
            train_rows.extend([r] * copies)

    rng.shuffle(train_rows)

    def _clean(r: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": r["messages"]}

    _write_jsonl([_clean(r) for r in train_rows], TRAIN_OUT)
    _write_jsonl([_clean(r) for r in valid_rows], VALID_OUT)
    _write_jsonl(topup_needed, GATE_FIX_DUMP)

    tier_train = Counter(r["_meta"]["tier"] for r in train_rows)
    aug_train = sum(1 for r in train_rows if r["_meta"]["augmented_takeaway"])
    manifest = {
        "source": str(CANDIDATES_V3.relative_to(_ROOT)),
        "candidates": n_cands,
        "kept_unique_rows": len(rows),
        "rejects_by_reason": dict(reason_hist),
        "board_coherence_prefix": dict(Counter(v for v in board_class.values() if v)),
        "dropped_collapse_BA_intermediate": dropped_collapse,
        "dropped_allsame_downweight": dropped_allsame,
        "allsame_drop_frac": args.allsame_drop,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_by_tier": dict(tier_train),
        "train_takeaway_augmented": aug_train,
        "valid_frac": args.valid_frac,
        "prompt_format": "build_grounded_user (facts + render_user_prompt + FORMAT_INSTRUCTION)",
        "fixes": [
            "lead_restatement_dangling_cleaner",
            "beginner_tempo_initiative_scrub",
            "collapse_BA_intermediate_dropped",
            "deterministic_takeaway_principle_gate",
            "downweight_all_same_boards",
            "upweight_full_gradient_triads_and_beginner_discriminating",
        ],
        "topup_needed_file": str(GATE_FIX_DUMP.relative_to(_ROOT)),
        "seed": SEED,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== 4B iter1 build summary ===")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote train -> {TRAIN_OUT} ({len(train_rows)} rows)")
    print(f"wrote valid -> {VALID_OUT} ({len(valid_rows)} rows)")
    print(f"wrote manifest -> {MANIFEST}")
    return 0


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="Measure gates + coherence (no writes).")
    pa.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("build", help="Write train_4b_iter1 / valid_4b_iter1 + manifest.")
    pb.add_argument("--valid-frac", type=float, default=0.05)
    pb.add_argument("--allsame-drop", type=float, default=0.5,
                    help="Fraction of non-differentiating all-same boards to drop (down-weight).")
    pb.set_defaults(func=cmd_build)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
