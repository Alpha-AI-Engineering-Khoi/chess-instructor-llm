#!/usr/bin/env python3
"""Re-score COMMITTED generations against the CORRECTED v6 labels (local, free).

Continuity cross-check for Stage-4: the shipped v4 headline was published on the
v4-era labels. Because the 120 held-out TEST FENs are STABLE (0 FEN changes; only
canonical/engine_best re-derived under deep search), we can re-score the already
committed generations against ``scenarios_v6.jsonl`` and compare directly to both
the published v4 number and the freshly generated Stage-4 numbers.

Metric definitions are byte-identical to ``scripts/stage4_eval.score_condition``
and use the same vendored extractor as ``scripts/reproduce_v4.py``
(``src.eval.evaluate.extract_recommended_move``).

Run::

    python scripts/stage4_rescore_committed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eval.evaluate import extract_recommended_move  # noqa: E402

TIERS = ("beginner", "intermediate", "advanced")
SCEN_V6 = _ROOT / "data" / "benchmark_gap803" / "scenarios_v6.jsonl"
GEN_DIR = _ROOT / "data" / "benchmark_honest" / "gen"
OUT = _ROOT / "data" / "benchmark_gap803" / "stage4" / "rescore_committed.json"

# committed grounded gens over the 120 val (TEST) positions -> label for the report
COMMITTED = {
    "v4_grounded_committed": GEN_DIR / "ours_v4.jsonl",
    "base_grounded_committed": GEN_DIR / "q3_32b.jsonl",
}


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _index_scen() -> Dict[str, dict]:
    return {s["id"]: s for s in _load_jsonl(SCEN_V6)}


def _canon_by_pos(scen_by_id: Dict[str, dict]) -> Dict[str, Dict[str, Optional[str]]]:
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for s in scen_by_id.values():
        if s.get("is_val"):
            out.setdefault(s["pos_id"], {})[s["tier"]] = s.get("canonical_uci")
    return out


def score_committed(gen_rows: List[dict], scen_by_id: Dict[str, dict],
                    canon_pos: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Any]:
    by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIERS}
    sound = [0, 0]
    named = [0, 0]
    fmt = [0, 0]
    preds_by_pos: Dict[str, Dict[str, Optional[str]]] = {}
    n = 0
    for g in gen_rows:
        sid = g.get("scenario_id") or g.get("id")
        s = scen_by_id.get(sid)
        if s is None or not s.get("is_val"):
            continue
        n += 1
        tier = s["tier"]
        _san, uci = extract_recommended_move(
            g.get("output", ""), s["fen"], s["student_move"].get("uci") or "")
        if tier in by_tier:
            by_tier[tier][1] += 1
            if uci and uci == s.get("canonical_uci"):
                by_tier[tier][0] += 1
        sound[1] += 1
        if uci and uci in {p["uci"] for p in s.get("sound_pool", []) if p.get("uci")}:
            sound[0] += 1
        named[1] += 1
        if uci:
            named[0] += 1
        fmt[1] += 1
        text = g.get("output", "") or ""
        if uci and ("I'd play" in text or "I\u2019d play" in text) and "Takeaway:" in text:
            fmt[0] += 1
        preds_by_pos.setdefault(s["pos_id"], {})[tier] = uci

    per_tier = {t: (by_tier[t][0] / by_tier[t][1]) for t in TIERS if by_tier[t][1]}
    diff = dist = 0
    for pid, cd in canon_pos.items():
        cb, ca = cd.get("beginner"), cd.get("advanced")
        if not (cb and ca and cb != ca):
            continue
        diff += 1
        mb = preds_by_pos.get(pid, {}).get("beginner")
        ma = preds_by_pos.get(pid, {}).get("advanced")
        if mb and ma and mb != ma:
            dist += 1
    return {
        "tier_policy_match": round(mean(per_tier.values()), 4) if per_tier else 0.0,
        "per_tier": {t: round(v, 4) for t, v in per_tier.items()},
        "per_tier_counts": {t: by_tier[t] for t in TIERS if by_tier[t][1]},
        "move_sound": round(sound[0] / sound[1], 4) if sound[1] else 0.0,
        "named_rate": round(named[0] / named[1], 4) if named[1] else 0.0,
        "format_rate": round(fmt[0] / fmt[1], 4) if fmt[1] else 0.0,
        "distinct_rate": round(dist / diff, 4) if diff else 0.0,
        "distinct_counts": [dist, diff],
        "n": n,
    }


def main() -> int:
    scen_by_id = _index_scen()
    canon_pos = _canon_by_pos(scen_by_id)
    scores: Dict[str, Any] = {}
    for name, path in COMMITTED.items():
        if not path.exists():
            print(f"SKIP {name}: missing {path}")
            continue
        scores[name] = score_committed(_load_jsonl(path), scen_by_id, canon_pos)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"benchmark": "scenarios_v6 (corrected) re-score of committed gens",
                               "scores": scores}, indent=2), encoding="utf-8")
    print("=== committed generations re-scored on corrected v6 labels (120 TEST) ===")
    hdr = f"{'model/condition':28} {'tier_fit':>8} {'sound':>7} {'distinct':>9} {'named':>7} {'format':>7}"
    print(hdr)
    for name, s in scores.items():
        print(f"{name:28} {s['tier_policy_match']:>8.4f} {s['move_sound']:>7.4f} "
              f"{s['distinct_rate']:>9.4f} {s['named_rate']:>7.4f} {s['format_rate']:>7.4f}  "
              f"per_tier={s['per_tier']} n={s['n']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
