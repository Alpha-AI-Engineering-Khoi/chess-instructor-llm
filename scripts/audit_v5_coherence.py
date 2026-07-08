#!/usr/bin/env python3
"""Diagnose tier-coherence defects in a dataset (default train_v4). Read-only.

Distinguishes BENIGN all-same (only one sound move -> every tier must pick it)
from a MISSED-DIFFERENTIATION all-same (pool had >=2 sound moves but all tiers
got the same one). Also dumps the pathological B=A!=I collapses in full so they
can be eyeballed / regenerated.
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_v5_instructiveness import parse_user, parse_assistant  # noqa: E402


def load(path):
    out = []
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
            u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if u and a:
                out.append((u, a))
    return out


def main(argv):
    path = Path(argv[0]) if argv else ROOT / "data" / "dataset" / "train_v4.jsonl"
    rows = load(path)
    seen = set(); uniq = []
    for u, a in rows:
        if (u, a) in seen:
            continue
        seen.add((u, a)); uniq.append((u, a))

    # board -> tier -> list of (rec, pool_size, pool_sans, maia_sans, student)
    board = defaultdict(lambda: defaultdict(list))
    for u, a in uniq:
        pu = parse_user(u); pa = parse_assistant(a)
        if pu["board"] and pu["tier"] and pa["rec"]:
            board[pu["board"]][pu["tier"]].append({
                "rec": pa["rec"],
                "pool": [p[0] for p in pu["pool"]],
                "maia": [m[0] for m in pu["maia"]],
                "student": pu["student"],
            })

    all3 = {b: tm for b, tm in board.items() if len(tm) == 3}

    allsame_benign = 0
    allsame_missed = 0
    collapse_BA = []
    for b, tm in all3.items():
        B = Counter(r["rec"] for r in tm["beginner"]).most_common(1)[0][0]
        I = Counter(r["rec"] for r in tm["intermediate"]).most_common(1)[0][0]
        A = Counter(r["rec"] for r in tm["advanced"]).most_common(1)[0][0]
        # min pool size seen across the tier rows for this board
        min_pool = min(min(len(r["pool"]) for r in tm[t]) for t in tm)
        if B == I == A:
            if min_pool <= 1:
                allsame_benign += 1
            else:
                allsame_missed += 1
        elif B == A and B != I:
            if len(collapse_BA) < 4:
                collapse_BA.append((b, tm, B, I, A))

    print(f"FILE {path.name}: unique={len(uniq)}  all-3-tier boards={len(all3)}\n")
    print("=== all-same (B=I=A) decomposition ===")
    tot_same = allsame_benign + allsame_missed
    print(f"  total all-same:            {tot_same}")
    print(f"  BENIGN (pool size <=1):    {allsame_benign}  (forced: only one sound move)")
    print(f"  MISSED-DIFF (pool >=2):    {allsame_missed}  (could have differentiated)")

    print("\n=== pathological B=A != I collapses (full) ===")
    for b, tm, B, I, A in collapse_BA:
        print("\n" + "-" * 70)
        print(b)
        print(f"  beginner   rec={B}  (pool {tm['beginner'][0]['pool'][:6]}  maia {tm['beginner'][0]['maia'][:4]})")
        print(f"  intermediate rec={I}  (pool {tm['intermediate'][0]['pool'][:6]}  maia {tm['intermediate'][0]['maia'][:4]})")
        print(f"  advanced   rec={A}  (pool {tm['advanced'][0]['pool'][:6]}  maia {tm['advanced'][0]['maia'][:4]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
