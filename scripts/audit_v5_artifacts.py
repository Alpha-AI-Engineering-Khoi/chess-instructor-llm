#!/usr/bin/env python3
"""Scan for concrete, cheaply-fixable target-text artifacts in a dataset
(default train_v4). Read-only. These are the "raise conciseness/grounding +
polish" levers that need NO teacher re-spend (a render/regex fix + rebuild).
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_v5_instructiveness import parse_user, parse_assistant, PRINCIPLE_FAMILIES, TACTIC_FAMILIES, principle_hits  # noqa: E402

REC_RE = re.compile(r"^I'?d play ([^.]+?)\.\s*(.*)", re.DOTALL)

# artifact patterns applied to the text immediately AFTER "I'd play X."
DANGLING = re.compile(r"^\s*[—\-–]\s*(and|but|so|then|in fact)\b", re.I)
RESTATE = re.compile(r"^\s*(THE MOVE\s*:|The move is\b|This is the move\b|Play\s+\S+\.|Consider\s+\S+\.|Let'?s play\b|The move to (?:play|learn|focus)\b|I'?d play\b)", re.I)
DOUBLE_SPACE = re.compile(r"  +")
CENTIPAWN_WORD = re.compile(r"\bcentipawn|\bcp\b|\beval", re.I)


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

    n = len(uniq)
    art = Counter()
    examples = {"dangling": [], "restate": [], "rec_repeat": [], "no_named_any": []}
    beg_tactic_only = 0
    beg_strat = 0
    beg_named_any = 0
    beg_n = 0

    for u, a in uniq:
        pu = parse_user(u); pa = parse_assistant(a)
        tier = pu["tier"]
        m = REC_RE.match(a.strip())
        if not m:
            art["no_leading_move"] += 1
            continue
        rec, after = m.group(1).strip(), m.group(2)
        if DANGLING.match(after):
            art["dangling_connector"] += 1
            if len(examples["dangling"]) < 6:
                examples["dangling"].append(a[:150])
        if RESTATE.match(after):
            art["restated_move"] += 1
            if len(examples["restate"]) < 6:
                examples["restate"].append(a[:150])
        # recommended SAN repeated many times in the body
        reps = len(re.findall(re.escape(rec), a))
        if reps >= 4:
            art["rec_san_ge4x"] += 1
            if len(examples["rec_repeat"]) < 4:
                examples["rec_repeat"].append((rec, reps, a[:150]))
        if DOUBLE_SPACE.search(a):
            art["double_space"] += 1
        if CENTIPAWN_WORD.search(a):
            art["engine_word"] += 1

        # named-principle-anywhere (strategic OR tactic) coverage
        strat = principle_hits(a, PRINCIPLE_FAMILIES)
        tac = principle_hits(a, TACTIC_FAMILIES)
        if not (strat or tac):
            if len(examples["no_named_any"]) < 8:
                examples["no_named_any"].append((tier, a[:170]))
        if tier == "beginner":
            beg_n += 1
            if strat:
                beg_strat += 1
            if strat or tac:
                beg_named_any += 1
            if tac and not strat:
                beg_tactic_only += 1

    print(f"FILE {path.name}: {len(rows)} raw, {n} unique\n")
    print("=== target-text artifacts (cheap render/regex fixes; no teacher re-spend) ===")
    for k, v in art.most_common():
        print(f"  {k:<22} {v:>6}  ({100.0*v/n:4.1f}%)")

    print("\n  -- dangling-connector samples --")
    for s in examples["dangling"]:
        print(f"    {s!r}")
    print("\n  -- restated-move samples --")
    for s in examples["restate"]:
        print(f"    {s!r}")
    print("\n  -- rec-SAN repeated >=4x --")
    for rec, reps, s in examples["rec_repeat"]:
        print(f"    ({rec} x{reps}) {s!r}")

    print("\n=== beginner named-principle composition ===")
    print(f"  beginner rows: {beg_n}")
    print(f"  name a STRATEGIC principle:        {beg_strat} ({100.0*beg_strat/beg_n:.1f}%)")
    print(f"  name ANY principle (strat|tactic): {beg_named_any} ({100.0*beg_named_any/beg_n:.1f}%)")
    print(f"  tactic-only (no strategic):        {beg_tactic_only} ({100.0*beg_tactic_only/beg_n:.1f}%)")

    print("\n=== rows naming NO principle at all (strat or tactic) — samples ===")
    for tier, s in examples["no_named_any"]:
        print(f"    [{tier}] {s!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
