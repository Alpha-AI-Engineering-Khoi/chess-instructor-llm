#!/usr/bin/env python3
"""Deeper, read-only QUALITY probes on a dataset (default train_v4).

Goes past presence -> genuineness / correctness:
  * E4 method quality: generic reusable checklist vs board-narration vs stub.
  * E2 gap samples: beginner rows whose takeaway names NO transferable principle.
  * Correctness sniff: "trade/exchange" advice cross-referenced with the position
    eval sign (best-move cp, side-to-move POV) -> flags possible "trade when
    behind" anti-heuristic (should be trade when AHEAD).
  * Beginner forbidden-vocabulary breakdown (tier_guides violations).
  * Collapse B=A!=I boards shown with the student move per tier.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_v5_instructiveness import (  # noqa: E402
    parse_user, parse_assistant, PRINCIPLE_FAMILIES, principle_hits,
    BEGINNER_FORBIDDEN_VOCAB,
)

TRADE_RE = re.compile(r"\b(trad\w+|exchang\w+|swap\w+|simplif\w+|off the board)\b", re.I)
# method "reusable routine" markers vs pure board narration
ROUTINE_RE = re.compile(
    r"\b(ask yourself|ask,|before you|before playing|next time|whenever|every time|"
    r"routine|checklist|habit|rule of thumb|first .{0,10}(?:check|look|ask)|"
    r"scan (?:for|the)|look for|make it a habit|get in the habit|step 1|"
    r"start by|always (?:check|look|ask))\b", re.I)
GENERIC_QUESTION_RE = re.compile(r"\?")


def load(path: Path):
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
    # dedup
    seen = set(); uniq = []
    for u, a in rows:
        if (u, a) in seen:
            continue
        seen.add((u, a)); uniq.append((u, a))
    print(f"FILE {path.name}: {len(rows)} raw, {len(uniq)} unique\n")

    method_bucket = Counter()
    method_len = []
    method_examples = {"routine": [], "narration": [], "stub": []}
    beg_no_principle_takeaways = []
    forbidden_counter = Counter()
    trade_flags = []
    trade_total = 0
    by_tier = Counter()

    for u, a in uniq:
        pu = parse_user(u); pa = parse_assistant(a)
        tier = pu["tier"]; by_tier[tier] += 1
        method = pa["method"] or ""
        # ---- method quality ----
        if len(method) < 20:
            method_bucket["stub(<20c)"] += 1
            if len(method_examples["stub"]) < 4:
                method_examples["stub"].append((tier, method))
        else:
            method_len.append(len(method))
            has_routine = bool(ROUTINE_RE.search(method))
            has_q = bool(GENERIC_QUESTION_RE.search(method))
            if has_routine or has_q:
                method_bucket["reusable-routine"] += 1
                if len(method_examples["routine"]) < 3:
                    method_examples["routine"].append((tier, method[:240]))
            else:
                method_bucket["board-narration-only"] += 1
                if len(method_examples["narration"]) < 4:
                    method_examples["narration"].append((tier, method[:240]))

        # ---- beginner takeaway without a named principle ----
        if tier == "beginner":
            tk = pa["takeaway"] or ""
            if not principle_hits(tk, PRINCIPLE_FAMILIES):
                if len(beg_no_principle_takeaways) < 12:
                    beg_no_principle_takeaways.append(tk[:160])
            for m in BEGINNER_FORBIDDEN_VOCAB.finditer(a):
                forbidden_counter[m.group(0).lower()] += 1

        # ---- trade-advice correctness sniff ----
        full = pa["full"]
        if TRADE_RE.search(full):
            trade_total += 1
            pool = pu["pool"]
            best_cp = pool[0][1] if pool else None
            # only flag ADVOCACY to trade (not "don't trade"/"avoid trading")
            advocates = re.search(r"\b(trade|exchange|swap|simplif\w+)\b", full, re.I)
            discourages = re.search(r"\b(don'?t|do not|avoid|resist|not?\s+the time|keep|without) (?:the )?(?:trad|exchang|swap|simplif)", full, re.I)
            if best_cp is not None and best_cp <= -60 and advocates and not discourages:
                if len(trade_flags) < 15:
                    trade_flags.append((tier, best_cp, pa["rec"], full[:220]))

    print("=== E4 method quality (unique rows) ===")
    tot = sum(method_bucket.values())
    for k, v in method_bucket.most_common():
        print(f"  {k:<22} {v:>6}  ({100.0*v/tot:4.1f}%)")
    if method_len:
        method_len.sort()
        print(f"  method length chars: median={method_len[len(method_len)//2]}  "
              f"p10={method_len[len(method_len)//10]}  p90={method_len[9*len(method_len)//10]}")
    print("\n  -- narration-only method samples (weak: not a reusable routine) --")
    for tier, m in method_examples["narration"]:
        print(f"    [{tier}] {m}")
    print("\n  -- reusable-routine method samples (strong) --")
    for tier, m in method_examples["routine"]:
        print(f"    [{tier}] {m}")

    print("\n=== E2 gap: beginner takeaways naming NO transferable principle ===")
    for tk in beg_no_principle_takeaways:
        print(f"    - {tk}")

    print("\n=== E5 beginner forbidden-vocabulary breakdown (tier_guides violations) ===")
    for k, v in forbidden_counter.most_common():
        print(f"    {k:<22} {v}")

    print(f"\n=== correctness sniff: trade/exchange advice while side-to-move is WORSE ===")
    print(f"  rows mentioning trade/exchange/simplify: {trade_total}")
    print(f"  flagged (advocates trade AND best-move cp <= -60): {len(trade_flags)} shown below")
    for tier, cp, rec, txt in trade_flags:
        print(f"    [{tier}] bestcp={cp} rec={rec}: {txt}")

    print(f"\n  tier counts: {dict(by_tier)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
