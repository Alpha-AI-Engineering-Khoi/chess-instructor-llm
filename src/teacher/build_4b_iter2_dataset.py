#!/usr/bin/env python3
"""Build the **4B iteration-2** SFT set — an eval-driven, contrastive rebalance.

iter-1 eval (val=120, honest gated pipeline) showed the tuned 4B beats the untuned
base on every deterministic axis BUT badly misses two completion criteria:

* tier-fit 0.386 (target >=0.60)
* **distinct-moves-per-level 0.26 (target >=0.95)** — the model is ~68% FLAT
  (gives the SAME move to beginner/intermediate/advanced on a position).

So the single weak axis is **tier differentiation**. This build attacks it with a
DATA-only intervention (no new teacher spend, reusing the iter-1 gate + render):

* **DROP every flat-teaching board class**: ``all_same`` (b==i==a) AND ``collapse_BA``
  (b==a!=i — regenerating its intermediate would only create the "nonsensical
  zigzag" the spec penalizes, so we drop the whole position). iter-1 merely
  down-weighted all_same 50% and kept collapse_BA's b/a rows — both taught the
  model that tiers share a move.
* **Hard up-weight the genuinely contrastive signal**: ``full`` (b!=i!=a) triads
  x``--full-copies`` (default 3) and beginner-DISCRIMINATING rows (beginner pick !=
  engine best) x``--disc-copies`` (default 3); keep the partial-gradient ``BI`` /
  ``IA`` (both still teach beginner!=advanced) x``--bi-ia-copies`` (default 1).

Everything else (gates, faithfulness, takeaway-principle, prompt format) is
inherited byte-for-byte from ``build_4b_dataset`` so base-vs-tuned stays clean.

    python -m src.teacher.build_4b_iter2_dataset build
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List

from config import settings
from src.teacher.build_4b_dataset import (
    _board_class, _gather, _principle_families, _write_jsonl,
)

log = logging.getLogger("teacher.build_4b_iter2")

TRAIN_OUT = settings.DATASET / "train_4b_iter2.jsonl"
VALID_OUT = settings.DATASET / "valid_4b_iter2.jsonl"
MANIFEST = settings.GENERATED / "4b_iter2_manifest.json"
SEED = 3407
DROP_CLASSES = ("all_same", "collapse_BA")   # flat-teaching -> removed entirely


def cmd_build(a: argparse.Namespace) -> int:
    fams = _principle_families()
    rows, reason_hist, picks_by_base, n_cands = _gather(fams)
    board_class = {b: _board_class(p) for b, p in picks_by_base.items()}

    by_base: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_base[r["_meta"]["base_id"]].append(r)
    base_ids = sorted(by_base)
    rng = random.Random(SEED)
    rng.shuffle(base_ids)
    n_valid = max(1, int(len(base_ids) * a.valid_frac))
    valid_bases = set(base_ids[:n_valid])

    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    dropped: Counter = Counter()
    upweighted: Counter = Counter()

    for bid in base_ids:
        cls = board_class.get(bid)
        is_valid = bid in valid_bases
        for r in by_base[bid]:
            m = r["_meta"]
            # DROP flat-teaching classes entirely (attack the 68% flat collapse).
            if cls in DROP_CLASSES:
                dropped[cls] += 1
                continue
            if is_valid:
                valid_rows.append(r)
                continue
            copies = 1
            if cls == "full":
                copies = a.full_copies
            elif cls in ("BI", "IA"):
                copies = a.bi_ia_copies
            if m["discriminating"]:            # beginner pick != engine best
                copies = max(copies, a.disc_copies)
            if copies > 1:
                upweighted[cls or "?"] += (copies - 1)
            train_rows.extend([r] * copies)

    rng.shuffle(train_rows)
    _write_jsonl([{"messages": r["messages"]} for r in train_rows], TRAIN_OUT)
    _write_jsonl([{"messages": r["messages"]} for r in valid_rows], VALID_OUT)

    tier_train = Counter(r["_meta"]["tier"] for r in train_rows)
    cls_train = Counter(board_class.get(r["_meta"]["base_id"]) for r in train_rows)
    manifest = {
        "iteration": "4b-iter2",
        "parent": "4b-iter1",
        "eval_driver": "distinct-moves 0.26 / tier-fit 0.386 (both below target) -> attack tier flatness",
        "source": "data/generated/candidates_v3.jsonl",
        "candidates": n_cands,
        "kept_unique_rows": len(rows),
        "board_coherence_prefix": dict(Counter(v for v in board_class.values() if v)),
        "dropped_flat_classes": dict(dropped),
        "mix": {"full_copies": a.full_copies, "bi_ia_copies": a.bi_ia_copies,
                "disc_copies": a.disc_copies, "dropped": list(DROP_CLASSES)},
        "upweight_extra_rows_by_class": dict(upweighted),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_by_tier": dict(tier_train),
        "train_by_board_class": {str(k): v for k, v in cls_train.items()},
        "rejects_by_reason": dict(reason_hist),
        "seed": SEED,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== 4B iter2 build summary ===")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote train -> {TRAIN_OUT} ({len(train_rows)} rows)")
    print(f"wrote valid -> {VALID_OUT} ({len(valid_rows)} rows)")
    print(f"wrote manifest -> {MANIFEST}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build", help="Write train_4b_iter2 / valid_4b_iter2 + manifest.")
    pb.add_argument("--valid-frac", type=float, default=0.05)
    pb.add_argument("--full-copies", type=int, default=3, help="copies of full (b!=i!=a) triads.")
    pb.add_argument("--bi-ia-copies", type=int, default=1, help="copies of partial-gradient BI/IA rows.")
    pb.add_argument("--disc-copies", type=int, default=3, help="copies of beginner-discriminating rows.")
    pb.set_defaults(func=cmd_build)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(a.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
