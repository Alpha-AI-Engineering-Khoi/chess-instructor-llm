#!/usr/bin/env python3
"""Apply the gate caches to web/public/showcase.json (from the pristine backup).

Reads data/showcase/showcase.pregate.json (untouched original) + the three job
caches (gated_cells.{local,frontier,open}.jsonl) and writes the gated showcase:

  coaching          -> GATED text (new default)
  raw_coaching      -> the original ungated text (model-capacity reference)
  raw_fabricated    -> verify_text_ext on the RAW text (pre-gate)
  gate_attempts     -> attempt # that produced the kept text (1 = raw was clean)
  verified_fallback -> True iff no re-sample verified and the engine-derived text
                       was used
  fabricated        -> POST-gate residual (verify_text_ext on the gated text)
  n_violations / violations -> post-gate receipts (usually empty)

Every other field (move/move_uci/sound/tier_fit/council_*) is preserved verbatim,
so the array stays contract-compatible with web/src/lib/showcase.ts. Refuses to
write unless EVERY cell with text has a cache record (no partial merges). Also
emits data/showcase/gate/gate_stats.json (per-model raw->gated, regens, cost).

Run:  ~/.venvs/mlx/bin/python data/showcase/gate/merge.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gate_lib as G  # noqa: E402
from src.engine.faithfulness_ext import verify_text_ext  # noqa: E402

PREGATE = G.ROOT / "data" / "showcase" / "showcase.pregate.json"
WEB_SHOWCASE = G.ROOT / "web" / "public" / "showcase.json"
STATS_OUT = HERE / "gate_stats.json"


def load_cache() -> Dict[Tuple[int, str, str], Dict[str, Any]]:
    """Merge every gated_cells.*.jsonl (all job caches); last record wins."""
    out: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    files = sorted(HERE.glob("gated_cells.*.jsonl"))
    if not files:
        print("WARNING: no gated_cells.*.jsonl caches found", file=sys.stderr)
    for path in files:
        n = 0
        for r in G.read_jsonl(path):
            out[(r["pi"], r["key"], r["tier"])] = r  # last wins
            n += 1
        print(f"[merge] {path.name}: {n} records", file=sys.stderr)
    return out


def main(argv=None) -> int:
    positions: List[Dict[str, Any]] = json.loads(PREGATE.read_text(encoding="utf-8"))
    name_to_key = G.name_to_key_map()
    cache = load_cache()

    expected = 0
    applied = 0
    missing: List[Tuple[int, str, str]] = []
    mismatch = 0

    stats = defaultdict(lambda: {
        "cells": 0, "text": 0, "raw_fab": 0, "gated_fab": 0, "regens": 0,
        "fallbacks": 0, "attempts_sum": 0, "prompt_tokens": 0,
        "completion_tokens": 0, "usd": 0.0, "local": False, "family": "",
    })

    for pi, pos in enumerate(positions):
        fen = pos["fen"]
        for m in pos.get("models", []):
            name = m["name"]
            key = name_to_key.get(name, name)
            st = stats[name]
            st["local"] = bool(m.get("local"))
            st["family"] = m.get("family", "")
            for tier in G.TIERS:
                cell = (m.get("byTier") or {}).get(tier)
                if not cell:
                    continue
                st["cells"] += 1
                coaching = cell.get("coaching")
                if not coaching or not str(coaching).strip():
                    # No text -> nothing to gate; annotate as clean-empty.
                    cell["raw_coaching"] = coaching or ""
                    cell["raw_fabricated"] = False
                    cell["gate_attempts"] = 0
                    cell["verified_fallback"] = False
                    cell["fabricated"] = False
                    cell["n_violations"] = 0
                    cell["violations"] = []
                    continue
                st["text"] += 1
                expected += 1
                rec = cache.get((pi, key, tier))
                if rec is None:
                    missing.append((pi, key, tier))
                    continue
                applied += 1

                raw_text = str(coaching)
                gated_text = rec["coaching"]
                move_uci = cell.get("move_uci")
                # Authoritative post-gate verification (single source of truth).
                vr = verify_text_ext(gated_text, fen, recommended_uci=move_uci)
                fabricated = not vr.ok
                if bool(rec.get("fabricated")) != fabricated:
                    mismatch += 1

                cell["raw_coaching"] = raw_text
                cell["raw_fabricated"] = bool(rec["raw_fabricated"])
                cell["gate_attempts"] = int(rec["gate_attempts"])
                cell["verified_fallback"] = bool(rec["verified_fallback"])
                cell["coaching"] = gated_text
                cell["fabricated"] = fabricated
                cell["n_violations"] = len(vr.violations)
                cell["violations"] = [
                    {"sentence": v.sentence, "reason": v.reason} for v in vr.violations[:5]
                ]

                st["raw_fab"] += int(bool(rec["raw_fabricated"]))
                st["gated_fab"] += int(fabricated)
                st["regens"] += int(rec.get("regens", 0))
                st["fallbacks"] += int(bool(rec["verified_fallback"]))
                st["attempts_sum"] += int(rec["gate_attempts"])
                st["prompt_tokens"] += int(rec.get("prompt_tokens", 0))
                st["completion_tokens"] += int(rec.get("completion_tokens", 0))
                st["usd"] += float(rec.get("usd", 0.0))

    print(f"[merge] cells-with-text expected={expected} applied={applied} "
          f"missing={len(missing)} fab-flag-mismatch={mismatch}", file=sys.stderr)
    if missing:
        print(f"[merge] REFUSING to write: {len(missing)} cells have no cache record "
              f"(jobs still running / incomplete). Examples: {missing[:5]}", file=sys.stderr)
        return 1

    # ---- write showcase.json (same array + indent=1 as the assembler) ----
    WEB_SHOWCASE.write_text(json.dumps(positions, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    # validate: reloads as a non-empty array
    check = json.loads(WEB_SHOWCASE.read_text(encoding="utf-8"))
    assert isinstance(check, list) and check, "showcase.json is not a non-empty array"

    # ---- per-model stats ----
    def rate(a, b):
        return round(a / b, 4) if b else 0.0

    per_model = {}
    tot = {"text": 0, "raw_fab": 0, "gated_fab": 0, "regens": 0, "fallbacks": 0, "usd": 0.0}
    for name, s in sorted(stats.items()):
        per_model[name] = {
            "family": s["family"], "local": s["local"],
            "cells": s["cells"], "text": s["text"],
            "raw_fab": s["raw_fab"], "raw_fab_rate": rate(s["raw_fab"], s["text"]),
            "gated_fab": s["gated_fab"], "gated_fab_rate": rate(s["gated_fab"], s["text"]),
            "regens": s["regens"], "fallbacks": s["fallbacks"],
            "fallback_rate": rate(s["fallbacks"], s["text"]),
            "avg_attempts": round(s["attempts_sum"] / s["text"], 3) if s["text"] else 0,
            "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["completion_tokens"],
            "usd": round(s["usd"], 4),
        }
        for k in ("text", "raw_fab", "gated_fab", "regens", "fallbacks"):
            tot[k] += s[k]
        tot["usd"] += s["usd"]

    stats_doc = {
        "n_positions": len(positions),
        "n_cells_with_text": expected,
        "totals": {
            "text": tot["text"],
            "raw_fab": tot["raw_fab"], "raw_fab_rate": rate(tot["raw_fab"], tot["text"]),
            "gated_fab": tot["gated_fab"], "gated_fab_rate": rate(tot["gated_fab"], tot["text"]),
            "regens": tot["regens"], "fallbacks": tot["fallbacks"],
            "new_spend_usd": round(tot["usd"], 2),
        },
        "per_model": per_model,
    }
    STATS_OUT.write_text(json.dumps(stats_doc, indent=2), encoding="utf-8")

    print(f"[merge] wrote {WEB_SHOWCASE} ({len(check)} positions)")
    print(f"[merge] wrote {STATS_OUT}")
    print(json.dumps(stats_doc["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
