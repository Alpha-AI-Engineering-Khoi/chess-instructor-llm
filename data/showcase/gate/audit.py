#!/usr/bin/env python3
"""FREE, deterministic audit of web/public/showcase.json before gating.

Scopes the Phase-A workload exactly and at zero cost:
  * counts positions / models / cells (present, non-null, with text);
  * maps every cell back to its showcase scenario (so re-gen can reuse the
    identical grounded prompt) and reports any cell that cannot be mapped;
  * runs the widened deterministic checker verify_text_ext on the RAW coaching
    (recommended_uci = the cell's own move_uci) to get the honest raw fabrication
    rate per model — the model-capacity metric;
  * splits the flagged (=> needs re-gen) cells into LOCAL (free MLX: ours/base)
    vs TFY (paid frontier/open) so the real dollar exposure is known up front.

Writes data/showcase/gate/audit.json (summary + flagged cell list). No network.
Run:  ~/.venvs/mlx/bin/python data/showcase/gate/audit.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
PIPE = ROOT / "data" / "showcase" / "pipeline"
for p in (str(ROOT), str(PIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import FIELD, LOCAL_KEYS, SPLIT_DIRS, model_meta, read_jsonl  # noqa: E402
from src.engine.faithfulness_ext import verify_text_ext  # noqa: E402

TIERS = ("beginner", "intermediate", "advanced")
WEB_SHOWCASE = ROOT / "web" / "public" / "showcase.json"
OUT = Path(__file__).resolve().parent / "audit.json"


def build_scn_index() -> Dict[str, Dict[Tuple[str, str], Dict[str, Any]]]:
    idx: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}
    for split_name, split_dir in SPLIT_DIRS.items():
        d: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for s in read_jsonl(split_dir / "scenarios.jsonl"):
            d[(s.get("pos_id", s["id"]), s["tier"])] = s
        idx[split_name] = d
    return idx


def main() -> int:
    name_to_key = {model_meta(k)["name"]: k for k in FIELD}
    scn_index = build_scn_index()
    # a global (pos_id,tier)->scn fallback for any split-label mismatch
    scn_global: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in scn_index.values():
        scn_global.update(d)

    positions: List[Dict[str, Any]] = json.loads(WEB_SHOWCASE.read_text(encoding="utf-8"))

    n_cells = 0
    n_text = 0
    n_empty = 0
    unmapped: List[List[str]] = []
    unknown_models: set = set()

    per_model = defaultdict(lambda: {
        "cells": 0, "text": 0, "flagged_raw": 0, "local": False, "family": "",
    })
    flagged: List[Dict[str, Any]] = []  # cells that need re-gen

    for pi, pos in enumerate(positions):
        split_source = pos.get("split_source") or ("train" if pos.get("split") == "train" else "test_new")
        fen = pos["fen"]
        pos_id = pos["id"]
        for m in pos.get("models", []):
            name = m["name"]
            key = name_to_key.get(name)
            if key is None:
                unknown_models.add(name)
            local = bool(m.get("local")) or (key in LOCAL_KEYS if key else False)
            for tier in TIERS:
                cell = (m.get("byTier") or {}).get(tier)
                if not cell:
                    continue
                n_cells += 1
                pm = per_model[name]
                pm["cells"] += 1
                pm["local"] = local
                pm["family"] = m.get("family", "")
                coaching = cell.get("coaching")
                move_uci = cell.get("move_uci")
                if not coaching or not str(coaching).strip():
                    n_empty += 1
                    continue
                n_text += 1
                pm["text"] += 1
                res = verify_text_ext(str(coaching), fen, recommended_uci=move_uci)
                if not res.ok:
                    pm["flagged_raw"] += 1
                    scn = scn_index.get(split_source, {}).get((pos_id, tier)) or scn_global.get((pos_id, tier))
                    mapped = scn is not None
                    if not mapped:
                        unmapped.append([pos_id, name, tier])
                    flagged.append({
                        "pi": pi, "pos_id": pos_id, "tier": tier, "model": name,
                        "key": key, "local": local, "split_source": split_source,
                        "mapped": mapped, "n_violations": len(res.violations),
                    })

    # ---- summarise ----
    def rate(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    local_flagged = sum(1 for f in flagged if f["local"])
    tfy_flagged = sum(1 for f in flagged if not f["local"])
    tfy_flagged_mapped = sum(1 for f in flagged if not f["local"] and f["mapped"])

    summary = {
        "n_positions": len(positions),
        "n_cells": n_cells,
        "n_cells_with_text": n_text,
        "n_cells_empty": n_empty,
        "n_flagged_raw": len(flagged),
        "raw_fab_rate_overall": rate(len(flagged), n_text),
        "flagged_local_free": local_flagged,
        "flagged_tfy_paid": tfy_flagged,
        "flagged_tfy_paid_mapped": tfy_flagged_mapped,
        "n_unmapped_flagged": len(unmapped),
        "unknown_models": sorted(unknown_models),
        "per_model": {
            name: {
                "family": pm["family"], "local": pm["local"],
                "cells": pm["cells"], "text": pm["text"],
                "flagged_raw": pm["flagged_raw"],
                "raw_fab_rate": rate(pm["flagged_raw"], pm["text"]),
            }
            for name, pm in sorted(per_model.items())
        },
    }

    OUT.write_text(json.dumps({"summary": summary, "flagged": flagged,
                               "unmapped": unmapped}, indent=1), encoding="utf-8")

    # ---- print ----
    print(json.dumps(summary, indent=2))
    print(f"\n[audit] wrote {OUT}")
    print("\nPer-model raw fabrication (verify_text_ext on RAW coaching):")
    print(f"{'model':<32}{'kind':<6}{'cells':>7}{'text':>7}{'flag':>7}{'raw_fab%':>10}")
    for name, r in summary["per_model"].items():
        kind = "local" if r["local"] else "tfy"
        print(f"{name:<32}{kind:<6}{r['cells']:>7}{r['text']:>7}"
              f"{r['flagged_raw']:>7}{r['raw_fab_rate']*100:>9.1f}%")
    print(f"\nRe-gen workload: {local_flagged} LOCAL (free) + {tfy_flagged} TFY (paid; "
          f"{tfy_flagged_mapped} mapped to a scenario).")
    if unmapped:
        print(f"WARNING: {len(unmapped)} flagged cells could not be mapped to a scenario.")
    if unknown_models:
        print(f"WARNING: unknown model names (no key): {sorted(unknown_models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
