#!/usr/bin/env python3
"""Stage-4 eval INPUT builder (local, free) — the corrected-v6 held-out TEST slice.

Builds the two model-facing prompts for every one of the 120 held-out TEST
positions x 3 tiers (360 scenarios) on the CORRECTED v6 benchmark, so the Modal
generation job (``scripts/stage4_eval.py``) reads a single self-contained inputs
file and never has to import the label/prompt pipeline.

For each scenario it emits, byte-identically to the v6 training/DPO pipeline:

* ``grounded_system`` / ``grounded_user`` — the deployable GROUNDED coach prompt
  (verified sound-pool + Maia + facts), built with
  ``build_v6._scn_for_prompt`` -> ``src.eval.benchmark.prompts.build_grounded_user``
  over the DEEP v6 labels (``data/generated/v6_labels.jsonl``). This is exactly the
  prompt v6-dpo/v4 were trained against; the tier selection reproduces
  ``scenarios_v6.jsonl``'s canonical labels 360/360 (validated).
* ``nog_system`` / ``nog_user`` — the NO-GROUNDING distillation prompt from
  ``scripts/distill_v6_format.build_nogrounding_user`` (position facts a player
  sees; no engine, no Maia). This is the distill thesis condition.

Plus the deterministic scoring targets from the corrected benchmark:
``canonical_uci`` (tier-policy), ``sound_ucis`` (soundness), ``student_uci``,
``engine_best_uci``, ``phase``, and per-position canonical beginner/advanced (for
distinct-moves-per-level).

Run::

    python scripts/stage4_build_inputs.py            # -> data/benchmark_gap803/stage4_eval_inputs.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import settings  # noqa: E402
import build_v6 as B  # noqa: E402  (reuse the exact v6 label/prompt machinery)
from src.eval.benchmark.prompts import build_grounded_user, load_system_prompt  # noqa: E402
from src.teacher.tier_select_v6 import select_tiers_v6  # noqa: E402
from distill_v6_format import build_nogrounding_user, SYSTEM_PROMPT as NOG_SYSTEM  # noqa: E402

V6_LABELS = settings.GENERATED / "v6_labels.jsonl"
SCEN_V6 = settings.DATA / "benchmark_gap803" / "scenarios_v6.jsonl"
OUT = settings.DATA / "benchmark_gap803" / "stage4_eval_inputs.jsonl"

TIERS = ("beginner", "intermediate", "advanced")


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    labels = _load_jsonl(V6_LABELS)
    lab_by_board = {B.board_key(L["fen"]): L for L in labels if L.get("sound_pool")}
    scen = _load_jsonl(SCEN_V6)
    val = [s for s in scen if s.get("is_val")]
    print(f"deep labels={len(labels)}  scenarios_v6={len(scen)}  held-out TEST scenarios={len(val)}")

    # per-position canonical beginner/advanced (for distinct-moves-per-level)
    canon_by_pos: Dict[str, Dict[str, str]] = {}
    for s in val:
        canon_by_pos.setdefault(s["pos_id"], {})[s["tier"]] = s.get("canonical_uci")

    grounded_system = load_system_prompt()
    rows: List[Dict[str, Any]] = []
    canon_ok = 0
    sel_cache: Dict[str, dict] = {}
    for s in sorted(val, key=lambda r: (r["pos_id"], TIERS.index(r["tier"]))):
        bk = B.board_key(s["fen"])
        L = lab_by_board.get(bk)
        if L is None:
            raise SystemExit(f"BLOCKED: no deep label for board of {s['id']} ({bk})")
        tier = s["tier"]
        if bk not in sel_cache:
            sel_cache[bk] = select_tiers_v6(L["sound_pool"], L["maia"], L["engine_best"])
        pick = sel_cache[bk]["picks"][tier]
        # fidelity guard: our reconstruction MUST reproduce the committed benchmark label
        if pick["uci"] == s.get("canonical_uci"):
            canon_ok += 1
        else:
            raise SystemExit(
                f"BLOCKED: canonical mismatch for {s['id']}: rebuilt {pick['uci']} != "
                f"benchmark {s.get('canonical_uci')} (prompt would be inconsistent)"
            )
        student = B._student_for(L, tier) or {
            "uci": s["student_move"]["uci"], "san": s["student_move"].get("san"),
            "cp_loss": 0, "severity": "none", "synthetic": True,
        }
        scn = B._scn_for_prompt(L, tier, student)
        grounded_user = build_grounded_user(scn)
        nog_user = build_nogrounding_user(s["fen"], tier, s["student_move"].get("san"))

        sound_ucis = [p["uci"] for p in s.get("sound_pool", []) if p.get("uci")]
        cb = canon_by_pos.get(s["pos_id"], {}).get("beginner")
        ca = canon_by_pos.get(s["pos_id"], {}).get("advanced")
        rows.append({
            "id": s["id"], "pos_id": s["pos_id"], "tier": tier, "fen": s["fen"],
            "phase": s.get("phase"),
            "canonical_uci": s.get("canonical_uci"),
            "canonical_beginner_uci": cb, "canonical_advanced_uci": ca,
            "engine_best_uci": s.get("engine_best_uci"),
            "sound_ucis": sound_ucis,
            "student_uci": s["student_move"]["uci"],
            "grounded_system": grounded_system, "grounded_user": grounded_user,
            "nog_system": NOG_SYSTEM, "nog_user": nog_user,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    n_pos = len({r["pos_id"] for r in rows})
    diff = sum(1 for pid, cd in canon_by_pos.items()
               if cd.get("beginner") and cd.get("advanced") and cd["beginner"] != cd["advanced"])
    print(f"wrote {len(rows)} scenarios ({n_pos} positions) -> {OUT}")
    print(f"canonical reproduced: {canon_ok}/{len(rows)}")
    print(f"tiers: {dict(Counter(r['tier'] for r in rows))}")
    print(f"positions with canonical beginner!=advanced (distinct denominator): {diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
