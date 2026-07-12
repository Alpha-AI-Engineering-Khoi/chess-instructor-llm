#!/usr/bin/env python3
"""Precompute the GROUNDED coach prompts for the FULL 803 x 3 corrected benchmark.

This is the offline half of the Colab full-803 eval of v6-dpo2: it produces a single
self-contained, ready-to-generate JSONL so the Colab notebook never has to install a
chess engine (Stockfish) or lc0/Maia. It is a strict generalization of
``scripts/stage4_build_inputs.py`` (which only builds the 120 held-out TEST slice) to
ALL 803 positions x 3 tiers (2409 scenarios).

Why no engines are needed here
------------------------------
The corrected v6 grounding is already COMMITTED to the repo:

* ``data/generated/v6_labels.jsonl``  — the deep, WDL/tablebase-verified sound pool +
  per-tier Maia policies for every board (the exact inputs the v6 pipeline trained /
  evaluated on), and
* ``data/benchmark_gap803/scenarios_v6.jsonl`` — the 803 x 3 flattened benchmark with
  the corrected ``canonical_uci`` (tier-policy target), ``sound_pool`` (soundness) and
  the played ``student_move``.

So the grounded prompt is a PURE, deterministic function of those committed files (the
only chess computation is python-chess re-deriving the verified fact sheet, which never
touches an engine). Re-running Stockfish/Maia here would be both unnecessary AND less
faithful (a fresh deep search could drift from the committed labels), so we rebuild the
prompt from the committed grounding via the EXACT stage-4 machinery:

    build_v6._scn_for_prompt(deep_label, tier, student)
        -> src.eval.benchmark.prompts.build_grounded_user   (facts + render_user_prompt
                                                              + FORMAT_INSTRUCTION)

That is byte-identical to what ``stage4_build_inputs.py`` fed the model to produce the
published ``RESULTS_STAGE4_CORRECTED.md`` numbers, so a model generated on THIS file and
scored with the same vendored extractor yields a v6-dpo2 803 row that is directly
comparable to the corrected-benchmark tables.

Two guards make the fidelity provable (both must pass, or the script BLOCKS):

1. **Canonical reproduction.** For every scenario, the tier move reconstructed from the
   deep label (``select_tiers_v6``) MUST equal the committed ``canonical_uci``. This is
   the same guard stage-4 uses; it proves the prompt is consistent with the benchmark
   target the row will be scored against.
2. **Byte-identity to the committed 360.** For the 120 held-out TEST scenarios that
   already exist in ``stage4_eval_inputs.jsonl``, this script's ``grounded_system`` /
   ``grounded_user`` MUST match that committed file byte-for-byte. That proves the full
   803 prompts are built by the identical path as the published Stage-4 slice.

Output
------
``data/benchmark_gap803/eval803_grounded_prompts.jsonl`` (2409 rows). Each row carries
the model-facing prompt (``grounded_system`` / ``grounded_user``) plus the deterministic
scoring targets (``canonical_uci``, per-position ``canonical_beginner_uci`` /
``canonical_advanced_uci`` for distinct-moves, ``sound_ucis``, ``student_uci``,
``engine_best_uci``, ``phase``, ``is_val``) — everything the Colab scorer needs and
nothing it does not.

Run (only python-chess + the repo needed; no engine, no GPU, ~a few seconds)::

    ~/.venvs/mlx/bin/python scripts/precompute_grounded_prompts_803.py
    # optional: --out <path> --limit N (smoke)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import settings  # noqa: E402
import build_v6 as B  # noqa: E402  (reuse the exact v6 label/prompt machinery)
from src.eval.benchmark.prompts import build_grounded_user, load_system_prompt  # noqa: E402
from src.teacher.tier_select_v6 import select_tiers_v6  # noqa: E402

TIERS = ("beginner", "intermediate", "advanced")

V6_LABELS = settings.GENERATED / "v6_labels.jsonl"
SCEN_V6 = settings.DATA / "benchmark_gap803" / "scenarios_v6.jsonl"
STAGE4_INPUTS = settings.DATA / "benchmark_gap803" / "stage4_eval_inputs.jsonl"
OUT_DEFAULT = settings.DATA / "benchmark_gap803" / "eval803_grounded_prompts.jsonl"


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _build_rows(limit: int = 0) -> List[Dict[str, Any]]:
    """Rebuild the grounded prompt + scoring targets for all 803 x 3 scenarios."""
    labels = _load_jsonl(V6_LABELS)
    lab_by_board = {B.board_key(L["fen"]): L for L in labels if L.get("sound_pool")}
    scen = _load_jsonl(SCEN_V6)
    print(f"deep labels={len(labels)} (with pool={len(lab_by_board)})  scenarios_v6={len(scen)}")

    # per-position canonical beginner/advanced (for distinct-moves-per-level denom)
    canon_by_pos: Dict[str, Dict[str, Optional[str]]] = {}
    for s in scen:
        canon_by_pos.setdefault(s["pos_id"], {})[s["tier"]] = s.get("canonical_uci")

    grounded_system = load_system_prompt()
    ordered = sorted(scen, key=lambda r: (r["pos_id"], TIERS.index(r["tier"])))
    if limit:
        ordered = ordered[:limit]

    rows: List[Dict[str, Any]] = []
    sel_cache: Dict[str, dict] = {}
    canon_ok = 0
    for s in ordered:
        bk = B.board_key(s["fen"])
        L = lab_by_board.get(bk)
        if L is None:
            raise SystemExit(f"BLOCKED: no deep label for board of {s['id']} ({bk})")
        tier = s["tier"]

        if bk not in sel_cache:
            sel_cache[bk] = select_tiers_v6(L["sound_pool"], L["maia"], L["engine_best"])
        pick = sel_cache[bk]["picks"][tier]
        # Fidelity guard #1: reconstruction MUST reproduce the committed benchmark label.
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

        sound_ucis = [p["uci"] for p in s.get("sound_pool", []) if p.get("uci")]
        cb = canon_by_pos.get(s["pos_id"], {}).get("beginner")
        ca = canon_by_pos.get(s["pos_id"], {}).get("advanced")
        rows.append({
            "id": s["id"], "pos_id": s["pos_id"], "tier": tier, "fen": s["fen"],
            "phase": s.get("phase"),
            "is_val": bool(s.get("is_val")),
            "canonical_uci": s.get("canonical_uci"),
            "canonical_beginner_uci": cb, "canonical_advanced_uci": ca,
            "engine_best_uci": s.get("engine_best_uci"),
            "sound_ucis": sound_ucis,
            "student_uci": s["student_move"]["uci"],
            "grounded_system": grounded_system, "grounded_user": grounded_user,
        })

    print(f"canonical reproduced: {canon_ok}/{len(rows)}")
    return rows


def _verify_byte_identity(rows: List[Dict[str, Any]]) -> None:
    """Guard #2: the 360 held-out TEST prompts must match stage4_eval_inputs.jsonl byte-for-byte."""
    if not STAGE4_INPUTS.exists():
        print(f"[verify] SKIP byte-identity check ({STAGE4_INPUTS.name} not present)")
        return
    committed = {r["id"]: r for r in _load_jsonl(STAGE4_INPUTS)}
    by_id = {r["id"]: r for r in rows}
    checked = mism = 0
    for cid, cr in committed.items():
        nr = by_id.get(cid)
        if nr is None:
            continue  # only present when running a --limit smoke that drops val rows
        checked += 1
        if nr["grounded_system"] != cr["grounded_system"] or nr["grounded_user"] != cr["grounded_user"]:
            mism += 1
            if mism <= 3:
                print(f"[verify] MISMATCH on {cid}")
    if checked and mism == 0:
        print(f"[verify] OK — {checked}/{len(committed)} committed TEST prompts reproduce byte-for-byte")
    elif mism:
        raise SystemExit(
            f"BLOCKED: {mism}/{checked} grounded prompts differ from the committed "
            f"stage4_eval_inputs.jsonl. The full-803 prompts are NOT byte-identical to the "
            f"published Stage-4 slice; do not upload."
        )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only build the first N scenarios")
    args = ap.parse_args(argv)

    rows = _build_rows(limit=args.limit)
    _verify_byte_identity(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = len({r["pos_id"] for r in rows})
    n_val = sum(1 for r in rows if r["is_val"])
    # distinct-moves denominator: positions whose canonical beginner != advanced
    by_pos: Dict[str, Dict[str, Optional[str]]] = {}
    for r in rows:
        by_pos.setdefault(r["pos_id"], {})[r["tier"]] = r["canonical_uci"]
    diff = sum(1 for cd in by_pos.values()
               if cd.get("beginner") and cd.get("advanced") and cd["beginner"] != cd["advanced"])

    print(f"\nwrote {len(rows)} scenarios ({n_pos} positions; {n_val} held-out TEST) -> {out}")
    print(f"tiers: {dict(Counter(r['tier'] for r in rows))}")
    print(f"positions with canonical beginner!=advanced (distinct denominator): {diff}")
    print(f"output size: {out.stat().st_size/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
