"""Deterministic, cloud-native data-improvement recipes for the 4B loop.

The autonomous loop's "improve the DATA" step, WITHOUT an agent and WITHOUT the
Mac: each recipe is a pure, reproducible re-curation of the Stockfish-verified v3
teacher candidates (``candidates_v3.jsonl``, seeded onto the ``chess-coach-lora``
Volume). It REUSES ``src.teacher.build_4b_dataset``'s row-level gate + v5 render
(the correctness-critical part — soundness, legality, no engine-speak, ply cap,
narrow faithfulness, principle-in-takeaway) and only varies the training MIX to
attack the exact weakness the previous eval reported:

* the "distinct-moves-per-level" moat — up-weight genuinely contrastive
  full-gradient (B!=I!=A) triads + beginner-discriminating rows, and drop more
  non-differentiating all-same boards;
* tier-fit — up-weight beginner-discriminating rows harder;
* principle-naming — keep only rows whose takeaway NATIVELY names a transferable
  principle (no auto-augmentation).

This is deliberately deterministic + robust ("deterministic + cloud-native beats
clever-but-fragile"): no fresh teacher generation is required in the running loop,
so the data step can never stall it. ``pick_recipe`` turns an eval report into the
next recipe. Everything imports cleanly off-Mac (build_4b_dataset's deps are pure
python-chess), so this runs inside a Modal CPU function.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Recipe knobs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Recipe:
    """A deterministic curation mix over the gated v3 candidate rows."""

    name: str
    allsame_drop: float = 0.5          # fraction of non-differentiating boards to drop
    full_triad_boost: int = 1          # extra copies of B!=I!=A contrastive triads
    discriminating_boost: int = 1      # extra copies of beginner pick != engine best
    drop_collapse_intermediate: bool = True   # drop the B==A!=I blend artifact rows
    require_native_principle: bool = False    # keep only non-augmented takeaways
    valid_frac: float = 0.05
    note: str = ""


RECIPES: Dict[str, Recipe] = {
    # Reproduce the iter-1 curation exactly (baseline / fallback).
    "v5_iter1": Recipe("v5_iter1", 0.5, 1, 1, True, False, 0.05,
                       "iter-1 baseline curation"),
    # iter-2: press the moat — stronger anti-collapse + contrastive up-weighting.
    "v5_moat": Recipe("v5_moat", 0.7, 2, 2, True, False, 0.05,
                      "stronger distinct-moves-per-level pressure (iter-2 top-up goal)"),
    # report-driven (iter>=3):
    "boost_distinct": Recipe("boost_distinct", 0.85, 3, 2, True, False, 0.05,
                             "max anti-collapse: crush B==A on differentiating positions"),
    "boost_tierfit": Recipe("boost_tierfit", 0.5, 2, 3, True, False, 0.05,
                            "up-weight beginner-discriminating rows for tier-fit"),
    "boost_principle": Recipe("boost_principle", 0.5, 1, 1, True, True, 0.05,
                              "keep only natively-principled takeaways"),
}


def pick_recipe(report: Optional[Dict[str, Any]]) -> str:
    """Deterministically choose the next recipe from the previous eval report.

    Reads ``ours_4b``'s reported weaknesses vs the completion criteria and targets
    the biggest gap. Falls back to pressing the moat when nothing is reported.
    """
    if not report:
        return "v5_moat"
    try:
        h = report["headline"]
        e = h["E_distinct_moves_per_level"].get("ours_4b", {})
        distinct = e.get("distinct_rate")
        tf = (report["per_model"].get("ours_4b") or {}).get("tier_fit")
        dims = (report["per_model"].get("ours_4b") or {}).get("coherence")  # noqa: F841
        principle = None
        pm = report["per_model"].get("ours_4b") or {}
        # transferable_principle dim (0-2) if present in per_model dims.
        principle = ((pm.get("dims") or {}) if isinstance(pm.get("dims"), dict) else {}).get(
            "transferable_principle")
    except Exception:  # noqa: BLE001 - a malformed report must not stall the loop
        return "v5_moat"

    if distinct is not None and distinct < 0.95:
        return "boost_distinct"
    if tf is not None and tf < 0.60:
        return "boost_tierfit"
    if principle is not None and principle < 1.5:
        return "boost_principle"
    return "boost_distinct"


def recipe_for_iter(iter_n: int, prev_report: Optional[Dict[str, Any]]) -> str:
    """The recipe queue: iter-1 baseline, iter-2 moat, then report-driven."""
    if iter_n <= 1:
        return "v5_iter1"
    if iter_n == 2:
        return "v5_moat"
    return pick_recipe(prev_report)


# --------------------------------------------------------------------------- #
# Build a dataset for a recipe (reuses build_4b_dataset's pure gate/render)
# --------------------------------------------------------------------------- #


def _drop_allsame(base_id: str, frac: float) -> bool:
    h = int(hashlib.sha256(base_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def build_dataset(
    recipe_name: str,
    candidates_path: str,
    out_train_path: str,
    out_valid_path: str,
    *,
    seed: int = 3407,
) -> Dict[str, Any]:
    """Curate ``candidates_v3`` into ``train``/``valid`` jsonl for ``recipe_name``.

    Returns a manifest. Reuses ``build_4b_dataset``'s per-row gate + v5 render
    verbatim; only the MIX (copies / drops / principle filter) varies by recipe.
    """
    import src.teacher.build_4b_dataset as B

    recipe = RECIPES.get(recipe_name)
    if recipe is None:
        raise ValueError(f"unknown recipe {recipe_name!r}")

    # Point the (module-constant) candidate source at the volume copy, in-process
    # only (never edits the file); then reuse the exact gate/render pipeline.
    B.CANDIDATES_V3 = Path(candidates_path)
    fams = B._principle_families()
    rows, reason_hist, picks_by_base, n_cands = B._gather(fams)
    board_class = {b: B._board_class(p) for b, p in picks_by_base.items()}

    by_base: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_base[r["_meta"]["base_id"]].append(r)
    base_ids = sorted(by_base)
    rng = random.Random(seed)
    rng.shuffle(base_ids)
    n_valid = max(1, int(len(base_ids) * recipe.valid_frac))
    valid_bases = set(base_ids[:n_valid])

    dropped_collapse = dropped_allsame = dropped_augmented = 0
    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []

    for bid in base_ids:
        cls = board_class.get(bid)
        is_valid = bid in valid_bases
        for r in by_base[bid]:
            m = r["_meta"]
            tier = m["tier"]
            if recipe.drop_collapse_intermediate and cls == "collapse_BA" and tier == "intermediate":
                dropped_collapse += 1
                continue
            if recipe.require_native_principle and m.get("augmented_takeaway"):
                dropped_augmented += 1
                continue
            if is_valid:
                valid_rows.append(r)
                continue
            if cls == "all_same" and _drop_allsame(bid, recipe.allsame_drop):
                dropped_allsame += 1
                continue
            copies = 1
            if cls == "full":
                copies += recipe.full_triad_boost
            if m["discriminating"]:
                copies += recipe.discriminating_boost
            train_rows.extend([r] * copies)

    rng.shuffle(train_rows)

    def _clean(r: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": r["messages"]}

    _write_jsonl([_clean(r) for r in train_rows], Path(out_train_path))
    _write_jsonl([_clean(r) for r in valid_rows], Path(out_valid_path))

    manifest = {
        "recipe": recipe.name,
        "recipe_note": recipe.note,
        "recipe_params": {
            "allsame_drop": recipe.allsame_drop,
            "full_triad_boost": recipe.full_triad_boost,
            "discriminating_boost": recipe.discriminating_boost,
            "drop_collapse_intermediate": recipe.drop_collapse_intermediate,
            "require_native_principle": recipe.require_native_principle,
            "valid_frac": recipe.valid_frac,
        },
        "candidates": n_cands,
        "kept_unique_rows": len(rows),
        "board_coherence": dict(Counter(v for v in board_class.values() if v)),
        "dropped_collapse_BA_intermediate": dropped_collapse,
        "dropped_allsame_downweight": dropped_allsame,
        "dropped_non_native_principle": dropped_augmented,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_by_tier": dict(Counter(r["_meta"]["tier"] for r in train_rows)),
        "train_discriminating": sum(1 for r in train_rows if r["_meta"]["discriminating"]),
        "seed": seed,
        "out_train": out_train_path,
        "out_valid": out_valid_path,
    }
    return manifest


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
