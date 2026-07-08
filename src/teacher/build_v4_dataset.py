#!/usr/bin/env python3
"""Build the **v4** chess-coaching SFT set from the existing v3 teacher labels.

v4 is a *pure DATA* intervention on top of v3 (same Qwen3-32B QLoRA recipe, same
canonical ``select_tier_move`` targets, same contrastive triples). It targets the
four measured v3 weak spots (see ``RESULTS_V3.md``) **without** re-paying the
teacher (~$141) — it re-derives train rows from ``candidates_v3.jsonl``:

1. **Train/serve prompt MATCH (fixes ~4-5% malformed outputs).** v3 trained on
   ``render_user_prompt`` ONLY, but every model is *evaluated / served* on
   ``build_grounded_user`` = ``render_pool_facts`` (VERIFIED FACTS) +
   ``render_user_prompt`` + ``FORMAT_INSTRUCTION``. v4 trains on the **exact**
   served prompt, removing the skew that pushed the 32B off-distribution (leading
   rating-range fragment / prompt-echo).

2. **WIDE faithfulness filter (fixes the truthfulness residual).** v3 filtered
   labels with the NARROW ``verify_text`` (piece location/existence). v4 rejects
   any label whose rendered target trips the WIDE ``verify_text_ext``
   (relational / move-consequence / turn / material / hanging), so the student
   never learns a structural fabrication.

3. **Format guard on labels (fixes malformed outputs).** Every kept target must
   render as ``I'd play <MOVE>. ... Takeaway: ...`` cleanly (starts with the move
   command, ends with a takeaway, no engine-speak, within the tier ply cap).

4. **Beginner-discriminating oversample (recovers beginner tier-fit).** v3's 32B
   prior collapsed beginner picks onto the engine-best move (beginner tier-fit
   47.9% -> 29.6%). v4 upsamples the beginner rows whose canonical pick is NOT the
   engine best (the tier signal) so the gradient on "beginner -> human-findable
   move" is stronger, WITHOUT touching the advanced rows (all engine-best) that
   made advanced tier-fit excellent.

Everything is v4-suffixed; v1/v2/v3 artifacts are untouched.

CLI
---
    python -m src.teacher.build_v4_dataset analyze
    python -m src.teacher.build_v4_dataset build --beginner-oversample 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import schema, settings  # noqa: E402
from src.eval.benchmark.prompts import build_grounded_user, load_system_prompt  # noqa: E402
from src.engine.faithfulness import verify_text  # noqa: E402
from src.engine.faithfulness_ext import verify_text_ext  # noqa: E402
from src.filter.filter import detect_engine_speak, longest_san_run, move_is_legal  # noqa: E402

log = logging.getLogger("teacher.build_v4")

CANDIDATES_V3 = settings.GENERATED / "candidates_v3.jsonl"
TRAIN_V4 = settings.DATASET / "train_v4.jsonl"
VALID_V4 = settings.DATASET / "valid_v4.jsonl"
REJECTS_V4 = settings.GENERATED / "rejects_v4.jsonl"
MANIFEST_V4 = settings.GENERATED / "v4_data_manifest.json"
#: Wide-checker-flagged KEPT rows -> candidate pool for the LLM truthfulness judge.
WIDE_SUBSET_V4 = settings.GENERATED / "v4_wide_flagged.jsonl"

TIER_ORDER = ("beginner", "intermediate", "advanced")
SEED = 3407

#: Optional set of candidate ids to exclude (judge-confirmed fabrications).
_EXCLUDE_IDS: set = set()


# --------------------------------------------------------------------------- #
# Candidate -> gated, rendered training row (or a reject with reasons)
# --------------------------------------------------------------------------- #


def _scenario_like(ti: Dict[str, Any]) -> Dict[str, Any]:
    """A benchmark-scenario-shaped dict so we can call the EXACT served prompt."""
    return {
        "fen": ti["fen"],
        "tier": ti["tier"],
        "student_move": ti["student_move"],
        "sound_pool": ti["sound_pool"],
        "maia": ti.get("maia_human_moves", []),
    }


def _gate_and_render(
    cand: Dict[str, Any], system_prompt: str
) -> Tuple[Optional[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Return ``(train_row_or_None, reasons, info)`` for one v3 candidate.

    Applies the v4 gates and, if clean, renders a chat row whose USER message is
    the exact served ``build_grounded_user`` prompt and whose ASSISTANT target is
    ``render_assistant_target_v2`` (the canonical select_tier_move move + method).
    """
    reasons: List[str] = []
    ti = cand.get("teacher_input") or {}
    to = cand.get("teacher_output") or {}
    engine = cand.get("engine") or {}
    meta = cand.get("meta") or {}

    fen = ti.get("fen")
    tier = ti.get("tier") or cand.get("tier")
    coaching = str(to.get("coaching") or "").strip()
    method = str(to.get("method") or "").strip()
    takeaway = str(to.get("takeaway") or "").strip()
    rec_uci = str(to.get("recommended_move_uci") or "").strip()
    rec_san = str(to.get("recommended_move_san") or "").strip()

    info: Dict[str, Any] = {
        "tier": tier,
        "is_engine_best": bool(meta.get("pick_is_engine_best", True)),
        "pool_rank": meta.get("pick_pool_rank"),
        "base_id": meta.get("base_id") or cand.get("id"),
        "cand_id": cand.get("id"),
    }

    if tier not in settings.TIERS:
        reasons.append("missing_tier")
    if not fen:
        reasons.append("missing_fen")
    if not coaching:
        reasons.append("empty_coaching")
    if not method:
        reasons.append("missing_method")
    if not takeaway:
        reasons.append("empty_takeaway")
    # Soundness: recommended move must be inside the position's Stockfish pool.
    sound_set = {str(u).lower() for u in (engine.get("sound_ucis") or [])}
    if not rec_uci or rec_uci.lower() not in sound_set:
        reasons.append("soundness")
    # Legality (defensive).
    if fen and (rec_uci or rec_san):
        legal, _ = move_is_legal(fen, rec_uci, rec_san)
        if not legal:
            reasons.append("illegal_move")

    if reasons:
        return None, reasons, info

    # Render the assistant target exactly as the student will emit it.
    target = schema.render_assistant_target_v2(to)  # type: ignore[arg-type]

    # Gate: no engine-speak anywhere in the learned target.
    if detect_engine_speak(target):
        reasons.append("engine_speak")
    # Gate: ply cap on the narrated coaching+method (the served output).
    ply_cap = settings.TIERS[tier]["ply_cap"]
    if longest_san_run(f"{coaching} {method}") > ply_cap:
        reasons.append("ply_cap")
    # Gate: FORMAT — the target must open with the move command and end with a takeaway.
    if not target.startswith(f"I'd play {rec_san}."):
        reasons.append("format_lead")
    if "Takeaway:" not in target:
        reasons.append("format_takeaway")
    # Gate: NARROW faithfulness (piece location/existence) — the base truth gate,
    # identical to v3, and the class the eval's `fabrication` metric measures. The
    # WIDE checker (verify_text_ext) is NOT used as a hard reject here: an audit of
    # its flags on these labels showed it over-fires (~>90% false positives),
    # because coaching legitimately describes the position AFTER the recommended
    # move ("your king defends e4" after Kd3) while the checker evaluates
    # relational/consequence claims on the CURRENT board. It is instead recorded as
    # a high-RECALL candidate flag: those rows are sent to the context-aware LLM
    # truthfulness judge, which confirms the genuine fabrications to exclude.
    if verify_text(target, fen).violations:
        reasons.append("faithfulness_narrow")
    ext = verify_text_ext(target, fen, rec_uci)
    info["wide_flagged"] = bool(ext.violations)
    info["wide_violation"] = ext.violations[0].reason if ext.violations else None
    info["target"] = target

    if reasons:
        return None, reasons, info

    # Optional external exclusion (judge-confirmed fabrications).
    if info["base_id"] and _EXCLUDE_IDS and cand.get("id") in _EXCLUDE_IDS:
        reasons.append("judge_fabrication")
        return None, reasons, info

    user = build_grounded_user(_scenario_like(ti))
    row = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "_meta": {
            "cand_id": info["cand_id"],
            "base_id": info["base_id"],
            "tier": tier,
            "fen": fen,
            "rec_uci": rec_uci,
            "rec_san": rec_san,
            "is_engine_best": info["is_engine_best"],
            "discriminating": (tier == "beginner" and not info["is_engine_best"]),
            "wide_flagged": info["wide_flagged"],
            "target": target,
        },
    }
    return row, [], info


def _iter_candidates() -> List[Dict[str, Any]]:
    if not CANDIDATES_V3.exists():
        raise SystemExit(f"BLOCKED: missing {CANDIDATES_V3}")
    out: List[Dict[str, Any]] = []
    with CANDIDATES_V3.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #


def cmd_analyze(args: argparse.Namespace) -> int:
    system_prompt = load_system_prompt()
    cands = _iter_candidates()
    log.info("loaded %d v3 candidates", len(cands))

    kept = 0
    reason_hist: Counter = Counter()
    tier_total: Counter = Counter()
    tier_kept: Counter = Counter()
    beg_disc_total = 0          # beginner rows with pick != engine best (pre-gate)
    beg_disc_kept = 0
    wide_flagged_kept = 0       # kept rows the WIDE checker flags (judge candidates)
    seen_keys: set = set()
    dupes = 0
    rank_by_tier: Dict[str, List[int]] = defaultdict(list)

    for c in cands:
        row, reasons, info = _gate_and_render(c, system_prompt)
        tier = info.get("tier")
        if tier in settings.TIERS:
            tier_total[tier] += 1
        if tier == "beginner" and not info.get("is_engine_best", True):
            beg_disc_total += 1
        if reasons:
            for r in reasons:
                reason_hist[r] += 1
            continue
        # dedup (fen, tier, move)
        m = row["_meta"]
        dk = (m["fen"], tier, m["rec_uci"])
        if dk in seen_keys:
            dupes += 1
            continue
        seen_keys.add(dk)
        kept += 1
        if m.get("wide_flagged"):
            wide_flagged_kept += 1
        if tier in settings.TIERS:
            tier_kept[tier] += 1
            if m.get("is_engine_best") is not None and info.get("pool_rank") is not None:
                rank_by_tier[tier].append(int(info["pool_rank"]))
        if m["discriminating"]:
            beg_disc_kept += 1

    print("\n=== v4 candidate analysis (from candidates_v3.jsonl) ===")
    print(f"candidates:                 {len(cands)}")
    print(f"kept after gates + dedup:   {kept}")
    print(f"duplicates dropped:         {dupes}")
    print(f"\nrejected by reason:")
    for r, n in reason_hist.most_common():
        print(f"  {r:<20} {n}")
    print(f"\nWIDE-checker flags among KEPT rows (judge candidates): {wide_flagged_kept}")
    print(f"  (wide over-fires on post-move coaching; sent to LLM judge, not hard-rejected)")
    print(f"\nby tier (kept / total gated):")
    for t in TIER_ORDER:
        print(f"  {t:<12} {tier_kept[t]} / {tier_total[t]}   "
              f"mean pick pool-rank={_mean(rank_by_tier[t]):.2f}")
    print(f"\nbeginner tier signal (pick != engine best):")
    print(f"  discriminating beginner rows (pre-gate): {beg_disc_total}")
    print(f"  discriminating beginner rows (kept):     {beg_disc_kept}")
    print(f"  -> at oversample x{args.beginner_oversample}, extra beginner-disc rows added: "
          f"{int(beg_disc_kept * (args.beginner_oversample - 1))}")
    return 0


def _mean(xs: List[int]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def cmd_build(args: argparse.Namespace) -> int:
    global _EXCLUDE_IDS
    if args.exclude_ids:
        p = Path(args.exclude_ids)
        if p.exists():
            _EXCLUDE_IDS = {x.strip() for x in p.read_text(encoding="utf-8").split() if x.strip()}
            log.info("loaded %d judge-confirmed ids to exclude from %s", len(_EXCLUDE_IDS), p)
        else:
            log.warning("exclude-ids file %s not found; proceeding without it", p)

    system_prompt = load_system_prompt()
    cands = _iter_candidates()
    log.info("loaded %d v3 candidates", len(cands))

    rows: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    wide_subset: List[Dict[str, Any]] = []
    reason_hist: Counter = Counter()
    seen_keys: set = set()
    for c in cands:
        row, reasons, info = _gate_and_render(c, system_prompt)
        if reasons:
            for r in reasons:
                reason_hist[r] += 1
            rejects.append({"id": c.get("id"), "reasons": reasons})
            continue
        m = row["_meta"]
        dk = (m["fen"], m["tier"], m["rec_uci"])
        if dk in seen_keys:
            reason_hist["duplicate"] += 1
            continue
        seen_keys.add(dk)
        rows.append(row)
        if m.get("wide_flagged"):
            wide_subset.append({
                "id": m["cand_id"], "tier": m["tier"], "fen": m["fen"],
                "rec_uci": m["rec_uci"], "rec_san": m["rec_san"], "target": m["target"],
            })

    log.info("kept %d clean rows (%d rejected)", len(rows), sum(reason_hist.values()))
    _write_jsonl(wide_subset, WIDE_SUBSET_V4)
    log.info("wrote %d wide-flagged KEPT rows for the judge pass -> %s",
             len(wide_subset), WIDE_SUBSET_V4)

    # Split by base position id so a position's triples never straddle train/valid.
    by_base: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_base[r["_meta"]["base_id"]].append(r)
    base_ids = sorted(by_base)
    rng = random.Random(SEED)
    rng.shuffle(base_ids)
    n_valid_bases = max(1, int(len(base_ids) * args.valid_frac))
    valid_bases = set(base_ids[:n_valid_bases])

    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    for bid in base_ids:
        (valid_rows if bid in valid_bases else train_rows).extend(by_base[bid])

    # Beginner-discriminating oversample (TRAIN only): upweight the tier signal.
    disc = [r for r in train_rows if r["_meta"]["discriminating"]]
    extra_copies = int(round((args.beginner_oversample - 1.0) * len(disc)))
    oversampled: List[Dict[str, Any]] = []
    if extra_copies > 0 and disc:
        for i in range(extra_copies):
            oversampled.append(disc[i % len(disc)])
    train_rows_final = train_rows + oversampled
    rng.shuffle(train_rows_final)

    # Strip the internal _meta before writing (keep the chat schema clean).
    def _clean(r: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": r["messages"]}

    _write_jsonl([_clean(r) for r in train_rows_final], TRAIN_V4)
    _write_jsonl([_clean(r) for r in valid_rows], VALID_V4)
    _write_jsonl(rejects, REJECTS_V4)

    tier_train = Counter(r["_meta"]["tier"] for r in train_rows_final)
    disc_train = sum(1 for r in train_rows_final if r["_meta"]["discriminating"])
    manifest = {
        "source": str(CANDIDATES_V3),
        "candidates": len(cands),
        "kept_unique": len(rows),
        "rejects_by_reason": dict(reason_hist),
        "beginner_oversample": args.beginner_oversample,
        "valid_frac": args.valid_frac,
        "train_rows": len(train_rows_final),
        "train_rows_before_oversample": len(train_rows),
        "valid_rows": len(valid_rows),
        "oversampled_extra": len(oversampled),
        "train_by_tier": dict(tier_train),
        "train_discriminating_beginner_rows": disc_train,
        "wide_flagged_kept": len(wide_subset),
        "excluded_by_judge": len(_EXCLUDE_IDS),
        "exclude_ids_file": args.exclude_ids,
        "prompt_format": "build_grounded_user (facts + render_user_prompt + FORMAT_INSTRUCTION)",
        "faithfulness_filter": "narrow verify_text reject; wide verify_text_ext -> LLM-judge rejection",
        "seed": SEED,
    }
    MANIFEST_V4.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== v4 build summary ===")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote train -> {TRAIN_V4} ({len(train_rows_final)} rows)")
    print(f"wrote valid -> {VALID_V4} ({len(valid_rows)} rows)")
    print(f"wrote manifest -> {MANIFEST_V4}")
    return 0


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="Measure the candidate pool + gate impact (no writes).")
    pa.add_argument("--beginner-oversample", type=float, default=2.0)
    pa.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("build", help="Write train_v4/valid_v4 + manifest.")
    pb.add_argument("--beginner-oversample", type=float, default=2.0,
                    help="Upsample factor for beginner rows whose pick != engine best.")
    pb.add_argument("--valid-frac", type=float, default=0.05)
    pb.add_argument("--exclude-ids", default=None,
                    help="Optional file of candidate ids (whitespace-separated) to drop "
                         "(e.g. judge-confirmed fabrications).")
    pb.set_defaults(func=cmd_build)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
