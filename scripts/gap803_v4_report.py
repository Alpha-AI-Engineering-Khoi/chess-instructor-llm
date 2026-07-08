#!/usr/bin/env python3
"""Aggregate the ISOLATED v4 803-eval into RESULTS_V4.md — v4 vs v3 vs untuned-32B.

Runs entirely inside ``data/benchmark_v4/`` (never touches the shared
``data/benchmark_gap803`` files or the shared report docs). Reuses the trusted
scoring PRIMITIVES from ``scripts.gap803_report`` (imported, not edited):
``compute_tier_metrics`` (the moat, re-extracted picks vs ``select_tier_move``),
``compute_objective_metrics`` (fabrication / no-engine-speak / ply / move_sound
via the reused objective scorer), ``compute_council`` (instructiveness), and
``compute_safety`` (blunder-only). It then writes a focused, honest per-metric
comparison table + the balanced score, treating ``ours_v4`` as a local model.

Pipeline (all with BENCH_DIR=data/benchmark_v4):
    ~/.venvs/mlx/bin/python -m scripts.gap803_report merge       # -> generations.jsonl
    ~/.venvs/mlx/bin/python -m scripts.gap803_report objective   # -> objective.jsonl
    ~/.venvs/mlx/bin/python -m scripts.gap803_safety             # -> move_safety.json
    ~/.venvs/mlx/bin/python -m scripts.gap803_v4_council --items 120   # -> council.jsonl
    ~/.venvs/mlx/bin/python -m scripts.gap803_v4_report          # -> RESULTS_V4.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# Point the reused gap803_report module at the isolated v4 dir BEFORE importing it.
os.environ["BENCH_DIR"] = str(_ROOT / "data" / "benchmark_v4")

from src.eval.benchmark import config as bcfg  # noqa: E402
from scripts import gap803_report as R  # noqa: E402

BENCH = Path(os.environ["BENCH_DIR"])
REPORT_MD = Path(os.environ.get("BENCH_REPORT_MD", str(_ROOT / "RESULTS_V4.md")))
LEADERBOARD = BENCH / "leaderboard_v4.json"

MODEL_ORDER: Tuple[str, ...] = (
    "ours_v4", "ours_v3", "ours", "base", "q3_32b",
    "gemma3_27b", "q3_next80b", "llama33_70b", "dsv32", "glm5",
    "mistral3", "kimi25", "dsr1", "gpt", "claude", "gemini",
)
DISPLAY: Dict[str, str] = {
    "ours_v4": "OURS-v4 (Qwen3-32B tuned)",
    "ours_v3": "OURS-v3 (Qwen3-32B tuned)",
    "ours": "OURS-v2 (Qwen3-1.7B tuned)", "base": "BASE (Qwen3-1.7B untuned)",
    "q3_32b": "Qwen3-32B (untuned v3/v4 base)", "q3_next80b": "Qwen3-Next-80B-A3B",
    "gemma3_27b": "Gemma-3-27B-it", "llama33_70b": "Llama-3.3-70B",
    "dsv32": "DeepSeek-V3.2", "glm5": "GLM-5", "mistral3": "Mistral-Large-3 (675B)",
    "kimi25": "Kimi-K2.5", "dsr1": "DeepSeek-R1",
    "gpt": "GPT-5.5", "claude": "Claude Opus 4.8", "gemini": "Gemini 3.1 Pro",
}
LOCAL_SCORE: Dict[str, float] = {
    "ours_v4": 1.0, "ours_v3": 1.0, "ours": 1.0, "base": 1.0, "q3_32b": 1.0,
    "gemma3_27b": 1.0, "q3_next80b": 0.6, "llama33_70b": 0.6,
    "dsv32": 0.0, "glm5": 0.0, "mistral3": 0.0, "kimi25": 0.0, "dsr1": 0.0,
    "gpt": 0.0, "claude": 0.0, "gemini": 0.0,
}


def _blended_cost(mk: str) -> float:
    if mk not in bcfg.MODELS:      # ours_v4 (+ ours_v3) are local -> free
        return 0.0
    m = bcfg.MODELS[mk]
    return float(m.price_in) + float(m.price_out)


def _pct(x: Optional[float]) -> str:
    return f"{100*x:.1f}%" if isinstance(x, (int, float)) else "—"


def _num(x: Optional[float], nd: int = 2) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def build_scores(tier, obj, council, safety) -> Dict[str, Dict[str, Any]]:
    models = [m for m in MODEL_ORDER if m in obj]
    blended = {m: _blended_cost(m) for m in models}
    cmax = max(blended.values()) if blended else 1.0
    cost_score = {m: (1.0 - blended[m] / cmax) if cmax > 0 else 1.0 for m in models}
    scored: Dict[str, Dict[str, Any]] = {}
    for m in models:
        t = tier.get(m, {})
        tvals = [x for x in (t.get("tier_fit_mean"), t.get("diff_rate"), t.get("direction"))
                 if x is not None]
        tier_score = sum(tvals) / len(tvals) if tvals else None
        c = council.get(m, {})
        instr_score = (1.0 - c["norm_rank"]) if c else None
        fab = obj[m].get("fabrication")
        fab_score = (1.0 - fab) if fab is not None else None
        loc = LOCAL_SCORE.get(m, 0.0)
        practical = 0.6 * loc + 0.4 * cost_score.get(m, 0.0)
        safe = safety.get(m)
        noes = obj[m].get("no_engine_speak")
        gate_ok = (safe is None or safe >= R.GATE_SAFE) and (noes is None or noes >= R.GATE_NOES)

        def _wsum(weights: Dict[str, float]) -> Optional[float]:
            parts = {"tier": tier_score, "instr": instr_score, "fab": fab_score,
                     "practical": practical}
            num = den = 0.0
            for k, w in weights.items():
                v = parts[k]
                if v is None:
                    continue
                num += w * v
                den += w
            return (num / den) if den else None

        scored[m] = {
            "tier_score": tier_score, "instr_score": instr_score, "fab_score": fab_score,
            "practical": practical, "gate_ok": gate_ok, "safe": safe, "no_engine_speak": noes,
            "balanced": _wsum(R.W_BALANCED), "base_fit": _wsum(R.W_BASE),
        }
    return scored


def _lead_cleaned_stats() -> Dict[str, Any]:
    """Malformed-output telemetry from the v4 gen (raw vs cleaned + narrow parse)."""
    p = BENCH / "gen" / "ours_v4.jsonl"
    if not p.exists():
        return {}
    n = lead = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        n += 1
        if d.get("lead_cleaned"):
            lead += 1
    return {"n": n, "lead_cleaned": lead, "lead_cleaned_pct": (lead / n if n else 0.0)}


def main() -> int:
    scns_by_id = R._scenarios_by_id()
    tier = R.compute_tier_metrics(scns_by_id)
    obj = R.compute_objective_metrics()
    council = R.compute_council()
    safety = R.compute_safety()
    scored = build_scores(tier, obj, council, safety)
    lead = _lead_cleaned_stats()

    bundle = {"tier": tier, "objective": obj, "council": council, "safety": safety,
              "scored": scored, "lead_cleaned": lead}
    LEADERBOARD.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {LEADERBOARD}")

    _write_markdown(tier, obj, council, safety, scored, lead)
    print(f"wrote {REPORT_MD}")
    return 0


def _row_metrics(mk: str, tier, obj, council, safety, scored) -> Dict[str, Any]:
    t = tier.get(mk, {})
    fit = t.get("tier_fit_by_tier", {}) or {}
    o = obj.get(mk, {})
    c = council.get(mk, {})
    s = scored.get(mk, {})
    return {
        "tier_fit": t.get("tier_fit_mean"),
        "beg": fit.get("beginner"), "int": fit.get("intermediate"), "adv": fit.get("advanced"),
        "council_rank": c.get("mean_rank"), "top1": (c.get("top1_pct") / 100 if c.get("top1_pct") is not None else None),
        "fabrication": o.get("fabrication"), "move_sound": o.get("move_sound"),
        "no_jargon": o.get("no_engine_speak"), "safety": safety.get(mk),
        "balanced": (s.get("balanced") * 100 if s.get("balanced") is not None else None),
    }


def _write_markdown(tier, obj, council, safety, scored, lead) -> None:
    m4 = _row_metrics("ours_v4", tier, obj, council, safety, scored)
    m3 = _row_metrics("ours_v3", tier, obj, council, safety, scored)
    mb = _row_metrics("q3_32b", tier, obj, council, safety, scored)

    def delta(a, b, pts=True, invert=False):
        if a is None or b is None:
            return "—"
        d = (a - b) * (100 if pts else 1)
        better = (d < 0) if invert else (d > 0)
        sign = "+" if d >= 0 else ""
        tag = "better" if better else ("flat" if abs(d) < (0.5 if pts else 0.01) else "worse")
        unit = " pts" if pts else ""
        return f"{sign}{d:.1f}{unit} ({tag})"

    L: List[str] = []
    L.append("# Results — v4 Chess Coach (Qwen3-32B), v3→v4 deltas\n")
    L.append("**v4 is a pure DATA intervention on v3** (identical Qwen3-32B QLoRA recipe, "
             "same canonical `select_tier_move` targets, same contrastive triples). It targets "
             "v3's four measured weak spots without re-paying the teacher:\n")
    L.append("1. **Train/serve prompt MATCH** — v3 trained on `render_user_prompt` only but is "
             "served the full `build_grounded_user` (VERIFIED FACTS + prompt + FORMAT_INSTRUCTION); "
             "v4 trains on the exact served prompt (fixes the ~4–5% malformed leading fragments).\n")
    L.append("2. **Beginner-discriminating oversample** — beginner rows whose canonical pick ≠ the "
             "engine best are upsampled 2× to recover beginner tier-fit (v3 fell to 29.6%).\n")
    L.append("3. **Narrow faithfulness reject + format guard** on every label; the WIDE "
             "`verify_text_ext` was found to over-fire on post-move-consequence coaching "
             "(~>90% false positives on a hand-checked sample), so it is used as a high-recall "
             "flag for the LLM judge, not a hard reject.\n")
    L.append("4. Instructiveness preserved (v3's method-teaching labels; the grounding prefix "
             "gives the student true facts to copy).\n")

    L.append("\n## Headline — OURS-v4 vs OURS-v3 vs untuned Qwen3-32B\n")
    L.append("| Metric | untuned 32B | OURS-v3 | **OURS-v4** | v3→v4 Δ |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(f"| Tier-fit (moat, mean of 3) | {_pct(mb['tier_fit'])} | {_pct(m3['tier_fit'])} | "
             f"**{_pct(m4['tier_fit'])}** | {delta(m4['tier_fit'], m3['tier_fit'])} |")
    L.append(f"| — tier-fit @ beginner | {_pct(mb['beg'])} | {_pct(m3['beg'])} | "
             f"**{_pct(m4['beg'])}** | {delta(m4['beg'], m3['beg'])} |")
    L.append(f"| — tier-fit @ intermediate | {_pct(mb['int'])} | {_pct(m3['int'])} | "
             f"**{_pct(m4['int'])}** | {delta(m4['int'], m3['int'])} |")
    L.append(f"| — tier-fit @ advanced | {_pct(mb['adv'])} | {_pct(m3['adv'])} | "
             f"**{_pct(m4['adv'])}** | {delta(m4['adv'], m3['adv'])} |")
    L.append(f"| Instructiveness (council rank ↓) | {_num(mb['council_rank'])} | {_num(m3['council_rank'])} | "
             f"**{_num(m4['council_rank'])}** | {delta(m4['council_rank'], m3['council_rank'], invert=True)} |")
    L.append(f"| Instructiveness top-1 ↑ | {_pct(mb['top1'])} | {_pct(m3['top1'])} | "
             f"**{_pct(m4['top1'])}** | {delta(m4['top1'], m3['top1'])} |")
    L.append(f"| Fabrication ↓ (narrow verifier) | {_pct(mb['fabrication'])} | {_pct(m3['fabrication'])} | "
             f"**{_pct(m4['fabrication'])}** | {delta(m4['fabrication'], m3['fabrication'], invert=True)} |")
    L.append(f"| Move-sound ↑ | {_pct(mb['move_sound'])} | {_pct(m3['move_sound'])} | "
             f"**{_pct(m4['move_sound'])}** | {delta(m4['move_sound'], m3['move_sound'])} |")
    L.append(f"| No-engine-jargon ↑ | {_pct(mb['no_jargon'])} | {_pct(m3['no_jargon'])} | "
             f"**{_pct(m4['no_jargon'])}** | {delta(m4['no_jargon'], m3['no_jargon'])} |")
    L.append(f"| Move-safety (blunder-free) ↑ | {_pct(mb['safety'])} | {_pct(m3['safety'])} | "
             f"**{_pct(m4['safety'])}** | {delta(m4['safety'], m3['safety'])} |")
    L.append(f"| Balanced score ↑ | {_num(mb['balanced'],1)} | {_num(m3['balanced'],1)} | "
             f"**{_num(m4['balanced'],1)}** | {delta(m4['balanced'], m3['balanced'], pts=False)} |")

    if lead:
        L.append(f"\n**Malformed-output telemetry (v4):** of {lead.get('n')} raw v4 generations, "
                 f"only {lead.get('lead_cleaned')} ({_pct(lead.get('lead_cleaned_pct'))}) needed the "
                 f"leading-garble cleanup — the train/serve match largely removed the v3 failure mode.")

    L.append("\n## Full field (balanced, this isolated v4 run)\n")
    L.append("| Model | tier-fit↑ | council↓ | top1↑ | fab↓ | move-sound↑ | no-jargon↑ | balanced↑ | gate |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for mk in MODEL_ORDER:
        if mk not in obj:
            continue
        r = _row_metrics(mk, tier, obj, council, safety, scored)
        s = scored.get(mk, {})
        gate = "pass" if s.get("gate_ok") else "FAIL"
        star = "**" if mk in ("ours_v4",) else ""
        L.append(f"| {star}{DISPLAY.get(mk, mk)}{star} | {_pct(r['tier_fit'])} | {_num(r['council_rank'])} | "
                 f"{_pct(r['top1'])} | {_pct(r['fabrication'])} | {_pct(r['move_sound'])} | "
                 f"{_pct(r['no_jargon'])} | {_num(r['balanced'],1)} | {gate} |")

    L.append("\n## Notes\n")
    L.append("- Objective (tier-fit / fabrication / move-sound / no-jargon / safety) is over the "
             "**full 803×3** flattened positions; instructiveness is the stratified ~120-item, "
             "16-model blinded cross-family council (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro).")
    L.append("- This uses the existing (stratified) council path in an isolated `data/benchmark_v4` "
             "dir. If the separate full-council harness upgrade is available, prefer it for the "
             "final instructiveness number; the deltas here are within-run apples-to-apples.")
    L.append("- Fabrication is the deterministic NARROW verifier (the metric the benchmark uses); "
             "the cross-family LLM-judge truthfulness residual is reported separately.")

    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
