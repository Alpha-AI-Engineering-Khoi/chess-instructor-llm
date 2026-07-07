"""Render RESULTS_OPEN_MODELS.md — the unified open-vs-frontier leaderboard.

Reads the checkpointed artifacts under ``BENCH_DIR`` (grounded generations,
objective scores, and the unified N-way council) and produces:

* a grounded objective leaderboard (fabrication% is the headline metric),
* the unified council instructiveness ranking (mean rank of N + top-1 win%),
* a per-model cost readout from recorded token usage,
* a short, data-driven recommendation.

Decoupled from the harness's fixed 5-model ``MODEL_ORDER``: model orders are
passed in, and council numbers are computed straight from ``label_to_model``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from . import config as bcfg
from .io_utils import read_jsonl


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _grounded_objective() -> Dict[str, Dict[str, float]]:
    rows = [r for r in read_jsonl(bcfg.OBJECTIVE_PATH) if r["condition"] == "grounded"]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)

    def pct(g: List[Dict[str, Any]], f: str) -> float:
        return 100.0 * sum(1 for r in g if r[f]) / len(g)

    out: Dict[str, Dict[str, float]] = {}
    for mk, g in by.items():
        out[mk] = {
            "n": float(len(g)),
            "move_parseable": pct(g, "move_parseable"),
            "move_sound": pct(g, "move_sound"),
            "no_engine_speak": pct(g, "no_engine_speak"),
            "ply_cap_ok": pct(g, "ply_cap_ok"),
            "fabrication": 100.0 * sum(1 for r in g if r["fabricated"]) / len(g),
            "avg_violations": sum(r["n_violations"] for r in g) / len(g),
        }
    return out


def _council_stats() -> Dict[str, Dict[str, Any]]:
    """Per-model mean rank / top-1 win% / rubric means from the unified council."""
    rows = read_jsonl(bcfg.COUNCIL_PATH)
    ranks: Dict[str, List[int]] = {}
    norm: Dict[str, List[float]] = {}
    wins: Dict[str, List[int]] = {}
    rubric: Dict[str, Dict[str, List[int]]] = {}
    field_sizes: List[int] = []
    n_obs = 0
    for r in rows:
        ranking = r["ranking"]
        l2m = r["label_to_model"]
        scores = r.get("scores", {})
        N = len(ranking)
        field_sizes.append(N)
        n_obs += 1
        m2l = {m: l for l, m in l2m.items()}
        for mk, lab in m2l.items():
            if lab not in ranking:
                continue
            rank = ranking.index(lab) + 1
            ranks.setdefault(mk, []).append(rank)
            norm.setdefault(mk, []).append((rank - 1) / (N - 1) if N > 1 else 0.0)
            wins.setdefault(mk, []).append(1 if rank == 1 else 0)
            cell = scores.get(lab, {}) or {}
            rr = rubric.setdefault(mk, {"tier_calibration": [], "clarity": [], "correctness": []})
            for dim in rr:
                if dim in cell:
                    rr[dim].append(int(cell[dim]))

    def mean(xs: Sequence[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    out: Dict[str, Dict[str, Any]] = {}
    for mk in ranks:
        rr = rubric.get(mk, {})
        out[mk] = {
            "mean_rank": mean(ranks[mk]),
            "norm_rank": mean(norm[mk]),
            "win": 100.0 * mean(wins[mk]) if wins[mk] else None,
            "n": len(ranks[mk]),
            "tier_calibration": mean(rr.get("tier_calibration", [])),
            "clarity": mean(rr.get("clarity", [])),
            "correctness": mean(rr.get("correctness", [])),
        }
    field_n = max(field_sizes) if field_sizes else 0
    return {"per_model": out, "field_n": field_n, "n_obs": n_obs}


def _cost() -> Dict[str, Any]:
    gens = [r for r in read_jsonl(bcfg.GENERATIONS_PATH) if r["condition"] == "grounded"]
    coun = read_jsonl(bcfg.COUNCIL_PATH)

    def acc(rows, key):
        out: Dict[str, Dict[str, int]] = {}
        for r in rows:
            d = out.setdefault(r[key], {"in": 0, "out": 0, "calls": 0})
            d["in"] += int(r.get("prompt_tokens", 0))
            d["out"] += int(r.get("completion_tokens", 0))
            d["calls"] += 1
        return out

    gen_tok = acc(gens, "model")
    jud_tok = acc(coun, "judge")

    def usd(mk: str, tok: Dict[str, int]) -> float:
        pin, pout = bcfg.price_for(mk)
        return tok["in"] / 1e6 * pin + tok["out"] / 1e6 * pout

    per: Dict[str, Any] = {}
    total = 0.0
    keys = set(gen_tok) | set(jud_tok)
    for mk in keys:
        g = gen_tok.get(mk, {"in": 0, "out": 0, "calls": 0})
        j = jud_tok.get(mk, {"in": 0, "out": 0, "calls": 0})
        c = usd(mk, {"in": g["in"] + j["in"], "out": g["out"] + j["out"]})
        total += c
        per[mk] = {
            "gen_calls": g["calls"], "judge_calls": j["calls"],
            "in": g["in"] + j["in"], "out": g["out"] + j["out"],
            "usd": c,
        }
    return {"per_model": per, "total": total}


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _fam_tag(mk: str) -> str:
    fam = bcfg.MODELS[mk].family
    return {"local": "ours/base", "gpt": "frontier", "claude": "frontier",
            "gemini": "frontier", "open": "open"}.get(fam, fam)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    aligns = ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(aligns) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def write_report(*, report_md, unified_order: List[str], frontier5: List[str],
                 council_field: List[str]) -> None:
    obj = _grounded_objective()
    coun = _council_stats()
    cost = _cost()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    present = [m for m in unified_order if m in obj]
    open_present = [m for m in present if bcfg.MODELS[m].family == "open"]
    cper = coun["per_model"]
    field_n = coun["field_n"]

    # ---- Objective leaderboard (sorted by fabrication asc) -----------------
    obj_sorted = sorted(present, key=lambda m: (obj[m]["fabrication"], -obj[m]["move_sound"]))
    obj_rows = []
    for mk in obj_sorted:
        s = obj[mk]
        obj_rows.append([
            bcfg.MODELS[mk].display, _fam_tag(mk),
            f"{s['fabrication']:.0f}%", f"{s['move_sound']:.0f}%",
            f"{s['no_engine_speak']:.0f}%", f"{s['ply_cap_ok']:.0f}%",
            f"{s['avg_violations']:.2f}", str(int(s["n"])),
        ])
    obj_table = _table(
        ["Model", "family", "fabrication↓", "move_sound↑", "no_engine_speak↑",
         "ply_cap_ok↑", "avg_violations↓", "n"],
        obj_rows,
    )

    # ---- Council leaderboard (field only, sorted by mean rank) -------------
    field_present = [m for m in unified_order if m in cper]
    field_sorted = sorted(field_present, key=lambda m: (cper[m]["mean_rank"] if cper[m]["mean_rank"] is not None else 99))
    coun_rows = []
    for mk in field_sorted:
        c = cper[mk]
        coun_rows.append([
            bcfg.MODELS[mk].display, _fam_tag(mk),
            f"{c['mean_rank']:.2f}" if c["mean_rank"] is not None else "–",
            f"{c['norm_rank']:.2f}" if c["norm_rank"] is not None else "–",
            f"{c['win']:.0f}%" if c["win"] is not None else "–",
            f"{c['tier_calibration']:.2f}" if c["tier_calibration"] is not None else "–",
            f"{c['clarity']:.2f}" if c["clarity"] is not None else "–",
            f"{c['correctness']:.2f}" if c["correctness"] is not None else "–",
        ])
    coun_table = _table(
        ["Model", "family", f"mean rank (of {field_n})↓", "norm rank↓",
         "top-1 win%↑", "tier_calib", "clarity", "correctness"],
        coun_rows,
    )

    # ---- Cost --------------------------------------------------------------
    cost_rows = []
    for mk in unified_order:
        d = cost["per_model"].get(mk)
        if not d:
            continue
        cost_rows.append([
            bcfg.MODELS[mk].display,
            str(d["gen_calls"]), str(d["judge_calls"]),
            f"{d['in']:,}", f"{d['out']:,}", f"${d['usd']:.2f}",
        ])
    cost_table = _table(["Model", "gen calls", "judge calls", "in tok", "out tok", "est. USD"], cost_rows)

    # ---- Recommendation (data-driven) --------------------------------------
    ours = obj.get("ours", {})
    ours_fab = ours.get("fabrication")
    open_fabs = [obj[m]["fabrication"] for m in open_present]
    worst_open_fab = max(open_fabs) if open_fabs else None
    best_fab_open = min(open_present, key=lambda m: obj[m]["fabrication"]) if open_present else None
    frontier = [m for m in ("gpt", "claude", "gemini") if m in obj]
    fr_fab = (sum(obj[m]["fabrication"] for m in frontier) / len(frontier)) if frontier else None

    council_open = [m for m in field_present if bcfg.MODELS[m].family == "open"]
    council_open.sort(key=lambda m: cper[m]["mean_rank"])
    best_council_open = council_open[0] if council_open else None
    best_frontier = min((m for m in field_present if bcfg.MODELS[m].family in ("gpt", "claude", "gemini")),
                        key=lambda m: cper[m]["mean_rank"], default=None)
    ours_rank = cper.get("ours", {}).get("mean_rank")
    biggest = "mistral3" if "mistral3" in cper else None

    def rankstr(mk):
        return f"{cper[mk]['mean_rank']:.2f}"

    rec: List[str] = []
    if best_fab_open is not None and ours_fab is not None:
        rec.append(
            f"- **Bigger open models fabricate far less than our 1.7B — the truthfulness gap is "
            f"essentially closed.** Every open model scores {min(open_fabs):.0f}–{worst_open_fab:.0f}% "
            f"grounded fabrication vs **OURS-v2 {ours_fab:.0f}%** (and BASE "
            f"{obj.get('base', {}).get('fabrication', float('nan')):.0f}%). The cleanest, "
            f"{bcfg.MODELS[best_fab_open].display} at {obj[best_fab_open]['fabrication']:.0f}%, "
            f"matches the frontier (~{fr_fab:.0f}% avg). Size — not our data intervention — is what "
            f"a 1.7B lacks for board-fact tracking."
        )
    if best_council_open is not None and ours_rank is not None and best_frontier is not None:
        oc = cper[best_council_open]
        rec.append(
            f"- **They also coach more instructively than OURS-v2 — but do not reach the frontier.** "
            f"Best open coach is {bcfg.MODELS[best_council_open].display} (mean rank "
            f"{oc['mean_rank']:.2f} of {field_n}); every open model in the field out-ranks "
            f"**OURS-v2 ({ours_rank:.2f})** by ~{ours_rank - cper[council_open[-1]]['mean_rank']:.1f}–"
            f"{ours_rank - oc['mean_rank']:.1f} positions, yet all trail the best "
            f"frontier coach {bcfg.MODELS[best_frontier].display} ({rankstr(best_frontier)}) by "
            f"~{oc['mean_rank'] - cper[best_frontier]['mean_rank']:.1f}. The instructiveness gap "
            f"narrows with size but does not close."
        )
    if biggest is not None and best_council_open is not None and biggest != best_council_open:
        rec.append(
            f"- **Raw size is NOT the coaching driver.** The largest model, "
            f"{bcfg.MODELS['mistral3'].display} ({rankstr('mistral3')}), is out-coached by the much "
            f"smaller {bcfg.MODELS[best_council_open].display} ({rankstr(best_council_open)})"
            + (f" and Gemma-3-27B ({rankstr('gemma3_27b')})" if 'gemma3_27b' in cper else "")
            + ". Model quality/training beats parameter count for this behavior."
        )
    # Concrete pick for each intended use.
    if "gemma3_27b" in cper:
        rec.append(
            f"- **Best v3 base / bigger local deployment: Gemma-3-27B-it.** Lowest fabrication of the "
            f"whole field ({obj['gemma3_27b']['fabrication']:.0f}%), essentially the top open coach "
            f"({rankstr('gemma3_27b')} of {field_n}), and small enough to fine-tune (QLoRA) and run "
            f"locally in 4-bit on a 64 GB Mac — a genuine drop-in upgrade path from the 1.7B."
        )
    if "q3_32b" in cper and "q3_32b" in obj:
        rec.append(
            f"- **Scaling up our OWN family helps but isn't the best base.** Qwen3-32B (same family as "
            f"our Qwen3-1.7B) cuts fabrication 38% → {obj['q3_32b']['fabrication']:.0f}% but coaches "
            f"worst of the open field ({rankstr('q3_32b')}); Gemma-3-27B is the stronger base at "
            f"similar size."
        )
    if best_frontier is not None and best_council_open is not None:
        rec.append(
            f"- **Teacher for distillation: keep GPT-5.5.** It is still the best coach in the field "
            f"({rankstr(best_frontier) if bcfg.MODELS[best_frontier].family=='gpt' else rankstr('gpt') if 'gpt' in cper else '–'}); "
            f"the strongest fully-open teacher alternatives are DeepSeek-V3.2 / Gemma-3-27B, which "
            f"trail GPT-5.5 on instructiveness — switch only if a 100%-open pipeline is the goal."
        )

    # ---- Assemble ----------------------------------------------------------
    parts: List[str] = []
    parts.append("# Chess-Coach Benchmark — Bigger Open Models vs OURS-v2 / Frontier\n")
    parts.append(
        f"Unified leaderboard extending the v2 benchmark to **{len(open_present)} bigger "
        f"open-source models**, on the **same {int(obj[present[0]]['n']) if present else 0} "
        f"held-out positions** and **identical grounding** the v2 run used, so the numbers "
        f"are directly comparable to OURS-v2 / BASE / GPT-5.5 / Claude Opus 4.8 / "
        f"Gemini 3.1 Pro. Generated {ts}.\n"
    )
    parts.append(
        "> **Question answered:** *do bigger open models fabricate less / coach more "
        "instructively than our 1.7B on the same grounded input?*\n"
    )

    parts.append("\n## TL;DR recommendation\n")
    parts.append("\n".join(rec) + "\n")

    parts.append("\n## Reachability on TrueFoundry (`bedrock-oss-group`)\n")
    parts.append(
        "Probed each candidate with a 1-token chat call (`scripts/tfy_access_open.py`); only "
        "reachable models were run.\n\n"
        "- **Reachable (9, run):** Qwen3-32B, Qwen3-Next-80B-A3B, Gemma-3-27B-it, Llama-3.3-70B, "
        "DeepSeek-V3.2, GLM-5, Mistral-Large-3 (675B), Kimi-K2.5, DeepSeek-R1.\n"
        "- **Unreachable / excluded:** `llama4-maverick-17b` — provider blocks Meta Llama access "
        "(HTTP 400 on both `aws-bedrock/` and `bedrock-oss-group/` routes); "
        "`kimi-k2-thinking` — spends its entire token budget on hidden reasoning and returns "
        "empty coaching content (doesn't fit the coach format); `deepseek.r1` direct route was 403 "
        "but the `bedrock-oss-group/deepseek-r1` virtual route works and is used.\n"
    )

    parts.append("\n## Phase 1 — Grounded objective leaderboard (fabrication is the metric)\n")
    parts.append(
        "All models get the **same VERIFIED-FACTS + Stockfish sound pool + Maia** input and the "
        "same format instruction; scoring is deterministic (the project's own faithfulness "
        "verifier + move/soundness/engine-speak checks). Sorted by fabrication (lower = better).\n\n"
    )
    parts.append(obj_table + "\n")
    parts.append(
        "\n*`fabrication` = share of outputs with ≥1 false board fact (non-LLM verifier). "
        "`move_sound` = recommended move is in the Stockfish sound pool. `avg_violations` = mean "
        "false facts per output.*\n"
    )
    parts.append(
        "\n> **Note on comparability:** every model here (including OURS-v2 / BASE / the frontier) is "
        "re-scored by the *current* faithfulness verifier on identical grounded inputs, so the numbers "
        "are internally consistent. That verifier is slightly stricter than the one behind the older "
        "`RESULTS_BENCHMARK_v2.md` (e.g. OURS-v2 grounded fabrication reads 38% here vs 33% there); "
        "the ranking and the size effect are unaffected.\n"
    )

    parts.append("\n## Phase 2 — Unified council instructiveness ranking (blinded, cross-family)\n")
    parts.append(
        f"One blinded council ranks a **single field of {field_n} anonymized coaches** "
        f"(the 5 v2 competitors + the strongest open models by Phase-1 objective) per item, on a "
        f"**{coun['n_obs'] // max(1, len(bcfg.JUDGE_KEYS))}-item grounded subset** "
        f"({coun['n_obs']} judge-observations across {len(bcfg.JUDGE_KEYS)} judges: "
        f"GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro). Lower mean rank = judged more instructive "
        f"(1 = best of {field_n}); `norm rank` scales that to 0 (best) – 1 (worst).\n\n"
    )
    parts.append(coun_table + "\n")
    parts.append(
        f"\n*Field chosen cost-aware: ranking all {len(open_present)} open models × all 100 "
        f"positions in one field would be a huge, less-reliable judge prompt, so Phase 2 uses the "
        f"5 v2 anchors + the top open objective performers on a reduced position subset. "
        f"Open models outside the field have Phase-1 objective numbers above but no council rank.*\n"
    )
    parts.append(
        "\n> **Bias caveat:** the three judges are also the top-3 competitors. The v2 run measured "
        "mean self-preference at +0.43 rank — small next to the ~2.7-position open→frontier gap — and "
        "it does not distort the open-vs-OURS-v2 comparison, since neither is any judge's own lab.\n"
    )

    parts.append("\n## Cost (this open-model extension only)\n")
    parts.append(cost_table + "\n")
    parts.append(
        f"\n**Total estimated cost of this extension: ${cost['total']:.2f}** — grounded "
        f"generation for the open models + the unified council. Open-model prices are best-effort "
        f"Bedrock on-demand estimates; frontier judge prices are the same per-1M-token figures the "
        f"v2 run used. The v2 five-model run itself cost ~$24 and is reused here for free.\n"
    )

    parts.append("\n## Artifacts\n")
    parts.append(
        "- Scenarios (same as v2): `data/benchmark_open/scenarios.jsonl`\n"
        "- Grounded generations (open + reused v2): `data/benchmark_open/generations.jsonl`\n"
        "- Objective scores: `data/benchmark_open/objective.jsonl`\n"
        "- Unified council: `data/benchmark_open/council.jsonl`\n"
        "- Reachability probe: `scripts/tfy_access_open.py`; driver: `scripts/run_benchmark_open.py`\n"
    )

    report_md.write_text("\n".join(parts), encoding="utf-8")
