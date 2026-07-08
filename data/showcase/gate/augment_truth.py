#!/usr/bin/env python3
"""Augment data/showcase/truthfulness.json with honest judge-detail breakdowns.

Adds, from data/showcase/gate/judge_raw.jsonl + the gated showcase.json:
  * per_judge_flag_rate      — how often each individual judge flagged (shows the
                               `any`-aggregation is a strict union; GPT is harshest);
  * panel_unanimous_frac     — fraction of sampled cells where all 3 judges agreed;
  * unanimous_untruthful_rate— fraction ALL 3 judges flagged (high-confidence floor);
  * by_cell_origin           — judge truthful-rate split by fallback / raw-clean /
                               regen (fallback should validate ~100%).
These make the residual interpretable and non-circular.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a binomial proportion k/n (small-n safe)."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import gate_lib as G  # noqa: E402

TRUTH = G.ROOT / "data" / "showcase" / "truthfulness.json"
JUDGE_RAW = HERE / "judge_raw.jsonl"
SHOWCASE = G.ROOT / "web" / "public" / "showcase.json"


def main() -> int:
    doc = json.loads(TRUTH.read_text())
    rows = [json.loads(l) for l in JUDGE_RAW.read_text().splitlines() if l.strip()]
    n2k = G.name_to_key_map()

    origin = {}
    for pi, p in enumerate(json.loads(SHOWCASE.read_text())):
        for m in p["models"]:
            k = n2k.get(m["name"])
            for t in G.TIERS:
                c = (m["byTier"] or {}).get(t)
                if not c:
                    continue
                sid = f"{pi}:{k}:{t}"
                origin[sid] = ("fallback" if c.get("verified_fallback")
                               else "raw_clean" if not c.get("raw_fabricated") else "regen")

    per_judge = defaultdict(lambda: {"calls": 0, "flagged": 0})
    bysid = defaultdict(list)
    for r in rows:
        b = per_judge[r["judge"]]
        b["calls"] += 1
        b["flagged"] += int(not r["truthful"])
        bysid[r["sid"]].append(r)

    unanimous_untruthful = 0
    unanimous_any = 0
    by_origin = defaultdict(lambda: {"n": 0, "truthful": 0})
    for sid, rs in bysid.items():
        truthful = all(x["truthful"] for x in rs)  # any-agg
        nt = sum(x["truthful"] for x in rs)
        if nt == 0:
            unanimous_untruthful += 1
        if nt == len(rs) or nt == 0:
            unanimous_any += 1
        o = by_origin[origin.get(sid, "raw_clean")]
        o["n"] += 1
        o["truthful"] += int(truthful)

    n = len(bysid)

    # Per-model truthful-rate under three panel-aggregation rules (nested lenience):
    #   any        = strict lower bound: truthful iff ALL 3 judges agree truthful
    #                (a single objection sinks the cell);
    #   majority   = truthful iff a strict majority (>=2 of 3) judges found it truthful;
    #   unanimous  = lenient upper bound: truthful UNLESS all 3 judges object
    #                (only a unanimous objection sinks the cell).
    per_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "any": 0, "maj": 0, "unan": 0})
    for sid, rs in bysid.items():
        name = rs[0].get("name", "?")
        nt = sum(1 for x in rs if x["truthful"])
        m = len(rs)
        b = per_model[name]
        b["n"] += 1
        if nt == m:
            b["any"] += 1
        if nt > m / 2:
            b["maj"] += 1
        if nt >= 1:
            b["unan"] += 1

    tot = {"n": 0, "any": 0, "maj": 0, "unan": 0}
    for name, b in per_model.items():
        for k in tot:
            tot[k] += b[k]
        jr = doc["judge_residual"].get(name)
        if jr is None:
            continue
        nn = b["n"]
        jr["truthful_rate_any"] = round(b["any"] / nn, 4) if nn else 0.0
        jr["any_ci95"] = _wilson(b["any"], nn)
        jr["truthful_rate_majority"] = round(b["maj"] / nn, 4) if nn else 0.0
        jr["majority_ci95"] = _wilson(b["maj"], nn)
        jr["truthful_rate_unanimous"] = round(b["unan"] / nn, 4) if nn else 0.0
        jr["unanimous_ci95"] = _wilson(b["unan"], nn)
    ov = doc["judge_residual"].get("_overall")
    if ov is not None and tot["n"]:
        ov["truthful_rate_any"] = round(tot["any"] / tot["n"], 4)
        ov["any_ci95"] = _wilson(tot["any"], tot["n"])
        ov["truthful_rate_majority"] = round(tot["maj"] / tot["n"], 4)
        ov["majority_ci95"] = _wilson(tot["maj"], tot["n"])
        ov["truthful_rate_unanimous"] = round(tot["unan"] / tot["n"], 4)
        ov["unanimous_ci95"] = _wilson(tot["unan"], tot["n"])

    doc["judge_detail"] = {
        "aggregations_note": ("Per-model truthful-rate is reported under three nested panel "
                              "rules (see judge_residual.*_rate_*): `any` (strict lower bound: "
                              "a single judge's objection sinks the cell), `majority` (>=2 of 3 "
                              "judges truthful), and `unanimous` (lenient upper bound: only a "
                              "unanimous 3/3 objection sinks the cell). Each carries a 95% Wilson "
                              "CI. Truth lies somewhere in the any..unanimous band."),
        "aggregation_note": ("`any` = a single cross-family judge's objection marks the "
                             "cell not-truthful (strict union). The judge flags concrete "
                             "FALSE or UNSUPPORTED-by-the-1-ply-verified-facts claims; it "
                             "is instructed NOT to flag general plans/principles or hedged "
                             "language. So this is a conservative floor on truthfulness, "
                             "not a claim that the rest are outright lies."),
        "per_judge_flag_rate": {
            j: {"calls": b["calls"], "flagged": b["flagged"],
                "flag_rate": round(b["flagged"] / b["calls"], 4) if b["calls"] else 0.0}
            for j, b in sorted(per_judge.items())
        },
        "panel_unanimous_frac": round(unanimous_any / n, 4) if n else 0.0,
        "unanimous_untruthful_rate": round(unanimous_untruthful / n, 4) if n else 0.0,
        "unanimous_untruthful_n": unanimous_untruthful,
        "by_cell_origin": {
            k: {"n": v["n"], "truthful_rate": round(v["truthful"] / v["n"], 4) if v["n"] else 0.0}
            for k, v in sorted(by_origin.items())
        },
    }
    TRUTH.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(doc["judge_detail"], indent=2))
    print(f"\naugmented {TRUTH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
