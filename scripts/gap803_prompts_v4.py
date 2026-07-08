#!/usr/bin/env python3
"""Build ``prompts_v4.jsonl`` for the Modal v4 eval generator (no model needed).

Identical render to ``gap803_prompts_v3`` (the SAME system + grounded
``build_grounded_user`` prompt every benchmark model gets), written into the
isolated v4 benchmark dir ``data/benchmark_v4/`` so the v4 eval never collides
with the shared ``data/benchmark_gap803`` files other workers use. One line per
(position, tier): ``{id, pos_id, tier, phase, severity, system, user}``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# Scenarios are the shared, static 803x3 set; prompts are model-agnostic.
os.environ.setdefault("BENCH_DIR", str(_ROOT / "data" / "benchmark_gap803"))

from src.eval.benchmark.prompts import build_user_prompt, load_system_prompt  # noqa: E402

SCN = Path(os.environ["BENCH_DIR"]) / "scenarios.jsonl"
OUT_DIR = _ROOT / "data" / "benchmark_v4"
OUT = OUT_DIR / "prompts_v4.jsonl"


def main() -> int:
    if not SCN.exists():
        raise SystemExit(f"missing {SCN}; the shared 803x3 scenarios must exist first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scns = [json.loads(l) for l in SCN.read_text(encoding="utf-8").splitlines() if l.strip()]
    system = load_system_prompt()
    with OUT.open("w", encoding="utf-8") as fh:
        for s in scns:
            fh.write(json.dumps({
                "id": s["id"], "pos_id": s["pos_id"], "tier": s["tier"],
                "phase": s["phase"], "severity": s["severity"],
                "system": system, "user": build_user_prompt(s, "grounded"),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(scns)} prompts -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
