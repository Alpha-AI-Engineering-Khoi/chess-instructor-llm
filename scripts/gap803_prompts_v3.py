#!/usr/bin/env python3
"""Build ``prompts_v3.jsonl`` for the Modal v3 eval generator (no model needed).

Reads the flattened 803x3 benchmark scenarios and renders, per scenario, the SAME
system prompt + grounded user prompt every other model in the benchmark gets
(``src.eval.benchmark.prompts``). One line per (position,tier):
``{id, pos_id, tier, phase, severity, system, user}``. This file is baked into the
``eval_modal_v3`` image so the A100 only needs to run ``model.generate``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("BENCH_DIR", str(_ROOT / "data" / "benchmark_gap803"))

from src.eval.benchmark.prompts import build_user_prompt, load_system_prompt  # noqa: E402

BENCH = Path(os.environ["BENCH_DIR"])
SCN = BENCH / "scenarios.jsonl"
OUT = BENCH / "prompts_v3.jsonl"


def main() -> int:
    if not SCN.exists():
        raise SystemExit(f"missing {SCN}; run `python -m scripts.gap803_gen seed` first.")
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
