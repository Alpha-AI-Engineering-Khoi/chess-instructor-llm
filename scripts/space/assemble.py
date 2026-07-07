"""Assemble the Space's index.html from the template + real data.

- ``DATA`` (the v1/v2 comparison blob) is reused **verbatim** from the currently
  deployed index.html (it is correct + curated), extracted once into
  ``data_blob.json`` so the build is reproducible from the repo afterwards.
- ``TIERDIFF`` is the divergence-harness differentiation (verified against
  ``data/analysis/divergence_compare_v2.json``).
- ``HEADTOHEAD`` is the real same-input faithfulness slice built by
  ``scripts/build_headtohead.py``.

Usage:
    python scripts/space/assemble.py --headtohead data/benchmark_v2/headtohead.json \
        --out /tmp/space/index.html [--old-index /tmp/space/index.html]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE / "index.template.html"
DATA_BLOB = HERE / "data_blob.json"
TIERDIFF = {"v1": 0.275, "v2": 0.392}  # divergence_compare_v2.json (120 matched)
# Blunder-only re-score ("Move safety — no blunders", cp-loss >= 250), produced by
# scripts/rescore_move_safety.py from the stored picks + Stockfish evals.
SAFETY = {
    "v2": REPO / "data" / "benchmark_v2" / "move_safety.json",
    "v1": REPO / "data" / "benchmark_v1" / "move_safety.json",
}


def inject_move_safety(data: dict) -> None:
    """Layer the re-scored ``move_safe`` metric onto each version's objective."""
    for ver, path in SAFETY.items():
        if ver in data and path.exists():
            data[ver]["objective"]["move_safe"] = json.loads(
                path.read_text(encoding="utf-8")
            )["move_safe"]
            print(f"[assemble] injected move_safe ({ver}) from {path}")


def extract_data_blob(old_index: Path) -> dict:
    text = old_index.read_text(encoding="utf-8")
    marker = "const DATA = "
    i = text.index(marker) + len(marker)
    j = text.index(";", i)
    return json.loads(text[i:j])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--headtohead", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--old-index", help="deployed index.html to extract the DATA blob from (first run)")
    args = p.parse_args()

    if args.old_index:
        data = extract_data_blob(Path(args.old_index))
        DATA_BLOB.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"[assemble] extracted DATA blob → {DATA_BLOB}")
    elif DATA_BLOB.exists():
        data = json.loads(DATA_BLOB.read_text(encoding="utf-8"))
        print(f"[assemble] reused DATA blob from {DATA_BLOB}")
    else:
        raise SystemExit("no --old-index and no data_blob.json; provide --old-index once")

    inject_move_safety(data)

    headtohead = json.loads(Path(args.headtohead).read_text(encoding="utf-8"))
    tpl = TEMPLATE.read_text(encoding="utf-8")
    html = (
        tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__TIERDIFF__", json.dumps(TIERDIFF))
        .replace("__HEADTOHEAD__", json.dumps(headtohead, ensure_ascii=False))
    )
    for ph in ("__DATA__", "__TIERDIFF__", "__HEADTOHEAD__"):
        assert ph not in html, f"placeholder {ph} not substituted"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[assemble] wrote {out} · {len(html):,} bytes · headtohead positions={len(headtohead)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
