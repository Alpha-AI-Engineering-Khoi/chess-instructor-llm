#!/usr/bin/env python3
"""Reseed web/public/library.json's OURS coaching through the LIVE v6-dpo2 endpoint.

Each library entry is a SINGLE-tier precomputed OURS coaching card (the `coach`
object == the /api/coach CoachResponse). This regenerates that `coach` object per
entry through the new v6-dpo2 Modal endpoint, replacing ONLY the OURS column
(there is no frontier/base/council data in library.json). The engine (Stockfish)
+ Maia grounding come back from the same gated pipeline, so the card stays
faithful. `meta.model` is normalized to the existing "OURS-vN (...)" convention;
the internal `label` (not rendered by the UI) is regenerated only when the move
changes, so the file stays self-consistent.

Cold-start aware: the first call after idle triggers a ~2.5-3 min Modal cold
start; we retry with a generous per-attempt timeout. Idempotent + re-runnable;
writes atomically. Use --limit to reseed a small measured subset first.

Usage::
    python scripts/reseed_library_v6dpo2.py \
        --endpoint https://chess-instructor-2--chess-coach-v6dpo2-4bit-maia-...modal.run \
        [--limit N] [--input web/public/library.json] [--out web/public/library.json]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL_LABEL = "OURS-v6-dpo2 (Qwen3-32B tuned)"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB = REPO_ROOT / "web" / "public" / "library.json"

# Cold-start-resilient call budget.
PER_ATTEMPT_TIMEOUT_S = 230
TOTAL_BUDGET_S = 360
BACKOFF_S = 12


def _post_coach(endpoint: str, fen: str, tier: str, student_move: str | None) -> dict[str, Any]:
    """POST /api/coach with cold-start retry. Returns the CoachResponse dict."""
    url = endpoint.rstrip("/") + "/api/coach"
    payload = {"fen": fen, "tier": tier}
    if student_move:
        payload["student_move"] = student_move
    body = json.dumps(payload).encode("utf-8")

    start = time.time()
    last_err: Exception | None = None
    attempt = 0
    while time.time() - start < TOTAL_BUDGET_S:
        attempt += 1
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=PER_ATTEMPT_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 4xx (except 408/425/429) is a hard client error: don't burn the budget.
            if 400 <= e.code < 500 and e.code not in (408, 425, 429):
                detail = e.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"HTTP {e.code} (hard): {detail}") from e
            last_err = e
        except Exception as e:  # noqa: BLE001 - network / timeout / cold-start
            last_err = e
        if time.time() - start >= TOTAL_BUDGET_S:
            break
        print(f"      ... waking/retry (attempt {attempt}, {int(time.time()-start)}s): {last_err}")
        time.sleep(BACKOFF_S)
    raise RuntimeError(f"coach call failed after {attempt} attempts: {last_err}")


def _relabel(entry: dict[str, Any], new_move: str, engine_best: str) -> str:
    """Single-tier label matching the existing convention:
    '{Tier} · {phase} · played {student} · picks {rec} over engine {best}'."""
    tier = str(entry.get("tier", "")).capitalize()
    phase = entry.get("phase", "")
    base = f"{tier} \u00b7 {phase}"
    sm = entry.get("student_move")
    if sm:
        base += f" \u00b7 played {sm}"
    base += f" \u00b7 picks {new_move} over engine {engine_best}"
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--input", default=str(DEFAULT_LIB))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="reseed only the first N entries (0=all)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else in_path
    lib: list[dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
    total = len(lib)
    n = args.limit if args.limit and args.limit < total else total

    print(f"reseeding {n}/{total} library entries via {args.endpoint}")
    changed_moves = 0
    t0 = time.time()
    for i, entry in enumerate(lib):
        if args.limit and i >= args.limit:
            break
        fen = entry["fen"]
        tier = entry["tier"]
        sm = entry.get("student_move")
        old_move = entry.get("coach", {}).get("recommended_move_san")
        t = time.time()
        resp = _post_coach(args.endpoint, fen, tier, sm)
        dt = time.time() - t
        new_move = resp.get("recommended_move_san")
        engine_best = resp.get("engine", {}).get("best_san", "?")
        # Normalize meta.model to the library convention; keep gate/tuned metadata.
        resp.setdefault("meta", {})
        resp["meta"]["model"] = MODEL_LABEL
        entry["coach"] = resp
        if new_move != old_move:
            changed_moves += 1
            entry["label"] = _relabel(entry, new_move, engine_best)
        flag = "  <-- move changed" if new_move != old_move else ""
        print(f"  [{i+1}/{n}] {tier:12s} {fen[:30]:30s} {old_move} -> {new_move} ({dt:.0f}s){flag}")
        # Persist incrementally so a mid-run stop still saves progress.
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(lib, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out_path)

    print(
        f"done: reseeded {n} entries in {int(time.time()-t0)}s "
        f"({changed_moves} moves changed) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
