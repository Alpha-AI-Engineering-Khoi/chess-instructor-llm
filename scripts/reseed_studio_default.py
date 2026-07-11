#!/usr/bin/env python3
"""Regenerate web/src/lib/studioDefault.ts through the LIVE v6-dpo2 endpoint.

studioDefault.ts bakes the Studio homepage's DEFAULT position (a king-and-pawn
endgame) with the tuned coach's precomputed per-tier answers, so the homepage
renders instantly (no cold-start wait) on mount. It shipped with the OURS-v4
cell; this re-runs the SAME position through the v6-dpo2 endpoint (one
/api/coach_all call) and rewrites the file with the real v6-dpo2 output, so the
homepage default is honestly the live-served model. Engine + Maia facts are
position-only (recomputed identically by the gated pipeline).

Usage::
    python scripts/reseed_studio_default.py --endpoint https://...modal.run
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
OUT = REPO_ROOT / "web" / "src" / "lib" / "studioDefault.ts"

FEN = "8/7b/5p2/P1kp3P/2pN1P2/4K3/8/8 w - - 1 39"
STUDENT_UCI = "d4e2"
STUDENT_SAN = "Ne2"

PER_ATTEMPT_TIMEOUT_S = 230
TOTAL_BUDGET_S = 420
BACKOFF_S = 12

HEADER = '''// AUTO-GENERATED cached-first seed for the Studio homepage.
//
// The DEFAULT king-and-pawn endgame (id vaLVwTHK_77) with the tuned coach's
// PRECOMPUTED answer at all three rating tiers, so the Studio renders the
// tier-adaptive move INSTANTLY on mount without any live call to the
// scale-to-zero Modal endpoint (which cold-starts in ~2-3 min).
//
// Provenance (all values are precomputed, never fabricated at runtime):
//   - per-tier recommended move + coaching prose + takeaway: regenerated LIVE
//     through the v6-dpo2 endpoint (/api/coach_all) -- the same tuned model the
//     live demo serves (Qwen3-32B + chess-coach-v6-dpo2 QLoRA).
//   - position-level engine facts (sound pool, best move, the student's Ne2 +
//     severity) and Maia human-frequency come back from the SAME gated pipeline,
//     computed once per position exactly as the live /api/coach_all does.
//
// Regenerate with: python scripts/reseed_studio_default.py --endpoint <url>
import type { CoachResponse, Tier } from "@/lib/api";

export const STUDIO_DEFAULT_FEN = "8/7b/5p2/P1kp3P/2pN1P2/4K3/8/8 w - - 1 39";
export const STUDIO_DEFAULT_STUDENT_UCI = "d4e2";

/** Precomputed tuned-coach answers for the default position, one per tier. */
export const STUDIO_DEFAULT_TIERS: Record<Tier, CoachResponse> = '''


def _post_coach_all(endpoint: str) -> dict[str, Any]:
    url = endpoint.rstrip("/") + "/api/coach_all"
    body = json.dumps({"fen": FEN, "student_move": STUDENT_SAN}).encode("utf-8")
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
            if 400 <= e.code < 500 and e.code not in (408, 425, 429):
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}") from e
            last_err = e
        except Exception as e:  # noqa: BLE001
            last_err = e
        print(f"  ... waking/retry (attempt {attempt}, {int(time.time()-start)}s): {last_err}")
        time.sleep(BACKOFF_S)
    raise RuntimeError(f"coach_all failed after {attempt} attempts: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    args = ap.parse_args()

    print(f"regenerating Studio default via {args.endpoint} ...")
    all3 = _post_coach_all(args.endpoint)
    tiers: dict[str, Any] = {}
    for tier in ("beginner", "intermediate", "advanced"):
        resp = all3[tier]
        resp.setdefault("meta", {})
        resp["meta"]["model"] = MODEL_LABEL
        tiers[tier] = resp
        print(f"  {tier:12s} -> {resp['recommended_move_san']} "
              f"(attempts={resp['meta'].get('attempts')}, "
              f"fallback={resp['meta'].get('verified_fallback')})")

    body = json.dumps(tiers, indent=2, ensure_ascii=False)
    OUT.write_text(HEADER + body + ";\n", encoding="utf-8")
    print(f"wrote {OUT}")
    moves = " / ".join(tiers[t]["recommended_move_san"] for t in ("beginner", "intermediate", "advanced"))
    print(f"default per-tier moves (B/I/A): {moves}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
