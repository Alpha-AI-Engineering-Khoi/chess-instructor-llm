#!/usr/bin/env python3
"""Phase A driver: gate every displayed cell for a set of models (resumable).

Reads the PRISTINE backup (data/showcase/showcase.pregate.json), verifies each
cell's raw coaching with verify_text_ext, and — only for the flagged ones —
re-samples via that model's own backend (identical grounded prompt), keeping the
first clean re-sample, else the verified engine-derived fallback. Writes one
record per cell to data/showcase/gate/gated_cells.<job>.jsonl (resumable, keyed
by pos-index/model/tier). No showcase.json write here — merge.py applies the
cache. Clean cells cost nothing; local re-gen is free; only flagged TFY cells
spend, and a hard --cost-cap guards against runaway.

Groups:
  local     -> ours, base            (sequential MLX, free)
  frontier  -> gpt, claude, gemini   (TFY gateway, paid)
  open      -> the 9 open competitors (TFY gateway, paid)

Run (background jobs):
  ~/.venvs/mlx/bin/python data/showcase/gate/gate.py --group local
  ~/.venvs/mlx/bin/python data/showcase/gate/gate.py --group frontier --concurrency 6
  ~/.venvs/mlx/bin/python data/showcase/gate/gate.py --group open --concurrency 10
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gate_lib as G  # noqa: E402
from common import LOCAL_KEYS, resolved_ident  # noqa: E402

PREGATE = G.ROOT / "data" / "showcase" / "showcase.pregate.json"
GROUPS = {
    "local": ["ours", "base"],
    "frontier": ["gpt", "claude", "gemini"],
    "open": ["q3_32b", "q3_next80b", "gemma3_27b", "llama33_70b", "dsv32",
             "glm5", "mistral3", "kimi25", "dsr1"],
}


def load_done(cache: Path) -> set:
    done = set()
    if cache.exists():
        for r in G.read_jsonl(cache):
            done.add((r["pi"], r["key"], r["tier"]))
    return done


def collect_cells(positions: List[Dict[str, Any]], keys: set,
                  name_to_key: Dict[str, str]) -> List[Dict[str, Any]]:
    """Every (pi, model, tier) cell whose model key is in ``keys`` and has text."""
    cells: List[Dict[str, Any]] = []
    for pi, pos in enumerate(positions):
        fen = pos["fen"]
        pos_id = pos["id"]
        split_source = pos.get("split_source") or ("train" if pos.get("split") == "train" else "test_new")
        for m in pos.get("models", []):
            key = name_to_key.get(m["name"])
            if key not in keys:
                continue
            for tier in G.TIERS:
                cell = (m.get("byTier") or {}).get(tier)
                if not cell:
                    continue
                coaching = cell.get("coaching")
                if not coaching or not str(coaching).strip():
                    continue
                cells.append({
                    "pi": pi, "pos_id": pos_id, "tier": tier, "key": key,
                    "name": m["name"], "fen": fen, "split_source": split_source,
                    "coaching": str(coaching), "move_uci": cell.get("move_uci"),
                })
    return cells


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=list(GROUPS))
    p.add_argument("--models", default="", help="comma keys (overrides --group)")
    p.add_argument("--job", default="", help="cache file suffix (defaults to group/models)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--cost-cap", type=float, default=200.0, help="hard USD ceiling")
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--min-interval", type=float, default=0.05)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-regen", action="store_true",
                   help="skip re-sampling; flagged cells go straight to the verified "
                        "fallback (use when a provider is down).")
    args = p.parse_args(argv)

    if args.models:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        job = args.job or "_".join(keys)
    elif args.group:
        keys = GROUPS[args.group]
        job = args.job or args.group
    else:
        print("need --group or --models", file=sys.stderr)
        return 2
    keyset = set(keys)

    positions = json.loads(PREGATE.read_text(encoding="utf-8"))
    name_to_key = G.name_to_key_map()
    scn_index = G.build_scn_index()
    prompts = G.PromptCache()

    cache = HERE / f"gated_cells.{job}.jsonl"
    done = load_done(cache)
    cells = collect_cells(positions, keyset, name_to_key)
    if args.limit:
        cells = cells[: args.limit]
    todo = [c for c in cells if (c["pi"], c["key"], c["tier"]) not in done]
    print(f"[gate/{job}] models={keys} cells={len(cells)} done={len(done)} todo={len(todo)}",
          file=sys.stderr)
    if not todo:
        print(f"[gate/{job}] nothing to do", file=sys.stderr)
        return 0

    write_lock = threading.Lock()
    cost_lock = threading.Lock()
    state = {"usd": 0.0, "regens": 0, "fallbacks": 0, "flagged": 0, "done": 0,
             "capped": False}
    fh = cache.open("a", encoding="utf-8")
    t0 = time.time()

    def scn_for(c) -> Optional[Dict[str, Any]]:
        return scn_index.get((c["pos_id"], c["tier"]))

    def write(rec: Dict[str, Any]) -> None:
        with write_lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    def finish_record(c, res: Dict[str, Any]) -> None:
        rec = {"pi": c["pi"], "pos_id": c["pos_id"], "tier": c["tier"],
               "key": c["key"], "name": c["name"], **res}
        write(rec)
        with cost_lock:
            state["done"] += 1
            state["usd"] += res.get("usd", 0.0)
            state["regens"] += res.get("regens", 0)
            if res.get("verified_fallback"):
                state["fallbacks"] += 1
            if res.get("raw_fabricated"):
                state["flagged"] += 1

    # Pre-split: cheap raw verification in the main thread.
    clean: List[Dict[str, Any]] = []
    flagged: List[Dict[str, Any]] = []
    for c in todo:
        if G.is_clean(c["coaching"], c["fen"], c["move_uci"]):
            clean.append(c)
        else:
            flagged.append(c)
    print(f"[gate/{job}] raw-clean={len(clean)} raw-flagged={len(flagged)} "
          f"(only flagged cost time/$)", file=sys.stderr)

    # Clean cells: write immediately (attempt #1 stands).
    for c in clean:
        finish_record(c, {
            "raw_fabricated": False, "gate_attempts": 1, "verified_fallback": False,
            "fabricated": False, "coaching": c["coaching"], "regens": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0,
        })

    is_local = keyset <= set(LOCAL_KEYS)

    def process(c, backend) -> None:
        scn = scn_for(c)
        if scn is None:  # should never happen (audit reported 0 unmapped)
            finish_record(c, {
                "raw_fabricated": True, "gate_attempts": 1, "verified_fallback": False,
                "fabricated": True, "coaching": c["coaching"], "regens": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0,
                "error": "no_scenario",
            })
            return
        res = G.gate_cell(coaching=c["coaching"], fen=c["fen"], move_uci=c["move_uci"],
                          scn=scn, backend=backend, key=c["key"], prompts=prompts,
                          raw_ok=False, allow_regen=not args.no_regen)
        finish_record(c, res)

    def progress() -> None:
        dt = time.time() - t0
        n = state["done"]
        rate = n / dt if dt else 0
        print(f"  [{job}] {n}/{len(todo)} done | flagged-regen={state['regens']} "
              f"fallbacks={state['fallbacks']} ${state['usd']:.2f} "
              f"({rate:.1f} cells/s, {dt:.0f}s)", file=sys.stderr)

    try:
        if is_local:
            # Sequential MLX, one model loaded at a time.
            by_key: Dict[str, List[Dict[str, Any]]] = {}
            for c in flagged:
                by_key.setdefault(c["key"], []).append(c)
            for key, group in by_key.items():
                print(f"[gate/{job}] loading local {key} for {len(group)} flagged cells",
                      file=sys.stderr)
                coach = G.LocalCoach(resolved_ident(key))
                for i, c in enumerate(group, 1):
                    process(c, coach)
                    if i % 25 == 0 or i == len(group):
                        progress()
                del coach
        else:
            backends = G.make_tfy_backends(keys, timeout=args.timeout,
                                           min_interval=args.min_interval,
                                           max_retries=args.max_retries)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futs = {}
                for c in flagged:
                    with cost_lock:
                        if state["usd"] >= args.cost_cap:
                            state["capped"] = True
                            break
                    futs[pool.submit(process, c, backends[c["key"]])] = c
                for i, fut in enumerate(as_completed(futs), 1):
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ! task error: {type(exc).__name__}: {exc}", file=sys.stderr)
                    if i % 50 == 0 or i == len(futs):
                        progress()
                    with cost_lock:
                        if state["usd"] >= args.cost_cap and not state["capped"]:
                            state["capped"] = True
                            print(f"[gate/{job}] COST CAP ${args.cost_cap} reached; "
                                  f"stopping new work.", file=sys.stderr)
    finally:
        fh.close()

    progress()
    print(f"[gate/{job}] DONE done={state['done']} flagged={state['flagged']} "
          f"regens={state['regens']} fallbacks={state['fallbacks']} "
          f"cost=${state['usd']:.2f}{' (CAPPED)' if state['capped'] else ''} "
          f"in {time.time()-t0:.0f}s -> {cache}", file=sys.stderr)
    if state["capped"]:
        print(f"[gate/{job}] WARNING: cost cap hit — re-run to finish remaining cells.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
