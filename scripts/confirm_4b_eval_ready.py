#!/usr/bin/env python3
"""Confirm the HONEST base-vs-tuned eval harness (``src.eval.honest`` +
``scripts.honest_eval``) will accept the **4B** contenders for a later loop's
eval RUN — WITHOUT rebuilding or duplicating the harness (a separate worker owns
it) and without downloading/loading weights.

It checks that the SAME machinery the 1.7B contenders already use works for the
4B pair:
  * ``base_4b``  = untuned ``mlx-community/Qwen3-4B-Instruct-2507-4bit`` (gated)
  * ``ours_4b``  = our tuned 4B, fused to MLX at models/mlx/chess-coach-4b-iter1

and prints the exact ``HONEST_MODELS`` registration + run commands to use once
training finishes. Read-only; safe to run anytime.

    ~/.venvs/mlx/bin/python -m scripts.confirm_4b_eval_ready
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BASE_4B_MLX = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
OURS_4B_MLX = "models/mlx/chess-coach-4b-iter1"   # produced by fusing the iter1 adapter


def main() -> int:
    ok = True

    # 1) The harness backend + driver import and expose the expected API.
    try:
        from src.eval.honest.gated import MLXSamplingCoach, generate  # noqa: F401
        from src.teacher.coach_gate import run_gate  # noqa: F401
        from scripts.honest_eval import HModel, HONEST_MODELS
        print("[OK] harness imports: src.eval.honest.gated + scripts.honest_eval")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] harness import: {type(exc).__name__}: {exc}")
        return 1

    # 2) MLXSamplingCoach accepts (model_path, adapter_path) — the exact mechanism
    #    the 1.7B base + tuned already use, so a 4B MLX path is accepted identically.
    sig = inspect.signature(MLXSamplingCoach.__init__)
    for p in ("model_path", "adapter_path"):
        if p in sig.parameters:
            print(f"[OK] MLXSamplingCoach accepts '{p}'")
        else:
            print(f"[FAIL] MLXSamplingCoach missing '{p}'")
            ok = False

    # 3) The existing 1.7B contenders prove the HModel(kind='mlx') contract; the 4B
    #    pair is byte-identical except the ident (model path) + tuned flag.
    for k in ("base_1p7", "ours_1p7"):
        m = HONEST_MODELS.get(k)
        if m and m.kind == "mlx":
            print(f"[OK] reference mlx contender present: {k} -> {m.ident}")
        else:
            print(f"[WARN] expected reference mlx contender {k} not found")

    # 4) Construct the 4B contenders with the SAME dataclass (no weights loaded).
    try:
        base_4b = HModel("base_4b", "BASE-4B (Qwen3-4B untuned, gated)", "mlx",
                         BASE_4B_MLX, False, "default")
        ours_4b = HModel("ours_4b", "OURS-4B (Qwen3-4B tuned, gated)", "mlx",
                         OURS_4B_MLX, True, "default")
        assert base_4b.kind == "mlx" and ours_4b.tuned is True
        print(f"[OK] constructed HModel contenders: {base_4b.key}, {ours_4b.key}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] could not construct 4B HModel contenders: {exc}")
        ok = False

    # 5) The untuned 4B base MLX repo exists (the tuned one is produced by training).
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(BASE_4B_MLX)
        print(f"[OK] untuned base MLX repo exists: {BASE_4B_MLX}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not verify {BASE_4B_MLX} on HF (offline?): {exc}")

    print("\n=== VERDICT ===")
    print("PASS — the honest harness accepts the 4B tuned + untuned-4B base as contenders"
          if ok else "ISSUES — see [FAIL] lines above")
    print("\nTo run the 4B eval AFTER training finishes (iteration >=2), add to "
          "scripts/honest_eval.HONEST_MODELS:")
    print('    "base_4b":  HModel("base_4b",  "BASE-4B (untuned, gated)", "mlx", '
          f'"{BASE_4B_MLX}", False, "default"),')
    print('    "ours_4b":  HModel("ours_4b",  "OURS-4B (tuned, gated)",   "mlx", '
          f'"{OURS_4B_MLX}", True,  "default"),')
    print('    "pbase_4b": HModel("pbase_4b", "PROMPT-BASE-4B (gated)",   "mlx", '
          f'"{BASE_4B_MLX}", False, "best_4b"),')
    print("  add base_4b/ours_4b/pbase_4b to FIELD, then:")
    print("    $P -m scripts.honest_eval gen --model ours_4b   # (fuse adapter -> MLX first)")
    print("    $P -m scripts.honest_eval gen --model base_4b")
    print("    $P -m scripts.honest_eval gen --model pbase_4b")
    print("    $P -m scripts.honest_eval judge && $P -m scripts.honest_eval report")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
