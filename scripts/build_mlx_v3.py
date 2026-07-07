#!/usr/bin/env python3
"""Build the local 4-bit MLX v3 model by fusing the trained PEFT LoRA into a 4-bit MLX base.

The Mac is too disk-constrained to hold the ~65GB fp16 merged 32B, so instead of
``mlx_lm.convert`` on a merged model we FUSE the small PEFT LoRA adapter (trained on
Modal) directly into the pre-quantized ``mlx-community/Qwen3-32B-4bit`` base. Peak
disk ~= 18GB (base) + 18GB (fused).

Steps:
  1. Convert the HF/PEFT adapter (``adapter_model.safetensors`` + ``adapter_config.json``)
     into the exact adapter format ``mlx_lm`` expects (``adapters.safetensors`` +
     ``adapter_config.json`` with ``lora_parameters``). The math (verified against
     ``mlx_lm/tuner/lora.py``): MLX ``lora_a``:(in,r) = PEFT ``A``ᵀ, MLX ``lora_b``:(r,out)
     = PEFT ``B``ᵀ, and MLX ``scale`` = PEFT ``alpha/r`` — so the fused weight delta
     ``scale·(lora_a@lora_b)ᵀ`` equals PEFT's ``(alpha/r)·(B@A)`` exactly.
  2. ``mlx_lm.fuse`` the converted adapter into the 4-bit MLX base (re-quantizes -> 4-bit out).
  3. Sanity-load + generate one coaching to confirm the fused model runs.

Usage::
    ~/.venvs/mlx/bin/python -m scripts.build_mlx_v3 \
        --adapter models/adapters/chess-coach-v3 \
        --base mlx-community/Qwen3-32B-4bit \
        --out models/mlx/chess-coach-v3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import mlx.core as mx

_ROOT = Path(__file__).resolve().parents[1]


def _map_path(peft_base: str) -> str:
    """``base_model.model.model.layers.N.self_attn.q_proj`` -> ``model.layers.N.self_attn.q_proj``."""
    p = peft_base
    for pref in ("base_model.model.",):
        if p.startswith(pref):
            p = p[len(pref):]
    return p


def convert_adapter(adapter_dir: Path, out_dir: Path) -> Tuple[int, int]:
    """Convert a PEFT LoRA adapter dir to an mlx_lm adapter dir. Returns (n_pairs, num_layers)."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    scale = alpha / r
    dropout = float(cfg.get("lora_dropout", 0.0) or 0.0)

    weights: Dict[str, mx.array] = mx.load(str(adapter_dir / "adapter_model.safetensors"))
    mlx_w: Dict[str, mx.array] = {}
    max_layer = -1
    a_keys = b_keys = 0
    for k, v in weights.items():
        if ".lora_A" in k:
            base = k.split(".lora_A")[0]
            path = _map_path(base) + ".lora_a"
            mlx_w[path] = v.T.astype(mx.float16)      # PEFT A:(r,in) -> lora_a:(in,r)
            a_keys += 1
        elif ".lora_B" in k:
            base = k.split(".lora_B")[0]
            path = _map_path(base) + ".lora_b"
            mlx_w[path] = v.T.astype(mx.float16)      # PEFT B:(out,r) -> lora_b:(r,out)
            b_keys += 1
        else:
            continue
        # track layer index for num_layers
        parts = _map_path(k).split(".")
        if "layers" in parts:
            try:
                max_layer = max(max_layer, int(parts[parts.index("layers") + 1]))
            except (ValueError, IndexError):
                pass
    if a_keys != b_keys or a_keys == 0:
        raise SystemExit(f"BLOCKED: lora_A/lora_B mismatch (A={a_keys} B={b_keys}) — inspect keys.")
    num_layers = max_layer + 1 if max_layer >= 0 else 999

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), mlx_w)
    mlx_cfg = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {"rank": r, "scale": scale, "dropout": dropout},
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(mlx_cfg, indent=2))
    print(f"[convert] {a_keys} LoRA pairs, r={r} alpha={alpha} scale={scale} num_layers={num_layers}")
    print(f"[convert] wrote {out_dir}/adapters.safetensors + adapter_config.json")
    return a_keys, num_layers


def fuse(base: str, mlx_adapter_dir: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", base,
        "--adapter-path", str(mlx_adapter_dir),
        "--save-path", str(out_dir),
    ]
    print(f"[fuse] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def sanity(out_dir: Path) -> str:
    from mlx_lm import generate, load
    model, tok = load(str(out_dir))
    messages = [
        {"role": "system", "content": "You are a concise chess coach."},
        {"role": "user", "content": "In one sentence, why control the center as a beginner?"},
    ]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    out = generate(model, tok, prompt=prompt, max_tokens=64, verbose=False)
    return out.strip()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, default=_ROOT / "models" / "adapters" / "chess-coach-v3")
    p.add_argument("--base", default="mlx-community/Qwen3-32B-4bit")
    p.add_argument("--out", type=Path, default=_ROOT / "models" / "mlx" / "chess-coach-v3")
    p.add_argument("--adapter-tmp", type=Path, default=_ROOT / "models" / "adapters" / "chess-coach-v3-mlxadapter")
    p.add_argument("--skip-fuse", action="store_true", help="only convert the adapter")
    args = p.parse_args(argv)

    if not (args.adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"BLOCKED: no PEFT adapter at {args.adapter} (download from Modal volume first)")

    convert_adapter(args.adapter, args.adapter_tmp)
    if args.skip_fuse:
        return 0
    fuse(args.base, args.adapter_tmp, args.out)
    print("[sanity] generating a short sample from the fused MLX model ...")
    print("[sanity] ->", sanity(args.out)[:200])
    print(f"\nDONE. Local 4-bit MLX v3 at: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
