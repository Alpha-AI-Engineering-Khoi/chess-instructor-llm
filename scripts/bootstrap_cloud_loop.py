#!/usr/bin/env python3
"""One-time bootstrap for the Mac-INDEPENDENT cloud training loop.

Two jobs, both run while the Mac is still up; afterwards the eval app + orchestrator
run entirely on Modal (chess-instructor-2) and survive a laptop restart:

1. **Secret** — create/refresh the Modal secret ``chess-eval-secrets`` (TrueFoundry
   gateway creds + optional HF token) from the gitignored ``.env``. Secret VALUES
   are read from ``.env`` and handed to ``modal secret create`` as process args;
   they are NEVER printed here or committed. This is how the cloud eval + teacher
   top-up reach the org-funded TrueFoundry council without any local file.
2. **Datasets on the Volume** — push the curated iter-1 dataset + the v3 teacher
   candidates + the iter-2 top-up target list onto the ``chess-coach-lora`` Volume
   under ``/datasets`` so the orchestrator's data-improvement step is cloud-native
   (the eval INPUTS are baked into the eval image, so nothing else needs seeding).

FOOTGUN handled internally: the bare ``MODAL_TOKEN_ID/SECRET`` (kim-lam's) would
override ``MODAL_PROFILE``; every modal subprocess is run with them stripped and
``MODAL_PROFILE`` pinned to the target workspace. Never targets kim-lam.

    ~/.venvs/mlx/bin/python scripts/bootstrap_cloud_loop.py --profile chess-instructor-2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = "kim-lam"
SECRET_NAME = "chess-eval-secrets"

# Secret keys to lift from .env into the Modal secret (names only ever printed).
SECRET_KEYS = [
    "TFY_API_KEY", "TFY_BASE_URL", "TFY_TEACHER_MODEL", "TFY_JUDGE_MODEL",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
]

LORA_VOLUME = "chess-coach-lora"
# (local path, remote path on the volume) — the orchestrator's data-step inputs.
DATASET_UPLOADS = [
    ("data/dataset/train_4b_iter1.jsonl", "/datasets/train_4b_iter1.jsonl"),
    ("data/dataset/valid_4b_iter1.jsonl", "/datasets/valid_4b_iter1.jsonl"),
    ("data/generated/candidates_v3.jsonl", "/datasets/candidates_v3.jsonl"),
    ("data/generated/4b_iter1_topup_needed.jsonl", "/datasets/4b_iter1_topup_needed.jsonl"),
    ("data/generated/4b_iter1_manifest.json", "/datasets/4b_iter1_manifest.json"),
]


def _modal_bin() -> str:
    sib = os.path.join(os.path.dirname(sys.executable), "modal")
    if os.path.exists(sib):
        return sib
    from shutil import which
    return which("modal") or "/Users/khoilam/.venvs/mlx/bin/modal"


def _clean_env(profile: str) -> dict:
    if profile == FORBIDDEN:
        raise SystemExit(f"refusing forbidden profile {profile!r}")
    env = dict(os.environ)
    env.pop("MODAL_TOKEN_ID", None)
    env.pop("MODAL_TOKEN_SECRET", None)
    env["MODAL_PROFILE"] = profile
    return env


def _run(args: list, profile: str, *, secret_args: list | None = None) -> int:
    """Run a modal CLI command; secret_args (VALUES) are appended but never logged."""
    printable = [_modal_bin(), *args] + (["<REDACTED_SECRET_ARGS>"] if secret_args else [])
    print("  $ " + " ".join(printable))
    proc = subprocess.run([_modal_bin(), *args, *(secret_args or [])],
                          env=_clean_env(profile))
    return proc.returncode


def make_secret(profile: str) -> None:
    from dotenv import dotenv_values

    env_path = ROOT / ".env"
    if not env_path.exists():
        raise SystemExit(f"BLOCKED: {env_path} not found (need TFY creds).")
    vals = dotenv_values(env_path)
    kv_args: list = []
    present, missing = [], []
    for k in SECRET_KEYS:
        v = vals.get(k) or os.environ.get(k)
        if v:
            kv_args.append(f"{k}={v}")
            present.append(k)
        else:
            missing.append(k)
    if not any(k.startswith("TFY_") for k in present):
        raise SystemExit("BLOCKED: no TFY_* keys found in .env — cannot reach the council.")
    print(f"[secret] {SECRET_NAME}: setting {present}" + (f" (missing: {missing})" if missing else ""))
    rc = _run(["secret", "create", SECRET_NAME, "--force"], profile, secret_args=kv_args)
    if rc != 0:
        raise SystemExit(f"secret create failed (rc={rc})")
    print(f"[secret] OK -> {SECRET_NAME} on {profile}")


def seed_datasets(profile: str) -> None:
    for local_rel, remote in DATASET_UPLOADS:
        local = ROOT / local_rel
        if not local.exists():
            print(f"[datasets] SKIP missing {local_rel}")
            continue
        rc = _run(["volume", "put", "--force", LORA_VOLUME, local.as_posix(), remote], profile)
        if rc != 0:
            print(f"[datasets] WARN upload failed for {local_rel} (rc={rc})")
        else:
            print(f"[datasets] OK {local_rel} -> {LORA_VOLUME}:{remote}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="chess-instructor-2")
    ap.add_argument("--skip-secret", action="store_true")
    ap.add_argument("--skip-datasets", action="store_true")
    a = ap.parse_args()
    print(f"=== bootstrap cloud loop on {a.profile} ===")
    if not a.skip_secret:
        make_secret(a.profile)
    if not a.skip_datasets:
        seed_datasets(a.profile)
    print("=== bootstrap done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
