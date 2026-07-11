#!/usr/bin/env python3
"""Deploy web/out to the HF Static Space, PRESERVING its README/config.

The Space (khoilamalphaai/chess-coach-studio) is a static Space whose README.md
carries the Space config frontmatter (sdk: static, title, emoji, ...) and whose
.gitattributes / .nojekyll are HF-static plumbing. `next build` (output: export)
does NOT emit those, so we download the CURRENT ones from the Space and drop them
into web/out before a full sync-upload (delete_patterns=["**"] makes the Space
match web/out exactly, clearing stale hashed chunks) — this preserves the README
and config verbatim while shipping the fresh v6-dpo2 build.

Usage:  python scripts/deploy_space.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SPACE_ID = "khoilamalphaai/chess-coach-studio"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "web" / "out"
PRESERVE = ["README.md", ".gitattributes", ".nojekyll"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    if not OUT.is_dir():
        raise SystemExit(f"missing build dir: {OUT} (run the Next build first)")

    api = HfApi(token=token)

    # Preserve the Space's README/config verbatim by copying the CURRENT ones into
    # the upload set (so the full-sync delete_patterns=['**'] never drops them).
    for fname in PRESERVE:
        dest = OUT / fname
        try:
            src = hf_hub_download(SPACE_ID, fname, repo_type="space", token=token)
            dest.write_bytes(Path(src).read_bytes())
            print(f"preserved {fname} from Space")
        except Exception as e:  # noqa: BLE001
            if fname == ".nojekyll":
                dest.write_text("")
                print("created empty .nojekyll (was absent)")
            else:
                print(f"WARN: could not fetch {fname} from Space ({e}); leaving as-is")

    n = sum(1 for _ in OUT.rglob("*") if _.is_file())
    print(f"uploading {n} files from {OUT} -> spaces/{SPACE_ID} (full sync)")
    if args.dry_run:
        print("(dry-run: not uploading)")
        return 0

    commit = api.upload_folder(
        folder_path=str(OUT),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Deploy v6-dpo2 static build (live coach -> v6-dpo2; Study library reseeded)",
        delete_patterns=["**"],  # exact sync: clear stale hashed chunks
    )
    print("committed:", getattr(commit, "commit_url", commit))
    print(f"Space: https://huggingface.co/spaces/{SPACE_ID}")
    print("Live:  https://khoilamalphaai-chess-coach-studio.static.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
