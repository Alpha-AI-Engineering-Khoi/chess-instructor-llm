"""Deploy the Chess-Coach Benchmark static Space (index.html + README.md).

Builds the static dashboard from ``index.template.html`` and pushes it to
``khoilamalphaai/chess-coach-benchmark`` together with a refreshed, honest
README. The Space's config frontmatter is PRESERVED verbatim (title, emoji,
colors, sdk, license) except for ``short_description``, which is set to the
current honest one-liner; only the README BODY is rewritten. Both files ship in
ONE atomic commit, so a concurrent editor cannot land a half-updated Space.

Transient errors (billing blips, connection resets, commit races) are retried:
each ``create_commit`` re-reads the branch head, so a retry naturally rebases on
whatever another worker just pushed.

Usage:
    set -a && source .env && set +a
    ~/.venvs/mlx/bin/python scripts/space/deploy.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assemble  # noqa: E402

SPACE_REPO = "khoilamalphaai/chess-coach-benchmark"
NAMESPACE = "khoilamalphaai"
BUILD_OUT = Path("/tmp/space_bench/index.html")

# HF caps short_description at 60 chars, so "tier-appropriate move selection" cannot
# fit; "the moat" is the page's defined shorthand for it. Kept honest and current.
SHORT_DESCRIPTION = "Chess-coach eval: OURS tops the moat; v4 base, v6-dpo2 live"

README_BODY = """# Chess-Coach Benchmark

A self-contained **static** dashboard (HTML and CSS, no runtime or GPU) for the honest
evaluation of our chess coach: a **Qwen3-32B QLoRA** model that picks the
tier-appropriate, instructive move for a player's level (beginner, intermediate,
advanced), grounded by Stockfish, Maia, and a truthfulness verifier.

## What the page shows

- **Current lineup on the corrected v6 benchmark** (120 held-out TEST positions across
  3 tiers, fresh grounding): base vs v4 vs v6-dpo vs v6-dpo2. **v4** is the SFT base and
  the model behind the full reproducible evaluation; **v6-dpo2** is the live-served
  best-DPO refinement of v4. Grounded tier-policy match runs 42.8% (base) to 89.2%
  (v6-dpo2), a +0.433 tuned-over-base gap.
- **v6-dpo2 vs the frontier, matched fresh grounding** (120 held-out TEST): the three
  frontier coaches (GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro) were generated on the SAME
  grounding pipeline that produced v6-dpo2's 89.2% and scored with the same extractor —
  the fair apples-to-apples cut, not a reuse of older cached gens. v6-dpo2 leads
  tier-policy match **89.2% vs 57.8-61.4%** for the frontier (+27.8 over the best);
  the frontier stays 100% sound and 100% names-a-move but differentiates tiers on only
  17-30% of positions (vs 98.7% for OURS).
- **The moat across the field:** tier-appropriate move selection over 803 held-out,
  zero-leakage positions for all 15 models under corrected v6 labels. OURS leads the
  field (family average 0.486, ahead of frontier 0.437 and open 0.310); best open coach
  is **GLM-5**.
- **Instructiveness:** a blinded cross-family council (GPT-5.5, Claude Opus 4.8,
  Gemini 3.1 Pro). The frontier leads on prose; both tuned models rank above the
  untuned base.
- **Faithfulness is a gated fairness floor:** after the verify-and-regenerate gate,
  every model ships zero verifier-detectable mechanical violations (0%). Raw pre-gate
  fabrication is intentionally not reported.
- **Cost** of the definitive eval, and the **version lineage** (current v4 and v6-dpo2
  vs historical, superseded v1, v2, v3, v5).

## Honest framing

OURS leads tier-appropriate move selection (the trained behavior, the moat). The
frontier leads judged instructiveness (prose). Safety, no-jargon, and faithfulness are
table-stakes floors, with faithfulness verifier-gated to 0 for every model. All numbers
are verbatim from the project's `RESULTS_STAGE4_CORRECTED.md` (120 TEST) and
`RESULTS_FULL_EVAL_803.md` (803 field).

Adapters: [chess-coach-32b-v4-qlora](https://huggingface.co/khoilamalphaai/chess-coach-32b-v4-qlora)
(SFT base), [chess-coach-32b-v6-dpo2](https://huggingface.co/khoilamalphaai/chess-coach-32b-v6-dpo2)
(live). Companion dataset:
[chess-coach-benchmark](https://huggingface.co/datasets/khoilamalphaai/chess-coach-benchmark).
"""


def _patch_frontmatter(current_readme: str) -> str:
    """Return a README that keeps the existing frontmatter (with short_description set)
    and replaces the body with the honest README_BODY. Falls back to a minimal
    frontmatter if the fetched card has none."""
    m = re.match(r"^---\n(.*?)\n---\n?", current_readme, flags=re.DOTALL)
    if m:
        fm = m.group(1)
    else:
        fm = (
            "title: Chess-Coach Benchmark\n"
            'emoji: "\u265f\ufe0f"\n'
            "colorFrom: indigo\ncolorTo: blue\nsdk: static\npinned: false\n"
            "license: cc-by-4.0"
        )
    sd_line = f'short_description: "{SHORT_DESCRIPTION}"'
    if re.search(r"^short_description:.*$", fm, flags=re.MULTILINE):
        fm = re.sub(r"^short_description:.*$", sd_line, fm, count=1, flags=re.MULTILINE)
    else:
        fm = fm.rstrip("\n") + "\n" + sd_line
    return f"---\n{fm}\n---\n\n{README_BODY}"


def _retry(fn, what: str, tries: int = 10, delay: float = 5.0):
    last = None
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — treat all HF/network errors as transient
            last = e
            print(f"[deploy] {what}: attempt {i}/{tries} failed: {e}", file=sys.stderr)
            if i < tries:
                time.sleep(delay)
    raise SystemExit(f"[deploy] {what}: gave up after {tries} attempts: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (run: set -a && source .env && set +a)")
    if len(SHORT_DESCRIPTION) > 60:
        raise SystemExit(
            f"short_description is {len(SHORT_DESCRIPTION)} chars; HF limit is 60"
        )

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    who = _retry(api.whoami, "whoami")
    print(f"[deploy] authenticated as {who['name']}")
    if who["name"] != NAMESPACE:
        raise SystemExit(f"unexpected namespace {who['name']} (want {NAMESPACE})")

    index_path = assemble.build(BUILD_OUT)

    # Preserve the live Space frontmatter, patch short_description, refresh the body.
    try:
        cur = Path(
            hf_hub_download(SPACE_REPO, "README.md", repo_type="space", token=token)
        ).read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[deploy] no live README fetched ({e}); using minimal frontmatter")
        cur = ""
    readme = _patch_frontmatter(cur)
    print(f"[deploy] short_description -> {SHORT_DESCRIPTION!r}")

    if args.dry_run:
        Path("/tmp/space_bench/README.md").write_text(readme, encoding="utf-8")
        print("[deploy] dry-run: wrote /tmp/space_bench/README.md; not uploading")
        return 0

    ops = [
        CommitOperationAdd("index.html", str(index_path)),
        CommitOperationAdd("README.md", readme.encode("utf-8")),
    ]
    commit = _retry(
        lambda: api.create_commit(
            repo_id=SPACE_REPO,
            repo_type="space",
            operations=ops,
            commit_message="Add v6-dpo2-vs-frontier matched-grounding panel (120 held-out TEST; frontier regenerated on the same grounding pipeline, honestly scoped)",
        ),
        "create_commit",
    )
    print("[deploy] committed:", getattr(commit, "commit_url", commit))
    print(f"[deploy] Space: https://huggingface.co/spaces/{SPACE_REPO}")
    print("[deploy] Live:  https://khoilamalphaai-chess-coach-benchmark.static.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
