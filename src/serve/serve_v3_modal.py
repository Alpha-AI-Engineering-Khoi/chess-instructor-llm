#!/usr/bin/env python3
"""Serve the **v3** chess coach (Qwen3-32B + QLoRA LoRA, 4-bit) as a scale-to-zero
Modal GPU endpoint that speaks the EXACT same coach contract as the local API.

Why this exists
---------------
v3 is a QLoRA LoRA adapter on ``unsloth/Qwen3-32B-unsloth-bnb-4bit`` (the best
*locally-runnable* model on the 803x3 benchmark, balanced 61.7). Its 4-bit MLX
form is too big/slow to serve on the Mac, so its deployable form is **base
Qwen3-32B in 4-bit (bitsandbytes) + the v3 LoRA adapter (peft) on a Modal GPU**.

What it reuses (model-agnostic — the whole point)
------------------------------------------------
The full gated coach pipeline lives in :mod:`src.api.server` and is entirely
backend-agnostic: Stockfish sound-pool + Maia (best-effort) + verified-facts
grounding (``render_pool_facts`` + ``render_user_prompt``) + a
VERIFY-AND-REGENERATE faithfulness gate (``verify_text_ext``, up to
``COACH_MAX_ATTEMPTS``) + a deterministic engine-derived fallback. We import that
FastAPI app verbatim and only swap the *generation backend*: instead of the local
``mlx_lm`` ``Coach``, we plug in a transformers+peft ``generate`` on the GPU. Every
route, response shape (``CoachResponse`` with ``meta.attempts`` /
``meta.verified_fallback``), and CORS rule is inherited unchanged.

Model load
----------
The base ``unsloth/Qwen3-32B-unsloth-bnb-4bit`` is baked into the image at build
time (so scale-to-zero cold starts don't re-download ~18 GB). At container start
we load it in 4-bit and apply the v3 LoRA adapter (peft) from the shared Modal
Volume ``chess-coach-lora:/chess-coach-v3/adapter``. Primary loader is plain
transformers+peft; if that cannot read the pre-quantized checkpoint we fall back
to Unsloth's ``FastLanguageModel`` (the exact loader the v3 eval used), so the
deploy is robust either way. Generation SAMPLES (temp 0.7 / top-p 0.8 / top-k 20,
+ light repetition guards) so the gate's re-generations actually differ.

Deploy
------
    # Modal auth from .env first (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET), then:
    modal deploy src/serve/serve_v3_modal.py

Maia note: lc0 + Maia nets are not installed on Modal, so the human-likelihood
signal degrades gracefully (``maia: []`` + a note) exactly as the local API does
when Maia is unavailable. Stockfish IS installed (apt) and used for real.

No secrets are hardcoded: the HF token (for the base pull) comes from a Modal
secret; Modal auth comes from the ambient profile / MODAL_TOKEN_* env.
"""
from __future__ import annotations

from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Names / paths
# --------------------------------------------------------------------------- #
APP_NAME: str = "chess-coach-v3-serve"
VOLUME_NAME: str = "chess-coach-lora"          # shared volume holding the v3 adapter
RUN_NAME: str = "chess-coach-v3"
VOL_MOUNT: str = "/vol"
ADAPTER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/adapter"

#: Base the v3 adapter was trained on (see adapter_config.json / eval_modal_v3.py).
BASE_MODEL: str = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
#: Human-readable label surfaced as ``meta.model`` / on /api/health.
MODEL_LABEL: str = "Qwen3-32B + chess-coach-v3 QLoRA (4-bit, Modal)"

#: 32B nf4 ~= 18-20 GB; A100-40GB has ample headroom for weights + KV + LoRA.
GPU: str = "A100-40GB"
#: Scale-to-zero after 5 min idle.
SCALEDOWN_S: int = 300
#: A single request may run the gate up to 6x; keep the ceiling generous.
REQUEST_TIMEOUT_S: int = 900

CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION: str = "3.11"
STOCKFISH_BIN: str = "/usr/games/stockfish"    # debian/ubuntu apt install location

# The v3 trainer/eval ML stack (same list -> layer-cache reuse + the EXACT peft
# (0.19.1) that wrote the adapter and the transformers/bitsandbytes that already
# load this pre-quantized 32B). Unsloth is kept as a proven fallback loader.
_ML_PIP: list[str] = [
    "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
    "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
]
# The thin HTTP + chess layer the repo's server needs (do NOT constrain the ML
# stack — these are added as a tail layer so the heavy layers stay cached).
_API_PIP: list[str] = ["python-chess", "fastapi", "uvicorn", "pydantic", "python-dotenv"]

if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = None

#: HF token for the base-model pull (public model, but attach a secret anyway so a
#: gated/rate-limited pull still works). Created from .env before deploy:
#:   modal secret create chess-hf HF_TOKEN=... HUGGING_FACE_HUB_TOKEN=...
hf_secret = modal.Secret.from_name("chess-hf")


def _bake_base_model() -> None:
    """Download the 4-bit base into the image's HF cache (baked in => fast cold start)."""
    import os

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(BASE_MODEL, token=token)
    print(f"[build] baked {BASE_MODEL} into image HF cache")


image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(*_ML_PIP)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    # --- tail layers (leave the heavy cached layers above untouched) ---------
    .apt_install("stockfish")
    .pip_install(*_API_PIP)
    .env({"STOCKFISH_PATH": STOCKFISH_BIN})
)
if modal.is_local():
    # Bake ONLY the packages the server imports (config / src / prompts) — keep the
    # image lean (no data/, models/, node_modules, .git).
    image = (
        image
        .add_local_dir((REPO_ROOT / "config").as_posix(), "/root/config", copy=True)
        .add_local_dir((REPO_ROOT / "src").as_posix(), "/root/src", copy=True)
        .add_local_dir((REPO_ROOT / "prompts").as_posix(), "/root/prompts", copy=True)
    )
# Pre-download the base weights LAST so a source edit never re-triggers the 18 GB pull.
image = image.run_function(_bake_base_model, secrets=[hf_secret])

volume = modal.Volume.from_name(VOLUME_NAME)   # existing; contains /chess-coach-v3/adapter
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Generation backend: a drop-in replacement for src.api.server.Coach that uses
# transformers+peft on the GPU instead of mlx_lm. Same .run(system, user) API.
# --------------------------------------------------------------------------- #
class _TransformersCoach:
    """Turns (system, user) into coaching text via a 4-bit Qwen3-32B + v3 LoRA.

    Mirrors ``src.api.server.Coach.run``: applies the model's chat template with
    ``enable_thinking=False``, generates with the same Qwen3 non-thinking sampling
    the local coach uses (so the faithfulness gate's re-samples differ), and strips
    any ``<think>`` block. Generation is serialized behind a lock (one GPU).
    """

    # Match src.api.server.GEN_* (Qwen3 non-thinking recommended sampling) and add
    # the light repetition guards the v3 eval used to tame 32B greedy degeneration.
    MAX_TOKENS: int = 640
    TEMP: float = 0.7
    TOP_P: float = 0.8
    TOP_K: int = 20
    REPETITION_PENALTY: float = 1.15
    NO_REPEAT_NGRAM: int = 4

    def __init__(self, model, tokenizer, strip_think) -> None:
        import threading

        self.model = model
        self.tok = tokenizer
        self._strip_think = strip_think
        self._lock = threading.Lock()

    def run(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            ids = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:  # tokenizer without the enable_thinking kwarg
            ids = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            )
        ids = ids.to(self.model.device)
        eos = self.tok.eos_token_id
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos
        with self._lock, torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.TEMP,
                top_p=self.TOP_P,
                top_k=self.TOP_K,
                repetition_penalty=self.REPETITION_PENALTY,
                no_repeat_ngram_size=self.NO_REPEAT_NGRAM,
                pad_token_id=pad,
            )
        text = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return self._strip_think(text)


def _load_transformers_peft():
    """Primary loader: plain transformers 4-bit base + peft LoRA. Returns (model, tok)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": 0}, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return model, tok


def _load_unsloth():
    """Fallback loader: the exact path eval_modal_v3.py used (base 4-bit + adapter)."""
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR, max_seq_length=3072, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@app.cls(
    image=image,
    gpu=GPU,
    volumes={VOL_MOUNT: volume},
    secrets=[hf_secret],
    scaledown_window=SCALEDOWN_S,
    timeout=REQUEST_TIMEOUT_S,
    max_containers=1,          # one GPU is plenty for a demo; caps idle/burst cost
)
@modal.concurrent(max_inputs=8)  # generation is lock-serialized; lets health/preflight through
class CoachV3:
    @modal.enter()
    def load(self) -> None:
        import os
        import shutil
        import sys
        import time

        # The repo uses namespace packages that assume the repo root is importable.
        sys.path.insert(0, "/root")

        # Set the server's env-driven config BEFORE importing it (module-level consts).
        os.environ["STOCKFISH_PATH"] = shutil.which("stockfish") or STOCKFISH_BIN
        os.environ["COACH_MODEL_PATH"] = MODEL_LABEL          # -> meta.model + tuned=True
        os.environ["COACH_ADAPTER_PATH"] = ADAPTER_DIR        # -> _is_tuned() True
        os.environ.setdefault("COACH_MAX_ATTEMPTS", "6")
        os.environ.setdefault("COACH_FAITHFULNESS_GATE", "1")

        volume.reload()  # make sure the committed adapter is visible

        t0 = time.time()
        try:
            model, tok = _load_transformers_peft()
            backend = "transformers+peft"
        except Exception as exc:  # noqa: BLE001 - fall back to the proven Unsloth loader
            print(f"[serve] transformers+peft load failed ({exc!r}); trying Unsloth")
            model, tok = _load_unsloth()
            backend = "unsloth(fallback)"
        print(f"[serve] loaded {BASE_MODEL} + v3 adapter via {backend} "
              f"in {time.time() - t0:.1f}s")

        # Import the repo's FastAPI app and swap ONLY the generation backend. The
        # app's lifespan calls ``Coach(COACH_MODEL_PATH, COACH_ADAPTER_PATH)`` — we
        # make that return our already-loaded GPU coach (no mlx_lm import, no reload).
        from src.api import server

        coach = _TransformersCoach(model, tok, server._strip_think)
        server.Coach = lambda *a, **k: coach
        self._app = server.app
        print("[serve] FastAPI coach app ready (v3 gated pipeline, transformers backend)")

    @modal.asgi_app()
    def fastapi_app(self):
        return self._app
