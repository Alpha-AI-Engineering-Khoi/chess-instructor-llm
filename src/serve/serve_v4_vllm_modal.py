#!/usr/bin/env python3
"""Serve the **v4** chess coach (Qwen3-32B + v4 QLoRA LoRA) as a scale-to-zero Modal
GPU endpoint using **vLLM** — the shipped/live product model.

This is a verbatim clone of ``serve_v3_vllm_modal.py`` with ONLY the adapter identity
swapped from v3 → v4. v4 is the chosen main model: a QLoRA LoRA (r=32) trained on
``unsloth/Qwen3-32B-unsloth-bnb-4bit`` (same base + rank + target modules as v3), so it
applies cleanly, by module name, to the canonical full-precision ``Qwen/Qwen3-32B`` in
BF16 on an H100-80GB. Everything else — the gated FastAPI pipeline, the ``CoachResponse``
contract, sampling, CORS, scale-to-zero — is inherited unchanged from the v3 vLLM serve.

What it reuses (model-agnostic — the whole point)
------------------------------------------------
The full gated coach pipeline lives in :mod:`src.api.server` and is entirely
backend-agnostic: Stockfish sound-pool + Maia (best-effort) + verified-facts grounding
(``render_pool_facts`` + ``render_user_prompt``) + a VERIFY-AND-REGENERATE faithfulness
gate (``verify_text_ext``, up to ``COACH_MAX_ATTEMPTS``) + a deterministic engine-derived
fallback. We import that FastAPI app verbatim and only replace the *generation backend*
with a vLLM ``LLM.generate`` + a per-request ``LoRARequest`` (the v4 adapter). Every
route, response shape (``CoachResponse`` with ``meta.attempts`` / ``meta.verified_fallback``),
and CORS rule is inherited unchanged.

Adapter source
--------------
The v4 LoRA lives on the Hub at ``khoilamalphaai/chess-coach-modal-backup`` under
``v4-lora-qwen3-32b/`` (the durable, account-independent backup). It is pulled into the
container at start (Volume copy preferred if present, HF fallback otherwise). The base
``Qwen/Qwen3-32B`` is baked into the image at build time so scale-to-zero cold starts
don't re-download ~64 GB.

Deploy (chess-instructor-3 workspace — the idle budget, so we don't contend with the
finish-v5 controller on chess-instructor-4)::

    export MODAL_PROFILE=chess-instructor-3
    modal deploy src/serve/serve_v4_vllm_modal.py

Maia note: lc0 + Maia nets are not installed on Modal, so the human-likelihood signal
degrades gracefully (``maia: []`` + a note) exactly as the local API does when Maia is
unavailable. Stockfish IS installed (apt) and used for real.

No secrets are hardcoded: the HF token (for the base pull / adapter fallback) comes from
the ``chess-hf`` Modal secret; Modal auth comes from the ambient profile.
"""
from __future__ import annotations

from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Names / paths
# --------------------------------------------------------------------------- #
#: v4 app name (the live product endpoint). Deployed on chess-instructor-3.
APP_NAME: str = "chess-coach-v4-vllm"
VOLUME_NAME: str = "chess-coach-lora"          # shared volume (v4 adapter pulled from HF)
RUN_NAME: str = "chess-coach-v4"
VOL_MOUNT: str = "/vol"
ADAPTER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/adapter"

#: vLLM-friendly base to serve. The v4 adapter was TRAINED on the bnb-4bit copy
#: (``unsloth/Qwen3-32B-unsloth-bnb-4bit``) but applies cleanly, by module name, to
#: the canonical full-precision weights. BF16 on an H100-80GB is the cleanest fit.
BASE_MODEL: str = "Qwen/Qwen3-32B"
#: Published HF fallback for the adapter (repo, subfolder) if the Volume copy is gone.
HF_ADAPTER_REPO: str = "khoilamalphaai/chess-coach-modal-backup"
HF_ADAPTER_SUBFOLDER: str = "v4-lora-qwen3-32b"
#: Where the HF-fallback adapter is materialized inside the container (if used).
HF_ADAPTER_LOCAL: str = "/root/_v4_adapter_hf"

#: Human-readable label surfaced as ``meta.model`` / on /api/health.
MODEL_LABEL: str = "Qwen3-32B + chess-coach-v4 QLoRA (vLLM BF16, Modal)"

#: LoRA rank from the v4 adapter_config.json (r=32). vLLM needs this at engine init.
LORA_RANK: int = 32

#: BF16 32B (~64 GB) needs an 80 GB card; H100-80GB is the fast, comfortable fit.
GPU: str = "H100"
#: Scale-to-zero after 5 min idle (cost-aware: this workspace has limited credits).
SCALEDOWN_S: int = 300
#: A single request runs the gate up to COACH_MAX_ATTEMPTS x; keep the ceiling generous.
REQUEST_TIMEOUT_S: int = 900

#: vLLM engine sizing. max_model_len covers the grounded prompt (~1.5-3k tok) + 640
#: gen with headroom; KV cache is sized by gpu_memory_utilization, not this cap.
MAX_MODEL_LEN: int = 6144
GPU_MEM_UTIL: float = 0.90
#: Eager mode avoids CUDA-graph capture memory spikes on a near-full 80 GB card
#: (correctness/robustness first); single-stream H100 decode is still fast.
ENFORCE_EAGER: bool = True

CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION: str = "3.11"
STOCKFISH_BIN: str = "/usr/games/stockfish"    # debian/ubuntu apt install location

#: vLLM (brings a compatible torch/transformers/fastapi/pydantic/uvicorn) + fast HF DL.
_ML_PIP: list[str] = ["vllm", "huggingface_hub", "hf_transfer"]
#: Thin HTTP + chess layer the repo's server needs (added as a tail layer so the heavy
#: vLLM layer stays cached). fastapi/uvicorn/pydantic are already pulled by vLLM; listed
#: unpinned here only so a missing extra can't break the import — pip keeps vLLM's pins.
_API_PIP: list[str] = ["python-chess", "python-dotenv", "fastapi", "uvicorn", "pydantic"]

if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = None

#: HF token for the base-model pull + adapter fallback. Created from .env before deploy:
#:   modal secret create chess-hf HF_TOKEN=... HUGGING_FACE_HUB_TOKEN=...
hf_secret = modal.Secret.from_name("chess-hf")


def _bake_base_model() -> None:
    """Download the BF16 base into the image's HF cache (baked in => fast cold start)."""
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
# Pre-download the base weights LAST so a source edit never re-triggers the 64 GB pull.
image = image.run_function(_bake_base_model, secrets=[hf_secret])

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Generation backend: a drop-in replacement for src.api.server.Coach that uses
# vLLM (base BF16 + per-request LoRA) instead of mlx_lm. Same .run(system, user) API.
# --------------------------------------------------------------------------- #
class _VLLMCoach:
    """Turns (system, user) into coaching text via Qwen3-32B (BF16) + the v4 LoRA on vLLM.

    Mirrors ``src.api.server.Coach.run``: applies the model's chat template with
    ``enable_thinking=False``, generates with the same Qwen3 non-thinking sampling the
    local coach uses (so the faithfulness gate's re-samples differ), and strips any
    ``<think>`` block. Generation is serialized behind a lock (one engine, one GPU);
    SamplingParams is left unseeded so each gate retry samples a genuinely new draft.
    """

    # Match src.api.server.GEN_* (Qwen3 non-thinking recommended sampling) + the light
    # repetition guard the v3/v4 eval used to tame 32B decode degeneration. (vLLM has no
    # no_repeat_ngram_size; repetition_penalty is the equivalent guard.)
    MAX_TOKENS: int = 640
    TEMP: float = 0.7
    TOP_P: float = 0.8
    TOP_K: int = 20
    REPETITION_PENALTY: float = 1.15

    def __init__(self, llm, tokenizer, lora_request, strip_think) -> None:
        import threading

        from vllm import SamplingParams

        self.llm = llm
        self.tok = tokenizer
        self.lora_request = lora_request
        self._strip_think = strip_think
        self._SamplingParams = SamplingParams
        self._lock = threading.Lock()

    def _render(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:  # tokenizer without the enable_thinking kwarg
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def run(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        prompt = self._render(system, user)
        params = self._SamplingParams(
            temperature=self.TEMP,
            top_p=self.TOP_P,
            top_k=self.TOP_K,
            repetition_penalty=self.REPETITION_PENALTY,
            max_tokens=max_tokens,
        )
        with self._lock:
            outputs = self.llm.generate(
                [prompt], params, lora_request=self.lora_request, use_tqdm=False,
            )
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return self._strip_think(text)


def _resolve_adapter_dir() -> str:
    """Return a local path to the v4 adapter — Volume copy preferred, HF fallback."""
    import os

    cfg = os.path.join(ADAPTER_DIR, "adapter_config.json")
    weights = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    if os.path.isfile(cfg) and os.path.isfile(weights):
        print(f"[serve] using v4 LoRA from Volume: {ADAPTER_DIR}")
        return ADAPTER_DIR

    # Fallback: pull the published adapter subfolder from the Hub into a local dir.
    from huggingface_hub import snapshot_download

    print(f"[serve] Volume adapter missing; pulling {HF_ADAPTER_REPO}"
          f"/{HF_ADAPTER_SUBFOLDER} from HF -> {HF_ADAPTER_LOCAL}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    local = snapshot_download(
        HF_ADAPTER_REPO, allow_patterns=[f"{HF_ADAPTER_SUBFOLDER}/*"], token=token,
    )
    resolved = os.path.join(local, HF_ADAPTER_SUBFOLDER)
    print(f"[serve] using v4 LoRA from HF fallback: {resolved}")
    return resolved


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
class CoachV4VLLM:
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
        # Keep the faithfulness gate ON; 4 attempts gives timeout-safety margin on vLLM.
        os.environ.setdefault("COACH_MAX_ATTEMPTS", "4")
        os.environ.setdefault("COACH_FAITHFULNESS_GATE", "1")

        volume.reload()  # make sure any committed adapter is visible

        # Resolve the adapter (may pull from HF) BEFORE going offline for the base load.
        adapter_dir = _resolve_adapter_dir()

        # The base weights are baked into the image cache — load them offline so a cold
        # start never blocks on a network round-trip to the Hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(adapter_dir)
        llm = LLM(
            model=BASE_MODEL,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=LORA_RANK,
            max_loras=1,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            enforce_eager=ENFORCE_EAGER,
            tensor_parallel_size=1,
            disable_log_stats=True,
        )
        lora_request = LoRARequest("chess-coach-v4", 1, adapter_dir)
        print(f"[serve] loaded {BASE_MODEL} (BF16) + v4 LoRA via vLLM "
              f"in {time.time() - t0:.1f}s")

        # Import the repo's FastAPI app and swap ONLY the generation backend. The app's
        # lifespan calls ``Coach(COACH_MODEL_PATH, COACH_ADAPTER_PATH)`` — we make that
        # return our already-loaded vLLM coach (no mlx_lm import, no reload).
        from src.api import server

        coach = _VLLMCoach(llm, tok, lora_request, server._strip_think)
        server.Coach = lambda *a, **k: coach
        self._app = server.app
        print("[serve] FastAPI coach app ready (v4 gated pipeline, vLLM backend)")

    @modal.asgi_app()
    def fastapi_app(self):
        return self._app
