"""Modal container images.

Two images, because the CPU phases must not wait on a multi-gigabyte CUDA
image and the GPU phases must not resolve torch on the laptop.

  cpu_image  - phase 0 (streaming ingest, LID, stratification) and phase 5
  gpu_image  - phases 1, 3, 4 (routing capture, PTQ, vLLM evaluation)

Version pins live in pyproject.toml's `gpu` extra and are mirrored here.
Keep the two in sync; `tests/test_images.py` asserts they match.
"""

from __future__ import annotations

import modal

PYTHON_VERSION = "3.12"

# vLLM 0.27.1 is the floor for the hybrid Mamba-2 serving path that
# Nemotron 3.5 Lightning requires. NVFP4 tensor cores additionally require
# Blackwell SM100+; the image will build on any host but the kernels only
# exist on B200/B300/GB200.
VLLM_VERSION = "0.27.1"

# transformers 5.x FUSES MoE experts into batched 3D parameters - a single
# Qwen2MoeExperts module carrying gate_up_proj of shape
# (num_experts, 2*intermediate, hidden) - instead of one nn.Linear per expert.
# ModelOpt's Linear-targeting configs then skip every routed expert, and PTQ
# silently produces a checkpoint whose experts are still full precision.
#
# Measured on Qwen1.5-MoE-A2.7B, routed-expert nn.Linear modules:
#   transformers 5.15.0 -> 0
#   transformers 4.57.1 -> 4320   (24 layers x 60 experts x 3 projections)
#
# Serving needs transformers 5.x (vLLM 0.27.1 requires it) while PTQ needs
# 4.x. Since PTQ never imports vLLM, the two live in separate images.
PTQ_TRANSFORMERS_VERSION = "4.57.1"
# vLLM 0.27.1 resolves its own torch; the observed runtime version inside the
# image is 2.13.0+cu130 regardless of what is requested here, so this pin is
# a floor rather than an exact match. Recorded because a silent torch swap
# changes kernel selection.
TORCH_VERSION = "2.9.0"
MODELOPT_VERSION = "0.40.0"
CUDA_TAG = "12.8.1"

_HOST_DEPS = [
    "pydantic>=2.9",
    "pyyaml>=6.0",
    "typer>=0.15",
    "rich>=13.9",
    "pandas>=2.2",
    "pyarrow>=18.0",
    "numpy>=2.1",
    "scipy>=1.14",
    "datasets>=3.2",
    "huggingface-hub[hf_transfer]>=0.27",
    "transformers>=4.48",
    "mlflow>=2.19",
    "tqdm>=4.67",
    "tabulate>=0.9",
]

_ENV = {
    # Streaming the 349 GB Malaysian corpus is bandwidth-bound; hf_transfer
    # roughly triples throughput on Modal's network.
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HOME": "/hf-cache",
    # MLflow 3 refuses to start on a filesystem store unless this is set.
    # The file store is the right backend here despite the deprecation:
    # phase 3 and 4 fan out across many containers writing concurrently to
    # the shared volume, which a single sqlite file would serialise badly.
    "MLFLOW_ALLOW_FILE_STORE": "true",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}


cpu_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "build-essential")
    .pip_install(*_HOST_DEPS)
    # fastText wheels are unreliable on 3.12; build from the pinned source.
    .pip_install("fasttext-wheel>=0.9.2")
    .env(_ENV)
    # Both packages: `modal_app` because app.py imports images.py from it,
    # `ilmu_glossary` because that is the pipeline itself.
    .add_local_python_source("ilmu_glossary", "modal_app")
    # Phase 3 reads recipes/*.yaml at run time; the python-source mount does
    # not carry them, and a missing recipe fails hours into a run.
    .add_local_dir("recipes", remote_path="/root/recipes")
)


gpu_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_TAG}-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    .apt_install("git", "build-essential", "libopenmpi-dev")
    .pip_install(*_HOST_DEPS)
    .pip_install(
        f"torch=={TORCH_VERSION}",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # NB: the [hf] extra conflicts with vllm 0.27.1 (it pins transformers<5
    # while vllm requires 5.x). ModelOpt warns that transformers 5 is untested;
    # the mtq path used here works regardless.
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"nvidia-modelopt=={MODELOPT_VERSION}",
        "accelerate>=1.2",
        "flashinfer-python>=0.2",
    )
    # NVIDIA AIPerf drives tier 4e. It replaces genai-perf and speaks the
    # OpenAI-compatible endpoint vLLM exposes.
    .pip_install("aiperf>=0.1")
    .env(
        {
            **_ENV,
            # NVFP4 kernels are Blackwell-only. Building for SM100 keeps the
            # image from silently falling back to W4A16 Marlin paths.
            "TORCH_CUDA_ARCH_LIST": "10.0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    .add_local_python_source("ilmu_glossary", "modal_app")
    .add_local_dir("recipes", remote_path="/root/recipes")
)


# --------------------------------------------------------------------------
# PTQ image - phases 1 and 3. transformers 4.x, ModelOpt, no vLLM.
# --------------------------------------------------------------------------
#
# Phase 1 shares this image so the expert indices it profiles come from the
# same module layout phase 3 quantizes. Profiling under one layout and
# quantizing under another would silently misalign every expert id.

ptq_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_TAG}-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    .apt_install("git", "build-essential")
    .pip_install(*[d for d in _HOST_DEPS if not d.startswith("transformers")])
    .pip_install(f"transformers=={PTQ_TRANSFORMERS_VERSION}")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(f"nvidia-modelopt=={MODELOPT_VERSION}", "accelerate>=1.2")
    .env({**_ENV, "TORCH_CUDA_ARCH_LIST": "10.0"})
    .add_local_python_source("ilmu_glossary", "modal_app")
    .add_local_dir("recipes", remote_path="/root/recipes")
)


__all__ = [
    "CUDA_TAG",
    "MODELOPT_VERSION",
    "PTQ_TRANSFORMERS_VERSION",
    "PYTHON_VERSION",
    "TORCH_VERSION",
    "VLLM_VERSION",
    "cpu_image",
    "gpu_image",
    "ptq_image",
]
