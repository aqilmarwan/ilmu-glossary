"""vLLM server lifecycle.

Every checkpoint is served with **identical flags** apart from the model path
and the Mamba SSM state dtype arm. If serving settings drifted between
checkpoints the tier 4e throughput comparison would measure the flags rather
than the quantization, and spec section 4e exists precisely to confirm that
recalibration costs no throughput.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ilmu_glossary.config import Config, MambaStateDtype

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerHandle:
    """A running vLLM server."""

    base_url: str
    model_path: str
    mamba_state: MambaStateDtype
    process: subprocess.Popen[bytes] | None

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"


def _clamp_to_model_capacity(model_path: str, requested: int, cfg: Config) -> int:
    """Clamp the serving context to what the model actually supports.

    vLLM refuses to start when max_model_len exceeds the checkpoint's
    max_position_embeddings, and the documented override
    (VLLM_ALLOW_LONG_MAX_MODEL_LEN) risks NaNs under RoPE or out-of-bounds
    reads under absolute position encoding - not something to enable silently
    in a measurement pipeline.

    Reading the capacity from the checkpoint keeps one config value working
    across a 32K proxy and a 1M-context Nemotron.
    """
    try:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=cfg.model.trust_remote_code
        )
    except Exception as exc:
        logger.warning("Could not read %s config (%r); serving at %d", model_path, exc, requested)
        return requested

    derived = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(derived, int | float) or derived <= 0:
        return requested

    capacity = int(derived)
    if requested <= capacity:
        return requested

    logger.warning(
        "Requested max_model_len=%d exceeds the model's %d; serving at %d. "
        "Evaluation inputs longer than this are truncated, so record it.",
        requested,
        capacity,
        capacity,
    )
    return capacity


def _wait_for_health(
    base_url: str,
    *,
    timeout_s: int = 1800,
    interval_s: float = 5.0,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    """Block until the server answers /health.

    A 30B hybrid Mamba model with a long context takes minutes to load, and
    NVFP4 kernel autotuning adds more, so the timeout is generous. It is only
    a backstop though: if vLLM exits, waiting the full 30 minutes for a
    process that is already dead burns half an hour of B200 time for nothing,
    so the subprocess is polled every interval and its exit is fatal
    immediately.
    """
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    waited = 0.0

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {process.returncode} before becoming "
                f"healthy, after {waited:.0f}s. Its traceback is above."
            )
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
                if response.status == 200:
                    logger.info("vLLM healthy at %s after %.0fs", base_url, waited)
                    return
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
        time.sleep(interval_s)
        waited += interval_s
        # Startup is long and quiet; say something so a stalled load is
        # distinguishable from a slow one.
        if waited % 60 == 0:
            logger.info("waiting for vLLM (%.0fs elapsed)", waited)

    raise TimeoutError(f"vLLM did not become healthy within {timeout_s}s: {last_error!r}")


@contextlib.contextmanager
def serve(
    cfg: Config,
    model_path: str,
    *,
    mamba_state: MambaStateDtype = MambaStateDtype.FP16_SR,
    max_model_len: int | None = None,
    reuse_existing: bool = True,
) -> Iterator[ServerHandle]:
    """Start vLLM, yield a handle, and shut it down cleanly.

    `reuse_existing` lets a developer attach to a server they started by hand
    rather than paying the load time again.
    """
    base_url = f"http://127.0.0.1:{cfg.vllm.port}"

    if reuse_existing:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=3) as response:
                if response.status == 200:
                    logger.info("Reusing the vLLM server already listening on %s", base_url)
                    yield ServerHandle(base_url, model_path, mamba_state, None)
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass

    requested = max_model_len or cfg.model.eval_max_model_len
    effective = _clamp_to_model_capacity(model_path, requested, cfg)

    vllm_cfg = cfg.vllm
    if cfg.dry_run and not vllm_cfg.enforce_eager:
        # Graph capture is minutes of B200 time that a plumbing run does not
        # need, and it hung outright on the proxy model.
        vllm_cfg = vllm_cfg.model_copy(update={"enforce_eager": True, "max_num_seqs": 32})
        logger.info("Dry run: serving eager with reduced concurrency")

    args = vllm_cfg.serve_args(
        model_path,
        mamba_state=mamba_state,
        max_model_len=effective,
        max_logprobs=cfg.eval.kl_top_k,
        # The dry-run proxy is a plain transformer MoE; Mamba/Nemotron flags
        # would make vLLM refuse to start.
        hybrid_mamba=not cfg.dry_run,
    )
    if cfg.model.trust_remote_code:
        args.append("--trust-remote-code")

    logger.info("Starting vLLM: %s", " ".join(args))
    # INFO, not WARNING: startup is the part that fails, and suppressing it
    # makes a crashed load indistinguishable from a slow one in the logs.
    env = {**os.environ, "VLLM_LOGGING_LEVEL": "INFO"}
    process = subprocess.Popen(args, env=env)

    try:
        _wait_for_health(base_url, process=process)
        yield ServerHandle(base_url, model_path, mamba_state, process)
    finally:
        logger.info("Stopping vLLM (pid %s)", process.pid)
        process.terminate()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM did not exit within 120s; killing")
            process.kill()
            process.wait(timeout=30)


# --------------------------------------------------------------------------
# request helpers
# --------------------------------------------------------------------------


def post_json(url: str, payload: dict[str, Any], *, timeout: int = 600) -> dict[str, Any]:
    """POST JSON and parse the response."""
    import json

    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result: dict[str, Any] = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # vLLM returns the reason in the body. Letting the bare status through
        # turns a one-line config problem ("logprobs exceeds max_logprobs")
        # into a blind debugging session.
        try:
            detail = exc.read().decode()[:600]
        except Exception:
            detail = "<no body>"
        raise RuntimeError(f"vLLM returned HTTP {exc.code} for {url}: {detail}") from exc
    return result


def completion_logprobs(
    handle: ServerHandle,
    prompt: str,
    *,
    top_k: int,
    max_tokens: int = 0,
    echo: bool = True,
) -> list[dict[str, Any]]:
    """Score a prompt and return per-token top-K logprobs.

    `max_tokens=0` with `echo=True` scores the prompt itself without
    generating, which is what both the KL and perplexity tiers need - they
    measure how the model assigns probability to given text, not what it
    would produce.
    """
    payload = {
        "model": handle.model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "echo": echo,
        "logprobs": top_k,
        "temperature": 0.0,
    }
    response = post_json(handle.completions_url, payload)
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}

    tokens = logprobs.get("tokens", [])
    top_logprobs = logprobs.get("top_logprobs") or []
    token_logprobs = logprobs.get("token_logprobs") or []

    out: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        # With echo=True the first prompt token has no predecessor, so vLLM
        # reports `null` for both its logprob and its top_logprobs rather than
        # an empty dict. Normalising here keeps every caller from having to
        # know that.
        distribution = top_logprobs[i] if i < len(top_logprobs) else None
        out.append(
            {
                "token": token,
                "logprob": token_logprobs[i] if i < len(token_logprobs) else None,
                "top_logprobs": distribution if isinstance(distribution, dict) else {},
            }
        )
    return out


def model_id(handle: ServerHandle) -> str:
    """The id vLLM reports, which may differ from the path passed in."""
    try:
        with urllib.request.urlopen(f"{handle.base_url}/v1/models", timeout=30) as response:
            import json

            payload = json.loads(response.read())
        return str(payload["data"][0]["id"])
    except Exception:
        return handle.model_path


__all__ = [
    "ServerHandle",
    "completion_logprobs",
    "model_id",
    "post_json",
    "serve",
]
