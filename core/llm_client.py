"""
Thin wrapper around the local Ollama HTTP API.

Runs on the USER'S machine (not this sandbox) — this sandbox has no
network access to reach localhost:11434 on the user's side, so this code
is written but not executed here. The user runs it and pastes back output.
"""
import json
import os
import time
import urllib.request
import urllib.error

OLLAMA_HOST = os.environ.get("VLEG_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL_NAME = "llama3.3:70b"
EMBED_MODEL_NAME = "nomic-embed-text"


class OllamaError(RuntimeError):
    pass


# --- Token usage tracking, added for Experiment 4 (Parameter Sensitivity
# and Overhead Analysis, paper Section IV-E). Ollama's /api/chat response
# includes prompt_eval_count and eval_count (input/output token counts)
# when available. Accumulates into a module-level counter that callers
# can reset/read around a block of work.
_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def reset_token_usage() -> None:
    _token_usage["prompt_tokens"] = 0
    _token_usage["completion_tokens"] = 0
    _token_usage["calls"] = 0


def get_token_usage() -> dict:
    return dict(_token_usage)


def chat(messages, model: str = MODEL_NAME, temperature: float = 0.0,
         timeout: int = 600, max_retries: int = 3, num_ctx: int | None = None, num_predict: int | None = None) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Returns the assistant's text content.
    Raises OllamaError with a clear message if all retries are exhausted.

    timeout default raised from 120s to 600s: with large local models
    (e.g. 70B), a single call — especially HDR's global-hypothesis calls,
    which embed the ENTIRE labeled training set in one prompt — can
    legitimately take several minutes. 120s was tuned for 7B and is too
    tight for larger models, and a mid-run timeout previously discarded
    an entire experiment's already-completed results (see
    DEVIATIONS_LOG.md #19) because nothing was persisted until the very
    end. Retries handle transient timeouts/connection blips without
    losing the whole run to a single flaky call.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if num_ctx is not None:
        payload["options"]["num_ctx"] = num_ctx
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict
    data = json.dumps(payload).encode("utf-8")

    last_error = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break  # success
        except urllib.error.URLError as e:
            last_error = OllamaError(
                f"Could not reach Ollama at {OLLAMA_HOST}. "
                f"Is `ollama serve` running? Underlying error: {e}"
            )
        except Exception as e:
            last_error = OllamaError(f"Ollama request failed: {e}")

        if attempt < max_retries:
            wait_s = 10 * attempt  # 10s, 20s, ... linear backoff
            print(f"  [llm_client] chat() attempt {attempt}/{max_retries} failed "
                  f"({last_error}); retrying in {wait_s}s...")
            time.sleep(wait_s)
    else:
        raise last_error

    if "message" not in body or "content" not in body.get("message", {}):
        raise OllamaError(f"Unexpected Ollama response shape: {body}")

    _token_usage["prompt_tokens"] += body.get("prompt_eval_count", 0) or 0
    _token_usage["completion_tokens"] += body.get("eval_count", 0) or 0
    _token_usage["calls"] += 1

    return body["message"]["content"]


def embed(text: str, model: str = EMBED_MODEL_NAME, timeout: int = 120, max_retries: int = 3) -> list[float]:
    """Returns the embedding vector for `text` using Ollama's embeddings API."""
    payload = {"model": model, "prompt": text}
    data = json.dumps(payload).encode("utf-8")

    last_error = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/embeddings",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.URLError as e:
            last_error = OllamaError(
                f"Could not reach Ollama at {OLLAMA_HOST} for embeddings. "
                f"Is `ollama serve` running? Underlying error: {e}"
            )
        except Exception as e:
            last_error = OllamaError(f"Ollama embeddings request failed: {e}")

        if attempt < max_retries:
            wait_s = 10 * attempt
            print(f"  [llm_client] embed() attempt {attempt}/{max_retries} failed "
                  f"({last_error}); retrying in {wait_s}s...")
            time.sleep(wait_s)
    else:
        raise last_error

    if "embedding" not in body:
        raise OllamaError(f"Unexpected Ollama embeddings response shape: {body}")
    return body["embedding"]


def check_connection(model: str = MODEL_NAME) -> None:
    """Quick smoke test — raises OllamaError with a clear message if unusable."""
    reply = chat(
        [{"role": "user", "content": "Reply with exactly: OK"}],
        model=model,
    )
    if "OK" not in reply.upper():
        print(f"[warn] Ollama reachable but reply looked unexpected: {reply!r}")
    else:
        print(f"[ok] Ollama reachable, model={model!r} responded correctly.")

    vec = embed("connection test", model=EMBED_MODEL_NAME)
    print(f"[ok] Embeddings reachable, model={EMBED_MODEL_NAME!r}, "
          f"vector dim={len(vec)}.")
