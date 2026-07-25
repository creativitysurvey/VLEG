"""
Thin wrapper around the local Ollama HTTP API.

Runs on the USER'S machine (not this sandbox) — this sandbox has no
network access to reach localhost:11434 on the user's side, so this code
is written but not executed here. The user runs it and pastes back output.
"""
import json
import urllib.request
import urllib.error

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b-instruct"  # matches user's `ollama list` output
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
         timeout: int = 120) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Returns the assistant's text content.
    Raises OllamaError with a clear message on connection/HTTP failure.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            f"Is `ollama serve` running? Underlying error: {e}"
        ) from e
    except Exception as e:
        raise OllamaError(f"Ollama request failed: {e}") from e

    if "message" not in body or "content" not in body.get("message", {}):
        raise OllamaError(f"Unexpected Ollama response shape: {body}")

    _token_usage["prompt_tokens"] += body.get("prompt_eval_count", 0) or 0
    _token_usage["completion_tokens"] += body.get("eval_count", 0) or 0
    _token_usage["calls"] += 1

    return body["message"]["content"]


def embed(text: str, model: str = EMBED_MODEL_NAME, timeout: int = 60) -> list[float]:
    """Returns the embedding vector for `text` using Ollama's embeddings API."""
    payload = {"model": model, "prompt": text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/embeddings",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_HOST} for embeddings. "
            f"Is `ollama serve` running? Underlying error: {e}"
        ) from e
    except Exception as e:
        raise OllamaError(f"Ollama embeddings request failed: {e}") from e

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
