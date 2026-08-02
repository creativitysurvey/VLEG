"""
Ollama HTTP client with vision support, for the real-image (CLEVR)
version of the pipeline. Parallel to core/llm_client.py, not shared with
it, so the already-validated symbolic 7B/70B results stay untouched.

Ollama's /api/chat accepts an "images" field per message: a list of
base64-encoded image strings (no data URI prefix, just raw base64).
"""
import base64
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "qwen2.5vl:72b"
EMBED_MODEL_NAME = "nomic-embed-text"  # embeddings stay text-only (over reflections)


class OllamaError(RuntimeError):
    pass


_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def reset_token_usage() -> None:
    _token_usage["prompt_tokens"] = 0
    _token_usage["completion_tokens"] = 0
    _token_usage["calls"] = 0


def get_token_usage() -> dict:
    return dict(_token_usage)


def encode_image(path: str) -> str:
    """Reads an image file and returns raw base64 (no data-URI prefix)."""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def chat(messages, model: str = MODEL_NAME, temperature: float = 0.0,
         timeout: int = 600, max_retries: int = 3, images: list = None) -> str:
    """
    messages: list of {"role": ..., "content": ...}
    images: optional list of local file paths -- attached to the LAST
            message in `messages` (matching Ollama's per-message
            "images" field convention). Pass paths, not base64 -- this
            function does the encoding.
    """
    msgs = [dict(m) for m in messages]  # shallow copy so we don't mutate caller's list
    if images:
        msgs[-1] = dict(msgs[-1])
        msgs[-1]["images"] = [encode_image(p) for p in images]

    payload = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {"temperature": temperature},
    }
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
            break
        except urllib.error.URLError as e:
            last_error = OllamaError(
                f"Could not reach Ollama at {OLLAMA_HOST}. "
                f"Is `ollama serve` running? Underlying error: {e}"
            )
        except Exception as e:
            last_error = OllamaError(f"Ollama request failed: {e}")

        if attempt < max_retries:
            wait_s = 10 * attempt
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


def embed(text: str, model: str = EMBED_MODEL_NAME, timeout: int = 120, max_retries: int = 3) -> list:
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
    reply = chat([{"role": "user", "content": "Reply with exactly: OK"}], model=model)
    if "OK" not in reply.upper():
        print(f"[warn] Ollama reachable but reply looked unexpected: {reply!r}")
    else:
        print(f"[ok] Ollama reachable, model={model!r} responded correctly.")
    vec = embed("connection test", model=EMBED_MODEL_NAME)
    print(f"[ok] Embeddings reachable, model={EMBED_MODEL_NAME!r}, vector dim={len(vec)}.")


def check_vision(sample_image_path: str) -> None:
    """Sanity check that the model can actually see an attached image."""
    reply = chat(
        [{"role": "user", "content": "What color and shape is the object in this image? "
                                       "Answer in exactly 3 words: <color> <shape>, nothing else."}],
        images=[sample_image_path],
    )
    print(f"[vision check] image={sample_image_path} -> model said: {reply.strip()!r}")
