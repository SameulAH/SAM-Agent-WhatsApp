import json
import re
import requests
from .base import ModelBackend
from .types import ModelRequest, ModelResponse

# ──────────────────────────────────────────────────────────────────────────────
# Import the authoritative SYSTEM_PROMPT from the prompt builder layer.
# This replaces the previously inline _SYSTEM_PROMPT so the behavioral
# contract is defined in a single place.
# ──────────────────────────────────────────────────────────────────────────────
from agent.prompting.prompt_builder import SYSTEM_PROMPT as _SYSTEM_PROMPT  # noqa: E402

# Regex to locate the [TOOL_CALL] marker (case-insensitive for robustness)
_TOOL_CALL_MARKER_RE = re.compile(r"\[TOOL_CALL\]", re.IGNORECASE)
# Fallback: pull the query field out of truncated JSON.
# The closing quote is optional so truncated values like "define got your messa)
# (missing the closing ") are still captured.
_QUERY_FALLBACK_RE = re.compile(r'"query"\s*:\s*"([^"]*)"?')


def _extract_tool_call(output: str):
    """
    Find [TOOL_CALL]{...} in model output using brace-counting.

    The old single-regex approach ({.*?}) stops at the FIRST closing brace,
    which is the inner 'arguments' object — producing invalid JSON every time.
    Brace-counting correctly handles nested objects.

    Returns:
        (tool_call_dict, cleaned_output) on success
        (None,           cleaned_output) if marker found but JSON unrecoverable
        (None,           original_output) if no marker at all
    """
    marker = _TOOL_CALL_MARKER_RE.search(output)
    if not marker:
        return None, output

    # Always strip [TOOL_CALL] and everything after it from the visible output
    clean = output[: marker.start()].strip()

    brace_start = output.find("{", marker.end())
    if brace_start == -1:
        return None, clean  # marker present but no JSON body

    # Walk forward counting braces to find the matching close
    depth = 0
    brace_end = -1
    for i, ch in enumerate(output[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    json_str = output[brace_start:brace_end] if brace_end != -1 else output[brace_start:]

    try:
        return json.loads(json_str), clean
    except json.JSONDecodeError:
        # Truncated / malformed JSON — try to at least salvage the query string
        q = _QUERY_FALLBACK_RE.search(json_str)
        if q:
            return {
                "name": "web_search",
                "arguments": {"query": q.group(1)},
            }, clean
        return None, clean  # nothing usable, but marker is still hidden from user


class OllamaModelBackend(ModelBackend):
    """
    Ollama backend for local model inference.

    Uses /api/chat so a system prompt can be included.
    Parses [TOOL_CALL]{...} markers from the model output and stores the
    decoded tool call in ModelResponse.metadata["tool_call"] so the
    LangGraph orchestrator can route to the tool execution node.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama backend.

        Args:
            model_name: Name of the model (e.g. "phi3:mini", "llama3")
            base_url:   Base URL of the Ollama service
        """
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Generate a response using Ollama /api/chat.

        Flow:
          1. Build a messages list: [system, (optional context), user]
          2. POST to /api/chat
          3. Parse response for [TOOL_CALL] marker
          4. If found → set metadata["tool_call"] so orchestrator routes to tool node
          5. If request.context is provided (second call after tool execution) →
             prepend tool results to the user message so the model can cite them.

        Args:
            request: ModelRequest with prompt, optional context, and timeout

        Returns:
            ModelResponse with output and tool_call in metadata when applicable
        """
        base_metadata = {
            "backend": "ollama",
            "model": self.model_name,
            "trace_id": request.trace_id,
        }

        try:
            messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

            # Build user message — inject tool results on second call
            if request.context:
                user_content = (
                    f"Search results retrieved for your query:\n\n"
                    f"{request.context}\n\n"
                    f"---\n"
                    f"Using the above results, please answer:\n{request.prompt}"
                )
            else:
                user_content = request.prompt

            messages.append({"role": "user", "content": user_content})

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
            }

            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=request.timeout_s,
            )

            resp.raise_for_status()
            data = resp.json()
            output: str = data.get("message", {}).get("content", "")

            # ── Parse [TOOL_CALL] marker ────────────────────────────────────
            metadata = dict(base_metadata)
            tool_call_data, output = _extract_tool_call(output)
            if tool_call_data:
                metadata["tool_call"] = tool_call_data

            return ModelResponse(
                status="success",
                output=output,
                metadata=metadata,
            )

        except requests.Timeout:
            return ModelResponse(
                status="recoverable_error",
                error_type="timeout",
                metadata=base_metadata,
            )

        except Exception as e:
            return ModelResponse(
                status="fatal_error",
                error_type="backend_unavailable",
                metadata={**base_metadata, "error": str(e)},
            )
