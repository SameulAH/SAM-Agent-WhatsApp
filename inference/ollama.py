import json
import re
import requests
from .base import ModelBackend
from .types import ModelRequest, ModelResponse

# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions (injected into the system prompt so the model knows what's
# available and when to call them).
# ──────────────────────────────────────────────────────────────────────────────
_TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for real-time or up-to-date information. "
            "Use this tool whenever the user asks about: current news, recent events, "
            "live prices, product availability, stock quotes, weather, sports scores, "
            "latest releases, or any fact that may have changed after your training cutoff."
        ),
        "parameters": {
            "query": "The search query string (concise, specific, in the user's language)"
        },
    }
]

_TOOL_JSON = json.dumps(_TOOL_DEFINITIONS, indent=2)

_SYSTEM_PROMPT = f"""You are SAM, a helpful and concise AI assistant. You have access to the following tools:

{_TOOL_JSON}

TOOL USAGE RULES:
- When you need real-time or current information, respond with EXACTLY this format and NOTHING else:
  [TOOL_CALL]{{"name": "web_search", "arguments": {{"query": "<your search query>"}}}}
- Do NOT add any explanation or text around the [TOOL_CALL] block.
- After receiving tool results, answer the user's question using those results.
- If you do NOT need a tool, respond normally without any [TOOL_CALL] marker.
"""

# Regex to extract [TOOL_CALL]{...} from model output
_TOOL_CALL_RE = re.compile(r"\[TOOL_CALL\]\s*(\{.*?\})", re.DOTALL)


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
            match = _TOOL_CALL_RE.search(output)
            if match:
                try:
                    tool_call_data = json.loads(match.group(1))
                    metadata["tool_call"] = tool_call_data
                    # Remove the raw marker from the visible output
                    output = _TOOL_CALL_RE.sub("", output).strip()
                except json.JSONDecodeError:
                    pass  # Malformed JSON — treat as plain text, no tool call

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
