"""
MCP Guardrails.

Enforces hard limits on tool usage to prevent:
- Infinite loops
- Unbounded content injection
- Memory mutation inside tool nodes
- Auto-retry abuse

All violations emit a GuardrailViolation and are handled gracefully.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.mcp.external_client import BrowserBaseResult


class GuardrailViolation(Exception):
    """
    Raised when a guardrail constraint is violated.

    Non-fatal by design: callers must catch and continue agent flow.
    """

    def __init__(self, rule: str, detail: str = ""):
        self.rule = rule
        self.detail = detail
        super().__init__(f"GuardrailViolation[{rule}]: {detail}")


class MCPGuardrails:
    """
    Stateless guardrail enforcement for MCP tool calls.

    Rules:
    - Max 1 tool call per turn
    - Max 5 results per call
    - Snippet length ≤ 300 chars
    - Total char budget ≤ 1500 chars
    - Timeout fallback (no auto-retry)
    - No memory mutation inside tool node
    """

    MAX_TOOL_CALLS_PER_TURN: int = 1
    MAX_RESULTS: int = 5
    MAX_SNIPPET_LEN: int = 300
    MAX_TOTAL_CHARS: int = 1500
    MCP_TIMEOUT_S: float = 5.0

    @classmethod
    def check_tool_call_limit(cls, tool_call_count: int) -> None:
        """
        Enforce max 1 tool call per turn.

        Args:
            tool_call_count: Number of tool calls already made this turn.

        Raises:
            GuardrailViolation: If limit exceeded.
        """
        if tool_call_count >= cls.MAX_TOOL_CALLS_PER_TURN:
            raise GuardrailViolation(
                "MAX_TOOL_CALLS_PER_TURN",
                f"Tool call limit ({cls.MAX_TOOL_CALLS_PER_TURN}) already reached this turn.",
            )

    @classmethod
    def sanitize_results(cls, results: list) -> list:
        """
        Sanitize and bound MCP results.

        Enforces:
        - URL must start with http
        - Snippet truncated to 300 chars
        - Max 5 results
        - Total char budget ≤ 1500

        Args:
            results: List of BrowserBaseResult objects.

        Returns:
            Sanitized list (may be smaller than input).
        """
        sanitized = []
        total_chars = 0

        for result in results[: cls.MAX_RESULTS]:
            # Reject invalid URLs
            if not result.url.startswith("http"):
                continue

            # Truncate snippet
            snippet = result.snippet[: cls.MAX_SNIPPET_LEN]

            # Char budget check
            item_chars = len(result.title) + len(result.url) + len(snippet)
            if total_chars + item_chars > cls.MAX_TOTAL_CHARS:
                break

            total_chars += item_chars

            # Import here to avoid circular; create new instance
            from agent.mcp.external_client import BrowserBaseResult  # noqa: PLC0415

            sanitized.append(
                BrowserBaseResult(
                    title=result.title,
                    url=result.url,
                    snippet=snippet,
                )
            )

        return sanitized

    @classmethod
    def format_tool_context(cls, results: list, max_chars: int = 2048) -> str:
        """
        Format sanitized results as injection-safe context string.

        Args:
            results: Sanitized BrowserBaseResult list.
            max_chars: Max character budget for context injection (≈512 tokens).

        Returns:
            Formatted context string, truncated if needed.
        """
        if not results:
            return ""

        lines = ["Web search results:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.title} — {r.snippet}")

        context = "\n".join(lines)

        # Enforce token budget (512 tokens ≈ 2048 chars)
        if len(context) > max_chars:
            context = context[:max_chars]

        return context
