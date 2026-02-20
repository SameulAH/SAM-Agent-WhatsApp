"""
MCP (Model Context Protocol) Client Layer.

Provides structured access to external tool servers (e.g. BrowserBase).
All clients are:
- Async-first with sync wrappers
- Schema-validated via Pydantic
- Guardrail-enforced (char budget, result count, snippet length)
- Trace-propagation-aware
"""

from agent.mcp.external_client import (
    BrowserBaseArgs,
    BrowserBaseResult,
    MCPResponse,
    MCPClient,
)
from agent.mcp.guardrails import GuardrailViolation, MCPGuardrails

__all__ = [
    "BrowserBaseArgs",
    "BrowserBaseResult",
    "MCPResponse",
    "MCPClient",
    "GuardrailViolation",
    "MCPGuardrails",
]
