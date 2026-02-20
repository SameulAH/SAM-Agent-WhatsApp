"""
Prompt Builder Layer
====================

Assembles structured user-facing prompts for the model backend.

Responsibilities:
- Defines the authoritative SYSTEM_PROMPT behavioral contract
- Assembles memory context + tool results + user input into a bounded prompt
- Enforces hard character-budget limits to prevent context bloat
- Provides the system prompt for model backends (e.g. OllamaModelBackend)

Invariants:
- memory_context is capped at _MAX_MEMORY_CHARS before injection
- tool_context is capped at _MAX_TOOL_CHARS before injection
- Combined injected context never exceeds _MAX_TOTAL_INJECT_CHARS
- tool_context has priority over memory_context when budget is tight
- system_prompt parameter is accepted by build_prompt() for API symmetry but
  is injected by the model backend (OllamaModelBackend system role), NOT
  embedded in the returned string — avoids double-injection.
"""

from typing import Optional

# ── Budget Constants ──────────────────────────────────────────────────────────
_MAX_MEMORY_CHARS: int = 2048    # ≈512 tokens: cap on memory context
_MAX_TOOL_CHARS: int = 2048      # ≈512 tokens: cap on tool results
_MAX_TOTAL_INJECT_CHARS: int = 1500  # Hard cap on combined injected context

# ── Behavioral Contract ───────────────────────────────────────────────────────
# This is the authoritative system prompt used across all model backends.
# OllamaModelBackend imports this to replace its inline _SYSTEM_PROMPT.
# The [TOOL_CALL] format matches the Ollama brace-counting parser.
SYSTEM_PROMPT = """You are SAM, a high-performance personal assistant.

Core Behavior:
- Be concise (max 5 sentences unless explicitly asked for more).
- No filler. No greetings. No meta-commentary.
- Do not explain your internal reasoning.
- Do not say you might use a tool — decide and act.

Tool Usage:
- If the question refers to: today, latest, current, recent, breaking, or news → use web_search.
- If the answer requires up-to-date information not in your training data → use web_search.
- When using a tool, respond ONLY with the exact tool call below and NOTHING else.

Tool Call Format (copy exactly, no extra text):
[TOOL_CALL]{"name": "web_search", "arguments": {"query": "<your concise search query>"}}

Personal Memory:
- If the user shares personal facts (birthday, preferences, workplace, etc.) → acknowledge briefly.
- Recalled personal facts are provided in Memory Context — use them naturally."""


def build_prompt(
    system_prompt: str,
    user_input: str,
    memory_context: Optional[str] = None,
    tool_context: Optional[str] = None,
) -> str:
    """
    Assemble the user-facing portion of the prompt.

    The system_prompt is injected by the model backend into the 'system' role
    (e.g., as {"role": "system", "content": SYSTEM_PROMPT} in Ollama /api/chat).
    This function builds the structured user message: memory context + tool
    results + user input + "Answer:" marker.

    Budget enforcement (priority: tool_context > memory_context):
    - memory_context: hard-capped at _MAX_MEMORY_CHARS
    - tool_context: hard-capped at _MAX_TOOL_CHARS
    - Combined: never exceeds _MAX_TOTAL_INJECT_CHARS; when over budget,
      memory is trimmed first, then tool context as last resort.

    Args:
        system_prompt: The behavioral contract string (handled by model backend;
                       accepted here for API symmetry and future non-chat backends).
        user_input: The preprocessed user message (never truncated).
        memory_context: Optional string of retrieved memory facts / STM context.
        tool_context: Optional string of tool execution results.

    Returns:
        Structured prompt string ready to pass as ModelRequest.prompt.
    """
    # ── Individual budget caps ────────────────────────────────────────────────
    if memory_context:
        memory_context = memory_context[:_MAX_MEMORY_CHARS]
    if tool_context:
        tool_context = tool_context[:_MAX_TOOL_CHARS]

    # ── Combined injection budget ─────────────────────────────────────────────
    mc_len = len(memory_context) if memory_context else 0
    tc_len = len(tool_context) if tool_context else 0

    if mc_len + tc_len > _MAX_TOTAL_INJECT_CHARS:
        if tool_context and memory_context:
            # Tool results take priority — trim memory to fit remaining budget
            budget_for_memory = max(0, _MAX_TOTAL_INJECT_CHARS - tc_len)
            memory_context = memory_context[:budget_for_memory] if budget_for_memory > 0 else None
        elif memory_context:
            memory_context = memory_context[:_MAX_TOTAL_INJECT_CHARS]
        elif tool_context:
            tool_context = tool_context[:_MAX_TOTAL_INJECT_CHARS]

    # ── Assemble structured prompt ────────────────────────────────────────────
    parts: list[str] = []

    if memory_context and memory_context.strip():
        parts.append(f"Memory Context:\n{memory_context.strip()}")

    if tool_context and tool_context.strip():
        parts.append(f"Tool Results:\n{tool_context.strip()}")

    parts.append(f"User:\n{user_input}")
    parts.append("Answer:")

    return "\n\n".join(parts)
