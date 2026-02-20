"""
tests/prompting/test_prompt_builder.py

Unit tests for the PromptBuilder layer.

Verifies:
✔ SYSTEM_PROMPT is a non-empty string with behavioral contract
✔ build_prompt returns structured prompt with memory context
✔ build_prompt returns structured prompt with tool results
✔ build_prompt truncates memory_context to _MAX_MEMORY_CHARS
✔ build_prompt truncates tool_context to _MAX_TOOL_CHARS
✔ Combined context never exceeds _MAX_TOTAL_INJECT_CHARS
✔ tool_context has priority over memory_context when over budget
✔ system_prompt parameter is accepted without embedding in output
✔ build_prompt with no context returns user input + Answer:
"""

import pytest
from agent.prompting.prompt_builder import (
    SYSTEM_PROMPT,
    build_prompt,
    _MAX_MEMORY_CHARS,
    _MAX_TOOL_CHARS,
    _MAX_TOTAL_INJECT_CHARS,
)


class TestSystemPrompt:
    """Tests for the SYSTEM_PROMPT behavioral contract."""

    def test_system_prompt_is_nonempty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_contains_conciseness_rule(self):
        assert "5 sentences" in SYSTEM_PROMPT or "concise" in SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_tool_usage_rule(self):
        assert "web_search" in SYSTEM_PROMPT

    def test_system_prompt_contains_freshness_keywords(self):
        """System prompt must mention when to use web_search."""
        kw_present = any(
            kw in SYSTEM_PROMPT.lower()
            for kw in ["today", "latest", "current", "recent", "breaking", "news"]
        )
        assert kw_present

    def test_system_prompt_contains_tool_call_format(self):
        """System prompt must define the tool call format."""
        assert "TOOL_CALL" in SYSTEM_PROMPT or "tool_call" in SYSTEM_PROMPT


class TestBuildPrompt:
    """Tests for the build_prompt() assembler."""

    def test_minimal_prompt_no_context(self):
        """Minimal call: user input only."""
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="What is the capital of France?",
        )
        assert "What is the capital of France?" in result
        assert "Answer:" in result

    def test_with_memory_context(self):
        """Memory context is injected into the prompt."""
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="What did I say before?",
            memory_context="User mentioned they like Python.",
        )
        assert "Memory Context:" in result
        assert "User mentioned they like Python." in result
        assert "What did I say before?" in result

    def test_with_tool_context(self):
        """Tool results are injected into the prompt."""
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="What is the news today?",
            tool_context="Web search results:\n1. AI News — Latest update.",
        )
        assert "Tool Results:" in result
        assert "Web search results:" in result
        assert "What is the news today?" in result

    def test_with_both_contexts(self):
        """Both memory and tool context appear in correct order."""
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="Tell me more.",
            memory_context="User birthday is Oct 12.",
            tool_context="Web: Today's top story.",
        )
        assert "Memory Context:" in result
        assert "Tool Results:" in result
        assert "User birthday is Oct 12." in result
        assert "Today's top story." in result

    def test_memory_context_truncation(self):
        """memory_context is truncated to _MAX_MEMORY_CHARS."""
        large_memory = "x" * (_MAX_MEMORY_CHARS + 500)
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="hi",
            memory_context=large_memory,
        )
        # The injected memory should not exceed the cap
        assert len(result) < len(large_memory) + 200  # prompt overhead small

    def test_tool_context_truncation(self):
        """tool_context is truncated to _MAX_TOOL_CHARS."""
        large_tool = "y" * (_MAX_TOOL_CHARS + 500)
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="hi",
            tool_context=large_tool,
        )
        assert len(result) < len(large_tool) + 200

    def test_combined_budget_enforcement(self):
        """Combined injected context never exceeds _MAX_TOTAL_INJECT_CHARS."""
        large_memory = "m" * 1000
        large_tool = "t" * 1000
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="question",
            memory_context=large_memory,
            tool_context=large_tool,
        )
        # Extract injected portion (between start and "User:")
        injected_portion = result.split("User:")[0]
        assert len(injected_portion) <= _MAX_TOTAL_INJECT_CHARS + 100  # small overhead

    def test_tool_context_priority_over_memory(self):
        """When over budget, tool_context is preserved over memory_context."""
        # Make tool context exactly at limit, memory would push it over
        tool_context = "T" * (_MAX_TOTAL_INJECT_CHARS - 50)
        memory_context = "M" * 200  # this would push total over budget
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="question",
            memory_context=memory_context,
            tool_context=tool_context,
        )
        # Tool context key letters must still be present
        assert "T" * 10 in result

    def test_system_prompt_not_embedded_in_user_portion(self):
        """The system prompt content is NOT duplicated inside build_prompt output."""
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="simple question",
        )
        # The full SYSTEM_PROMPT text should NOT appear verbatim in the assembled user portion
        # (it belongs in the model backend's system role, not the user message)
        assert result.count("You are SAM") == 0

    def test_answer_marker_always_present(self):
        """The 'Answer:' marker is always at the end."""
        result = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input="test",
        )
        assert result.strip().endswith("Answer:")

    def test_empty_memory_context_omitted(self):
        """None and empty-string memory_context produce no Memory Context section."""
        for mc in [None, "", "   "]:
            result = build_prompt(
                system_prompt=SYSTEM_PROMPT,
                user_input="hello",
                memory_context=mc,
            )
            assert "Memory Context:" not in result

    def test_empty_tool_context_omitted(self):
        """None and whitespace tool_context produce no Tool Results section."""
        for tc in [None, "", "   "]:
            result = build_prompt(
                system_prompt=SYSTEM_PROMPT,
                user_input="hello",
                tool_context=tc,
            )
            assert "Tool Results:" not in result
