"""
tests/integration/test_tool_two_pass.py

Integration tests for the TWO-PASS TOOL EXECUTION CYCLE FIX.

These tests specifically cover the NEW behaviour introduced in this fix:

  • _try_parse_tool_call() detects LOOSE tool syntax (no [TOOL_CALL] marker,
    no metadata["tool_call"]) that qwen2.5:7b and similar models emit.

  • _model_call_node_impl SHORT-CIRCUITS directly to tool_execution_node
    when a tool call is detected, bypassing result_handling_node entirely
    so that raw tool text NEVER becomes final_output.

  • The two-pass cycle completes cleanly:
      model_call (tool detected) → tool_execution → model_call (summary) → output

  • No infinite loops: tool_executed=True guard prevents re-execution on the
    second model call even if the model emits tool syntax again.

  • Raw tool text in ANY recognised format never appears in final_output.

Complements (do NOT duplicate):
  • tests/integration/test_tool_intent_flow.py   — metadata["tool_call"] path
  • tests/integration/test_full_pipeline_tool.py — full trace + memory path
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from inference import ModelBackend, ModelRequest, ModelResponse
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.intelligence.tools import ToolResult


# ─────────────────────────────────────────────────────────────────────────────
# Test stub model backends
# ─────────────────────────────────────────────────────────────────────────────


class LooseBraceSyntaxModel(ModelBackend):
    """
    Emits  web_search{"query": "..."}  on first call — NO metadata["tool_call"].
    Tests Strategy 2 of _try_parse_tool_call (loose brace syntax).
    Second call returns a clean summary.
    """

    def __init__(self, raw_output: str = 'web_search{"query": "Trump news today"}'):
        self._calls = 0
        self._raw_output = raw_output

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1
        if self._calls == 1:
            # Raw loose tool syntax — deliberately NO tool_call in metadata
            return ModelResponse(
                status="success",
                output=self._raw_output,
                metadata={},
            )
        # Second call: clean summary (no tool syntax)
        return ModelResponse(
            status="success",
            output="Trump is expected to return to the White House after winning the 2024 election.",
            metadata={},
        )


class StructuredJsonSyntaxModel(ModelBackend):
    """
    Emits  {"name": "web_search", "arguments": {...}}  on first call.
    Tests Strategy 1 of _try_parse_tool_call (raw structured JSON).
    """

    def __init__(self):
        self._calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1
        if self._calls == 1:
            return ModelResponse(
                status="success",
                output='{"name": "web_search", "arguments": {"query": "Trump latest news"}}',
                metadata={},
            )
        return ModelResponse(
            status="success",
            output="The latest news: Trump won the 2024 presidential election.",
            metadata={},
        )


class FunctionCallStyleModel(ModelBackend):
    """
    Emits  web_search({"query": "..."})  on first call.
    Tests function-call variant of Strategy 2.
    """

    def __init__(self):
        self._calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1
        if self._calls == 1:
            return ModelResponse(
                status="success",
                output='web_search({"query": "AI news 2025"})',
                metadata={},
            )
        return ModelResponse(
            status="success",
            output="The latest AI news: ChatGPT-5 was released.",
            metadata={},
        )


class AlwaysLooseToolModel(ModelBackend):
    """
    Always emits loose tool syntax — even on the second call.
    Tests that tool_executed=True prevents infinite re-execution.
    """

    def __init__(self):
        self._calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1
        return ModelResponse(
            status="success",
            output='web_search{"query": "always search"}',
            metadata={},
        )


class ContextCapturingModel(ModelBackend):
    """
    Records every prompt it receives so tests can assert on tool context injection.
    First call: emits loose tool syntax.
    Second call: returns clean summary.
    """

    def __init__(self):
        self._calls = 0
        self.received_prompts: List[str] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1
        self.received_prompts.append(request.prompt or "")
        if self._calls == 1:
            return ModelResponse(
                status="success",
                output='web_search{"query": "Trump news"}',
                metadata={},
            )
        return ModelResponse(
            status="success",
            output="Here is the latest news: Trump won.",
            metadata={},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_registry() -> MagicMock:
    """
    Create a mock ToolRegistry that returns canned web search results.

    get.return_value=None triggers the WebSearchTool registration path inside
    _tool_execution_node_impl; the actual registration call is absorbed by the
    MagicMock (no real WebSearchTool is stored).  execute() returns our canned
    ToolResult regardless.
    """
    registry = MagicMock()
    registry.get.return_value = None  # triggers registration branch
    registry.execute.return_value = ToolResult(
        success=True,
        data={
            "results": [
                {
                    "title": "Trump Wins 2024 Election",
                    "url": "https://example.com/trump-2024",
                    "snippet": "Donald Trump won the 2024 US presidential election.",
                }
            ],
            "query": "Trump news today",
        },
        execution_time_ms=42,
    )
    return registry


def _run(orch: SAMAgentOrchestrator, query: str) -> Dict[str, Any]:
    """Synchronously invoke the orchestrator (mirrors existing integration test style)."""
    return asyncio.run(orch.invoke(query))


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite 1 — Loose tool syntax detection
# ─────────────────────────────────────────────────────────────────────────────


class TestLooseToolSyntaxDetection:
    """
    _try_parse_tool_call detects raw tool text and short-circuits to execute_tool.
    These tests specifically exercise the path where the backend sets NO
    metadata["tool_call"] — requiring the orchestrator-level parser to fire.
    """

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_loose_brace_syntax_completes_two_pass(self, mock_get_registry):
        """
        web_search{"query": "..."} (no metadata) →
          _try_parse_tool_call detects → tool executes → second model call → clean output.
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        # Query without freshness keywords so the freshness guardrail is not
        # the confounding trigger here — the tool syntax alone must drive the flow.
        model = LooseBraceSyntaxModel('web_search{"query": "Trump electoral history"}')
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "Tell me about Trump's electoral history")

        # Two model calls: one with tool detected + one for summary
        assert model._calls == 2, f"Expected 2 model calls, got {model._calls}"
        assert result["status"] == "success"
        output = result["output"] or ""
        # Raw tool syntax must NEVER appear in final output
        assert 'web_search{' not in output, f"Tool syntax escaped to output: {output!r}"
        assert "[TOOL_CALL]" not in output

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_structured_json_syntax_completes_two_pass(self, mock_get_registry):
        """
        {"name": "web_search", "arguments": {...}} (no metadata) →
          _try_parse_tool_call Strategy 1 fires → two-pass cycle completes.
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = StructuredJsonSyntaxModel()
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "Tell me about Trump's policies")

        assert model._calls == 2, f"Expected 2 model calls, got {model._calls}"
        assert result["status"] == "success"
        output = result["output"] or ""
        assert '"name": "web_search"' not in output
        assert '"arguments"' not in output

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_function_call_style_syntax_completes_two_pass(self, mock_get_registry):
        """
        web_search({"query": "..."}) (function-call style, no metadata) →
          _try_parse_tool_call detects → two-pass cycle completes.
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = FunctionCallStyleModel()
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "What happened with AI research recently?")

        assert model._calls == 2, f"Expected 2 model calls, got {model._calls}"
        output = result["output"] or ""
        assert "web_search(" not in output


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite 2 — Raw tool text never escapes to final output
# ─────────────────────────────────────────────────────────────────────────────


class TestRawToolTextNeverEscapes:
    """
    In ALL detection paths, raw tool invocation text must never appear in
    the response sent to the user (Telegram / API caller).
    """

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_loose_brace_not_in_output(self, mock_get_registry):
        """web_search{...} never reaches final_output."""
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = LooseBraceSyntaxModel('web_search{"query": "AI safety"}')
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "Tell me about AI safety research")

        output = result.get("output") or ""
        assert 'web_search{' not in output, f"Brace syntax escaped: {output!r}"
        assert '{"query":' not in output

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_structured_json_not_in_output(self, mock_get_registry):
        """{"name": "web_search", ...} never reaches final_output."""
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = StructuredJsonSyntaxModel()
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "Tell me about AI safety")

        output = result.get("output") or ""
        assert '"name": "web_search"' not in output, f"JSON syntax escaped: {output!r}"
        assert '"arguments"' not in output

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_function_call_style_not_in_output(self, mock_get_registry):
        """web_search({...}) never reaches final_output."""
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = FunctionCallStyleModel()
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "What happened with AI research")

        output = result.get("output") or ""
        assert 'web_search(' not in output, f"Function-call syntax escaped: {output!r}"

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_tool_call_marker_not_in_output(self, mock_get_registry):
        """
        Even if [TOOL_CALL] marker leaks through to result_handling somehow,
        the orchestrator must not forward it.  This test uses a model that
        sets the standard [TOOL_CALL] format in metadata (covered by existing
        tests) but verifies the constraint once more in this suite.
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        class MetadataToolModel(ModelBackend):
            _calls = 0

            def generate(self, request: ModelRequest) -> ModelResponse:
                self._calls += 1
                if self._calls == 1:
                    return ModelResponse(
                        status="success",
                        output="",  # backend already cleaned the output
                        metadata={
                            "tool_call": {
                                "name": "web_search",
                                "arguments": {"query": "AI news"},
                            }
                        },
                    )
                return ModelResponse(
                    status="success",
                    output="AI has made tremendous progress in 2025.",
                    metadata={},
                )

        model = MetadataToolModel()
        orch = SAMAgentOrchestrator(model_backend=model)
        result = _run(orch, "What's happening in AI?")

        output = result.get("output") or ""
        assert "[TOOL_CALL]" not in output
        assert "web_search" not in output or "AI" in output


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite 3 — No infinite loop
# ─────────────────────────────────────────────────────────────────────────────


class TestNoInfiniteLoop:
    """
    Even when the model always emits tool syntax, the graph must terminate
    after exactly 1 tool execution (MAX_TOOL_CALLS_PER_TURN = 1).
    """

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_tool_executes_at_most_once(self, mock_get_registry):
        """
        AlwaysLooseToolModel emits tool syntax on every call.
        registry.execute must be called ≤ 1 time.
        Graph must terminate without RecursionError.
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = AlwaysLooseToolModel()
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "What's happening?")

        assert registry.execute.call_count <= 1, (
            f"Tool executed {registry.execute.call_count} times (expected ≤1)"
        )
        assert result is not None, "Graph must terminate with a result dict"

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_model_called_at_most_twice(self, mock_get_registry):
        """
        Normal tool flow: first call detects tool, second call summarises.
        Model must never be invoked a third time.
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = LooseBraceSyntaxModel()
        orch = SAMAgentOrchestrator(model_backend=model)

        _run(orch, "Tell me about Trump's policies")

        assert model._calls <= 2, (
            f"Model called {model._calls} times (expected ≤2)"
        )

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_always_tool_model_terminates(self, mock_get_registry):
        """
        AlwaysLooseToolModel: graph terminates without RecursionError or hang.
        Output may be None/empty — that is acceptable.  What is NOT acceptable
        is a crash or infinite loop.
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = AlwaysLooseToolModel()
        orch = SAMAgentOrchestrator(model_backend=model)

        # Must NOT raise RecursionError or loop indefinitely
        result = _run(orch, "Tell me about Trump")
        assert result is not None

        output = result.get("output") or ""
        # Raw tool syntax must still not escape even in this edge case
        assert 'web_search{' not in output, f"Tool text escaped: {output!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite 4 — Two-pass cycle end-to-end
# ─────────────────────────────────────────────────────────────────────────────


class TestTwoPassCycleEndToEnd:
    """
    Full two-pass integration: loose syntax detected → tool executes →
    second model call receives tool context → clean summary in output.
    """

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_second_model_call_receives_tool_context_in_prompt(self, mock_get_registry):
        """
        After tool execution the second model call prompt must contain
        'Tool Results:' (injected by build_prompt via tool_context).
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = ContextCapturingModel()
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "Tell me about Trump's history")

        assert model._calls == 2, f"Expected exactly 2 model calls, got {model._calls}"

        second_prompt = model.received_prompts[1] if len(model.received_prompts) > 1 else ""
        assert "Tool Results:" in second_prompt, (
            f"Second model call did not receive tool context.\n"
            f"Second prompt snippet: {second_prompt[:300]!r}"
        )

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_final_output_is_summary_not_tool_syntax(self, mock_get_registry):
        """
        Final output must be the human-readable summary from the second model call,
        not any form of raw tool invocation syntax.
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = LooseBraceSyntaxModel()
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "Tell me about Trump's electoral history")

        assert result["status"] == "success"
        output = result["output"] or ""

        # Must be non-empty summary text
        assert len(output) > 10, f"Output suspiciously short: {output!r}"

        # Must NOT contain any raw tool invocation patterns
        for bad_pattern in ('web_search{', 'web_search(', '"name": "web_search"', "[TOOL_CALL]"):
            assert bad_pattern not in output, (
                f"Pattern {bad_pattern!r} found in final output: {output!r}"
            )

    @patch("agent.intelligence.tools.get_tool_registry")
    def test_trump_news_query_with_freshness_keywords(self, mock_get_registry):
        """
        Spec test 1 — replicated exactly:
        Input: "Can you search the latest news about Trump today?"
        Expected: tool executes, second model call, clean summary, no raw web_search{}.

        Note: this query has freshness keywords ('latest', 'today'), so the
        freshness guardrail may ALSO be a valid trigger path.  Either way,
        the two-pass cycle must complete and output must be clean.
        """
        registry = _make_mock_registry()
        mock_get_registry.return_value = registry

        model = LooseBraceSyntaxModel('web_search{"query": "Trump today latest"}')
        orch = SAMAgentOrchestrator(model_backend=model)

        result = _run(orch, "Can you search the latest news about Trump today?")

        assert result["status"] == "success"
        output = result["output"] or ""

        assert 'web_search{' not in output, f"Raw syntax in output: {output!r}"
        assert "[TOOL_CALL]" not in output
        # Tool must have been invoked (registry.execute called)
        assert registry.execute.call_count >= 1, "Tool was never executed"
