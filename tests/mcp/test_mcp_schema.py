"""
tests/mcp/test_mcp_schema.py

Unit tests for MCP schema validation.

Verifies:
✔ Valid BrowserBaseArgs accepted
✔ Query too short rejected (< 3 chars)
✔ max_results out of range rejected
✔ URL must start with http (guardrail)
✔ Snippet truncated at 300 chars
✔ MCPResponse parses success + error status
✔ Invalid schema rejected at model_validate
✔ Total char budget enforced (≤ 1500)
"""

import pytest
from pydantic import ValidationError

from agent.mcp.external_client import BrowserBaseArgs, BrowserBaseResult, MCPResponse
from agent.mcp.guardrails import MCPGuardrails


# ─────────────────────────────────────────────────────
# BrowserBaseArgs validation
# ─────────────────────────────────────────────────────


class TestBrowserBaseArgs:
    def test_valid_args(self):
        args = BrowserBaseArgs(query="AI regulation", max_results=3)
        assert args.query == "AI regulation"
        assert args.max_results == 3

    def test_query_too_short_rejected(self):
        with pytest.raises(ValidationError):
            BrowserBaseArgs(query="AI")  # 2 chars, min is 3

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            BrowserBaseArgs(query="")

    def test_max_results_too_low_rejected(self):
        with pytest.raises(ValidationError):
            BrowserBaseArgs(query="test query", max_results=0)

    def test_max_results_too_high_rejected(self):
        with pytest.raises(ValidationError):
            BrowserBaseArgs(query="test query", max_results=6)

    def test_max_results_defaults_to_3(self):
        args = BrowserBaseArgs(query="hello world")
        assert args.max_results == 3

    def test_max_results_boundary_1(self):
        args = BrowserBaseArgs(query="hello world", max_results=1)
        assert args.max_results == 1

    def test_max_results_boundary_5(self):
        args = BrowserBaseArgs(query="hello world", max_results=5)
        assert args.max_results == 5


# ─────────────────────────────────────────────────────
# BrowserBaseResult validation
# ─────────────────────────────────────────────────────


class TestBrowserBaseResult:
    def test_valid_result(self):
        r = BrowserBaseResult(
            title="EU AI Act",
            url="https://example.com/ai-act",
            snippet="The EU AI Act regulates...",
        )
        assert r.title == "EU AI Act"

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            BrowserBaseResult(title="Test", url="https://example.com")  # missing snippet


# ─────────────────────────────────────────────────────
# MCPResponse validation
# ─────────────────────────────────────────────────────


class TestMCPResponse:
    def test_success_response(self):
        data = {
            "status": "success",
            "results": [
                {
                    "title": "Test",
                    "url": "https://example.com",
                    "snippet": "A snippet",
                }
            ],
        }
        resp = MCPResponse.model_validate(data)
        assert resp.status == "success"
        assert len(resp.results) == 1

    def test_error_response_with_empty_results(self):
        data = {"status": "error", "results": []}
        resp = MCPResponse.model_validate(data)
        assert resp.status == "error"
        assert resp.results == []

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            MCPResponse.model_validate({"status": "unknown", "results": []})

    def test_missing_results_defaults_to_empty(self):
        resp = MCPResponse.model_validate({"status": "success"})
        assert resp.results == []


# ─────────────────────────────────────────────────────
# MCPGuardrails sanitize_results
# ─────────────────────────────────────────────────────


class TestMCPGuardrailsSanitize:
    def _make_result(self, title="T", url="https://example.com", snippet="S"):
        return BrowserBaseResult(title=title, url=url, snippet=snippet)

    def test_url_not_http_filtered_out(self):
        results = [
            self._make_result(url="ftp://bad.com"),
            self._make_result(url="https://good.com"),
        ]
        sanitized = MCPGuardrails.sanitize_results(results)
        assert len(sanitized) == 1
        assert sanitized[0].url == "https://good.com"

    def test_snippet_truncated_at_300(self):
        long_snippet = "x" * 400
        results = [self._make_result(snippet=long_snippet)]
        sanitized = MCPGuardrails.sanitize_results(results)
        assert len(sanitized[0].snippet) == 300

    def test_max_5_results_enforced(self):
        results = [self._make_result(title=f"Result {i}") for i in range(10)]
        sanitized = MCPGuardrails.sanitize_results(results)
        assert len(sanitized) <= 5

    def test_char_budget_enforced(self):
        # Each result is ~400 chars — 4 should exceed the 1500 budget
        big_snippet = "y" * 295
        results = [
            self._make_result(title=f"Title{i}", snippet=big_snippet)
            for i in range(5)
        ]
        sanitized = MCPGuardrails.sanitize_results(results)
        # Total chars should not exceed 1500
        total = sum(
            len(r.title) + len(r.url) + len(r.snippet) for r in sanitized
        )
        assert total <= MCPGuardrails.MAX_TOTAL_CHARS

    def test_empty_results_returns_empty(self):
        assert MCPGuardrails.sanitize_results([]) == []


# ─────────────────────────────────────────────────────
# MCPGuardrails format_tool_context
# ─────────────────────────────────────────────────────


class TestMCPGuardrailsFormat:
    def _make_result(self, title="T", url="https://example.com", snippet="S"):
        return BrowserBaseResult(title=title, url=url, snippet=snippet)

    def test_format_basic(self):
        results = [self._make_result(title="EU AI Act", snippet="A new law...")]
        ctx = MCPGuardrails.format_tool_context(results)
        assert "Web search results:" in ctx
        assert "EU AI Act" in ctx

    def test_format_empty_returns_empty_string(self):
        assert MCPGuardrails.format_tool_context([]) == ""

    def test_format_truncated_at_max_chars(self):
        results = [
            self._make_result(title=f"Title {i}", snippet="s" * 200) for i in range(5)
        ]
        ctx = MCPGuardrails.format_tool_context(results, max_chars=100)
        assert len(ctx) <= 100
