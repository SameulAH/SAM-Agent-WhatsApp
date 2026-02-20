"""
tests/mcp/test_mcp_timeout.py

Tests for MCPClient timeout handling and error paths.

Verifies:
✔ Timeout returns MCPResponse(status="error")
✔ HTTP error returns MCPResponse(status="error")
✔ Malformed JSON returns MCPResponse(status="error")
✔ Invalid base URL returns error immediately
✔ trace events emitted: mcp_request_sent, tool_execution_failed
✔ Never raises exceptions
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent.mcp.external_client import BrowserBaseArgs, BrowserBaseResult, MCPClient, MCPResponse
from agent.tracing.tracer import TraceMetadata


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────


def make_trace_metadata():
    return TraceMetadata(
        trace_id="test-trace-001",
        conversation_id="conv-001",
    )


def make_mock_tracer():
    tracer = MagicMock()
    tracer.record_event = MagicMock()
    return tracer


# ─────────────────────────────────────────────────────
# Invalid base URL
# ─────────────────────────────────────────────────────


class TestMCPClientInvalidURL:
    @pytest.mark.asyncio
    async def test_invalid_base_url_returns_error(self):
        client = MCPClient(base_url="ftp://bad-url", timeout=1.0)
        result = await client.search(BrowserBaseArgs(query="AI regulation"))
        assert result.status == "error"
        assert result.results == []

    @pytest.mark.asyncio
    async def test_empty_base_url_returns_error(self):
        client = MCPClient(base_url="", timeout=1.0)
        result = await client.search(BrowserBaseArgs(query="AI regulation"))
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_invalid_url_emits_trace_event(self):
        tracer = make_mock_tracer()
        meta = make_trace_metadata()

        client = MCPClient(base_url="ftp://bad", timeout=1.0)
        await client.search(BrowserBaseArgs(query="AI regulation"), tracer=tracer, trace_metadata=meta)

        # Should emit tool_execution_failed
        calls = [c.args[0] for c in tracer.record_event.call_args_list]
        assert "tool_execution_failed" in calls


# ─────────────────────────────────────────────────────
# Timeout
# ─────────────────────────────────────────────────────


class TestMCPClientTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_error_status(self):
        with patch("httpx.AsyncClient") as mock_class:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.post.side_effect = httpx.TimeoutException("timed out")
            mock_class.return_value = mock_instance

            client = MCPClient(base_url="https://mcp.example.com", timeout=0.1)
            result = await client.search(BrowserBaseArgs(query="AI regulation"))

        assert result.status == "error"
        assert result.results == []

    @pytest.mark.asyncio
    async def test_timeout_emits_trace_event(self):
        tracer = make_mock_tracer()
        meta = make_trace_metadata()

        with patch("httpx.AsyncClient") as mock_class:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.post.side_effect = httpx.TimeoutException("timed out")
            mock_class.return_value = mock_instance

            client = MCPClient(base_url="https://mcp.example.com", timeout=0.1)
            await client.search(
                BrowserBaseArgs(query="AI regulation"),
                tracer=tracer,
                trace_metadata=meta,
            )

        emitted = [c.args[0] for c in tracer.record_event.call_args_list]
        assert "tool_execution_failed" in emitted


# ─────────────────────────────────────────────────────
# HTTP errors
# ─────────────────────────────────────────────────────


class TestMCPClientHTTPErrors:
    @pytest.mark.asyncio
    async def test_http_500_returns_error(self):
        with patch("httpx.AsyncClient") as mock_class:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response
            )
            mock_instance.post.return_value = mock_response
            mock_class.return_value = mock_instance

            client = MCPClient(base_url="https://mcp.example.com", timeout=1.0)
            result = await client.search(BrowserBaseArgs(query="AI regulation"))

        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_malformed_json_returns_error(self):
        with patch("httpx.AsyncClient") as mock_class:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.side_effect = ValueError("Bad JSON")
            mock_instance.post.return_value = mock_response
            mock_class.return_value = mock_instance

            client = MCPClient(base_url="https://mcp.example.com", timeout=1.0)
            result = await client.search(BrowserBaseArgs(query="AI regulation"))

        assert result.status == "error"


# ─────────────────────────────────────────────────────
# Successful response + trace events
# ─────────────────────────────────────────────────────


class TestMCPClientSuccess:
    @pytest.mark.asyncio
    async def test_success_emits_request_and_response_events(self):
        tracer = make_mock_tracer()
        meta = make_trace_metadata()

        # MCP JSON-RPC tools/call response shape
        valid_response_data = {
            "jsonrpc": "2.0",
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps([
                            {
                                "title": "EU AI Act",
                                "url": "https://example.com/ai-act",
                                "snippet": "The EU AI Act is...",
                            }
                        ]),
                    }
                ],
                "isError": False,
            },
            "id": 1,
        }

        with patch("httpx.AsyncClient") as mock_class:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = valid_response_data
            mock_instance.post.return_value = mock_response
            mock_class.return_value = mock_instance

            # Pass fake credentials so the guard doesn't block
            client = MCPClient(
                base_url="https://mcp.example.com",
                smithery_api_key="test_smithery_key_xyz",
                api_key="test_key_abc",
                project_id="test_project_123",
                timeout=1.0,
            )
            result = await client.search(
                BrowserBaseArgs(query="AI regulation"),
                tracer=tracer,
                trace_metadata=meta,
            )

        assert result.status == "success"
        assert len(result.results) == 1

        emitted = [c.args[0] for c in tracer.record_event.call_args_list]
        assert "mcp_request_sent" in emitted
        assert "mcp_response_received" in emitted

    @pytest.mark.asyncio
    async def test_sync_wrapper_works(self):
        """search_sync must work in a non-async context."""
        valid_response_data = {
            "jsonrpc": "2.0",
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps([
                            {
                                "title": "Test",
                                "url": "https://example.com",
                                "snippet": "Some text here",
                            }
                        ]),
                    }
                ],
                "isError": False,
            },
            "id": 1,
        }

        with patch("httpx.AsyncClient") as mock_class:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = valid_response_data
            mock_instance.post.return_value = mock_response
            mock_class.return_value = mock_instance

            client = MCPClient(
                base_url="https://mcp.example.com",
                smithery_api_key="test_smithery_key_xyz",
                api_key="test_key_abc",
                project_id="test_project_123",
                timeout=1.0,
            )
            result = client.search_sync(BrowserBaseArgs(query="hello world"))

        assert result.status == "success"
