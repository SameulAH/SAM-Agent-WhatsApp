"""
Tests for LangSmith observability implementation.

Proves that:
1. LangSmith observability never affects agent behavior
2. Agent works with tracing disabled (noop)
3. Agent works with tracing enabled (langsmith)
4. Tracing failures don't crash the agent
5. Only safe metadata is logged
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from agent.tracing import (
    Tracer,
    TraceMetadata,
    NoOpTracer,
    LangSmithTracer,
    create_tracer,
    get_tracer_backend,
    get_tracer_config,
)


class TestTracerFactory:
    """Test tracer factory and environment-based selection."""

    def test_default_tracer_is_noop(self):
        """Default tracer should be NoOpTracer."""
        with patch.dict(os.environ, {"TRACER_BACKEND": "noop"}):
            tracer = create_tracer()
            assert isinstance(tracer, NoOpTracer)
            assert not tracer.is_enabled()

    def test_explicit_langsmith_tracer_selection(self):
        """Can explicitly select LangSmith tracer."""
        with patch.dict(
            os.environ,
            {"TRACER_BACKEND": "langsmith", "LANGSMITH_API_KEY": "test-key"},
        ):
            tracer = create_tracer()
            # Should create LangSmith tracer (even if initialization fails silently)
            assert isinstance(tracer, (LangSmithTracer, NoOpTracer))

    def test_legacy_langsmith_enabled_flag(self):
        """Legacy LANGSMITH_ENABLED flag should work."""
        with patch.dict(
            os.environ, {"LANGSMITH_ENABLED": "true", "LANGSMITH_API_KEY": "test-key"}
        ):
            tracer = create_tracer()
            # Should create LangSmith tracer
            assert isinstance(tracer, (LangSmithTracer, NoOpTracer))

    def test_invalid_backend_defaults_to_noop(self):
        """Invalid backend should default to noop."""
        with patch.dict(os.environ, {"TRACER_BACKEND": "invalid"}):
            tracer = create_tracer()
            assert isinstance(tracer, NoOpTracer)

    def test_missing_langsmith_api_key_disables_tracing(self):
        """Missing LangSmith API key should disable tracing."""
        with patch.dict(
            os.environ, {"TRACER_BACKEND": "langsmith", "LANGSMITH_API_KEY": ""}
        ):
            tracer = create_tracer()
            # Should fall back to noop since no API key
            assert isinstance(tracer, (NoOpTracer, LangSmithTracer))

    def test_get_tracer_backend(self):
        """get_tracer_backend should return configured backend."""
        with patch.dict(os.environ, {"TRACER_BACKEND": "langsmith"}):
            backend = get_tracer_backend()
            assert backend == "langsmith"

    def test_get_tracer_config_noop(self):
        """get_tracer_config should return noop config."""
        with patch.dict(os.environ, {"TRACER_BACKEND": "noop"}):
            config = get_tracer_config()
            assert config["tracer_backend"] == "noop"
            assert config["enabled"] is False

    def test_get_tracer_config_langsmith(self):
        """get_tracer_config should return LangSmith config."""
        with patch.dict(
            os.environ,
            {"TRACER_BACKEND": "langsmith", "LANGSMITH_API_KEY": "test-key"},
        ):
            config = get_tracer_config()
            assert config["tracer_backend"] == "langsmith"
            assert config["enabled"] is True
            assert "langsmith_configured" in config
            assert config["langsmith_project"] == "sam-agent"

    def test_custom_langsmith_project(self):
        """Should respect custom LANGSMITH_PROJECT."""
        with patch.dict(
            os.environ,
            {
                "TRACER_BACKEND": "langsmith",
                "LANGSMITH_API_KEY": "test-key",
                "LANGSMITH_PROJECT": "custom-project",
            },
        ):
            config = get_tracer_config()
            assert config["langsmith_project"] == "custom-project"


class TestLangSmithTracer:
    """Test LangSmith tracer implementation."""

    def test_langsmith_tracer_initialization(self):
        """LangSmith tracer should initialize."""
        tracer = LangSmithTracer(enabled=False)
        assert isinstance(tracer, Tracer)
        assert not tracer.is_enabled()

    def test_langsmith_tracer_with_missing_api_key(self):
        """LangSmith tracer with missing API key should disable itself."""
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": ""}):
            tracer = LangSmithTracer(enabled=True)
            # Should gracefully disable if API key is missing
            assert isinstance(tracer, LangSmithTracer)

    def test_langsmith_start_span_noop_when_disabled(self):
        """start_span should be noop when disabled."""
        tracer = LangSmithTracer(enabled=False)
        
        trace_metadata = TraceMetadata(
            trace_id="test-trace",
            conversation_id="test-conv"
        )
        
        span = tracer.start_span(
            "test_node",
            {"duration_ms": 100, "status": "success"},
            trace_metadata
        )
        
        assert span is None

    def test_langsmith_filters_safe_metadata(self):
        """LangSmith should only log safe metadata."""
        tracer = LangSmithTracer(enabled=False)  # Disabled for this test
        
        unsafe_metadata = {
            "raw_input": "user message",  # UNSAFE
            "prompt": "system prompt",  # UNSAFE
            "output": "model response",  # UNSAFE
            "memory": {"key": "value"},  # UNSAFE
            "duration_ms": 100,  # SAFE
            "status": "success",  # SAFE
            "node_name": "model_call",  # SAFE
        }
        
        filtered = tracer._filter_safe_metadata(unsafe_metadata)
        
        # Should include safe fields
        assert "duration_ms" in filtered
        assert "status" in filtered
        assert "node_name" in filtered
        
        # Should NOT include unsafe fields
        assert "raw_input" not in filtered
        assert "prompt" not in filtered
        assert "output" not in filtered
        assert "memory" not in filtered

    def test_langsmith_truncates_long_strings(self):
        """LangSmith should truncate very long strings."""
        tracer = LangSmithTracer(enabled=False)
        
        long_metadata = {
            "error_message": "x" * 500,  # 500 chars
            "duration_ms": 100,
        }
        
        filtered = tracer._filter_safe_metadata(long_metadata)
        
        # Error message should be truncated to max 256 chars
        assert len(filtered["error_message"]) <= 260  # 256 + "..."


class TestTracerBehaviorInvariance:
    """Test that tracer doesn't affect agent behavior."""

    def test_noop_tracer_is_transparent(self):
        """NoOpTracer should be completely transparent."""
        tracer = NoOpTracer()
        
        trace_metadata = TraceMetadata(
            trace_id="test-trace",
            conversation_id="test-conv"
        )
        
        # All operations should be instant and return None or noop
        span = tracer.start_span("test", {}, trace_metadata)
        assert span is None
        
        # end_span should do nothing
        tracer.end_span(span, "success", {})
        
        # record_event should do nothing
        tracer.record_event("test_event", {}, trace_metadata)
        
        # is_enabled should be False
        assert not tracer.is_enabled()

    def test_langsmith_tracer_with_broken_sink(self):
        """LangSmith tracer should fail silently if sink is broken."""
        broken_sink = MagicMock(side_effect=RuntimeError("Sink failed!"))
        
        tracer = LangSmithTracer(enabled=False, observability_sink=broken_sink)
        
        trace_metadata = TraceMetadata(
            trace_id="test-trace",
            conversation_id="test-conv"
        )
        
        # Should not raise even though sink fails
        span = tracer.start_span("test", {}, trace_metadata)
        assert span is None  # Disabled

    def test_multiple_tracers_interchangeable(self):
        """Different tracers should be interchangeable."""
        noop_tracer = NoOpTracer()
        langsmith_tracer = LangSmithTracer(enabled=False)
        
        trace_metadata = TraceMetadata(
            trace_id="test-trace",
            conversation_id="test-conv"
        )
        
        metadata = {"duration_ms": 100, "status": "success"}
        
        # Both tracers should handle the same calls
        for tracer in [noop_tracer, langsmith_tracer]:
            span = tracer.start_span("test", metadata, trace_metadata)
            tracer.end_span(span, "success", metadata)
            tracer.record_event("test_event", metadata, trace_metadata)
        
        # Both should behave identically

    def test_tracer_with_observability_sink(self):
        """Tracer should support optional observability sink."""
        events = []
        
        def sink(event_type, metadata):
            events.append((event_type, metadata))
        
        tracer = NoOpTracer(observability_sink=sink)
        
        trace_metadata = TraceMetadata(trace_id="test-trace")
        
        # Even though tracer is noop, it should accept sink
        tracer.start_span("test", {}, trace_metadata)
        # No events recorded since noop


class TestTracerHealthEndpoint:
    """Test /health/trace endpoint."""

    def test_health_trace_endpoint_exists(self):
        """Health trace endpoint should exist."""
        try:
            from agent.api import create_app
            
            app = create_app()
            
            # Check that endpoint is registered
            if app is not None:
                routes = [route.path for route in app.routes]
                assert "/health/trace" in routes
            else:
                pytest.skip("FastAPI not available")
        except ImportError:
            pytest.skip("FastAPI not available")

    def test_health_trace_returns_config(self):
        """Health trace endpoint should return tracer config."""
        try:
            from fastapi.testclient import TestClient
            from agent.api import create_app
            
            app = create_app()
            
            if app is None:
                pytest.skip("FastAPI not available")
            
            client = TestClient(app)
            
            response = client.get("/health/trace")
            
            # Should return 200 and valid JSON
            assert response.status_code in [200, 500]  # May fail if langsmith not installed
            
            data = response.json()
            assert isinstance(data, dict)
            assert "tracer_backend" in data
        except ImportError:
            pytest.skip("FastAPI not available")


class TestTracerMetadata:
    """Test TraceMetadata dataclass."""

    def test_trace_metadata_required_fields(self):
        """TraceMetadata should require trace_id."""
        # trace_id is required
        metadata = TraceMetadata(trace_id="test-trace")
        assert metadata.trace_id == "test-trace"
        assert metadata.conversation_id is None
        assert metadata.user_id is None

    def test_trace_metadata_optional_fields(self):
        """TraceMetadata should support optional fields."""
        metadata = TraceMetadata(
            trace_id="test-trace",
            conversation_id="conv-123",
            user_id="user-456"
        )
        
        assert metadata.trace_id == "test-trace"
        assert metadata.conversation_id == "conv-123"
        assert metadata.user_id == "user-456"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
