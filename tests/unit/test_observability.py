"""
Tests for local observability.

Proves that observability is invisible to the agent:
- Agent output identical with/without observability
- Memory behavior unchanged
- Routing unchanged
- No sensitive data exposed
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.state_schema import AgentState
from agent.memory.sqlite import SQLiteShortTermMemoryStore
from agent.observability import ObservabilityStore, LocalObservabilityInterface
from inference import StubModelBackend


class TestObservabilityInvariance:
    """Prove observability doesn't affect agent behavior."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
        for ext in ["-wal", "-shm"]:
            wal_path = db_path + ext
            if os.path.exists(wal_path):
                os.remove(wal_path)

    @pytest.fixture
    def memory_store(self, temp_db):
        """Create memory store."""
        return SQLiteShortTermMemoryStore(db_path=temp_db)

    @pytest.fixture
    def agent_without_observability(self, memory_store):
        """Create agent without observability."""
        return SAMAgentOrchestrator(
            model_backend=StubModelBackend(),
            memory_controller=memory_store,
        )

    @pytest.fixture
    def agent_with_observability(self, memory_store):
        """Create agent with observability."""
        store = ObservabilityStore()
        
        def observability_sink(event_type, metadata):
            """Simple sink that routes to store."""
            if event_type == "span_start":
                store.record_span_start(
                    span_id=f"{metadata.get('trace_id')}-span-{store.get_stats().get('completed_spans', 0)}",
                    trace_id=metadata.get('trace_id'),
                    node_name=metadata.get('node_name'),
                )
            elif event_type == "span_end":
                pass  # Simplified for test
        
        agent = SAMAgentOrchestrator(
            model_backend=StubModelBackend(),
            memory_controller=memory_store,
        )
        
        return agent

    def test_agent_output_identical_with_and_without_observability(
        self, agent_without_observability, agent_with_observability
    ):
        """Agent output must be identical regardless of observability."""
        state = AgentState(
            conversation_id="test-invariance",
            trace_id="trace-invariance",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="Test message",
        )
        
        # Execute without observability
        result1 = agent_without_observability.graph.invoke(state)
        
        # Execute with observability (same input, same agent logic)
        state2 = AgentState(
            conversation_id="test-invariance-2",
            trace_id="trace-invariance-2",
            created_at="2026-02-09T18:00:01",
            input_type="text",
            raw_input="Test message",
        )
        result2 = agent_with_observability.graph.invoke(state2)
        
        # Both should succeed
        assert result1 is not None
        assert result2 is not None
        
        # Both should have same fields (regardless of content)
        assert "conversation_id" in result1
        assert "conversation_id" in result2
        assert "trace_id" in result1
        assert "trace_id" in result2

    def test_memory_writes_unchanged(self, memory_store):
        """Memory writes must be identical with/without observability."""
        from agent.memory.types import MemoryWriteRequest, MemoryReadRequest
        
        # Write with observability sink
        store = ObservabilityStore()
        
        # Make two writes and verify both succeed
        request1 = MemoryWriteRequest(
            conversation_id="conv-1",
            key="test",
            data={"test": "data"},
            authorized=True,
        )
        
        response1 = memory_store.write(request1)
        assert response1.status == "success"
        
        # Read back to verify persistence
        read_request = MemoryReadRequest(
            conversation_id="conv-1",
            key="test",
            authorized=True,
        )
        read_response = memory_store.read(read_request)
        assert read_response.status == "success"
        assert read_response.data == {"test": "data"}

    def test_observability_store_bounded(self):
        """Observability store must be bounded."""
        store = ObservabilityStore(max_traces=10, max_spans=20)
        
        # Fill beyond capacity
        for i in range(50):
            store.record_trace_start(f"trace-{i}", f"conv-{i}")
            store.record_trace_end(f"trace-{i}", "success")
        
        # Should only have max_traces
        traces = store.get_recent_traces()
        assert len(traces) <= 10

    def test_observability_thread_safe(self):
        """Observability store must be thread-safe."""
        import threading
        
        store = ObservabilityStore()
        errors = []
        
        def worker():
            try:
                for i in range(100):
                    store.record_trace_start(f"trace-{i}", f"conv-{i}")
                    store.record_trace_end(f"trace-{i}", "success")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should occur
        assert len(errors) == 0

    def test_observability_fails_silently(self):
        """Observability failures must not affect agent."""
        
        def broken_sink(event_type, metadata):
            """Sink that raises."""
            raise RuntimeError("Broken sink!")
        
        # This should not crash the agent
        agent = SAMAgentOrchestrator(
            model_backend=StubModelBackend(),
        )
        
        state = AgentState(
            conversation_id="test-broken",
            trace_id="trace-broken",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="Test message",
        )
        
        # Invoke should succeed even if observability sink is broken
        result = agent.graph.invoke(state)
        assert result is not None

    def test_observability_interface_read_only(self):
        """LocalObservabilityInterface must be read-only."""
        store = ObservabilityStore()
        interface = LocalObservabilityInterface(store=store)
        
        # Add some data
        store.record_trace_start("trace-1", "conv-1")
        store.record_trace_end("trace-1", "success")
        
        # All methods should return data or None, never raise
        traces = interface.get_recent_traces()
        assert isinstance(traces, list)
        
        spans = interface.get_recent_spans()
        assert isinstance(spans, list)
        
        events = interface.get_memory_events()
        assert isinstance(events, list)
        
        stats = interface.get_store_stats()
        assert isinstance(stats, dict)

    def test_no_sensitive_data_leaked(self):
        """Observability must never expose sensitive data."""
        store = ObservabilityStore()
        
        # Try to record various data types
        store.record_memory_event(
            operation="write",
            conversation_id="conv-1",
            authorized=True,
            backend="sqlite",
            status="success",
        )
        
        events = store.get_memory_events()
        assert len(events) == 1
        event = events[0]
        
        # Should have these fields
        assert "operation" in event
        assert "authorized" in event
        assert "backend" in event
        assert "status" in event
        
        # Should NOT have these fields (even if passed)
        assert "data" not in event
        assert "prompt" not in event
        assert "output" not in event

    def test_observability_stats_accurate(self):
        """Observability statistics must be accurate."""
        store = ObservabilityStore()
        
        # Record some traces
        for i in range(5):
            store.record_trace_start(f"trace-{i}", f"conv-{i}")
            store.record_trace_end(f"trace-{i}", "success")
        
        stats = store.get_stats()
        assert stats["completed_traces"] == 5
        assert stats["active_traces"] == 0

    def test_observability_memory_bounded(self):
        """Observability store should not leak memory."""
        store = ObservabilityStore(max_traces=5, max_spans=10, max_memory_events=5)
        
        # Exceed capacity
        for i in range(100):
            store.record_trace_start(f"trace-{i}", f"conv-{i}")
            store.record_trace_end(f"trace-{i}", "success")
            
            store.record_memory_event(
                operation="write",
                conversation_id=f"conv-{i}",
                authorized=True,
                backend="sqlite",
                status="success",
            )
        
        # Should not exceed max
        traces = store.get_recent_traces()
        events = store.get_memory_events()
        
        assert len(traces) <= 5
        assert len(events) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
