"""
Integration tests for SQLite short-term memory with LangGraph agent.

Tests full request flow to verify memory writes are triggered and persisted.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.state_schema import AgentState
from agent.memory.sqlite import SQLiteShortTermMemoryStore
from inference import StubModelBackend


class TestSqliteIntegration:
    """Integration tests for SQLite memory with agent."""

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
        """Create memory store with temporary database."""
        return SQLiteShortTermMemoryStore(db_path=temp_db)

    @pytest.fixture
    def agent(self, memory_store):
        """Create agent with memory enabled."""
        model_backend = StubModelBackend()
        return SAMAgentOrchestrator(
            model_backend=model_backend,
            memory_controller=memory_store,
        )

    def test_memory_write_when_authorized(self, agent, memory_store):
        """
        Test A: Full path where decision_logic_node authorizes write.
        
        Verify that:
        1. memory_write_node is called
        2. Data is written to SQLite
        3. Data persists after request completes
        """
        conversation_id = "test-conv-write-authorized"
        
        # Create initial state with authorization
        state = AgentState(
            conversation_id=conversation_id,
            trace_id="trace-123",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="Test message for memory",
            memory_write_authorized=True,  # Authorized
            final_output="Agent response",
        )
        
        # Call memory_write_node directly
        result = agent.memory_nodes.memory_write_node(state)
        
        # Verify write succeeded
        assert result['memory_write_status'] == 'success'
        assert result['memory_available'] is True
        
        # Verify data was persisted by querying directly
        from agent.memory.types import MemoryReadRequest
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key="conversation_context",
            authorized=True,
            reason="test",
        )
        
        response = memory_store.read(read_request)
        assert response.status == "success"
        assert response.data is not None
        assert "final_output" in response.data

    def test_memory_write_unauthorized_skipped(self, agent, memory_store):
        """
        Test that memory_write_node returns early if not authorized.
        
        Verify that:
        1. Write is skipped
        2. Database remains empty
        3. Agent continues (non-fatal)
        """
        conversation_id = "test-conv-unauthorized"
        
        state = AgentState(
            conversation_id=conversation_id,
            trace_id="trace-456",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="Test message",
            memory_write_authorized=False,  # NOT authorized
            final_output="Agent response",
        )
        
        # Call memory_write_node
        result = agent.memory_nodes.memory_write_node(state)
        
        # Should return early with None status
        assert result['memory_write_status'] is None
        
        # Verify nothing was written
        from agent.memory.types import MemoryReadRequest
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key="conversation_context",
            authorized=True,
            reason="test",
        )
        
        response = memory_store.read(read_request)
        assert response.status == "not_found"

    def test_memory_write_adapter_failure_graceful_degradation(self, agent, memory_store):
        """
        Test B: Adapter failure is handled gracefully.
        
        Verify that:
        1. Adapter raises/returns error
        2. memory_write_node captures error
        3. Agent continues and returns success (degraded mode)
        4. Error is logged but doesn't block execution
        """
        conversation_id = "test-conv-adapter-error"
        
        state = AgentState(
            conversation_id=conversation_id,
            trace_id="trace-789",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="Test message",
            memory_write_authorized=True,
            final_output="Agent response",
            memory_available=True,
        )
        
        # Mock the adapter to raise an exception
        with patch.object(memory_store, 'write') as mock_write:
            mock_write.side_effect = Exception("Simulated database failure")
            
            # Call memory_write_node
            result = agent.memory_nodes.memory_write_node(state)
            
            # Should handle gracefully
            assert result['memory_write_status'] == 'failed'
            assert result['memory_available'] is False
            
            # Agent should NOT crash (we're just returning the result)
            # In a real scenario, the graph would continue to next node

    def test_multiple_writes_same_conversation(self, agent, memory_store):
        """Test multiple authorized writes to the same conversation."""
        conversation_id = "test-conv-multiple"
        
        # First write
        state1 = AgentState(
            conversation_id=conversation_id,
            trace_id="trace-1",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="First message",
            memory_write_authorized=True,
            final_output="First response",
        )
        
        result1 = agent.memory_nodes.memory_write_node(state1)
        assert result1['memory_write_status'] == 'success'
        
        # Second write (should overwrite with same key)
        state2 = AgentState(
            conversation_id=conversation_id,
            trace_id="trace-2",
            created_at="2026-02-09T18:00:01",
            input_type="text",
            raw_input="Second message",
            memory_write_authorized=True,
            final_output="Second response",
        )
        
        result2 = agent.memory_nodes.memory_write_node(state2)
        assert result2['memory_write_status'] == 'success'
        
        # Read and verify we have the latest
        from agent.memory.types import MemoryReadRequest
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key="conversation_context",
            authorized=True,
            reason="test",
        )
        
        response = memory_store.read(read_request)
        assert response.status == "success"
        # Should have the second response
        assert response.data['final_output'] == "Second response"

    def test_memory_write_node_logging(self, agent, memory_store, caplog):
        """Test that memory_write_node produces appropriate log messages."""
        import logging
        caplog.set_level(logging.INFO)
        
        conversation_id = "test-conv-logging"
        
        state = AgentState(
            conversation_id=conversation_id,
            trace_id="trace-log",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="Test message",
            memory_write_authorized=True,
            final_output="Agent response",
        )
        
        # Call memory_write_node
        result = agent.memory_nodes.memory_write_node(state)
        
        # Check logs
        log_output = caplog.text
        assert "memory_write_node" in log_output.lower()
        assert conversation_id in log_output

    def test_isolated_conversations_database(self, agent, memory_store):
        """Test that different conversations don't interfere."""
        conv1 = "test-conv-isolated-1"
        conv2 = "test-conv-isolated-2"
        
        # Write to conversation 1
        state1 = AgentState(
            conversation_id=conv1,
            trace_id="trace-conv1",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="Message 1",
            memory_write_authorized=True,
            final_output="Response 1",
        )
        agent.memory_nodes.memory_write_node(state1)
        
        # Write to conversation 2
        state2 = AgentState(
            conversation_id=conv2,
            trace_id="trace-conv2",
            created_at="2026-02-09T18:00:01",
            input_type="text",
            raw_input="Message 2",
            memory_write_authorized=True,
            final_output="Response 2",
        )
        agent.memory_nodes.memory_write_node(state2)
        
        # Verify isolation
        from agent.memory.types import MemoryReadRequest
        
        read1 = MemoryReadRequest(
            conversation_id=conv1,
            key="conversation_context",
            authorized=True,
            reason="test",
        )
        response1 = memory_store.read(read1)
        assert response1.data['final_output'] == "Response 1"
        
        read2 = MemoryReadRequest(
            conversation_id=conv2,
            key="conversation_context",
            authorized=True,
            reason="test",
        )
        response2 = memory_store.read(read2)
        assert response2.data['final_output'] == "Response 2"


class TestMemoryWriteLatching:
    """Regression tests for memory write execution latching.
    
    Verifies that memory write executes exactly once per invocation
    and the graph converges properly without infinite loops.
    """

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
        """Create memory store with temporary database."""
        return SQLiteShortTermMemoryStore(db_path=temp_db)

    @pytest.fixture
    def agent(self, memory_store):
        """Create agent with memory enabled."""
        model_backend = StubModelBackend()
        return SAMAgentOrchestrator(
            model_backend=model_backend,
            memory_controller=memory_store,
        )

    def test_memory_write_executes_once(self, agent, memory_store):
        """
        Test: Memory write executes exactly once per invocation.
        
        Verify that:
        1. Agent graph executes successfully
        2. Exactly one row is written to SQLite
        3. No infinite loops or multiple writes
        """
        from agent.memory.types import MemoryReadRequest
        
        conversation_id = "test-once-write"
        
        state = AgentState(
            conversation_id=conversation_id,
            trace_id="trace-once",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="Test message for once",
        )
        
        # Invoke graph - returns a dict (merged state)
        result = agent.graph.invoke(state)
        
        # Verify invocation succeeded
        assert result is not None
        assert isinstance(result, dict)
        
        # Check SQLite directly for row count
        import sqlite3
        conn = sqlite3.connect(memory_store.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM short_term_memory WHERE conversation_id = ?", (conversation_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        # Must be exactly 1 row (memory write happened once, not repeated)
        assert count == 1, f"Expected 1 memory write, got {count}"

    def test_graph_converges_after_memory_write(self, agent):
        """
        Test: Graph converges to format_response_node after memory write.
        
        Verify that:
        1. Agent invocation completes (no infinite loop)
        2. Returns result dict with all required fields
        3. No GraphRecursionError
        """
        state = AgentState(
            conversation_id="test-convergence",
            trace_id="trace-convergence",
            created_at="2026-02-09T18:00:00",
            input_type="text",
            raw_input="Does the graph converge?",
        )
        
        # This should complete without hitting recursion limit
        # If it hits recursion limit, this will raise GraphRecursionError
        result = agent.graph.invoke(state)
        
        # Verify success - result is dict with required fields
        assert result is not None
        assert isinstance(result, dict)
        assert 'conversation_id' in result
        assert 'trace_id' in result
        assert result['conversation_id'] == "test-convergence"
        assert result['trace_id'] == "trace-convergence"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
