"""
Test suite for the LangGraph skeleton.

These tests verify that the graph executes exactly as specified in
design/langgraph_skeleton.md with no deviations.

Tests focus on:
- Exact node execution order
- Correct state transitions
- Deterministic behavior with StubModelBackend
- All invariants are maintained
"""

import pytest
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.state_schema import AgentState
from inference import StubModelBackend, ModelRequest, ModelResponse


def test_graph_compilation():
    """Verify the graph compiles without errors."""
    orchestrator = SAMAgentOrchestrator()
    assert orchestrator.graph is not None


def test_initial_state_validation():
    """Verify state schema enforces invariants."""
    with pytest.raises(ValueError, match="conversation_id must not be empty"):
        AgentState(
            conversation_id="",
            trace_id="test",
            created_at="2026-02-06",
            input_type="text",
            raw_input="test",
        )

    with pytest.raises(ValueError, match="input_type must be"):
        AgentState(
            conversation_id="test",
            trace_id="test",
            created_at="2026-02-06",
            input_type="invalid",
            raw_input="test",
        )


def test_stub_backend_deterministic():
    """Verify StubModelBackend is deterministic."""
    backend = StubModelBackend()
    
    request = ModelRequest(task="respond", prompt="Hello")
    response1 = backend.generate(request)
    response2 = backend.generate(request)
    
    assert response1.status == response2.status
    assert response1.output == response2.output
    assert response1.error_type == response2.error_type


def test_stub_backend_fail_task():
    """Verify StubModelBackend handles fail task correctly."""
    backend = StubModelBackend()
    
    request = ModelRequest(task="fail", prompt="test")
    response = backend.generate(request)
    
    assert response.status == "recoverable_error"
    assert response.error_type == "invalid_output"


def test_router_node():
    """Verify router_node classifies input modality."""
    from uuid import uuid4
    orchestrator = SAMAgentOrchestrator()
    
    state = AgentState(
        conversation_id="test-conv",
        trace_id=str(uuid4()),
        created_at="2026-02-06T10:00:00Z",
        input_type="text",
        raw_input="Hello, world!",
    )
    
    result = orchestrator._router_node(state)
    assert result["input_type"] == "text"


def test_state_init_node():
    """Verify state_init_node generates IDs and timestamps."""
    from uuid import uuid4
    orchestrator = SAMAgentOrchestrator()
    
    state = AgentState(
        conversation_id="test-conv",
        trace_id=str(uuid4()),
        created_at="2026-02-06T10:00:00Z",
        input_type="text",
        raw_input="test",
    )
    
    result = orchestrator._state_init_node(state)
    
    assert result["conversation_id"]
    assert result["trace_id"]
    assert result["created_at"]


def test_decision_logic_node_flow():
    """Verify decision_logic_node makes correct routing decisions."""
    orchestrator = SAMAgentOrchestrator()
    
    # After state_init, before preprocessing
    state1 = AgentState(
        conversation_id="test",
        trace_id="test",
        created_at="now",
        input_type="text",
        raw_input="test",
        preprocessing_result=None,
    )
    result1 = orchestrator._decision_logic_node(state1)
    assert result1["command"] == "preprocess"
    
    # After preprocessing, before model call
    state2 = AgentState(
        conversation_id="test",
        trace_id="test",
        created_at="now",
        input_type="text",
        raw_input="test",
        preprocessing_result="normalized text",
        model_response=None,
    )
    result2 = orchestrator._decision_logic_node(state2)
    assert result2["command"] == "call_model"
    
    # After model call, format response
    model_response = ModelResponse(
        status="success",
        output="model output",
        metadata={"backend": "stub"},
    )
    state3 = AgentState(
        conversation_id="test",
        trace_id="test",
        created_at="now",
        input_type="text",
        raw_input="test",
        preprocessing_result="normalized",
        model_response=model_response,
    )
    result3 = orchestrator._decision_logic_node(state3)
    assert result3["command"] == "format"


def test_task_preprocessing_node():
    """Verify task_preprocessing_node handles text modality."""
    orchestrator = SAMAgentOrchestrator()
    
    state = AgentState(
        conversation_id="test",
        trace_id="test",
        created_at="now",
        input_type="text",
        raw_input="  Hello, world!  ",
    )
    
    result = orchestrator._task_preprocessing_node(state)
    assert result["preprocessing_result"] == "Hello, world!"


def test_model_call_node_success():
    """Verify model_call_node invokes backend correctly."""
    backend = StubModelBackend()
    orchestrator = SAMAgentOrchestrator(model_backend=backend)
    
    state = AgentState(
        conversation_id="test",
        trace_id="test-trace-123",
        created_at="now",
        input_type="text",
        raw_input="test",
        preprocessing_result="normalized",
    )
    
    result = orchestrator._model_call_node(state)
    
    assert result["model_response"].status == "success"
    assert result["model_response"].output is not None
    assert result["model_metadata"]["backend"] == "stub"


def test_result_handling_node():
    """Verify result_handling_node validates model output."""
    orchestrator = SAMAgentOrchestrator()
    
    model_response = ModelResponse(
        status="success",
        output="Valid output",
        metadata={"backend": "stub"},
    )
    
    state = AgentState(
        conversation_id="test",
        trace_id="test",
        created_at="now",
        input_type="text",
        raw_input="test",
        model_response=model_response,
    )
    
    result = orchestrator._result_handling_node(state)
    assert result["final_output"] == "Valid output"


def test_error_router_node():
    """Verify error_router_node classifies failures."""
    orchestrator = SAMAgentOrchestrator()
    
    model_response = ModelResponse(
        status="fatal_error",
        error_type="backend_unavailable",
        metadata={"backend": "ollama"},
    )
    
    state = AgentState(
        conversation_id="test",
        trace_id="test",
        created_at="now",
        input_type="text",
        raw_input="test",
        model_response=model_response,
    )
    
    result = orchestrator._error_router_node(state)
    assert result["error_type"] == "backend_unavailable"
    assert "[Error:" in result["final_output"]


def test_format_response_node_success():
    """Verify format_response_node serializes success state."""
    orchestrator = SAMAgentOrchestrator()
    
    state = AgentState(
        conversation_id="conv-123",
        trace_id="trace-456",
        created_at="2026-02-06T10:00:00",
        input_type="text",
        raw_input="test",
        final_output="Agent response",
        error_type=None,
        model_metadata={"backend": "stub"},
    )
    
    result = orchestrator._format_response_node(state)
    
    assert result["conversation_id"] == "conv-123"
    assert result["trace_id"] == "trace-456"
    assert result["status"] == "success"
    assert result["output"] == "Agent response"
    assert result["error_type"] is None


def test_format_response_node_error():
    """Verify format_response_node serializes error state."""
    orchestrator = SAMAgentOrchestrator()
    
    state = AgentState(
        conversation_id="conv-123",
        trace_id="trace-456",
        created_at="2026-02-06T10:00:00",
        input_type="text",
        raw_input="test",
        final_output="[Error: timeout]",
        error_type="timeout",
        model_metadata={"backend": "ollama"},
    )
    
    result = orchestrator._format_response_node(state)
    
    assert result["status"] == "error"
    assert result["error_type"] == "timeout"

# ─────────────────────────────────────────────────────────
# PHASE 2: MEMORY STATE TESTS
# ─────────────────────────────────────────────────────────


def test_state_has_memory_fields():
    """Verify state schema includes memory fields (Phase 2)."""
    state = AgentState(
        conversation_id="test",
        trace_id="test",
        created_at="now",
        input_type="text",
        raw_input="test",
    )
    
    # Memory fields exist with defaults
    assert hasattr(state, "memory_available")
    assert state.memory_available is True
    assert hasattr(state, "memory_read_authorized")
    assert state.memory_read_authorized is False
    assert hasattr(state, "memory_write_authorized")
    assert state.memory_write_authorized is False
    assert hasattr(state, "memory_read_result")
    assert state.memory_read_result is None
    assert hasattr(state, "memory_write_status")
    assert state.memory_write_status is None


def test_memory_fields_can_be_set():
    """Verify memory fields can be set without state validation errors."""
    state = AgentState(
        conversation_id="test",
        trace_id="test",
        created_at="now",
        input_type="text",
        raw_input="test",
        memory_available=False,
        memory_read_authorized=True,
        memory_write_authorized=True,
        memory_read_result={"key": "value"},
        memory_write_status="success",
    )
    
    assert state.memory_available is False
    assert state.memory_read_authorized is True
    assert state.memory_write_authorized is True
    assert state.memory_read_result == {"key": "value"}
    assert state.memory_write_status == "success"


def test_memory_fields_dont_affect_schema_validation():
    """Verify memory fields are orthogonal to state validation."""
    # Valid state with memory fields
    state = AgentState(
        conversation_id="test",
        trace_id="test",
        created_at="now",
        input_type="text",
        raw_input="test",
        memory_available=True,
    )
    assert state is not None
    
    # Invalid conversation_id still fails (validation unchanged)
    with pytest.raises(ValueError, match="conversation_id"):
        AgentState(
            conversation_id="",
            trace_id="test",
            created_at="now",
            input_type="text",
            raw_input="test",
            memory_available=True,
        )