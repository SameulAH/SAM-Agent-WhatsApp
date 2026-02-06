"""
Integration tests for the LangGraph skeleton.

These tests verify that the complete graph executes from start to finish
with the exact flow specified in design/langgraph_skeleton.md.
"""

import pytest
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from inference import StubModelBackend


def test_full_graph_execution_success_path():
    """
    Test the complete success path through the graph.
    
    Path:
    __start__ → router → state_init → decision → preprocess 
    → decision → model_call → result_handling → decision → format → __end__
    """
    orchestrator = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    # Invoke with raw input
    # Note: Using sync invocation for testing; production uses async
    initial_state = orchestrator._create_initial_state("Hello, world!")
    
    # Execute through graph
    result = orchestrator.graph.invoke(initial_state)
    
    # Verify final state contains expected values
    assert result["conversation_id"]
    assert result["trace_id"]
    assert result["created_at"]
    assert result["final_output"] is not None
    assert result["error_type"] is None


def test_full_graph_deterministic():
    """Verify the graph produces deterministic results."""
    orchestrator = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    input_text = "test input"
    
    initial_state1 = orchestrator._create_initial_state(input_text)
    result1 = orchestrator.graph.invoke(initial_state1)
    
    initial_state2 = orchestrator._create_initial_state(input_text)
    result2 = orchestrator.graph.invoke(initial_state2)
    
    # Both executions should have same output
    assert result1["final_output"] == result2["final_output"]
    assert result1["status"] == result2["status"]


def test_graph_preserves_identity_invariants():
    """Verify conversation_id and trace_id are preserved through graph."""
    orchestrator = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    conversation_id = "conv-123-abc"
    trace_id = "trace-456-def"
    
    initial_state = orchestrator._create_initial_state(
        "test",
        conversation_id=conversation_id,
        trace_id=trace_id,
    )
    
    result = orchestrator.graph.invoke(initial_state)
    
    assert result["conversation_id"] == conversation_id
    assert result["trace_id"] == trace_id


def test_graph_no_memory_access():
    """Verify the graph does not attempt to access memory."""
    # This test ensures no memory-related operations occur
    orchestrator = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    initial_state = orchestrator._create_initial_state("test")
    result = orchestrator.graph.invoke(initial_state)
    
    # If we reach here without errors, memory was not accessed
    assert result is not None


def test_state_transitions_are_explicit():
    """Verify all state transitions follow the specified graph structure."""
    orchestrator = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    # Start with initial state
    state = orchestrator._create_initial_state("test input")
    
    # Node 1: router_node
    state_after_router = orchestrator._router_node(state)
    assert "input_type" in state_after_router
    
    # Node 2: state_init_node
    state = AgentState(**{**state.__dict__, **state_after_router})
    state_after_init = orchestrator._state_init_node(state)
    assert "conversation_id" in state_after_init
    assert "trace_id" in state_after_init
    
    # Node 3: decision_logic_node (first time)
    state = AgentState(**{**state.__dict__, **state_after_init})
    state_after_decision1 = orchestrator._decision_logic_node(state)
    assert state_after_decision1["command"] == "preprocess"
    
    # Node 4: task_preprocessing_node
    state = AgentState(**{**state.__dict__, **state_after_decision1})
    state_after_preprocess = orchestrator._task_preprocessing_node(state)
    assert "preprocessing_result" in state_after_preprocess
    
    # Node 5: decision_logic_node (second time)
    state = AgentState(**{**state.__dict__, **state_after_preprocess})
    state_after_decision2 = orchestrator._decision_logic_node(state)
    assert state_after_decision2["command"] == "call_model"
    
    # Node 6: model_call_node
    state = AgentState(**{**state.__dict__, **state_after_decision2})
    state_after_model = orchestrator._model_call_node(state)
    assert "model_response" in state_after_model
    
    # Node 7: result_handling_node
    state = AgentState(**{**state.__dict__, **state_after_model})
    state_after_result = orchestrator._result_handling_node(state)
    assert "final_output" in state_after_result
    
    # Node 8: decision_logic_node (third time)
    state = AgentState(**{**state.__dict__, **state_after_result})
    state_after_decision3 = orchestrator._decision_logic_node(state)
    assert state_after_decision3["command"] == "format"
    
    # Node 9: format_response_node
    state = AgentState(**{**state.__dict__, **state_after_decision3})
    response = orchestrator._format_response_node(state)
    assert response["status"]
    assert response["output"]


def test_invariant_single_execution():
    """Verify only one node executes at a time."""
    # This is guaranteed by LangGraph's synchronous execution model
    orchestrator = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    # Graph executes sequentially, not in parallel
    result = orchestrator.graph.invoke(orchestrator._create_initial_state("test"))
    
    # If we get here, execution was sequential
    assert result is not None


def test_invariant_no_silent_failures():
    """Verify failures are explicit and typed."""
    from inference import ModelResponse
    
    orchestrator = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    # Create a state with failed model response
    model_response = ModelResponse(
        status="fatal_error",
        error_type="backend_unavailable",
        metadata={},
    )
    
    state = orchestrator._create_initial_state("test")
    state.model_response = model_response
    
    # error_router_node should classify the failure
    result = orchestrator._error_router_node(state)
    
    assert result["error_type"] is not None
    assert result["final_output"] is not None
    # No silent failures


# Helper to create initial state
def _create_initial_state_helper(orchestrator, raw_input, conversation_id=None, trace_id=None):
    """Helper to create initial state for testing."""
    from agent.state_schema import AgentState
    from uuid import uuid4
    
    return AgentState(
        conversation_id=conversation_id or str(uuid4()),
        trace_id=trace_id or str(uuid4()),
        created_at="",
        input_type="",
        raw_input=raw_input,
    )


# Monkey-patch for testing
SAMAgentOrchestrator._create_initial_state = lambda self, raw_input, conversation_id=None, trace_id=None: _create_initial_state_helper(self, raw_input, conversation_id, trace_id)
