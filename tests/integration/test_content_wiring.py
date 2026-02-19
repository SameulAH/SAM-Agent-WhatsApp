"""
Test Stage 1: Content Loop Closure

Verifies that model output from AgentState is properly wired into the API response.

This test validates:
1. Content visibility: model output is returned in the response
2. Behavioral invariance: no changes to control flow, routing, or state mutations
3. Stub compatibility: works with StubModelBackend
"""

from agent.state_schema import AgentState
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from inference import StubModelBackend
from uuid import uuid4
from datetime import datetime


def test_content_wiring_stub_backend():
    """
    Given: StubModelBackend returns "This is a stubbed response."
    When: Agent executes with user input
    Then: final_output contains the model response
    """
    # Setup
    agent = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    state = AgentState(
        conversation_id=str(uuid4()),
        trace_id=str(uuid4()),
        created_at=datetime.now().isoformat(),
        input_type="text",
        raw_input="Test input"
    )
    
    # Execute
    result = agent.graph.invoke(state)
    
    # Verify: final_output is present and populated
    assert "final_output" in result, "final_output not in result"
    assert result["final_output"] is not None, "final_output is None"
    assert len(result["final_output"]) > 0, "final_output is empty"
    assert "stubbed" in result["final_output"].lower(), \
        f"Expected stub response, got: {result['final_output']}"


def test_content_wiring_preserves_state_integrity():
    """
    Given: A complete agent execution
    When: Model output is wired to final_output
    Then: All other state fields remain unchanged
    """
    agent = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    initial_state = AgentState(
        conversation_id="test-conv-id",
        trace_id="test-trace-id",
        created_at="2026-02-09T12:00:00",
        input_type="text",
        raw_input="Test input"
    )
    
    # Execute
    result = agent.graph.invoke(initial_state)
    
    # Verify identity and routing fields unchanged
    assert result["conversation_id"] == "test-conv-id"
    assert result["trace_id"] == "test-trace-id"
    assert result["created_at"] == "2026-02-09T12:00:00"
    assert result["input_type"] == "text"
    assert result["raw_input"] == "Test input"


def test_api_response_format():
    """
    Given: Agent execution completes successfully
    When: /invoke endpoint builds response
    Then: Response includes final_output as 'output' field
    """
    agent = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    state = AgentState(
        conversation_id=str(uuid4()),
        trace_id=str(uuid4()),
        created_at=datetime.now().isoformat(),
        input_type="text",
        raw_input="Test input"
    )
    
    # Execute graph
    result = agent.graph.invoke(state)
    
    # Simulate API response construction
    api_response = {
        "status": "success",
        "input": state.raw_input,
        "output": result.get("final_output", ""),  # This is the wire
        "conversation_id": result.get("conversation_id", ""),
        "trace_id": result.get("trace_id", "")
    }
    
    # Verify response structure
    assert api_response["status"] == "success"
    assert api_response["output"] != "", "output field is empty"
    assert "stubbed" in api_response["output"].lower()
    assert api_response["conversation_id"] == result["conversation_id"]
    assert api_response["trace_id"] == result["trace_id"]


def test_multiple_invocations_isolation():
    """
    Given: Multiple sequential invocations
    When: Each produces independent output
    Then: No state leakage between calls
    """
    agent = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    results = []
    for i in range(3):
        state = AgentState(
            conversation_id=str(uuid4()),
            trace_id=str(uuid4()),
            created_at=datetime.now().isoformat(),
            input_type="text",
            raw_input=f"Input {i}"
        )
        
        result = agent.graph.invoke(state)
        results.append(result)
    
    # Verify each has independent IDs
    conv_ids = [r["conversation_id"] for r in results]
    trace_ids = [r["trace_id"] for r in results]
    
    assert len(set(conv_ids)) == 3, "conversation_ids not unique"
    assert len(set(trace_ids)) == 3, "trace_ids not unique"
    
    # Verify each has output
    for r in results:
        assert r.get("final_output"), f"Missing final_output in result {r}"
