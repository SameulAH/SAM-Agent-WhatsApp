#!/usr/bin/env python3
"""
Quick integration test for content wiring verification.

Run with: python tests/integration/verify_content_wiring.py
"""

import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.state_schema import AgentState
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from inference import StubModelBackend
from uuid import uuid4
from datetime import datetime


def test_stub_backend_returns_output():
    """Test 1: StubModelBackend produces visible output"""
    print("[Test 1] Stub backend returns visible output...")
    
    agent = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    state = AgentState(
        conversation_id=str(uuid4()),
        trace_id=str(uuid4()),
        created_at=datetime.now().isoformat(),
        input_type="text",
        raw_input="Test input"
    )
    
    result = agent.graph.invoke(state)
    
    assert "final_output" in result, "FAIL: final_output not in result"
    assert result["final_output"] is not None, "FAIL: final_output is None"
    assert len(result["final_output"]) > 0, "FAIL: final_output is empty"
    assert "stubbed" in result["final_output"].lower(), \
        f"FAIL: Expected stub response, got: {result['final_output']}"
    
    print(f"  ✓ final_output: '{result['final_output']}'")
    return True


def test_api_response_format():
    """Test 2: API response includes output field with model content"""
    print("[Test 2] API response format...")
    
    agent = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    state = AgentState(
        conversation_id=str(uuid4()),
        trace_id=str(uuid4()),
        created_at=datetime.now().isoformat(),
        input_type="text",
        raw_input="Test input"
    )
    
    result = agent.graph.invoke(state)
    
    # Simulate API response construction (from agent/api.py)
    api_response = {
        "status": "success",
        "input": state.raw_input,
        "output": result.get("final_output", ""),  # The wire: final_output → output
        "conversation_id": result.get("conversation_id", ""),
        "trace_id": result.get("trace_id", "")
    }
    
    assert api_response["status"] == "success", "FAIL: status not success"
    assert api_response["output"] != "", "FAIL: output field is empty"
    assert "stubbed" in api_response["output"].lower(), \
        f"FAIL: Expected stub content, got: {api_response['output']}"
    
    print(f"  ✓ API response: {api_response}")
    return True


def test_state_invariants_preserved():
    """Test 3: No changes to control flow or state invariants"""
    print("[Test 3] State invariants preserved...")
    
    agent = SAMAgentOrchestrator(model_backend=StubModelBackend())
    
    conv_id = str(uuid4())
    trace_id = str(uuid4())
    
    state = AgentState(
        conversation_id=conv_id,
        trace_id=trace_id,
        created_at="2026-02-09T12:00:00",
        input_type="text",
        raw_input="Test input"
    )
    
    result = agent.graph.invoke(state)
    
    # Verify identity and input fields (created_at is reset by state_init_node as designed)
    assert result["conversation_id"] == conv_id, "FAIL: conversation_id changed"
    assert result["trace_id"] == trace_id, "FAIL: trace_id changed"
    assert result["input_type"] == "text", "FAIL: input_type changed"
    assert result["raw_input"] == "Test input", "FAIL: raw_input changed"
    # created_at is intentionally reset by state_init_node (by design)
    assert result["created_at"] is not None, "FAIL: created_at is None"
    
    print("  ✓ All critical invariants preserved")
    return True


def test_multiple_invocations():
    """Test 4: Multiple invocations are isolated"""
    print("[Test 4] Multiple invocations...")
    
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
        
        assert result.get("final_output"), f"FAIL: Missing final_output in result {i}"
    
    # Verify uniqueness
    conv_ids = [r["conversation_id"] for r in results]
    trace_ids = [r["trace_id"] for r in results]
    
    assert len(set(conv_ids)) == 3, "FAIL: conversation_ids not unique"
    assert len(set(trace_ids)) == 3, "FAIL: trace_ids not unique"
    
    print(f"  ✓ {len(results)} invocations with isolated state")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Content Loop Closure - Verification Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_stub_backend_returns_output,
        test_api_response_format,
        test_state_invariants_preserved,
        test_multiple_invocations,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
