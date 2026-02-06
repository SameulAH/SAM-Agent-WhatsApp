#!/usr/bin/env python3
"""
Verification script for LangGraph skeleton implementation.

This script verifies that:
1. All required files exist
2. All nodes are implemented
3. All tests can be imported
4. State schema is valid
5. Graph compiles successfully
"""

import sys
from pathlib import Path

def check_files():
    """Verify all required files exist."""
    required_files = [
        "design/langgraph_skeleton.md",
        "agent/langgraph_orchestrator.py",
        "agent/state_schema.py",
        "agent/orchestrator.py",
        "agent/README.md",
        "inference/base.py",
        "inference/stub.py",
        "inference/ollama.py",
        "inference/types.py",
        "inference/__init__.py",
        "tests/unit/test_langgraph_skeleton.py",
        "tests/integration/test_graph_execution.py",
        "ARCHITECTURE.md",
        "QUICK_REFERENCE.md",
        "SKELETON_IMPLEMENTATION.md",
        "IMPLEMENTATION_COMPLETE.md",
    ]
    
    print("📋 Checking required files...")
    all_exist = True
    for file in required_files:
        path = Path(file)
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
        all_exist = all_exist and exists
    
    return all_exist

def check_imports():
    """Verify core modules can be imported."""
    print("\n📦 Checking imports...")
    try:
        from agent.state_schema import AgentState
        print("  ✅ agent.state_schema.AgentState")
    except Exception as e:
        print(f"  ❌ agent.state_schema.AgentState: {e}")
        return False
    
    try:
        from agent.langgraph_orchestrator import SAMAgentOrchestrator
        print("  ✅ agent.langgraph_orchestrator.SAMAgentOrchestrator")
    except Exception as e:
        print(f"  ❌ agent.langgraph_orchestrator.SAMAgentOrchestrator: {e}")
        return False
    
    try:
        from agent.orchestrator import SAMOrchestrator
        print("  ✅ agent.orchestrator.SAMOrchestrator")
    except Exception as e:
        print(f"  ❌ agent.orchestrator.SAMOrchestrator: {e}")
        return False
    
    try:
        from inference import (
            ModelBackend,
            StubModelBackend,
            ModelRequest,
            ModelResponse,
        )
        print("  ✅ inference.ModelBackend")
        print("  ✅ inference.StubModelBackend")
        print("  ✅ inference.ModelRequest")
        print("  ✅ inference.ModelResponse")
    except Exception as e:
        print(f"  ❌ inference modules: {e}")
        return False
    
    return True

def check_nodes():
    """Verify all 8 nodes are implemented."""
    print("\n🧠 Checking nodes implementation...")
    try:
        from agent.langgraph_orchestrator import SAMAgentOrchestrator
        orchestrator = SAMAgentOrchestrator()
        
        nodes = [
            "_router_node",
            "_state_init_node",
            "_decision_logic_node",
            "_task_preprocessing_node",
            "_model_call_node",
            "_result_handling_node",
            "_error_router_node",
            "_format_response_node",
        ]
        
        all_exist = True
        for node in nodes:
            exists = hasattr(orchestrator, node)
            status = "✅" if exists else "❌"
            print(f"  {status} {node}")
            all_exist = all_exist and exists
        
        return all_exist
    except Exception as e:
        print(f"  ❌ Failed to check nodes: {e}")
        return False

def check_graph_compilation():
    """Verify graph compiles successfully."""
    print("\n🔀 Checking graph compilation...")
    try:
        from agent.langgraph_orchestrator import SAMAgentOrchestrator
        orchestrator = SAMAgentOrchestrator()
        
        if orchestrator.graph is None:
            print("  ❌ Graph is None")
            return False
        
        print("  ✅ Graph compiled successfully")
        return True
    except Exception as e:
        print(f"  ❌ Graph compilation failed: {e}")
        return False

def check_state_schema():
    """Verify state schema is valid."""
    print("\n🧪 Checking state schema...")
    try:
        from agent.state_schema import AgentState
        
        # Valid state
        try:
            state = AgentState(
                conversation_id="test",
                trace_id="test",
                created_at="2026-02-06",
                input_type="text",
                raw_input="test",
            )
            print("  ✅ Valid state creation")
        except Exception as e:
            print(f"  ❌ Valid state creation: {e}")
            return False
        
        # Invalid conversation_id
        try:
            state = AgentState(
                conversation_id="",
                trace_id="test",
                created_at="2026-02-06",
                input_type="text",
                raw_input="test",
            )
            print("  ❌ Invalid conversation_id not caught")
            return False
        except ValueError:
            print("  ✅ Invalid conversation_id caught")
        
        # Invalid input_type
        try:
            state = AgentState(
                conversation_id="test",
                trace_id="test",
                created_at="2026-02-06",
                input_type="invalid",
                raw_input="test",
            )
            print("  ❌ Invalid input_type not caught")
            return False
        except ValueError:
            print("  ✅ Invalid input_type caught")
        
        return True
    except Exception as e:
        print(f"  ❌ State schema check failed: {e}")
        return False

def check_stub_backend():
    """Verify StubModelBackend works."""
    print("\n🎬 Checking StubModelBackend...")
    try:
        from inference import StubModelBackend, ModelRequest
        
        backend = StubModelBackend()
        print("  ✅ StubModelBackend instantiation")
        
        request = ModelRequest(task="respond", prompt="test")
        response = backend.generate(request)
        
        if response.status != "success":
            print(f"  ❌ Unexpected status: {response.status}")
            return False
        print("  ✅ Generate method works")
        
        if response.output is None:
            print("  ❌ Output is None")
            return False
        print("  ✅ Response has output")
        
        return True
    except Exception as e:
        print(f"  ❌ StubModelBackend check failed: {e}")
        return False

def check_tests_importable():
    """Verify test files can be imported."""
    print("\n🧪 Checking tests...")
    try:
        import tests.unit.test_langgraph_skeleton
        print("  ✅ tests.unit.test_langgraph_skeleton")
    except Exception as e:
        print(f"  ❌ tests.unit.test_langgraph_skeleton: {e}")
        return False
    
    try:
        import tests.integration.test_graph_execution
        print("  ✅ tests.integration.test_graph_execution")
    except Exception as e:
        print(f"  ❌ tests.integration.test_graph_execution: {e}")
        return False
    
    return True

def main():
    """Run all verification checks."""
    print("=" * 80)
    print("LangGraph Skeleton Implementation Verification")
    print("=" * 80)
    
    checks = [
        ("Files", check_files),
        ("Imports", check_imports),
        ("Nodes", check_nodes),
        ("Graph Compilation", check_graph_compilation),
        ("State Schema", check_state_schema),
        ("StubModelBackend", check_stub_backend),
        ("Tests", check_tests_importable),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 80)
    if all_passed:
        print("✅ All checks passed! LangGraph skeleton is ready.")
        return 0
    else:
        print("❌ Some checks failed. See above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
