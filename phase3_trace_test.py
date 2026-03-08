"""
Minimal test to verify Phase 3 memory write cycle works.
Bypasses the model entirely, injects a fake ModelResponse into state
to simulate post-model Phase 3 routing.
"""
import sys
import sqlite3
import json
import logging

# Enable INFO logging so we can see memory_write_node events
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '/app')

from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.state_schema import AgentState
from inference import ModelResponse
from inference.ollama import OllamaModelBackend
from datetime import datetime

# Build the agent (uses real memory controller)
from agent.orchestrator import SAMOrchestrator
agent_wrapper = SAMOrchestrator()
orchestrator = agent_wrapper.langgraph_orchestrator

# Create a state that's already past the model call
# (preprocessing done, model_response set)
state = AgentState(
    conversation_id="telegram_phase3_test",
    trace_id="trace-phase3-001",
    created_at=datetime.utcnow().isoformat(),
    input_type="text",
    raw_input="What is 2 + 2?",
    preprocessing_result="What is 2 + 2?",
    model_response=ModelResponse(
        status="success",
        output="The answer is 4.",
        error_type=None,
        metadata={},
    ),
    memory_write_authorized=False,
    memory_write_status=None,
)

print("=== Phase 3 routing test ===")
print(f"State: model_response.output={state.model_response.output}, memory_write_authorized={state.memory_write_authorized}")

# Manually call decision_logic_node to see what Phase 3 returns
result = orchestrator._decision_logic_node(state)
print(f"Phase 3 result: {result}")

# Simulate what LangGraph does: apply result to state
for k, v in result.items():
    setattr(state, k, v)
print(f"After update: command={state.command}, memory_write_authorized={state.memory_write_authorized}")

if state.command == "memory_write":
    print("\n✅ Phase 3 routed to memory_write correctly")
    # Now simulate memory_write_node
    mem_result = orchestrator._memory_write_node_wrapper(state)
    print(f"memory_write_node result: {mem_result}")
    
    # Apply result
    for k, v in mem_result.items():
        setattr(state, k, v)
    
    # Check SQLite
    conn = sqlite3.connect('/app/data/memory.db')
    cur = conn.cursor()
    cur.execute("SELECT conversation_id, data FROM short_term_memory WHERE conversation_id='telegram_phase3_test'")
    row = cur.fetchone()
    conn.close()
    
    if row:
        data = json.loads(row[1]) if row[1] else {}
        print(f"\n✅ STM WRITTEN for {row[0]}:")
        print(f"   raw_input    : {data.get('raw_input')}")
        print(f"   final_output : {data.get('final_output')}")
    else:
        print("\n❌ STM entry NOT found!")
else:
    print(f"\n❌ Phase 3 did NOT route to memory_write! Got: {state.command}")
