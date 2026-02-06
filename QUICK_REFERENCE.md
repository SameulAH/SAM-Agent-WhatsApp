# Quick Reference: LangGraph Skeleton

## Core Principle

**Task nodes execute. Decision nodes route. Model outputs are data.**

---

## Node Checklist (8 Total)

| # | Node | Task | Emit/Output |
|----|------|------|-------------|
| 1 | `router_node` | Classify modality | input_type |
| 2 | `state_init_node` | Init identity | conversation_id, trace_id |
| 3 | `decision_logic_node` | Route control | command (preprocess\|call_model\|format) |
| 4 | `task_preprocessing_node` | Transform input | preprocessing_result |
| 5 | `model_call_node` | Call backend | model_response |
| 6 | `result_handling_node` | Validate output | final_output |
| 7 | `error_router_node` | Type error | error_type, fallback |
| 8 | `format_response_node` | Serialize | response dict |

---

## State Fields (12 Total)

### Identity (Immutable)
```python
conversation_id: str
trace_id: str
created_at: str
```

### Input
```python
input_type: str  # text | audio | image
raw_input: str
```

### Processing
```python
preprocessing_result: Optional[str]
```

### Model
```python
model_response: Optional[ModelResponse]
model_metadata: Optional[Dict[str, Any]]
```

### Output
```python
final_output: Optional[str]
error_type: Optional[str]
```

### Control
```python
command: Optional[str]  # preprocess | call_model | format
```

---

## Hard Rules (NON-NEGOTIABLE)

```
❌ Do not invent logic
❌ Do not merge responsibilities
❌ Do not call models except through boundary
❌ Do not read/write memory yet
❌ Do not add retries, heuristics, or prompts
❌ Do not have implicit transitions
❌ Do not mutate state outside agent core
```

If any rule is broken → architecture collapses.

---

## Success Path (9 Steps)

```
1. __start__
2. router_node         → input_type
3. state_init_node     → conversation_id, trace_id, created_at
4. decision_logic_node → command: preprocess
5. task_preprocessing_node → preprocessing_result
6. decision_logic_node → command: call_model
7. model_call_node     → model_response (success)
8. result_handling_node → final_output
9. decision_logic_node → command: format
10. format_response_node → response dict
11. __end__
```

---

## Failure Path (4 Steps)

```
7. model_call_node     → model_response (failure)
7b. error_router_node  → error_type, fallback output
8. format_response_node → response dict
9. __end__
```

---

## Invariants (6 MUST-PRESERVE)

1. **Single Execution**: Only one node at a time
2. **Explicit Transitions**: No implicit fallbacks
3. **Model Output is Data**: Never controls routing
4. **Controlled Mutation**: Only agent core nodes mutate
5. **No Memory Access**: Phase 2 feature
6. **Typed Failures**: error_router classifies

---

## Testing

### Unit Tests (13)
- Node behavior in isolation
- State schema validation
- Routing decisions
- Determinism with StubModelBackend

### Integration Tests (9)
- Full graph execution
- Success and failure paths
- Identity preservation
- Explicit transitions
- Invariant validation

---

## Files

```
design/langgraph_skeleton.md         ← Source of truth
agent/langgraph_orchestrator.py      ← Implementation
agent/state_schema.py                ← State type
agent/orchestrator.py                ← Public API
inference/                           ← Boundary layer
tests/unit/test_langgraph_skeleton.py
tests/integration/test_graph_execution.py
```

---

## Quick Start

```python
from agent.orchestrator import SAMOrchestrator
from inference import StubModelBackend

# Create orchestrator
orchestrator = SAMOrchestrator()

# Invoke
response = await orchestrator.invoke(
    raw_input="Hello, world!",
)

# Response structure
{
    "conversation_id": "...",
    "trace_id": "...",
    "status": "success",
    "output": "stubbed response",
    "error_type": None,
    "metadata": {"backend": "stub"}
}
```

---

## Decision Logic Algorithm

```
ENTRY 1: After state_init
  IF preprocessing_result is None
    command = "preprocess"

ENTRY 2: After task_preprocessing
  IF model_response is None
    command = "call_model"

ENTRY 3: After result_handling
  ELSE
    command = "format"
```

Pure control flow. No execution. No state mutation.

---

## Error Classification

```python
error_type in {
    "timeout",
    "backend_unavailable",
    "invalid_output",
    None  # success
}
```

Set by `error_router_node` only.

---

## Modality Handling

| Modality | Preprocessing | Future |
|----------|---------------|--------|
| text | `.strip()` | More normalization |
| audio | placeholder | STT integration |
| image | placeholder | Vision preprocessing |

---

## Why This Design?

✓ **Deterministic**: Same input → same output  
✓ **Testable**: Each node independently testable  
✓ **Traceable**: No implicit behavior  
✓ **Extensible**: Add features without changing graph  
✓ **Production-Ready**: No magic, no surprises  

---

## Phase 2 (Memory Integration)

Once skeleton is validated:
1. Access short-term memory (conversation history)
2. Access long-term memory (semantic search)
3. Store interaction outcomes

The graph structure does NOT change.

---

## Common Questions

**Q: Can decision_logic_node execute tasks?**
A: No. Task nodes execute. Decision nodes route.

**Q: Can task nodes decide next step?**
A: No. decision_logic_node decides ALL routing.

**Q: Can model output control routing?**
A: No. Model outputs are data. decision_logic_node decides.

**Q: Can we access memory yet?**
A: No. Phase 2 feature. Zero memory access in Phase 1.

**Q: Can we add retries?**
A: No. model_call_node delegates to backend. Retries happen in backend or LangGraph layer, not here.

**Q: Why so much duplication in nodes?**
A: Separation of concerns. Each node has ONE responsibility.

---

## Commit Message

```
feat: implement langgraph skeleton orchestration layer

- Define 8-node graph with exact specification from design/
- Implement state_schema.py with AgentState invariants
- Implement langgraph_orchestrator.py with all nodes
- Add 13 unit tests + 9 integration tests
- All hard rules enforced, all invariants preserved
- Graph is fully deterministic with StubModelBackend
- Phase 1 complete: orchestration layer ready for Phase 2 (memory)

Co-authored: Architecture skeleton definition
```

---

**This is boring. That's correct.**

If the skeleton felt clever → wrong  
If the skeleton felt minimal → correct ✓
