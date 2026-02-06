# LangGraph Skeleton: Orchestration Layer

## Overview

This is the orchestration layer of the SAM (Stateful Agent Model) agent. It implements a deterministic, testable, invariant-safe LangGraph that serves as the control flow backbone.

**Key Property**: The graph is boring. It does almost nothing useful. That's intentional.

## Architecture

The graph is defined in `design/langgraph_skeleton.md` (source of truth).

The implementation is in `agent/langgraph_orchestrator.py`.

State schema is in `agent/state_schema.py`.

## Graph Structure

### Nodes (Exact)

| Node | Responsibility | Input | Output |
|------|----------------|-------|--------|
| `router_node` | Classify input modality (text\|audio\|image) | Raw request | input_type annotation |
| `state_init_node` | Initialize state, enforce identity invariants | Annotated request | conversation_id, trace_id, created_at |
| `decision_logic_node` | Make routing decisions (pure control flow) | Current state | command: preprocess\|call_model\|format |
| `task_preprocessing_node` | Modality-specific preprocessing | State with input_type | preprocessing_result |
| `model_call_node` | Invoke ModelBackend.generate() | Preprocessed input | model_response (success or failure) |
| `result_handling_node` | Validate and store model output | model_response | final_output |
| `error_router_node` | Classify failure, produce fallback | Failed model_response | error_type, fallback output |
| `format_response_node` | Serialize final state to response | Final state | Response dict |

### Flow (Exact)

**Success Path**:
```
__start__
  ↓
router_node
  ↓
state_init_node
  ↓
decision_logic_node [emit: preprocess]
  ↓
task_preprocessing_node
  ↓
decision_logic_node [emit: call_model]
  ↓
model_call_node (success)
  ↓
result_handling_node
  ↓
decision_logic_node [emit: format]
  ↓
format_response_node
  ↓
__end__
```

**Failure Path**:
```
model_call_node (failure)
  ↓
error_router_node
  ↓
format_response_node
  ↓
__end__
```

## Hard Rules

❌ **Do not violate these**:

1. Task nodes execute without decision logic
2. `decision_logic_node` makes ALL routing decisions
3. Model is called only through `ModelBackend` boundary
4. Memory is NOT accessed yet
5. All failures are explicit and typed
6. State is mutated only in agent core nodes
7. No retries, heuristics, or prompt engineering

If any rule is broken → architecture collapses.

## State Schema

```python
@dataclass
class AgentState:
    # Identity
    conversation_id: str
    trace_id: str
    created_at: str
    
    # Input / Routing
    input_type: str  # text | audio | image
    raw_input: str
    
    # Processing
    preprocessing_result: Optional[str] = None
    
    # Model
    model_response: Optional[ModelResponse] = None
    model_metadata: Optional[Dict[str, Any]] = None
    
    # Output
    final_output: Optional[str] = None
    error_type: Optional[str] = None
    
    # Control
    command: Optional[str] = None  # preprocess | call_model | format
```

## Testing

### Unit Tests
Location: `tests/unit/test_langgraph_skeleton.py`

- Individual node behavior
- State schema validation
- Routing logic
- Determinism with StubModelBackend

### Integration Tests
Location: `tests/integration/test_graph_execution.py`

- Full graph execution (start to finish)
- Success and failure paths
- Identity invariant preservation
- Explicit state transitions

## Usage

```python
from agent.orchestrator import SAMOrchestrator
from inference import StubModelBackend

# Default: uses StubModelBackend (CI/testing)
orchestrator = SAMOrchestrator()

# Custom backend
orchestrator = SAMOrchestrator(model_backend=StubModelBackend())

# Invoke
response = await orchestrator.invoke(
    raw_input="Hello, world!",
    conversation_id="optional-id",
    trace_id="optional-trace",
)

# Response structure
{
    "conversation_id": "...",
    "trace_id": "...",
    "status": "success" or "error",
    "output": "final agent output or fallback",
    "error_type": None or "timeout|backend_unavailable|invalid_output",
    "metadata": {...}
}
```

## Design Rationale

### Why so simple?

- **Determinism**: No randomness, no branching logic in task nodes
- **Testability**: Each node is independently testable
- **Invariant Safety**: Hard rules prevent accidental violations
- **Extensibility**: Add new backends without changing graph logic

### What comes next?

1. **Memory Integration** (Phase 2): Access short-term and long-term memory
2. **Prompt Engineering** (Phase 3): Inject context, constraints into model calls
3. **Routing Heuristics** (Phase 4): Intelligent next-step selection
4. **Retry Logic** (Phase 5): Handle transient failures

Each phase is orthogonal to this skeleton. The graph structure stays the same.

## Files

```
.
├── agent/
│   ├── langgraph_orchestrator.py    # Graph implementation (authoritative)
│   ├── state_schema.py              # AgentState definition
│   └── orchestrator.py              # Public API
├── inference/
│   ├── base.py                      # ModelBackend abstract
│   ├── stub.py                      # StubModelBackend (default)
│   ├── ollama.py                    # OllamaModelBackend (optional)
│   └── types.py                     # Shared contract
├── design/
│   └── langgraph_skeleton.md        # Source of truth (this specification)
└── tests/
    ├── unit/test_langgraph_skeleton.py
    └── integration/test_graph_execution.py
```

## Invariants Checklist

- [x] Only one node executes at a time
- [x] All transitions are explicit
- [x] Model output never controls routing
- [x] State is mutated only in agent core nodes
- [x] Memory is NOT accessed yet
- [x] Failures are typed and routed
- [x] Graph is fully deterministic with StubModelBackend
- [x] No node violates its responsibility contract
