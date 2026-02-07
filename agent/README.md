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
4. Memory is accessed ONLY via MemoryNodeManager (when authorized by decision_logic_node)
5. **Observability is strictly passive** (never influences execution, decisions, or state)
6. All failures are explicit and typed
7. State is mutated only in agent core nodes
8. No retries, heuristics, or prompt engineering

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
    command: Optional[str] = None  # preprocess | call_model | format | memory_read | memory_write | long_term_memory_read | long_term_memory_write
    
    # Short-term memory (Phase 2)
    memory_available: bool = True
    memory_read_authorized: bool = False
    memory_write_authorized: bool = False
    memory_read_result: Optional[Dict[str, Any]] = None
    memory_write_status: Optional[str] = None
    
    # Long-term memory (Phase 3.2)
    long_term_memory_requested: bool = False
    long_term_memory_status: str = "available"
    long_term_memory_read_result: Optional[Dict[str, Any]] = None
    long_term_memory_write_status: Optional[str] = None
```

## Memory Integration

### Short-Term Memory (Phase 2)
Session-scoped context store for conversation continuity. See `agent/memory/` for details.

### Long-Term Memory (Phase 3.2)
Advisory, append-only store of stable facts with semantic search. Used for personalization without influencing routing.

**Key Invariant**: Long-term memory is never authoritative. It informs responses but never changes:
- Routing decisions
- Control flow
- What to do next
- Authorization levels

If long-term memory is removed, agent correctness is unchanged—only response personalization degrades.

## Observability Integration (Phase 4)

### Tracing (Tool-Agnostic)
Implements passive observability via `agent/tracing/` module.

**Key Guarantee**: Removing all tracing leaves agent behavior unchanged. Only visibility is lost.

Features:
- **Tracer ABC**: Tool-agnostic interface (no vendor lock-in)
- **LangTraceTracer**: LangSmith backend (gracefully degrades if unavailable)
- **InvariantAlarmSystem**: Detects constraint violations (smoke detectors, not sprinklers)
- **Data Safety**: DENY list enforces that prompts, outputs, secrets never logged

See [agent/tracing/README.md](tracing/README.md) for full documentation.

**Design Frozen**: [design/observability_invariants.md](../design/observability_invariants.md)

### Integration Points
- Node entry/exit spans (in `langgraph_orchestrator.py`)
- Trace identity propagation (trace_id, conversation_id, user_id)
- Invariant violation alarms (non-blocking)
- Silent failure on backend down (never blocks agent execution)

### Non-Negotiable Constraints
1. Tracing never affects control flow
2. Tracing failures are silent and non-fatal
3. Forbidden data (prompts, outputs, secrets) never traced
4. Alarms emit but never block or decide
5. Agent behavior identical with tracing ON/OFF

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
