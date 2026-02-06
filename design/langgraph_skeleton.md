# LangGraph Skeleton Architecture

**Source of Truth for the SAM Agent Control Flow**

This document specifies the exact graph structure, node responsibilities, and transition rules for the agent orchestrator.

## Node List (Authoritative)

The agent graph consists of exactly these nodes:

### Entry / Exit
- `__start__`: Graph entry point
- `__end__`: Graph exit point

### Routing
- `router_node`: Input modality classification

### Agent Core
- `state_init_node`: State initialization
- `decision_logic_node`: Control flow decisions
- `task_preprocessing_node`: Modality-specific preprocessing
- `model_call_node`: LLM invocation
- `result_handling_node`: Output validation and state update
- `error_router_node`: Error classification and fallback
- `format_response_node`: Response serialization

## Node Responsibilities

### router_node
**Input**: Raw invocation request  
**Output**: State with `input_type` annotated  
**Logic**:
- Classify input modality: `text` | `audio` | `image`
- Annotate state with detected input_type
- Forward to state_init_node

**Must NOT**:
- Call model
- Read/write memory
- Decide next step

---

### state_init_node
**Input**: Annotated request  
**Output**: Initialized state with identity invariants  
**Logic**:
- Initialize conversation_id (generate or from request)
- Initialize trace_id (generate or from request)
- Set created_at timestamp
- Initialize empty state fields (preprocessing_result, model_response, etc.)
- Enforce schema validation

**Must NOT**:
- Branch logic
- Call model
- Access memory

---

### decision_logic_node
**Input**: Current state  
**Output**: Routing command (command field updated)  
**Logic**:
- Examine state and routing context
- Emit one of: `preprocess`, `call_model`, `success`, `failure`
- Pure control flow — no execution

**Entry Point 1**: After state_init_node (→ preprocess)  
**Entry Point 2**: After task_preprocessing_node (→ call_model)  
**Entry Point 3**: After result_handling_node (→ success)

**Must NOT**:
- Execute tasks
- Call model
- Mutate state directly

---

### task_preprocessing_node
**Input**: State with input_type and raw content  
**Output**: State with preprocessing_result updated  
**Logic**:
- If input_type == "text": normalize text
- If input_type == "audio": placeholder for STT (not implemented)
- If input_type == "image": placeholder for preprocessing (not implemented)
- Store result in state.preprocessing_result

**Must NOT**:
- Decide control flow
- Call model
- Write memory
- Handle errors globally

---

### model_call_node
**Input**: State with preprocessed content  
**Output**: State with model_response updated or failure signal  
**Logic**:
- Create ModelRequest from state
- Call `self.model_backend.generate(request)`
- Store response in state.model_response
- If response.status == "success": forward to result_handling_node
- If response.status != "success": forward to error_router_node

**Must NOT**:
- Retry
- Decide routing (transitions are explicit)
- Mutate state beyond model_response

---

### result_handling_node
**Input**: State with model_response (success)  
**Output**: State with validated output  
**Logic**:
- Validate model_response.output is non-empty
- Update state.final_output = model_response.output
- Update state.model_metadata = model_response.metadata
- Forward to decision_logic_node with success signal

**Must NOT**:
- Call model again
- Access memory implicitly
- Model outputs are data, not control signals

---

### error_router_node
**Input**: State with failed model_response  
**Output**: State with fallback values  
**Logic**:
- Classify failure: timeout, backend_unavailable, invalid_output
- Set state.error_type = classified error
- Set state.final_output = "<fallback message>"
- Forward to format_response_node

**Must NOT**:
- Retry silently
- Mutate unrelated state

---

### format_response_node
**Input**: Final state (success or failure)  
**Output**: Response payload  
**Logic**:
- Convert state → response object
- Include: conversation_id, trace_id, output, status, metadata
- Return response

**Must NOT**:
- Decide logic
- Call model
- Write memory

---

## Graph Flow (Exact)

### Success Path
```
__start__
  ↓
router_node
  ↓
state_init_node
  ↓
decision_logic_node → [command: preprocess]
  ↓
task_preprocessing_node
  ↓
decision_logic_node → [command: call_model]
  ↓
model_call_node (success)
  ↓
result_handling_node
  ↓
decision_logic_node → [command: success]
  ↓
format_response_node
  ↓
__end__
```

### Failure Path
```
model_call_node (failure)
  ↓
error_router_node
  ↓
format_response_node
  ↓
__end__
```

---

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
    command: Optional[str] = None  # preprocess, call_model, success, failure
```

---

## Invariants (MUST BE PRESERVED)

1. **Only one node executes at a time** — No parallelization
2. **All transitions are explicit** — No implicit fallbacks
3. **Model output never controls routing** — decision_logic_node decides
4. **State is mutated only in agent core nodes** — Defined set of mutation points
5. **Memory is NOT accessed yet** — Phase 2 design
6. **Failures are typed and routed** — error_router_node classifies

---

## Implementation Notes

- No LangGraph channels beyond the state object
- No retry logic
- No prompt engineering
- No memory access
- All decisions are made by decision_logic_node
- Task nodes execute without branching
- Graph is fully deterministic with StubModelBackend
