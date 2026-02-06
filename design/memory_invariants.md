# Memory Invariants: Phase 2 Design

**Source of Truth for Memory Semantics**

This document specifies memory behavior before any storage implementation. It answers the hard questions about when, how, and why memory is accessed.

---

## Core Principle

**Memory is advisory, never authoritative.**

Memory may enhance responses, never determine control flow.

---

## Memory Access Rules

### Rule 1: Read Authorization

**When can memory be read?**

- ✅ ONLY when `decision_logic_node` explicitly authorizes it
- ✅ ONLY in explicit `memory_read_node` (separate node in graph)
- ❌ Task nodes may NEVER read memory directly
- ❌ No implicit reads on graph entry
- ❌ No auto-loading based on context

**Authorization Flow:**
```
decision_logic_node decides memory is needed
  ↓
sets memory_read_authorized = True in state
  ↓
graph routes to memory_read_node
  ↓
memory_read_node calls MemoryController.read()
  ↓
returns data or failure (safe)
  ↓
state updated with memory data (if success)
```

### Rule 2: Write Authorization

**When can memory be written?**

- ✅ ONLY when `decision_logic_node` explicitly authorizes it
- ✅ ONLY in explicit `memory_write_node` (separate node in graph)
- ❌ Task nodes may NEVER write memory directly
- ❌ No implicit writes on model output
- ❌ No auto-persisting based on action

**Authorization Flow:**
```
decision_logic_node decides outcome should be stored
  ↓
sets memory_write_authorized = True in state
  ↓
graph routes to memory_write_node
  ↓
memory_write_node calls MemoryController.write()
  ↓
returns success or failure (safe)
```

### Rule 3: Task Node Isolation

**Task nodes (preprocessing, model_call, result_handling) may NEVER:**

- ❌ Read memory
- ❌ Write memory
- ❌ Check memory state
- ❌ Request memory access
- ❌ Assume memory availability

**What task nodes can do:**

- ✅ Process data in state
- ✅ Call model through boundary
- ✅ Transform input/output
- ✅ Return results

Task nodes are deterministic executors, blind to memory.

---

## Data Storage Rules

### Rule 4: Content Type Restrictions

**Forbidden to store:**

- ❌ Raw user messages
- ❌ Raw assistant responses
- ❌ Chat logs or transcripts
- ❌ Audio/image data
- ❌ Full conversation history
- ❌ Token counts or API calls

These belong in short-term state, not long-term memory.

**Allowed to store:**

- ✅ Derived attributes ("user prefers concise responses")
- ✅ Stable facts ("user is a data scientist")
- ✅ User preferences ("language: Spanish")
- ✅ Interaction patterns ("most frequent query type")
- ✅ Summaries of past interactions
- ✅ Learned user intent

**Decision rule:** If it could change every message, don't store it. If it's stable across conversations, it belongs in memory.

### Rule 5: State Storage Restrictions

**State may contain:**

- ✅ conversation_id (pointer to memory)
- ✅ memory_available flag
- ✅ memory_read_authorized flag
- ✅ memory_write_authorized flag
- ✅ memory_read_result (after reading)
- ✅ memory_write_status (after writing)

**State may NOT contain:**

- ❌ Raw memory content
- ❌ Derived facts (those stay in memory service)
- ❌ Vectors or embeddings
- ❌ Persistence logic

**Principle:** State stores pointers and metadata, never knowledge.

---

## Failure Handling

### Rule 6: Memory Unavailability

**If memory is unavailable or corrupted:**

- ✅ Log/trace the failure
- ✅ Set memory_available = False
- ✅ Continue executing normally
- ✅ Agent behaves identically to memory_disabled

**NOT allowed:**

- ❌ Retry logic
- ❌ Blocking on memory
- ❌ Fallback routing
- ❌ Abort or error state

**Invariant:** Memory failure must never change control flow.

### Rule 7: Memory Consistency

**If memory read fails:**

- state.memory_read_result = None
- state.memory_read_authorized = False
- state.memory_available = False
- Agent continues without memory data
- No crash, no hang, no retry

**If memory write fails:**

- state.memory_write_status = "failed"
- Agent continues normally
- Log the error
- Next decision can retry or skip

---

## Control Flow Guarantees

### Rule 8: Memory Never Routes

**Memory may NEVER:**

- ❌ Change decision_logic_node decisions
- ❌ Route to different execution paths
- ❌ Trigger preprocessing or model calls
- ❌ Abort or redirect execution

**Correct mental model:**

Memory is read in `memory_read_node` and data is added to state.

But `decision_logic_node` always makes the same routing decision whether memory was read or not.

**Example:**

```python
# WRONG:
if memory.user_likes_concise_responses:
    route_to(quick_response)
else:
    route_to(detailed_response)

# CORRECT:
decision_logic_node always routes to model_call_node
memory data is in state
model_call_node receives memory data and uses it for response content
routing does not change
```

### Rule 9: Determinism with Memory ON/OFF

**Control flow must be identical whether memory is enabled or disabled.**

**What MUST match:**

- Same nodes execute
- Same routing decisions
- Same error handling paths
- Same model inputs (except memory context)

**What MAY differ:**

- Response text (personalization)
- Content richness (memory allows more tailored responses)
- Tone or style (memory may inform voice)

**Test requirement:** The agent graph should execute identically with `MemoryController` replaced by `DisabledMemoryController`.

---

## Authorization Model

### Rule 10: Explicit Intent

**Before any memory operation, state must contain explicit intent:**

```python
# For reading:
state.memory_read_authorized = True
state.memory_read_reason = "context_enrichment"  # or other reason

# For writing:
state.memory_write_authorized = True
state.memory_write_reason = "store_user_preference"  # or other reason
```

**Authorization is checked by MemoryController:**

```python
def read(self, request: MemoryReadRequest) -> MemoryReadResponse:
    if not request.authorized:
        raise UnauthorizedMemoryAccess()
    
    # Only then read
```

This prevents accidental memory access and makes all access auditable.

---

## Memory Interface Contract

### Rule 11: MemoryController Semantics

**MemoryController must support:**

```python
class MemoryController(ABC):
    # Read: retrieve derived facts
    def read(self, request: MemoryReadRequest) -> MemoryReadResponse:
        """Read authorized memory. Returns data or failure (not exception)."""
    
    # Write: store derived facts
    def write(self, request: MemoryWriteRequest) -> MemoryWriteResponse:
        """Write authorized memory. Returns success or failure (not exception)."""
```

**Key requirement:** Memory operations return `Response` objects, never raise exceptions. Failures are data, not errors.

---

## Graph Flow (Memory-Aware)

### Success Path with Optional Memory Reads

```
decision_logic_node
  ├─ IF memory_needed:
  │   ├─ set memory_read_authorized = True
  │   └─ route to memory_read_node
  │       ├─ call MemoryController.read()
  │       ├─ update state with result (success or failure)
  │       └─ always route to next processing node
  │
  └─ Always route to model_call_node
      └─ (model_call_node receives memory data if available)
```

### Write Path (After Execution)

```
result_handling_node
  └─ route to decision_logic_node (last time)
      ├─ IF outcome_should_be_stored:
      │   ├─ set memory_write_authorized = True
      │   └─ route to memory_write_node
      │       ├─ call MemoryController.write()
      │       ├─ update state with write_status
      │       └─ route to format_response_node
      │
      └─ Always route to format_response_node
```

**Critical:** Graph routes to memory nodes explicitly. No implicit reads/writes.

---

## Phase 2 Scope

**What IS implemented in Phase 2:**

- ✅ Memory invariants (this document)
- ✅ MemoryController abstract interface
- ✅ StubMemoryController (in-memory, deterministic)
- ✅ Minimal state extensions (pointers + flags)
- ✅ Explicit memory nodes in graph
- ✅ Tests proving memory is optional and non-fatal

**What is NOT implemented in Phase 2:**

- ❌ SQLite persistence
- ❌ Qdrant vectors
- ❌ Embeddings
- ❌ RAG
- ❌ Vector search
- ❌ Prompt engineering
- ❌ Memory-influenced heuristics

Those are Phase 3+.

---

## Testing Requirements

### Test 1: Unauthorized Read is Blocked
```python
request = MemoryReadRequest(authorized=False, ...)
response = memory_controller.read(request)
assert response.status == "unauthorized"
```

### Test 2: Unauthorized Write is Blocked
```python
request = MemoryWriteRequest(authorized=False, ...)
response = memory_controller.write(request)
assert response.status == "unauthorized"
```

### Test 3: Memory Failure is Non-Fatal
```python
# Simulate memory failure
orchestrator with broken memory controller
invoke agent
assert agent completes normally with memory_available=False
```

### Test 4: Control Flow Identical with Memory ON/OFF
```python
# Run agent with memory enabled
result1 = await orchestrator.invoke("test", memory_enabled=True)

# Run agent with memory disabled
result2 = await orchestrator.invoke("test", memory_enabled=False)

assert result1.control_flow == result2.control_flow  # Same nodes executed
assert result1.routing == result2.routing  # Same decisions made
```

### Test 5: Task Nodes Never See Memory
```python
# Instrument task nodes to check for memory access
assert preprocessing_node does not call memory
assert model_call_node does not call memory
assert result_handling_node does not call memory
```

---

## Checklist: Memory is Safe

- [x] Memory is opt-in (disabled by default in Phase 2)
- [x] Reads authorized only by decision_logic_node
- [x] Writes authorized only by decision_logic_node
- [x] Task nodes are memory-blind
- [x] No raw conversation history stored
- [x] Memory failure is non-fatal
- [x] Control flow identical with memory ON/OFF
- [x] State stores pointers, not knowledge
- [x] All memory access is explicit and auditable

---

## Summary

By the end of Phase 2:

✅ Memory behavior is fully specified (this document)

✅ Memory access is explicit (authorized by decision_logic_node)

✅ Memory is optional and replaceable (like ModelBoundary)

✅ Agent works identically with memory disabled

✅ Memory failures degrade safely

This design allows Phase 3+ to add SQLite/Qdrant without changing agent behavior.
