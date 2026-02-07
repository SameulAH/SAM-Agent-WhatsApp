# Agent Memory

## Overview

This module manages memory services for the agent. It is completely decoupled from agent logic and operates as an optional, non-fatal service.

**Key property**: If memory fails, the agent continues execution normally. Only continuity degrades.

## Phase Implementation Status

### Phase 3.1: SQLite Short-Term Memory (CLOSED)

SQLite short-term memory is a durable, session-scoped scratchpad that supports conversation continuity without ever influencing control flow, decisions, or authority.

**Short-term memory IS:**
- Session-scoped
- Overwritable
- Allowed to be wrong
- Durable across restarts
- Used only for recent context

**Short-term memory IS NOT:**
- Knowledge
- Personalization
- Long-term memory
- Routing input
- Decision input

#### Implementation

- **SQLiteShortTermMemoryStore**: Implements `MemoryController` interface
- **Data stored**: conversation_id, turn info, message metadata, timestamps
- **Data NOT stored**: long-term facts, preferences, embeddings, control signals
- **Failure mode**: Non-fatal (degraded continuity, not blocked execution)

#### Interface

```python
from agent.memory import SQLiteShortTermMemoryStore

# Create instance (auto-initializes schema)
memory = SQLiteShortTermMemoryStore(db_path="memory.db")

# Read session context (only when authorized)
read_request = MemoryReadRequest(
    conversation_id="conv-123",
    key="session_context",
    authorized=True,  # Set by decision_logic_node
)
response = memory.read(read_request)

# Write session context (only when authorized)
write_request = MemoryWriteRequest(
    conversation_id="conv-123",
    key="session_context",
    data={"turns": 5, "last_update": "2026-02-06T10:00:00Z"},
    authorized=True,  # Set by decision_logic_node
)
response = memory.write(write_request)
```

### Phase 3.2: Long-Term Memory (CLOSED)

Long-term memory is an advisory, append-only store of stable facts with semantic search capabilities. It enables cross-session personalization without ever influencing control flow or decisions.

**Long-term memory IS:**
- User-scoped (spans multiple conversations)
- Append-only (immutable history)
- Advisory-only (never influences routing)
- Semantically searchable (Qdrant-backed)
- Non-blocking (failures don't prevent execution)
- Authorization-based (decision_logic_node sole authorizer)

**Long-term memory IS NOT:**
- Updatable (facts can't be corrected retroactively)
- Authoritative (never decides what to do)
- Transactional (no rollbacks)
- Mandatory (gracefully degrades if unavailable)
- Emotional state (never stores urgency, emotions, or control directives)

#### Implementation

- **LongTermMemoryStore**: Abstract interface with write_fact() and retrieve_facts()
- **StubLongTermMemoryStore**: In-memory append-only list for testing
- **DisabledLongTermMemoryStore**: Disabled store for control-flow invariant testing
- **QdrantLongTermMemoryStore**: Qdrant-backed semantic search (production)

#### Data Model

```python
@dataclass
class MemoryFact:
    fact_type: str              # "preference", "pattern", "summary", etc.
    content: Dict[str, Any]     # JSON-serializable fact
    user_id: str                # Who owns this fact?
    confidence: float = 1.0     # 0.0-1.0 confidence score
    source: str = "conversation"  # Where did this fact come from?
    fact_id: Optional[str] = None   # UUID (set by storage)
    created_at: Optional[str] = None # ISO timestamp (set by storage)
```

#### Interface

```python
from agent.memory import (
    LongTermMemoryStore,
    StubLongTermMemoryStore,
    MemoryFact,
    LongTermMemoryWriteRequest,
    LongTermMemoryRetrievalQuery,
)

# Create instance
store = StubLongTermMemoryStore()

# Write a fact (only when authorized)
fact = MemoryFact(
    fact_type="preference",
    content={"preference": "concise responses"},
    user_id="user123",
    confidence=0.9,
    source="agent_inferred",
)
write_request = LongTermMemoryWriteRequest(
    user_id="user123",
    fact=fact,
    authorized=True,  # Set by decision_logic_node
    reason="agent_storing_interaction_outcome",
)
response = store.write_fact(write_request)
# Returns: LongTermMemoryWriteResponse with status and fact_id

# Retrieve facts (only when authorized)
query = LongTermMemoryRetrievalQuery(
    user_id="user123",
    fact_types=["preference"],  # Optional: filter by type
    limit=10,  # Default limit
    authorized=True,  # Set by decision_logic_node
)
response = store.retrieve_facts(query)
# Returns: LongTermMemoryRetrievalResponse with facts (oldest first)
```

#### Key Invariants

1. **Append-Only**: Facts are written once, never updated or deleted
2. **Advisory-Only**: Retrieval never influences decisions or routing
3. **Authorization-Based**: Only `decision_logic_node` can authorize reads/writes
4. **Non-Fatal**: Failures return Response objects, never raise exceptions
5. **User-Scoped**: Facts isolated per user_id
6. **Deterministic Ordering**: Results ordered by created_at (oldest first)

#### Testing

Long-term memory is tested with 21 comprehensive tests covering:
- Interface compatibility (Stub ↔ Qdrant swappable)
- Append-only semantics (no overwrites, full history)
- Authorization enforcement (unauthorized reads/writes rejected)
- Control-flow invariance (routing identical with memory ON/OFF)
- Failure safety (Qdrant unavailable, empty retrieval, etc.)

See `tests/unit/test_long_term_memory.py` for full test suite.

## Architecture

Memory is decoupled via abstract interfaces:

**Short-Term Memory** (MemoryController):
- **MemoryController**: Abstract base class for session-scoped context
- **StubMemoryController**: In-memory for testing
- **SQLiteShortTermMemoryStore**: Durable session context (Phase 3.1)

**Long-Term Memory** (LongTermMemoryStore):
- **LongTermMemoryStore**: Abstract base class for cross-session facts
- **StubLongTermMemoryStore**: In-memory append-only list for testing
- **DisabledLongTermMemoryStore**: Disabled store for invariant testing
- **QdrantLongTermMemoryStore**: Qdrant-backed semantic search (production)

All implementations:
- Return typed Response objects (never raise exceptions)
- Respect authorization flags (decision_logic_node controls access)
- Never influence control flow or routing
- Fail gracefully (status field indicates outcome)

## Usage

### From Decision Logic

```python
# In decision_logic_node:
if should_read_memory(state):
    state["memory_read_authorized"] = True
    state["command"] = "memory_read"
    return state
```

### From Memory Read Node

```python
# memory_read_node:
if state.memory_read_authorized:
    request = MemoryReadRequest(
        conversation_id=state.conversation_id,
        key="session_context",
        authorized=True,
    )
    response = self.memory_controller.read(request)
    state["memory_read_result"] = response.data
```

### From Memory Write Node

```python
# memory_write_node:
if state.memory_write_authorized:
    request = MemoryWriteRequest(
        conversation_id=state.conversation_id,
        key="session_context",
        data={"turns": state.turn_count, "timestamp": datetime.now().isoformat()},
        authorized=True,
    )
    response = self.memory_controller.write(request)
```

## Testing

```bash
# Phase 3.1 SQLite tests
pytest tests/unit/test_sqlite_memory.py -v

# Memory controller interface tests
pytest tests/unit/test_memory_controller.py -v

# Integration tests
pytest tests/integration/test_memory_integration.py -v
```

## Invariants

✅ **Control flow identical with memory ON/OFF**
✅ **Authorization-based access (decision_logic_node only)**
✅ **Non-fatal failures (graceful degradation)**
✅ **No implicit dependencies**
✅ **Pure plumbing (no business logic)**
