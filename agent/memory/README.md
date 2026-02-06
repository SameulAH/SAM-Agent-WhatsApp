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

### Phase 3.2: Long-Term Memory (PLANNED)

Qdrant-backed semantic knowledge base with embeddings, personalization, and cross-session learning.

## Architecture

Memory is decoupled via `MemoryController` abstract interface:

- **MemoryController**: Abstract base class
- **StubMemoryController**: In-memory for testing (Phase 2)
- **DisabledMemoryController**: Disabled for testing invariants (Phase 2)
- **SQLiteShortTermMemoryStore**: Durable session context (Phase 3.1)
- Future: **QdrantLongTermMemoryStore** (Phase 3.2)

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
