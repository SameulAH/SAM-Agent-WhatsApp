"""
Memory module exports.

Clean interface for agent to import memory components.
"""

from agent.memory.base import MemoryController
from agent.memory.stub import StubMemoryController, DisabledMemoryController
from agent.memory.sqlite import SQLiteShortTermMemoryStore
from agent.memory.types import (
    MemoryReadRequest,
    MemoryReadResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
    MemoryReadStatus,
    MemoryWriteStatus,
)
from agent.memory.long_term_base import LongTermMemoryStore
from agent.memory.long_term_stub import StubLongTermMemoryStore, DisabledLongTermMemoryStore
from agent.memory.long_term_qdrant import QdrantLongTermMemoryStore
from agent.memory.long_term_types import (
    MemoryFact,
    LongTermMemoryWriteRequest,
    LongTermMemoryWriteResponse,
    LongTermMemoryRetrievalQuery,
    LongTermMemoryRetrievalResponse,
    LongTermMemoryWriteStatus,
    LongTermMemoryReadStatus,
)

__all__ = [
    # Short-term memory (Phase 3.1)
    "MemoryController",
    "StubMemoryController",
    "DisabledMemoryController",
    "SQLiteShortTermMemoryStore",
    "MemoryReadRequest",
    "MemoryReadResponse",
    "MemoryWriteRequest",
    "MemoryWriteResponse",
    "MemoryReadStatus",
    "MemoryWriteStatus",
    # Long-term memory (Phase 3.2)
    "LongTermMemoryStore",
    "StubLongTermMemoryStore",
    "DisabledLongTermMemoryStore",
    "QdrantLongTermMemoryStore",
    "MemoryFact",
    "LongTermMemoryWriteRequest",
    "LongTermMemoryWriteResponse",
    "LongTermMemoryRetrievalQuery",
    "LongTermMemoryRetrievalResponse",
    "LongTermMemoryWriteStatus",
    "LongTermMemoryReadStatus",
]
