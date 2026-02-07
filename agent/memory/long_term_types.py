"""
Long-term memory service types and contracts.

Defines request/response types for long-term memory operations.
Frozen by design/long_term_memory_invariants.md.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime

LongTermMemoryWriteStatus = Literal["success", "failed", "unauthorized"]
LongTermMemoryReadStatus = Literal["success", "not_found", "unavailable", "unauthorized"]


@dataclass
class MemoryFact:
    """A stable fact in long-term memory."""
    
    fact_type: str                  # "preference", "pattern", "summary", etc.
    content: Dict[str, Any]         # The actual fact (JSON-serializable)
    user_id: str                    # Whose fact is this?
    confidence: float = 1.0         # 0.0-1.0 (optional, default 1.0)
    source: str = "conversation"    # Where did this fact come from?
    fact_id: Optional[str] = None   # UUID (set by storage)
    created_at: Optional[str] = None # ISO timestamp (set by storage)


@dataclass
class LongTermMemoryWriteRequest:
    """Request to write a fact to long-term memory."""
    
    user_id: str                    # Whose fact is this?
    fact: MemoryFact                # The fact to append
    authorized: bool = False        # Must be True (set by decision_logic_node)
    reason: Optional[str] = None    # Why are we writing this fact?


@dataclass
class LongTermMemoryWriteResponse:
    """Response from long-term memory write operation."""
    
    status: LongTermMemoryWriteStatus
    fact_id: Optional[str] = None   # UUID of written fact (if status=="success")
    error: Optional[str] = None     # Error description if status != "success"


@dataclass
class LongTermMemoryRetrievalQuery:
    """Request to retrieve facts from long-term memory."""
    
    user_id: str                    # Whose facts to retrieve?
    fact_types: Optional[List[str]] = None  # Filter by type (default: all)
    limit: int = 10                 # Max facts to return
    authorized: bool = False        # Must be True (set by decision_logic_node)


@dataclass
class LongTermMemoryRetrievalResponse:
    """Response from long-term memory retrieval operation."""
    
    status: LongTermMemoryReadStatus
    facts: Optional[List[MemoryFact]] = None  # Retrieved facts (oldest first)
    error: Optional[str] = None     # Error description if status != "success"
