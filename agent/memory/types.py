"""
Memory boundary layer types and contracts.

Defines the request/response types for memory operations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal

MemoryReadStatus = Literal["success", "not_found", "unavailable", "unauthorized"]
MemoryWriteStatus = Literal["success", "failed", "unauthorized"]


@dataclass
class MemoryReadRequest:
    """Request to read derived facts from memory."""
    
    conversation_id: str              # Which conversation
    key: str                          # What to read (e.g., "user_preferences")
    authorized: bool = False          # Must be True (set by decision_logic_node)
    reason: Optional[str] = None      # Why (for auditing)


@dataclass
class MemoryReadResponse:
    """Response from memory read operation."""
    
    status: MemoryReadStatus
    data: Optional[Dict[str, Any]] = None  # Retrieved facts (if status=="success")
    error: Optional[str] = None            # Error description if status != "success"


@dataclass
class MemoryWriteRequest:
    """Request to write derived facts to memory."""
    
    conversation_id: str              # Which conversation
    key: str                          # What to write (e.g., "user_preferences")
    data: Dict[str, Any]              # Facts to store
    authorized: bool = False          # Must be True (set by decision_logic_node)
    reason: Optional[str] = None      # Why (for auditing)


@dataclass
class MemoryWriteResponse:
    """Response from memory write operation."""
    
    status: MemoryWriteStatus
    error: Optional[str] = None  # Error description if status != "success"
