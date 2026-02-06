"""
Stub memory controller for testing and CI.

In-memory, deterministic, no external dependencies.
Perfect for Phase 2: tests prove memory is optional and safe.
"""

from agent.memory.base import MemoryController
from agent.memory.types import MemoryReadRequest, MemoryReadResponse, MemoryWriteRequest, MemoryWriteResponse


class StubMemoryController(MemoryController):
    """
    Deterministic fake memory for testing and CI.
    
    Properties:
    - Stores data in-memory (dict)
    - Deterministic: same input → same output
    - Never fails silently: all failures explicit
    - Authorization-aware: rejects unauthorized requests
    
    Phase 2 only: This is a placeholder.
    Phase 3+ will add persistence (SQLite, Qdrant, etc.)
    """

    def __init__(self):
        """Initialize stub memory (empty dict per conversation)."""
        self.storage = {}  # {conversation_id: {key: data}}

    def read(self, request: MemoryReadRequest) -> MemoryReadResponse:
        """
        Read derived facts from in-memory storage.
        
        Args:
            request: MemoryReadRequest
            
        Returns:
            MemoryReadResponse with success or explicit failure
        """
        # Check authorization
        if not request.authorized:
            return MemoryReadResponse(
                status="unauthorized",
                error="Memory read not authorized by decision_logic_node",
            )

        # Get conversation storage
        conv_data = self.storage.get(request.conversation_id)
        if conv_data is None:
            return MemoryReadResponse(
                status="not_found",
                error=f"No memory for conversation {request.conversation_id}",
            )

        # Get key
        data = conv_data.get(request.key)
        if data is None:
            return MemoryReadResponse(
                status="not_found",
                error=f"Key {request.key} not found in memory",
            )

        # Success
        return MemoryReadResponse(
            status="success",
            data=data,
        )

    def write(self, request: MemoryWriteRequest) -> MemoryWriteResponse:
        """
        Write derived facts to in-memory storage.
        
        Args:
            request: MemoryWriteRequest
            
        Returns:
            MemoryWriteResponse with success or explicit failure
        """
        # Check authorization
        if not request.authorized:
            return MemoryWriteResponse(
                status="unauthorized",
                error="Memory write not authorized by decision_logic_node",
            )

        # Create conversation storage if needed
        if request.conversation_id not in self.storage:
            self.storage[request.conversation_id] = {}

        # Write data
        try:
            self.storage[request.conversation_id][request.key] = request.data
            return MemoryWriteResponse(status="success")
        except Exception as e:
            return MemoryWriteResponse(
                status="failed",
                error=f"Memory write failed: {str(e)}",
            )


class DisabledMemoryController(MemoryController):
    """
    Memory disabled controller for testing control flow invariants.
    
    Used to verify that agent behavior is identical with memory ON/OFF.
    All operations fail gracefully and predictably.
    """

    def read(self, request: MemoryReadRequest) -> MemoryReadResponse:
        """Always return 'unavailable' (memory is disabled)."""
        return MemoryReadResponse(
            status="unavailable",
            error="Memory is disabled",
        )

    def write(self, request: MemoryWriteRequest) -> MemoryWriteResponse:
        """Always return 'failed' (memory is disabled)."""
        return MemoryWriteResponse(
            status="failed",
            error="Memory is disabled",
        )
