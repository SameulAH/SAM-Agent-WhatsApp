"""
Abstract memory controller interface.

Memory is a service, not state.
The agent depends only on this interface, not on specific implementations.
"""

from abc import ABC, abstractmethod
from agent.memory.types import MemoryReadRequest, MemoryReadResponse, MemoryWriteRequest, MemoryWriteResponse


class MemoryController(ABC):
    """
    Abstract memory boundary.
    Agent code must depend ONLY on this interface.
    
    Key properties:
    - Memory operations return Response objects, never raise exceptions
    - All failures are explicit and non-fatal
    - Memory is optional and can be disabled
    - Authorization is checked (not enforced here, but in caller)
    """

    @abstractmethod
    def read(self, request: MemoryReadRequest) -> MemoryReadResponse:
        """
        Read derived facts from memory.
        
        Args:
            request: MemoryReadRequest with conversation_id, key, authorized flag
            
        Returns:
            MemoryReadResponse with status and data (if successful) or error
            
        Never raises exceptions. All failures are returned as response status.
        """
        raise NotImplementedError

    @abstractmethod
    def write(self, request: MemoryWriteRequest) -> MemoryWriteResponse:
        """
        Write derived facts to memory.
        
        Args:
            request: MemoryWriteRequest with conversation_id, key, data, authorized flag
            
        Returns:
            MemoryWriteResponse with status and optional error
            
        Never raises exceptions. All failures are returned as response status.
        """
        raise NotImplementedError
