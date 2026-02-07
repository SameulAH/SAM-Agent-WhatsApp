"""
Abstract long-term memory service interface.

Long-term memory is advisory-only: it informs responses but never influences
control flow, decisions, or authority.

Key properties:
- Append-only (never update/delete)
- Authorization-based (decision_logic_node controls access)
- Non-fatal failures (never blocks execution)
- Advisory (responses can reference facts, but decisions never depend on them)
"""

from abc import ABC, abstractmethod
from agent.memory.long_term_types import (
    MemoryFact,
    LongTermMemoryWriteRequest,
    LongTermMemoryWriteResponse,
    LongTermMemoryRetrievalQuery,
    LongTermMemoryRetrievalResponse,
)


class LongTermMemoryStore(ABC):
    """
    Abstract long-term memory service.
    
    Memory is a boundary service: agent depends only on this interface.
    
    Key properties:
    - Append-only: facts are written once, never updated
    - Advisory: facts inform responses, never influence decisions
    - Authorized: only decision_logic_node can authorize access
    - Non-fatal: failures never block execution
    """

    @abstractmethod
    def write_fact(self, request: LongTermMemoryWriteRequest) -> LongTermMemoryWriteResponse:
        """
        Append a fact to long-term memory.
        
        Args:
            request: LongTermMemoryWriteRequest with user_id, fact, authorized flag
            
        Returns:
            LongTermMemoryWriteResponse with status and fact_id (if successful)
            
        Never raises exceptions. All failures are returned as response status.
        
        Rules:
        - Append-only: never overwrites existing facts
        - Authorized only: request.authorized must be True
        - Non-fatal: failures never prevent execution
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_facts(self, query: LongTermMemoryRetrievalQuery) -> LongTermMemoryRetrievalResponse:
        """
        Retrieve facts from long-term memory.
        
        Args:
            query: LongTermMemoryRetrievalQuery with user_id, fact_types, limit
            
        Returns:
            LongTermMemoryRetrievalResponse with facts (oldest first) or error
            
        Never raises exceptions. All failures are returned as response status.
        
        Rules:
        - Advisory-only: facts never influence routing or decisions
        - Authorized only: query.authorized must be True
        - Ordered: facts returned in creation order (oldest first)
        - Complete: all matching facts returned (up to limit)
        """
        raise NotImplementedError
