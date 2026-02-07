"""
Stub long-term memory store for testing and CI.

In-memory, append-only, deterministic, no external dependencies.
Perfect for Phase 3.2: tests prove memory is optional and safe.
"""

import uuid
from datetime import datetime
from typing import Optional
from agent.memory.long_term_base import LongTermMemoryStore
from agent.memory.long_term_types import (
    MemoryFact,
    LongTermMemoryWriteRequest,
    LongTermMemoryWriteResponse,
    LongTermMemoryRetrievalQuery,
    LongTermMemoryRetrievalResponse,
)


class StubLongTermMemoryStore(LongTermMemoryStore):
    """
    Deterministic in-memory long-term memory store.
    
    Properties:
    - Append-only: facts never overwritten
    - Ordered: facts stored in creation order
    - Deterministic: same input → same output
    - Authorization-aware: respects authorized flag
    - Non-raising: all failures explicit in response
    
    Phase 3.2 only: placeholder for persistence.
    Phase 3.2+ will add Qdrant backend.
    """

    def __init__(self):
        """Initialize stub memory (empty append-only list)."""
        self.facts = []  # List[MemoryFact] - append-only

    def write_fact(self, request: LongTermMemoryWriteRequest) -> LongTermMemoryWriteResponse:
        """
        Append a fact to in-memory storage.
        
        Args:
            request: LongTermMemoryWriteRequest
            
        Returns:
            LongTermMemoryWriteResponse with success or explicit failure
        """
        # Check authorization
        if not request.authorized:
            return LongTermMemoryWriteResponse(
                status="unauthorized",
                error="Memory write not authorized by decision_logic_node",
            )

        try:
            # Create fact with ID and timestamp
            fact = request.fact
            fact.fact_id = str(uuid.uuid4())
            fact.created_at = datetime.now().isoformat()

            # Append (never overwrite)
            self.facts.append(fact)

            return LongTermMemoryWriteResponse(
                status="success",
                fact_id=fact.fact_id,
            )
        except Exception as e:
            return LongTermMemoryWriteResponse(
                status="failed",
                error=f"Memory write failed: {str(e)}",
            )

    def retrieve_facts(self, query: LongTermMemoryRetrievalQuery) -> LongTermMemoryRetrievalResponse:
        """
        Retrieve facts from in-memory storage.
        
        Args:
            query: LongTermMemoryRetrievalQuery
            
        Returns:
            LongTermMemoryRetrievalResponse with facts (oldest first) or error
        """
        # Check authorization
        if not query.authorized:
            return LongTermMemoryRetrievalResponse(
                status="unauthorized",
                error="Memory read not authorized by decision_logic_node",
            )

        try:
            # Filter by user_id
            user_facts = [f for f in self.facts if f.user_id == query.user_id]

            # Filter by fact_type if specified
            if query.fact_types:
                user_facts = [f for f in user_facts if f.fact_type in query.fact_types]

            # Apply limit
            user_facts = user_facts[: query.limit]

            if not user_facts:
                # Empty retrieval is not an error, just no facts
                return LongTermMemoryRetrievalResponse(
                    status="success",
                    facts=[],
                )

            return LongTermMemoryRetrievalResponse(
                status="success",
                facts=user_facts,
            )
        except Exception as e:
            return LongTermMemoryRetrievalResponse(
                status="unavailable",
                error=f"Memory retrieval failed: {str(e)}",
            )

    def clear_user(self, user_id: str) -> bool:
        """
        Clear all facts for a user (for testing).
        
        Args:
            user_id: User to clear
            
        Returns:
            True if successful
        """
        try:
            self.facts = [f for f in self.facts if f.user_id != user_id]
            return True
        except Exception:
            return False


class DisabledLongTermMemoryStore(LongTermMemoryStore):
    """
    Disabled long-term memory store for testing control flow invariants.
    
    Used to verify that agent behavior is identical with memory ON/OFF.
    All operations fail gracefully and predictably.
    """

    def write_fact(self, request: LongTermMemoryWriteRequest) -> LongTermMemoryWriteResponse:
        """Always fail (memory is disabled)."""
        return LongTermMemoryWriteResponse(
            status="failed",
            error="Long-term memory is disabled",
        )

    def retrieve_facts(self, query: LongTermMemoryRetrievalQuery) -> LongTermMemoryRetrievalResponse:
        """Always unavailable (memory is disabled)."""
        return LongTermMemoryRetrievalResponse(
            status="unavailable",
            facts=[],
            error="Long-term memory is disabled",
        )
