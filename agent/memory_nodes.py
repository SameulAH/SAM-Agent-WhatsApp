"""
Memory node implementations for the LangGraph agent.

These nodes are explicit decision points in the graph.
Memory reads and writes are routed only when authorized by decision_logic_node.

Phase 2: Short-term memory (MemoryController)
Phase 3.2: Long-term memory (LongTermMemoryStore)
"""

import logging
from typing import Dict, Any, Optional

from agent.state_schema import AgentState
from agent.memory import (
    MemoryController,
    MemoryReadRequest,
    MemoryWriteRequest,
    StubMemoryController,
    LongTermMemoryStore,
    StubLongTermMemoryStore,
    MemoryFact,
    LongTermMemoryWriteRequest,
    LongTermMemoryRetrievalQuery,
)

# Get logger for memory operations
logger = logging.getLogger(__name__)


class MemoryNodeManager:
    """
    Manages memory read/write nodes in the graph.
    
    Responsibility:
    - Execute authorized SHORT-TERM memory reads (Phase 2)
    - Execute authorized SHORT-TERM memory writes (Phase 2)
    - Execute authorized LONG-TERM memory reads (Phase 3.2)
    - Execute authorized LONG-TERM memory writes (Phase 3.2)
    - Handle failures gracefully (non-fatal)
    - Never assume memory availability
    """

    def __init__(
        self,
        memory_controller: MemoryController,
        long_term_memory_store: Optional[LongTermMemoryStore] = None,
    ):
        """
        Initialize with memory controllers.
        
        Args:
            memory_controller: MemoryController instance (short-term)
            long_term_memory_store: LongTermMemoryStore instance (optional, Phase 3.2)
        """
        self.memory_controller = memory_controller
        self.long_term_memory_store = long_term_memory_store or StubLongTermMemoryStore()

    def memory_read_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute authorized memory read.
        
        Responsibility:
        - Check authorization
        - Call MemoryController.read()
        - Handle failure gracefully
        
        Rules:
        - Only runs if memory_read_authorized is True
        - Never crashes agent on failure
        - Sets memory_available=False if unavailable
        
        Returns:
            Dict with memory_read_result and memory_available
        """
        # Safety check: should only be called if authorized
        if not state.memory_read_authorized:
            return {
                "memory_read_result": None,
                "memory_available": state.memory_available,
            }

        # Build request
        request = MemoryReadRequest(
            conversation_id=state.conversation_id,
            key="conversation_context",  # Default key for context
            authorized=True,
            reason="agent_requested",
        )

        # Execute read (never crashes)
        try:
            response = self.memory_controller.read(request)

            # Handle response
            if response.status == "success":
                return {
                    "memory_read_result": response.data,
                    "memory_available": True,
                }
            elif response.status == "unavailable":
                return {
                    "memory_read_result": None,
                    "memory_available": False,
                }
            else:  # not_found, unauthorized, etc.
                return {
                    "memory_read_result": None,
                    "memory_available": True,
                }
        except Exception as e:
            # Memory failure is non-fatal
            return {
                "memory_read_result": None,
                "memory_available": False,
            }

    def memory_write_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute authorized memory write.
        
        Responsibility:
        - Check authorization
        - Call MemoryController.write()
        - Handle failure gracefully
        
        Rules:
        - Only runs if memory_write_authorized is True
        - Never crashes agent on failure
        - Sets memory_available=False if unavailable
        
        Returns:
            Dict with memory_write_status and memory_available
        """
        # Log entry to memory_write_node
        logger.info(
            f"memory_write_node: START (conversation_id={state.conversation_id}, "
            f"trace_id={state.trace_id}, authorized={state.memory_write_authorized})"
        )
        
        # Safety check: should only be called if authorized
        if not state.memory_write_authorized:
            logger.debug(
                f"memory_write_node: Not authorized for conversation {state.conversation_id}"
            )
            return {
                "memory_write_status": None,
                "memory_available": state.memory_available,
            }

        # Build request (store derived facts only)
        request = MemoryWriteRequest(
            conversation_id=state.conversation_id,
            key="conversation_context",  # Default key
            data={
                "final_output": state.final_output,
                "interaction_timestamp": state.created_at,
            },
            authorized=True,
            reason="agent_storing_outcome",
        )

        # Execute write (never crashes)
        try:
            logger.debug(
                f"memory_write_node: Calling write() for {state.conversation_id}"
            )
            response = self.memory_controller.write(request)

            if response.status == "success":
                logger.info(
                    f"memory_write_node: SUCCESS for {state.conversation_id}"
                )
                return {
                    "memory_write_status": "success",
                    "memory_available": True,
                }
            elif response.status == "failed":
                logger.warning(
                    f"memory_write_node: FAILED for {state.conversation_id}: {response.error}"
                )
                return {
                    "memory_write_status": "failed",
                    "memory_available": True,
                }
            else:  # unauthorized, etc.
                logger.warning(
                    f"memory_write_node: {response.status.upper()} for {state.conversation_id}"
                )
                return {
                    "memory_write_status": response.status,
                    "memory_available": True,
                }
        except Exception as e:
            # Memory failure is non-fatal
            logger.error(
                f"memory_write_node: EXCEPTION for {state.conversation_id}: {str(e)}"
            )
            return {
                "memory_write_status": "failed",
                "memory_available": False,
            }
    # ─────────────────────────────────────────────────────
    # LONG-TERM MEMORY NODES (Phase 3.2)
    # ─────────────────────────────────────────────────────

    def long_term_memory_read_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute authorized long-term memory retrieval.
        
        Responsibility:
        - Check authorization
        - Retrieve facts from LongTermMemoryStore
        - Handle failures gracefully (advisory-only)
        
        Rules:
        - Only runs if memory_read_authorized is True
        - Never crashes agent on failure
        - Sets long_term_memory_status="unavailable" if store unavailable
        - Always returns Dict (never raises exception)
        
        Returns:
            Dict with long_term_memory_read_result and long_term_memory_status
        """
        # Safety check: should only be called if authorized
        if not state.memory_read_authorized:
            return {
                "long_term_memory_read_result": None,
                "long_term_memory_status": state.long_term_memory_status,
            }

        # Build retrieval query
        query = LongTermMemoryRetrievalQuery(
            user_id=state.conversation_id,  # Use conversation_id as user_id
            fact_types=None,  # Retrieve all types
            limit=10,  # Default limit
            authorized=True,
        )

        # Execute retrieval (never crashes)
        try:
            response = self.long_term_memory_store.retrieve_facts(query)

            # Handle response
            if response.status == "success":
                return {
                    "long_term_memory_read_result": {
                        "facts": [
                            {
                                "fact_type": f.fact_type,
                                "content": f.content,
                                "confidence": f.confidence,
                                "created_at": f.created_at.isoformat() if hasattr(f.created_at, "isoformat") else str(f.created_at),
                            }
                            for f in response.facts
                        ]
                    },
                    "long_term_memory_status": "available",
                }
            elif response.status == "unavailable":
                return {
                    "long_term_memory_read_result": None,
                    "long_term_memory_status": "unavailable",
                }
            else:  # not_found, unauthorized, etc.
                return {
                    "long_term_memory_read_result": None,
                    "long_term_memory_status": "available",
                }
        except Exception as e:
            # Long-term memory failure is non-fatal
            return {
                "long_term_memory_read_result": None,
                "long_term_memory_status": "unavailable",
            }

    def long_term_memory_write_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute authorized long-term memory fact writing.
        
        Responsibility:
        - Check authorization
        - Write facts to LongTermMemoryStore
        - Handle failures gracefully (advisory-only)
        
        Rules:
        - Only runs if memory_write_authorized is True
        - Never crashes agent on failure
        - Sets long_term_memory_status="unavailable" if store unavailable
        - Always returns Dict (never raises exception)
        
        Returns:
            Dict with long_term_memory_write_status and long_term_memory_status
        """
        # Safety check: should only be called if authorized
        if not state.memory_write_authorized:
            return {
                "long_term_memory_write_status": None,
                "long_term_memory_status": state.long_term_memory_status,
            }

        # Build fact from conversation outcome
        if not state.final_output:
            # Nothing to write
            return {
                "long_term_memory_write_status": None,
                "long_term_memory_status": state.long_term_memory_status,
            }

        # Create a memory fact from the interaction
        fact = MemoryFact(
            fact_type="interaction_outcome",
            content=state.final_output,
            user_id=state.conversation_id,
            confidence=0.8,  # Default confidence for agent-generated facts
            source="agent_interaction",
        )

        # Build write request
        request = LongTermMemoryWriteRequest(
            user_id=state.conversation_id,
            fact=fact,
            authorized=True,
            reason="agent_storing_interaction_outcome",
        )

        # Execute write (never crashes)
        try:
            response = self.long_term_memory_store.write_fact(request)

            if response.status == "success":
                return {
                    "long_term_memory_write_status": "success",
                    "long_term_memory_status": "available",
                }
            elif response.status == "failed":
                return {
                    "long_term_memory_write_status": "failed",
                    "long_term_memory_status": "available",
                }
            else:  # unauthorized, etc.
                return {
                    "long_term_memory_write_status": response.status,
                    "long_term_memory_status": "available",
                }
        except Exception as e:
            # Long-term memory failure is non-fatal
            return {
                "long_term_memory_write_status": "failed",
                "long_term_memory_status": "unavailable",
            }