"""
Memory node implementations for the LangGraph agent.

These nodes are explicit decision points in the graph.
Memory reads and writes are routed only when authorized by decision_logic_node.
"""

from typing import Dict, Any

from agent.state_schema import AgentState
from agent.memory import MemoryController, MemoryReadRequest, MemoryWriteRequest, StubMemoryController


class MemoryNodeManager:
    """
    Manages memory read/write nodes in the graph.
    
    Responsibility:
    - Execute authorized memory reads
    - Execute authorized memory writes
    - Handle failures gracefully (non-fatal)
    - Never assume memory availability
    """

    def __init__(self, memory_controller: MemoryController):
        """
        Initialize with a memory controller.
        
        Args:
            memory_controller: MemoryController instance
        """
        self.memory_controller = memory_controller

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
        # Safety check: should only be called if authorized
        if not state.memory_write_authorized:
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
            response = self.memory_controller.write(request)

            if response.status == "success":
                return {
                    "memory_write_status": "success",
                    "memory_available": True,
                }
            elif response.status == "failed":
                return {
                    "memory_write_status": "failed",
                    "memory_available": True,
                }
            else:  # unauthorized, etc.
                return {
                    "memory_write_status": response.status,
                    "memory_available": True,
                }
        except Exception as e:
            # Memory failure is non-fatal
            return {
                "memory_write_status": "failed",
                "memory_available": False,
            }
