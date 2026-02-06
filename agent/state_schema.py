"""
Agent state schema and types.

The AgentState is the single source of truth for all agent execution context.
It is mutated only by specific nodes and never by task execution.

Phase 2 Addition: Memory-related fields store pointers and flags, never knowledge.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

from inference import ModelResponse


@dataclass
class AgentState:
    """
    Complete state schema for the SAM agent.
    
    Invariants:
    - conversation_id and trace_id are immutable once set
    - preprocessing_result, model_response, final_output are written only by specific nodes
    - command is the only control flow signal (decision_logic_node writes it)
    - error_type is set only by error_router_node
    - memory fields store pointers and metadata, never actual knowledge
    """

    # Identity
    conversation_id: str
    trace_id: str
    created_at: str

    # Input / Routing
    input_type: str  # text | audio | image
    raw_input: str

    # Processing
    preprocessing_result: Optional[str] = None

    # Model
    model_response: Optional[ModelResponse] = None
    model_metadata: Optional[Dict[str, Any]] = None

    # Output
    final_output: Optional[str] = None
    error_type: Optional[str] = None

    # Control
    command: Optional[str] = None  # preprocess | call_model | success | failure
    
    # Memory: Phase 2 Addition
    # These fields store metadata and pointers, never actual memory content
    memory_available: bool = True                    # Is memory service available?
    memory_read_authorized: bool = False             # Does this execution want to read memory?
    memory_write_authorized: bool = False            # Does this execution want to write memory?
    memory_read_result: Optional[Dict[str, Any]] = None  # Data from memory read (if authorized)
    memory_write_status: Optional[str] = None        # Status of memory write operation

    def __post_init__(self):
        """Validate state schema."""
        if not self.conversation_id:
            raise ValueError("conversation_id must not be empty")
        if not self.trace_id:
            raise ValueError("trace_id must not be empty")
        if self.input_type not in ("text", "audio", "image"):
            raise ValueError(f"input_type must be text, audio, or image; got {self.input_type}")
        if not self.raw_input:
            raise ValueError("raw_input must not be empty")
