"""
LangGraph-based agent orchestrator.

Implements the exact graph structure defined in design/langgraph_skeleton.md.

The graph is fully deterministic and stateless except for the AgentState.
All routing decisions are made by decision_logic_node.
Task nodes execute without branching.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from uuid import uuid4

from langgraph.graph import StateGraph

from inference import ModelBackend, ModelRequest, StubModelBackend
from agent.state_schema import AgentState
from agent.memory import MemoryController, StubMemoryController
from agent.memory_nodes import MemoryNodeManager


class SAMAgentOrchestrator:
    """
    LangGraph-based agent orchestrator.
    
    Materializes the control flow defined in design/langgraph_skeleton.md.
    
    Hard rules:
    - Task nodes execute without decision logic
    - decision_logic_node makes all routing decisions
    - Model is called only through the boundary
    - Memory is not accessed yet
    - All failures are explicit and typed
    """

    def __init__(self, model_backend: Optional[ModelBackend] = None, memory_controller: Optional[MemoryController] = None):
        """
        Initialize orchestrator with a model backend and memory controller.
        
        Args:
            model_backend: ModelBackend instance (StubModelBackend by default)
            memory_controller: MemoryController instance (StubMemoryController by default)
        """
        self.model_backend = model_backend or StubModelBackend()
        self.memory_controller = memory_controller or StubMemoryController()
        self.memory_nodes = MemoryNodeManager(self.memory_controller)
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledGraph:
        """
        Build the LangGraph graph with exact structure from skeleton.
        
        Phase 2: Includes optional memory read/write nodes.
        
        Returns:
            Compiled LangGraph graph
        """
        graph = StateGraph(AgentState)

        # Add all nodes
        graph.add_node("router_node", self._router_node)
        graph.add_node("state_init_node", self._state_init_node)
        graph.add_node("decision_logic_node", self._decision_logic_node)
        graph.add_node("task_preprocessing_node", self._task_preprocessing_node)
        graph.add_node("memory_read_node", self.memory_nodes.memory_read_node)  # Phase 2
        graph.add_node("model_call_node", self._model_call_node)
        graph.add_node("result_handling_node", self._result_handling_node)
        graph.add_node("memory_write_node", self.memory_nodes.memory_write_node)  # Phase 2
        graph.add_node("error_router_node", self._error_router_node)
        graph.add_node("format_response_node", self._format_response_node)

        # Entry point
        graph.set_entry_point("router_node")

        # Success path: __start__ → router → state_init → decision → preprocess → memory_read → decision → model → result → memory_write → decision → format → __end__
        graph.add_edge("router_node", "state_init_node")
        graph.add_edge("state_init_node", "decision_logic_node")
        
        # decision_logic_node branches based on command
        graph.add_conditional_edges(
            "decision_logic_node",
            self._route_from_decision,
            {
                "preprocess": "task_preprocessing_node",
                "memory_read": "memory_read_node",
                "call_model": "model_call_node",
                "memory_write": "memory_write_node",
                "format": "format_response_node",
            }
        )

        graph.add_edge("task_preprocessing_node", "decision_logic_node")
        graph.add_edge("memory_read_node", "decision_logic_node")
        
        # model_call_node branches on success/failure
        graph.add_conditional_edges(
            "model_call_node",
            self._route_from_model_call,
            {
                "success": "result_handling_node",
                "failure": "error_router_node",
            }
        )

        graph.add_edge("result_handling_node", "decision_logic_node")
        graph.add_edge("memory_write_node", "decision_logic_node")
        
        # error_router routes to format
        graph.add_edge("error_router_node", "format_response_node")
        
        # format_response is exit
        graph.set_finish_point("format_response_node")

        return graph.compile()

    def _route_from_decision(self, state: AgentState) -> str:
        """Route based on command from decision_logic_node."""
        if state.command == "preprocess":
            return "preprocess"
        elif state.command == "memory_read":
            return "memory_read"
        elif state.command == "call_model":
            return "call_model"
        elif state.command == "memory_write":
            return "memory_write"
        else:
            return "format"

    def _route_from_model_call(self, state: AgentState) -> str:
        """Route based on model response status."""
        if state.model_response and state.model_response.status == "success":
            return "success"
        else:
            return "failure"

    # ─────────────────────────────────────────────────────
    # NODE IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────

    def _router_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Classify input modality and annotate state.
        
        Responsibility: Detect input_type from raw_input
        
        Rules:
        - Must NOT call model
        - Must NOT read/write memory
        - Must NOT decide next step (that's decision_logic_node's job)
        """
        # Classify input type based on raw_input
        # For skeleton: assume text by default
        # TODO: Implement audio/image detection in future
        input_type = "text"

        return {
            "input_type": input_type,
        }

    def _state_init_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Initialize agent state with identity invariants.
        
        Responsibility: Set up conversation_id, trace_id, timestamps
        
        Rules:
        - Must NOT branch logic
        - Must NOT call model
        - Must NOT access memory
        """
        # Generate IDs if not present
        conversation_id = state.conversation_id or str(uuid4())
        trace_id = state.trace_id or str(uuid4())
        created_at = datetime.utcnow().isoformat()

        return {
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "created_at": created_at,
            "command": None,
        }

    def _decision_logic_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Make control flow decisions.
        
        Responsibility: Emit next command based on state
        
        Command sequence (with optional memory):
        1. After state_init: preprocess
        2. After preprocessing: memory_read (if needed) OR call_model
        3. After memory_read: call_model
        4. After model_call: memory_write (if needed) OR format
        5. After memory_write: format
        
        Rules:
        - Must NOT execute tasks
        - Must NOT call model
        - Must NOT mutate state directly (only set command)
        """
        # Decision logic: pure control flow
        if state.preprocessing_result is None:
            # After state_init, before preprocessing
            return {"command": "preprocess"}
        elif state.model_response is None:
            # After preprocessing, before model call
            # Check if memory read is needed (placeholder for future logic)
            if state.memory_read_authorized and state.memory_read_result is None:
                # Memory read was requested but not yet executed
                return {"command": "memory_read"}
            else:
                # No memory read needed, proceed to model
                return {"command": "call_model"}
        else:
            # After model call, check if memory write is needed
            # (placeholder for future logic)
            if state.memory_write_authorized:
                # Memory write was requested
                return {"command": "memory_write"}
            else:
                # No memory write needed, format response
                return {"command": "format"}

    def _task_preprocessing_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute modality-specific preprocessing.
        
        Responsibility: Transform raw_input based on input_type
        
        Rules:
        - Must NOT decide control flow
        - Must NOT call model
        - Must NOT write memory
        - Must NOT handle errors globally
        """
        if state.input_type == "text":
            # Text preprocessing: simple normalization
            preprocessing_result = state.raw_input.strip()
        elif state.input_type == "audio":
            # Audio preprocessing: placeholder for STT
            # TODO: Implement STT call in future
            preprocessing_result = state.raw_input
        elif state.input_type == "image":
            # Image preprocessing: placeholder
            # TODO: Implement image preprocessing in future
            preprocessing_result = state.raw_input
        else:
            preprocessing_result = state.raw_input

        return {
            "preprocessing_result": preprocessing_result,
        }

    def _model_call_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Call the model backend.
        
        Responsibility: Invoke ModelBackend.generate(), return structured response
        
        Rules:
        - Must NOT retry
        - Must NOT decide routing (transitions are explicit)
        - Must NOT mutate state beyond model_response
        """
        # Build request
        request = ModelRequest(
            task="respond",
            prompt=state.preprocessing_result or state.raw_input,
            context=None,
            timeout_s=30,
            trace_id=state.trace_id,
        )

        # Call model backend
        model_response = self.model_backend.generate(request)

        return {
            "model_response": model_response,
            "model_metadata": model_response.metadata,
        }

    def _result_handling_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate and handle model output.
        
        Responsibility: Validate model_response, update state with output
        
        Rules:
        - Must NOT call model again
        - Must NOT access memory implicitly
        - Model outputs are data, not control signals
        """
        if not state.model_response:
            raise ValueError("model_response is None in result_handling_node")

        if state.model_response.status == "success" and state.model_response.output:
            final_output = state.model_response.output
        else:
            # This shouldn't happen if routing is correct, but be explicit
            final_output = None

        return {
            "final_output": final_output,
        }

    def _error_router_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Classify failure and produce fallback state.
        
        Responsibility: Type the error, set fallback values
        
        Rules:
        - Must NOT retry silently
        - Must NOT mutate unrelated state
        """
        if not state.model_response:
            error_type = "unknown"
            fallback_output = "[Error: No model response]"
        else:
            error_type = state.model_response.error_type or "unknown"
            fallback_output = f"[Error: {error_type}]"

        return {
            "error_type": error_type,
            "final_output": fallback_output,
        }

    def _format_response_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Convert final state to response payload.
        
        Responsibility: Serialize state to response object
        
        Rules:
        - Must NOT decide logic
        - Must NOT call model
        - Must NOT write memory
        """
        response = {
            "conversation_id": state.conversation_id,
            "trace_id": state.trace_id,
            "status": "success" if state.error_type is None else "error",
            "output": state.final_output,
            "error_type": state.error_type,
            "metadata": state.model_metadata or {},
        }

        return response

    # ─────────────────────────────────────────────────────
    # HELPERS (for testing and internal use)
    # ─────────────────────────────────────────────────────

    def _create_initial_state(
        self,
        raw_input: str,
        conversation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> AgentState:
        """
        Create initial state for graph invocation.
        
        Args:
            raw_input: Raw input to process
            conversation_id: Optional conversation ID (generated if not provided)
            trace_id: Optional trace ID (generated if not provided)
            
        Returns:
            Initial AgentState
        """
        return AgentState(
            conversation_id=conversation_id or str(uuid4()),
            trace_id=trace_id or str(uuid4()),
            created_at="",
            input_type="",
            raw_input=raw_input,
        )

    # ─────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────

    async def invoke(self, raw_input: str, conversation_id: Optional[str] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a single agent invocation.
        
        Args:
            raw_input: Raw input to process
            conversation_id: Optional conversation ID (generated if not provided)
            trace_id: Optional trace ID (generated if not provided)
            
        Returns:
            Response dict with conversation_id, trace_id, status, output, etc.
        """
        initial_state = AgentState(
            conversation_id=conversation_id or "",
            trace_id=trace_id or "",
            created_at="",
            input_type="",
            raw_input=raw_input,
        )

        # Run graph
        result = self.graph.invoke(initial_state)

        # Extract response from format_response_node output
        if isinstance(result, dict):
            return result
        elif isinstance(result, AgentState):
            # If result is state, manually format
            return {
                "conversation_id": result.conversation_id,
                "trace_id": result.trace_id,
                "status": "success" if result.error_type is None else "error",
                "output": result.final_output,
                "error_type": result.error_type,
                "metadata": result.model_metadata or {},
            }
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")
