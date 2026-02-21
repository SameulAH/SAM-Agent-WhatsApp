"""
LangGraph-based agent orchestrator.

Implements the exact graph structure defined in design/langgraph_skeleton.md.

The graph is fully deterministic and stateless except for the AgentState.
All routing decisions are made by decision_logic_node.
Task nodes execute without branching.

Phase MCP: Added tool_execution_node and execute_tool routing.
- decision_logic_node detects tool_call in model metadata
- tool_execution_node executes the tool, stores result, clears model_response
- model_call_node is re-invoked with tool_context injected
- Max 1 tool call per turn enforced via guardrails
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime
from uuid import uuid4
import json
import re
import time

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph

from inference import ModelBackend, ModelRequest, ModelResponse, StubModelBackend
from agent.state_schema import AgentState
from agent.memory import MemoryController, StubMemoryController, LongTermMemoryStore, StubLongTermMemoryStore
from agent.memory_nodes import MemoryNodeManager
from agent.tracing import Tracer, TraceMetadata, NoOpTracer
from agent.mcp.guardrails import MCPGuardrails, GuardrailViolation
from agent.prompting.prompt_builder import SYSTEM_PROMPT, build_prompt


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

    # ── Output guardrails ─────────────────────────────────────────────────────
    MAX_OUTPUT_CHARS: int = 800          # Hard ceiling on final response length
    MAX_OUTPUT_SENTENCES: int = 5        # Soft ceiling (applied before char limit)

    # ── Freshness keywords that trigger forced web_search ────────────────────
    # Phase 5: only keep keywords that are truly time-sensitive.
    # Removed: "now", "currently", "update", "updates", "live" — too broad,
    # fired on conversational queries and caused an extra ~20s tool round-trip.
    _FRESHNESS_KEYWORDS: frozenset = frozenset({
        "today", "latest", "recent", "breaking", "news",
        "right now", "this week", "this month", "current events",
    })

    # ── Reference patterns that signal conversational back-references ─────────
    # When detected, STM + LTM context is retrieved before the model call so
    # the agent can ground its response in prior conversation history.
    _REFERENCE_PATTERNS: frozenset = frozenset({
        "that", "it", "this", "one day", "as we discussed",
        "like before", "earlier", "previously", "we talked",
        "you said", "remember", "last time", "you mentioned",
        "what i said", "what you said", "as i said", "as you said",
    })

    @classmethod
    def _requires_memory_retrieval(cls, text: str) -> bool:
        """Return True if text contains a conversational back-reference signal."""
        return any(pat in text for pat in cls._REFERENCE_PATTERNS)

    def __init__(
        self,
        model_backend: Optional[ModelBackend] = None,
        memory_controller: Optional[MemoryController] = None,
        long_term_memory_store: Optional[LongTermMemoryStore] = None,
        tracer: Optional[Tracer] = None,
    ):
        """
        Initialize orchestrator with a model backend and memory controllers.
        
        Args:
            model_backend: ModelBackend instance (StubModelBackend by default)
            memory_controller: MemoryController instance (StubMemoryController by default)
            long_term_memory_store: LongTermMemoryStore instance (StubLongTermMemoryStore by default)
            tracer: Tracer instance for observability (NoOpTracer by default)
        """
        self.model_backend = model_backend or StubModelBackend()
        self.memory_controller = memory_controller or StubMemoryController()
        self.long_term_memory_store = long_term_memory_store or StubLongTermMemoryStore()
        self.tracer = tracer or NoOpTracer()
        self.memory_nodes = MemoryNodeManager(self.memory_controller, self.long_term_memory_store)
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph graph with exact structure from skeleton.
        
        Phase 2: Includes optional memory read/write nodes.
        
        Returns:
            Compiled LangGraph graph
        """
        graph = StateGraph(AgentState)

        # Add all nodes (wrapped with tracing via _wrap_node_call)
        graph.add_node("router_node", self._router_node)
        graph.add_node("state_init_node", self._state_init_node)
        graph.add_node("decision_logic_node", self._decision_logic_node)
        graph.add_node("task_preprocessing_node", self._task_preprocessing_node)
        graph.add_node("memory_read_node", self._memory_read_node_wrapper)  # Phase 2
        graph.add_node("model_call_node", self._model_call_node)
        # result_handling_node removed (single-model-pass: model → format directly)
        graph.add_node("memory_write_node", self._memory_write_node_wrapper)  # Phase 2
        graph.add_node("long_term_memory_read_node", self._long_term_memory_read_node_wrapper)  # Phase 3.2
        graph.add_node("long_term_memory_write_node", self._long_term_memory_write_node_wrapper)  # Phase 3.2
        graph.add_node("tool_execution_node", self._tool_execution_node_wrapper)  # Phase MCP
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
                "long_term_memory_read": "long_term_memory_read_node",  # Phase 3.2
                "call_model": "model_call_node",
                "execute_tool": "tool_execution_node",  # Phase MCP
                "memory_write": "memory_write_node",
                "long_term_memory_write": "long_term_memory_write_node",  # Phase 3.2
                "format": "format_response_node",
            }
        )

        graph.add_edge("task_preprocessing_node", "decision_logic_node")
        graph.add_edge("memory_read_node", "decision_logic_node")
        graph.add_edge("long_term_memory_read_node", "decision_logic_node")  # Phase 3.2
        graph.add_edge("tool_execution_node", "decision_logic_node")  # Phase MCP

        # Single-model-pass: model_call_node exits directly to format (no tool loop-back)
        # Failures are handled by error_router_node via format_response_node
        graph.add_conditional_edges(
            "model_call_node",
            self._route_from_model_call,
            {
                "success": "format_response_node",
                "failure": "error_router_node",
            }
        )
        graph.add_edge("memory_write_node", "decision_logic_node")
        graph.add_edge("long_term_memory_write_node", "decision_logic_node")  # Phase 3.2
        
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
        elif state.command == "long_term_memory_read":  # Phase 3.2
            return "long_term_memory_read"
        elif state.command == "call_model":
            return "call_model"
        elif state.command == "execute_tool":  # Phase MCP
            return "execute_tool"
        elif state.command == "memory_write":
            return "memory_write"
        elif state.command == "long_term_memory_write":  # Phase 3.2
            return "long_term_memory_write"
        else:
            return "format"

    def _route_from_model_call(self, state: AgentState) -> str:
        """Route based on model response status.

        Single-model-pass architecture: model never loops back to tools.
          • "success" → format_response_node
          • "failure" → error_router_node
        """
        if state.model_response and state.model_response.status == "success":
            return "success"
        return "failure"

    def _create_trace_metadata(self, state: AgentState) -> TraceMetadata:
        """Extract trace metadata from state."""
        return TraceMetadata(
            trace_id=state.trace_id,
            conversation_id=state.conversation_id,
            user_id=None,  # Not yet in schema, reserved for future
        )

    def _wrap_node_execution(self, node_name: str, node_fn, state: AgentState) -> Dict[str, Any]:
        """
        Wrap a node execution with tracing.
        
        Spans at node entry/exit for observability.
        Tracing failures are silent and non-blocking.
        """
        trace_metadata = self._create_trace_metadata(state)
        span = None
        start_time = time.time()

        # Node entry span
        try:
            span = self.tracer.start_span(
                name=node_name,
                metadata={"node_name": node_name},
                trace_metadata=trace_metadata,
            )
        except Exception:
            # Tracing failure is non-fatal
            pass

        try:
            # Execute node
            result = node_fn(state)
            duration_s = time.time() - start_time
            duration_ms = duration_s * 1000

            # ── Phase 1: mandatory latency log ────────────────────────────
            logger.info(f"[LATENCY] {node_name} took {duration_s:.3f}s")

            # Emit node_latency tracer event (non-blocking)
            try:
                self.tracer.record_event(
                    name="node_latency",
                    metadata={
                        "node_name": node_name,
                        "duration_ms": duration_ms,
                    },
                    trace_metadata=trace_metadata,
                )
            except Exception:
                pass

            # Node exit span
            try:
                self.tracer.end_span(
                    span=span,
                    status="success",
                    metadata={"duration_ms": duration_ms},
                )
            except Exception:
                # Tracing failure is non-fatal
                pass

            return result
        except Exception as e:
            # Exception handling (node failure)
            duration_s = time.time() - start_time
            duration_ms = duration_s * 1000

            # ── Phase 1: mandatory latency log (failure path) ─────────────
            logger.info(f"[LATENCY] {node_name} FAILED after {duration_s:.3f}s ({type(e).__name__})")

            # Record failure span
            try:
                self.tracer.end_span(
                    span=span,
                    status="error",
                    metadata={"duration_ms": duration_ms, "error": type(e).__name__},
                )
            except Exception:
                # Tracing failure is non-fatal
                pass

            # Re-raise the exception (node failure propagates)
            raise

    # ─────────────────────────────────────────────────────
    # TOOL CALL PARSER (static, backend-agnostic)
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _try_parse_tool_call(output: str) -> Optional[Dict[str, Any]]:
        """
        Secondary parser for tool calls that don't use the [TOOL_CALL] marker.

        The primary parser lives in OllamaModelBackend._extract_tool_call and handles
        the [TOOL_CALL]{...} format.  This method is called by _model_call_node_impl
        as a fallback when metadata["tool_call"] is not already set, covering models
        (e.g. qwen2.5:7b) that emit tool syntax in other formats:

          • {"name": "web_search", "arguments": {"query": "…"}}  — raw JSON
          • web_search{"query": "…"}                              — loose syntax
          • web_search({"query": "…"})                           — function-call style

        Uses brace-counting for correctness (mirrors primary parser).

        Returns:
            Normalised {"name": "...", "arguments": {...}} dict, or None.
        """
        if not output or not output.strip():
            return None

        # ── Strategy 1: raw structured JSON {"name": "...", "arguments": {...}} ──
        m = re.search(
            r'\{\s*"name"\s*:\s*"([\w_]+)"\s*,\s*"arguments"\s*:\s*(\{)',
            output,
        )
        if m:
            tool_name = m.group(1)
            brace_pos = m.start(2)
            depth = 0
            brace_end = -1
            for i, ch in enumerate(output[brace_pos:], brace_pos):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end = i + 1
                        break
            if brace_end != -1:
                try:
                    args = json.loads(output[brace_pos:brace_end])
                    return {"name": tool_name, "arguments": args}
                except json.JSONDecodeError:
                    pass

        # ── Strategy 2: loose syntax — known_tool{...} or known_tool({...}) ──
        for tool_name in ("web_search",):
            pattern = re.compile(
                rf"\b{re.escape(tool_name)}\s*\(?\s*(\{{)",
                re.IGNORECASE,
            )
            m = pattern.search(output)
            if m:
                brace_pos = m.start(1)
                depth = 0
                brace_end = -1
                for i, ch in enumerate(output[brace_pos:], brace_pos):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            brace_end = i + 1
                            break
                if brace_end != -1:
                    json_str = output[brace_pos:brace_end]
                    try:
                        args = json.loads(json_str)
                        if "arguments" in args:
                            args = args["arguments"]
                        return {"name": tool_name, "arguments": args}
                    except json.JSONDecodeError:
                        qm = re.search(r'"query"\s*:\s*"([^"]*)"', json_str)
                        if qm:
                            return {
                                "name": tool_name,
                                "arguments": {"query": qm.group(1)},
                            }

        return None

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
        return self._wrap_node_execution("router_node", self._router_node_impl, state)

    def _router_node_impl(self, state: AgentState) -> Dict[str, Any]:
        """Router node implementation (unwrapped)."""
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
        return self._wrap_node_execution("state_init_node", self._state_init_node_impl, state)

    def _state_init_node_impl(self, state: AgentState) -> Dict[str, Any]:
        """State init node implementation (unwrapped)."""
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

        Single-model-pass architecture:
        - Tool routing is decided BEFORE the model is called (Phase 2b below)
        - model_call_node NEVER routes back to tools; it exits directly to format
        - decision_logic_node is the SOLE authority for execute_tool routing

        Command sequence:
        1. After state_init: preprocess
        2. After preprocessing:
           a. STM read (if authorized) → LTM read → tool check → call_model
           b. If freshness keyword (and no tool run yet) → execute_tool
        3. After tool_execution: call_model (tool_executed=True, model_response=None)
        4. After model_call: memory_write → long_term_memory_write → format
           (model_call_node now routes directly to format_response_node; this
            branch is only reached via the memory cycle path below)

        Rules:
        - Must NOT execute tasks
        - Must NOT call model
        - Must NOT mutate state directly (only set command + forced_tool metadata)
        - Tool call authority: tool_call_count < MAX_TOOL_CALLS_PER_TURN (max 1)
        
        Note: Tracing NOT wrapped here (decision logic is not observed externally).
        """
        trace_metadata = self._create_trace_metadata(state)

        # ── Phase 1: Pre-processing ──────────────────────────────────────────
        if state.preprocessing_result is None:
            return {"command": "preprocess"}

        # ── Phase 2: Before model call ───────────────────────────────────────
        if state.model_response is None:
            # Post-tool re-call: skip memory reads (already done in turn 1)
            if state.tool_executed:
                return {"command": "call_model"}

            # ── Reference-signal-gated STM read ──────────────────────────────
            # Detect conversational back-references BEFORE tool routing so the
            # model is grounded in prior context when it generates the answer.
            # Triggered once per turn: guarded by memory_read_authorized flag.
            user_text_mem = (state.preprocessing_result or state.raw_input or "").lower()
            if (
                not state.memory_read_authorized
                and state.memory_read_result is None
                and self._requires_memory_retrieval(user_text_mem)
            ):
                try:
                    self.tracer.record_event(
                        name="memory_retrieval_triggered",
                        metadata={
                            "trigger": "reference_signal",
                            "query_preview": user_text_mem[:80],
                        },
                        trace_metadata=trace_metadata,
                    )
                except Exception:
                    pass
                return {
                    "command": "memory_read",
                    "memory_read_authorized": True,
                }

            # LTM read: attempt once per turn if not yet done and store is available
            if (
                state.long_term_memory_read_result is None
                and state.long_term_memory_status == "available"
            ):
                return {"command": "long_term_memory_read"}

            # ── Phase 2b: Pre-model tool routing ─────────────────────────────
            # This is the ONLY place execute_tool is emitted.  Check freshness
            # keywords in the user query before the first model call.
            if state.tool_call_count < MCPGuardrails.MAX_TOOL_CALLS_PER_TURN:
                query_lower = (state.preprocessing_result or state.raw_input or "").lower()
                has_freshness = any(kw in query_lower for kw in self._FRESHNESS_KEYWORDS)

                if has_freshness:
                    forced_query = state.preprocessing_result or state.raw_input
                    try:
                        self.tracer.record_event(
                            name="tool_intent_detected",
                            metadata={
                                "query_preview": query_lower[:80],
                                "trigger": "freshness_keyword_pre_model",
                            },
                            trace_metadata=trace_metadata,
                        )
                        self.tracer.record_event(
                            name="forced_tool_call",
                            metadata={
                                "tool_name": "web_search",
                                "reason": "freshness_keyword_no_tool_call",
                            },
                            trace_metadata=trace_metadata,
                        )
                    except Exception:
                        pass
                    _forced_tc = {
                        "name": "web_search",
                        "arguments": {"query": forced_query},
                    }
                    return {
                        "command": "execute_tool",
                        "tool_call": _forced_tc,
                    }

            return {"command": "call_model"}

        # ── Phase 3: After model call — memory/format routing ────────────────
        # model_call_node now exits directly to format_response_node via a hard
        # graph edge, so this branch is only reached when the memory-write cycle
        # loops back here (memory_write_node → decision_logic_node).

        # Case 1: Memory write authorized but not yet executed
        if state.memory_write_authorized and state.memory_write_status is None:
            return {"command": "memory_write"}

        # Case 2: Memory write not yet authorized (authorize once)
        if not state.memory_write_authorized:
            return {
                "command": "memory_write",
                "memory_write_authorized": True,
            }

        # Case 3: STM write done — now check long-term memory write
        if state.long_term_memory_write_status is None:
            return {"command": "long_term_memory_write"}

        # Case 4: All memory done → format
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
        return self._wrap_node_execution("task_preprocessing_node", self._task_preprocessing_node_impl, state)

    def _task_preprocessing_node_impl(self, state: AgentState) -> Dict[str, Any]:
        """Preprocessing node implementation (unwrapped)."""
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
        return self._wrap_node_execution("model_call_node", self._model_call_node_impl, state)

    def _model_call_node_impl(self, state: AgentState) -> Dict[str, Any]:
        """Model call node implementation (unwrapped)."""
        # ── Derive memory context from state memory fields ────────────────────
        # Combine LTM (retrieved facts) and STM (recent context) into one string
        memory_context: Optional[str] = None
        memory_context_parts = []

        if state.long_term_memory_read_result:
            facts = state.long_term_memory_read_result.get("facts", [])
            # Phase 8: cap at 3 facts — large memory injection slows model pass
            facts = facts[:3]
            if facts:
                lines = [
                    f"- {f.get('content', f) if isinstance(f.get('content'), str) else str(f.get('content', ''))}"
                    for f in facts
                    if f.get("content") or (isinstance(f, dict) and f)
                ]
                if lines:
                    memory_context_parts.append("Remembered facts:\n" + "\n".join(lines))

        if state.memory_read_result:
            # Build a human-readable prior-turn summary from STM data
            stm_parts = []
            prior_input = state.memory_read_result.get("raw_input")
            prior_output = (
                state.memory_read_result.get("final_output")
                or state.memory_read_result.get("conversation_context")
            )
            if prior_input:
                stm_parts.append(f"User previously said: {prior_input}")
            if prior_output:
                stm_parts.append(f"You previously answered: {prior_output}")
            if stm_parts:
                memory_context_parts.append("Prior conversation:\n" + "\n".join(stm_parts))
            elif prior_output:
                # fallback: plain context string
                memory_context_parts.append(f"Prior conversation:\n{prior_output}")

        if memory_context_parts:
            memory_context = "\n\n".join(memory_context_parts)

        # ── Build structured prompt via PromptBuilder ─────────────────────────
        user_input = state.preprocessing_result or state.raw_input
        prompt = build_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_input=user_input,
            memory_context=memory_context,
            tool_context=state.tool_context,
        )

        trace_metadata = self._create_trace_metadata(state)

        # Emit prompt_built trace event
        try:
            self.tracer.record_event(
                name="prompt_built",
                metadata={
                    "has_memory_context": memory_context is not None,
                    "has_tool_context": state.tool_context is not None,
                    "prompt_length": len(prompt),
                },
                trace_metadata=trace_metadata,
            )
        except Exception:
            pass

        # Emit memory_injected trace event if memory was used
        if memory_context:
            try:
                self.tracer.record_event(
                    name="memory_injected",
                    metadata={"memory_context_length": len(memory_context)},
                    trace_metadata=trace_metadata,
                )
            except Exception:
                pass

        # ── Build ModelRequest (context already embedded in prompt) ───────────
        # Phase 3: 45s cap — tool context adds tokens so model needs more headroom.
        # Simple queries still return in <5s; freshness queries with context ~10-25s.
        request = ModelRequest(
            task="respond",
            prompt=prompt,
            context=None,   # tool_context is now embedded in prompt via build_prompt
            timeout_s=45,
            trace_id=state.trace_id,
        )

        start_time = time.time()

        # Model call span (metadata only, no prompts/outputs)
        try:
            self.tracer.record_event(
                name="model_call_attempted",
                metadata={"model_requested": True},
                trace_metadata=trace_metadata,
            )
        except Exception:
            # Tracing failure is non-fatal
            pass

        # Call model backend
        model_response = self.model_backend.generate(request)

        # Record model call result (metadata only)
        duration_ms = (time.time() - start_time) * 1000
        try:
            success_status = "success" if model_response.status == "success" else "failure"
            self.tracer.record_event(
                name="model_call_completed",
                metadata={
                    "status": success_status,
                    "duration_ms": duration_ms,
                    "error_type": model_response.error_type if model_response.status != "success" else None,
                },
                trace_metadata=trace_metadata,
            )
        except Exception:
            # Tracing failure is non-fatal
            pass

        # ── Tool call detection removed (single-model-pass architecture) ──────
        # model_call_node no longer detects or routes to tools.
        # decision_logic_node handles all tool routing BEFORE the model call.

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
        return self._wrap_node_execution("result_handling_node", self._result_handling_node_impl, state)

    def _result_handling_node_impl(self, state: AgentState) -> Dict[str, Any]:
        """Result handling node implementation (unwrapped)."""
        if not state.model_response:
            raise ValueError("model_response is None in result_handling_node")

        if state.model_response.status == "success" and state.model_response.output:
            final_output = state.model_response.output
        else:
            # This shouldn't happen if routing is correct, but be explicit
            final_output = None

        # ── Phase 5: Output conciseness enforcement ───────────────────────────
        if final_output:
            original_len = len(final_output)
            truncated = False

            # Soft limit: cap at MAX_OUTPUT_SENTENCES (split on sentence endings)
            import re as _re
            sentences = _re.split(r"(?<=[.!?])\s+", final_output.strip())
            if len(sentences) > self.MAX_OUTPUT_SENTENCES:
                final_output = " ".join(sentences[: self.MAX_OUTPUT_SENTENCES]).strip()
                truncated = True

            # Hard limit: absolute character ceiling
            if len(final_output) > self.MAX_OUTPUT_CHARS:
                final_output = final_output[: self.MAX_OUTPUT_CHARS] + "..."
                truncated = True

            if truncated:
                trace_metadata = self._create_trace_metadata(state)
                try:
                    self.tracer.record_event(
                        name="response_truncated",
                        metadata={
                            "original_length": original_len,
                            "final_length": len(final_output),
                            "max_chars": self.MAX_OUTPUT_CHARS,
                        },
                        trace_metadata=trace_metadata,
                    )
                except Exception:
                    pass

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
        return self._wrap_node_execution("error_router_node", self._error_router_node_impl, state)

    def _error_router_node_impl(self, state: AgentState) -> Dict[str, Any]:
        """Error router node implementation (unwrapped)."""
        if not state.model_response:
            error_type = "unknown"
            fallback_output = "Sorry, I couldn't get a response. Please try again."
        else:
            error_type = state.model_response.error_type or "unknown"
            if error_type == "timeout":
                fallback_output = "⏳ The response took too long. Please try again — the model may be warming up."
            elif error_type == "connection":
                fallback_output = "🔌 Couldn't connect to the AI model. Please try again shortly."
            else:
                fallback_output = "Sorry, something went wrong. Please try again."

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
        return self._wrap_node_execution("format_response_node", self._format_response_node_impl, state)

    def _format_response_node_impl(self, state: AgentState) -> Dict[str, Any]:
        """Format response node implementation (unwrapped)."""
        # ── Derive final_output from model_response if not already set ────────
        # model_call_node now routes directly here; final_output is not set by
        # a result_handling_node anymore, so we resolve it here.
        final_output = state.final_output
        if final_output is None and state.model_response:
            if state.model_response.status == "success" and state.model_response.output:
                final_output = state.model_response.output

        # Safety fallback: model emitted a tool call again (output stripped to "") but
        # we already have tool_context — synthesize a minimal answer from context.
        if not final_output and state.tool_context:
            import re as _re_fb
            # Extract first substantive sentence from tool_context
            sentences = _re_fb.split(r"(?<=[.!?])\s+", state.tool_context.strip())
            snippet = " ".join(sentences[:2]).strip()[:300]
            if snippet:
                final_output = f"Based on recent search results: {snippet}"

        # ── Output conciseness enforcement (migrated from result_handling_node) ─
        if final_output:
            original_len = len(final_output)
            truncated = False

            import re as _re
            sentences = _re.split(r"(?<=[.!?])\s+", final_output.strip())
            if len(sentences) > self.MAX_OUTPUT_SENTENCES:
                final_output = " ".join(sentences[: self.MAX_OUTPUT_SENTENCES]).strip()
                truncated = True

            if len(final_output) > self.MAX_OUTPUT_CHARS:
                final_output = final_output[: self.MAX_OUTPUT_CHARS] + "..."
                truncated = True

            if truncated:
                trace_metadata = self._create_trace_metadata(state)
                try:
                    self.tracer.record_event(
                        name="response_truncated",
                        metadata={
                            "original_length": original_len,
                            "final_length": len(final_output),
                            "max_chars": self.MAX_OUTPUT_CHARS,
                        },
                        trace_metadata=trace_metadata,
                    )
                except Exception:
                    pass

        # Write final_output into state so invoke() can read it via result.get("final_output")
        return {
            "final_output": final_output,
        }

    # ─────────────────────────────────────────────────────
    # TOOL EXECUTION NODE (Phase MCP)
    # ─────────────────────────────────────────────────────

    def _tool_execution_node_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """
        Wrap tool_execution_node with tracing.

        Phase MCP: Executes tool requested by model, stores result,
        clears model_response for re-call, formats tool_context.

        Guardrails enforced:
        - Max 1 tool call per turn
        - Max 5 results, snippet ≤ 300 chars, budget ≤ 1500 chars
        - Timeout fallback (no auto-retry)
        - No memory mutation
        """
        trace_metadata = self._create_trace_metadata(state)

        try:
            self.tracer.record_event(
                name="tool_execution_started",
                metadata={
                    "tool_call_count": state.tool_call_count,
                    "tool_executed_before": state.tool_executed,
                },
                trace_metadata=trace_metadata,
            )
        except Exception:
            pass

        return self._wrap_node_execution(
            "tool_execution_node", self._tool_execution_node_impl, state
        )

    def _tool_execution_node_impl(self, state: AgentState) -> Dict[str, Any]:
        """
        Tool execution node implementation (unwrapped).

        Responsibilities:
        1. Extract tool_call from model_response.metadata
        2. Enforce guardrail: tool_call_count < MAX
        3. Execute tool via ToolRegistry
        4. Format results as tool_context (injection-safe)
        5. Clear model_response for second model call
        6. Set tool_executed = True, increment tool_call_count

        Must NOT:
        - Write to any memory_* field
        - Change memory_write_authorized
        - Set command (decision_logic_node authority)
        - Override trace_id
        """
        trace_metadata = self._create_trace_metadata(state)

        # Extract tool call: prefer state.tool_call (set by _model_call_node_impl),
        # fall back to model_response.metadata for the legacy decision_logic path.
        tool_call = state.tool_call or (
            state.model_response.metadata.get("tool_call")
            if state.model_response and state.model_response.metadata
            else None
        )

        if not tool_call:
            # No tool call to execute — gracefully skip
            try:
                self.tracer.record_event(
                    name="tool_execution_failed",
                    metadata={"reason": "no_tool_call_in_metadata"},
                    trace_metadata=trace_metadata,
                )
            except Exception:
                pass
            return {
                "tool_call": None,
                "command": None,
                "tool_executed": True,
                "tool_call_count": state.tool_call_count + 1,
                "model_response": None,  # Clear so decision routes to call_model
            }

        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        # Guardrail check (belt + suspenders — decision_logic already checks)
        try:
            MCPGuardrails.check_tool_call_limit(state.tool_call_count)
        except GuardrailViolation as gv:
            try:
                self.tracer.record_event(
                    name="tool_execution_failed",
                    metadata={"reason": "guardrail_violation", "rule": gv.rule},
                    trace_metadata=trace_metadata,
                )
            except Exception:
                pass
            # Skip tool, continue flow without tool context
            return {
                "tool_call": None,
                "command": None,
                "tool_executed": True,
                "tool_call_count": state.tool_call_count + 1,
                "tool_result": {"error": str(gv)},
                "model_response": None,
            }

        # Execute via ToolRegistry
        tool_result_dict: Dict[str, Any] = {}
        tool_context: Optional[str] = None

        try:
            from agent.intelligence.tools import get_tool_registry  # noqa: PLC0415
            from agent.tools.web_search_tool import WebSearchTool    # noqa: PLC0415

            registry = get_tool_registry()

            # Register WebSearchTool with tracer if not yet registered
            if registry.get("web_search") is None or not isinstance(
                registry.get("web_search"), WebSearchTool
            ):
                registry.register(
                    WebSearchTool(
                        tracer=self.tracer,
                        trace_metadata=trace_metadata,
                    )
                )

            tool_result = registry.execute(tool_name, tool_args)
            tool_result_dict = tool_result.model_dump()

            if tool_result.success and tool_result.data.get("results"):
                # Format bounded tool context for model injection
                # Reconstruct BrowserBaseResult-like objects for formatter
                from agent.mcp.external_client import BrowserBaseResult  # noqa: PLC0415

                raw_results = tool_result.data["results"]
                result_objects = [
                    BrowserBaseResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("snippet", ""),
                    )
                    for r in raw_results
                ]
                tool_context = MCPGuardrails.format_tool_context(result_objects)

            # Fallback: if tool returned no usable results, give model a notice
            # so it answers from training knowledge rather than repeating the tool call
            if not tool_context:
                tool_context = (
                    "Note: The web search did not return usable results. "
                    "Please answer based on your training knowledge and state that "
                    "you could not retrieve live results."
                )

            try:
                self.tracer.record_event(
                    name="tool_execution_completed",
                    metadata={
                        "tool_name": tool_name,
                        "success": tool_result.success,
                        "result_count": len(tool_result.data.get("results", [])),
                        "execution_time_ms": tool_result.execution_time_ms,
                    },
                    trace_metadata=trace_metadata,
                )
            except Exception:
                pass

        except Exception as e:
            try:
                self.tracer.record_event(
                    name="tool_execution_failed",
                    metadata={"reason": "exception", "error": type(e).__name__},
                    trace_metadata=trace_metadata,
                )
            except Exception:
                pass
            tool_result_dict = {"error": str(e)}
            tool_context = (
                "Note: The web search encountered an error and could not retrieve results. "
                "Please answer based on your training knowledge."
            )

        return {
            # Tool execution state (Phase MCP)
            "tool_executed": True,
            "tool_call_count": state.tool_call_count + 1,
            "tool_result": tool_result_dict,
            "tool_context": tool_context,
            # Clear pending tool_call so it isn't re-executed on the next pass
            "tool_call": None,
            # Reset command so second _route_from_model_call doesn't short-circuit again
            "command": None,
            # Clear model_response so decision_logic routes back to call_model
            "model_response": None,
            # Clear final_output so result_handling_node sets it fresh
            "final_output": None,
            # DO NOT touch memory_* fields
        }

    # ─────────────────────────────────────────────────────
    # MEMORY NODE WRAPPERS (WITH TRACING)
    # ─────────────────────────────────────────────────────

    def _memory_read_node_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """Wrap memory_read_node with tracing."""
        trace_metadata = self._create_trace_metadata(state)
        
        # Record memory read attempt
        try:
            self.tracer.record_event(
                name="memory_read_attempted",
                metadata={"authorized": state.memory_read_authorized},
                trace_metadata=trace_metadata,
            )
        except Exception:
            # Tracing failure is non-fatal
            pass

        return self._wrap_node_execution("memory_read_node", self.memory_nodes.memory_read_node, state)

    def _memory_write_node_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """Wrap memory_write_node with tracing."""
        trace_metadata = self._create_trace_metadata(state)
        
        # Record memory write attempt
        try:
            self.tracer.record_event(
                name="memory_write_attempted",
                metadata={"authorized": state.memory_write_authorized},
                trace_metadata=trace_metadata,
            )
        except Exception:
            # Tracing failure is non-fatal
            pass

        return self._wrap_node_execution("memory_write_node", self.memory_nodes.memory_write_node, state)

    def _long_term_memory_read_node_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """Wrap long_term_memory_read_node with tracing (Phase 3.2)."""
        trace_metadata = self._create_trace_metadata(state)
        
        # Record long-term memory read attempt
        try:
            self.tracer.record_event(
                name="long_term_memory_read_attempted",
                metadata={"requested": state.long_term_memory_requested},
                trace_metadata=trace_metadata,
            )
        except Exception:
            # Tracing failure is non-fatal
            pass

        return self._wrap_node_execution("long_term_memory_read_node", self.memory_nodes.long_term_memory_read_node, state)

    def _long_term_memory_write_node_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """Wrap long_term_memory_write_node with tracing (Phase 3.2)."""
        trace_metadata = self._create_trace_metadata(state)
        
        # Record long-term memory write attempt
        try:
            self.tracer.record_event(
                name="long_term_memory_write_attempted",
                metadata={"requested": state.long_term_memory_requested},
                trace_metadata=trace_metadata,
            )
        except Exception:
            # Tracing failure is non-fatal
            pass

        return self._wrap_node_execution("long_term_memory_write_node", self.memory_nodes.long_term_memory_write_node, state)

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
            input_type="text",
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
            conversation_id=conversation_id or str(uuid4()),
            trace_id=trace_id or str(uuid4()),
            created_at="",
            input_type="text",
            raw_input=raw_input,
        )

        # Run graph
        result = self.graph.invoke(initial_state)

        # Extract response from format_response_node output
        if isinstance(result, dict):
            # LangGraph returns the final state as a plain dict — map to standard shape
            return {
                "conversation_id": result.get("conversation_id", ""),
                "trace_id": result.get("trace_id", ""),
                "status": "success" if result.get("error_type") is None else "error",
                "output": result.get("final_output"),
                "error_type": result.get("error_type"),
                "metadata": result.get("model_metadata") or {},
            }
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
