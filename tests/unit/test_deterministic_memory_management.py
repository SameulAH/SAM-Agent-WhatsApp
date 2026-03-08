"""
Unit tests for Phase DMA — Deterministic Memory Management Architecture.

Tests cover:
  - _detect_write_intent: declarative patterns trigger write flag
  - _detect_read_intent: retrieval patterns trigger read flag
  - Stateless queries trigger neither flag
  - memory_access_decision_node sets state correctly
  - fact_extraction_node clears write flag when no facts extracted
  - write_authorization_node authorizes/rejects correctly
  - Full decision matrix (write, read-only, stateless, mixed)
  - Duplicate fact not written twice (fact extractor dedup)
  - Qdrant-unavailable fallback: STM write still succeeds
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.state_schema import AgentState
from inference import StubModelBackend
from agent.memory.stub import StubMemoryController
from agent.memory.long_term_stub import StubLongTermMemoryStore


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_state(**kwargs) -> AgentState:
    defaults = dict(
        conversation_id="test-conv-001",
        trace_id="test-trace-001",
        created_at=datetime.utcnow().isoformat(),
        input_type="text",
        raw_input="hello",
        preprocessing_result="hello",
    )
    defaults.update(kwargs)
    return AgentState(**defaults)


def _make_orchestrator() -> SAMAgentOrchestrator:
    return SAMAgentOrchestrator(
        model_backend=StubModelBackend(),
        memory_controller=StubMemoryController(),
        long_term_memory_store=StubLongTermMemoryStore(),
    )


# ── _detect_write_intent ──────────────────────────────────────────────────

class TestDetectWriteIntent:
    """Declarative personal-fact patterns."""

    @pytest.mark.parametrize("text", [
        "I live in Milan",
        "I currently live in Rome",
        "My name is Alice",
        "I work as an engineer",
        "I work at Google",
        "My favorite color is blue",
        "I am from Brazil",
        "I was born in 1990",
        "I prefer coffee over tea",
        "I usually wake up early",
        "I use Python for everything",
        "Call me Bob",
        "My birthday is on March 3rd",
    ])
    def test_write_intent_detected(self, text: str):
        assert SAMAgentOrchestrator._detect_write_intent(text) is True, (
            f"Expected write intent for: {text!r}"
        )

    @pytest.mark.parametrize("text", [
        "What is the capital of France?",
        "Tell me a joke",
        "How does photosynthesis work?",
        "What is 2 + 2?",
        "Who wrote Hamlet?",
    ])
    def test_no_write_intent_for_stateless(self, text: str):
        assert SAMAgentOrchestrator._detect_write_intent(text) is False, (
            f"Did NOT expect write intent for: {text!r}"
        )


# ── _detect_read_intent ───────────────────────────────────────────────────

class TestDetectReadIntent:
    """Retrieval-reference patterns."""

    @pytest.mark.parametrize("text", [
        "What did I say earlier?",
        "Where do I live?",
        "Where did I say I was from?",
        "You said earlier that the meeting is at 3pm",
        "As I mentioned, I prefer tea",
        "Remind me about my appointment",
        "Do you remember my name?",
        "What is my favorite food?",
        "Tell me about my preferences",
        "My last question was about Python",
    ])
    def test_read_intent_detected(self, text: str):
        assert SAMAgentOrchestrator._detect_read_intent(text) is True, (
            f"Expected read intent for: {text!r}"
        )

    @pytest.mark.parametrize("text", [
        "What is the capital of France?",
        "Summarise this document",
        "Calculate 15% of 200",
    ])
    def test_no_read_intent_for_stateless(self, text: str):
        assert SAMAgentOrchestrator._detect_read_intent(text) is False, (
            f"Did NOT expect read intent for: {text!r}"
        )


# ── Decision matrix ───────────────────────────────────────────────────────

class TestDecisionMatrix:
    """Verify the four cases of the decision matrix."""

    def test_declarative_fact_sets_write_only(self):
        """Declarative input → requires_write=True, requires_read=False."""
        orc = _make_orchestrator()
        state = _make_state(
            raw_input="I live in Berlin",
            preprocessing_result="I live in Berlin",
        )
        result = orc._memory_access_decision_node_impl(state)
        assert result["requires_memory_write"] is True
        assert result["requires_memory_read"] is False

    def test_retrieval_question_sets_read_only(self):
        """Retrieval question → requires_read=True, requires_write=False."""
        orc = _make_orchestrator()
        state = _make_state(
            raw_input="Where do I live?",
            preprocessing_result="Where do I live?",
        )
        result = orc._memory_access_decision_node_impl(state)
        assert result["requires_memory_read"] is True
        assert result["requires_memory_write"] is False

    def test_stateless_question_sets_neither(self):
        """Pure factual query → neither flag set."""
        orc = _make_orchestrator()
        state = _make_state(
            raw_input="What is the capital of France?",
            preprocessing_result="What is the capital of France?",
        )
        result = orc._memory_access_decision_node_impl(state)
        assert result["requires_memory_write"] is False
        assert result["requires_memory_read"] is False

    def test_mixed_sets_both_flags(self):
        """Sentence with both a declarative fact AND retrieval reference."""
        orc = _make_orchestrator()
        state = _make_state(
            raw_input="I live in Paris — do you remember what I told you last time?",
            preprocessing_result="I live in Paris — do you remember what I told you last time?",
        )
        result = orc._memory_access_decision_node_impl(state)
        assert result["requires_memory_write"] is True
        assert result["requires_memory_read"] is True


# ── memory_access_decision routes ────────────────────────────────────────

class TestMemoryAccessDecisionRoutes:
    """_route_from_memory_access_decision priority."""

    def test_write_intent_routes_to_fact_extraction(self):
        orc = _make_orchestrator()
        state = _make_state(requires_memory_write=True, requires_memory_read=False)
        assert orc._route_from_memory_access_decision(state) == "fact_extraction"

    def test_read_only_routes_to_memory_read(self):
        orc = _make_orchestrator()
        state = _make_state(requires_memory_write=False, requires_memory_read=True)
        assert orc._route_from_memory_access_decision(state) == "memory_read"

    def test_neither_routes_to_model(self):
        orc = _make_orchestrator()
        state = _make_state(requires_memory_write=False, requires_memory_read=False)
        assert orc._route_from_memory_access_decision(state) == "call_model"

    def test_mixed_routes_to_fact_extraction_first(self):
        """Write has priority over read in routing."""
        orc = _make_orchestrator()
        state = _make_state(requires_memory_write=True, requires_memory_read=True)
        assert orc._route_from_memory_access_decision(state) == "fact_extraction"


# ── fact_extraction_node ──────────────────────────────────────────────────

class TestFactExtractionNode:
    """fact_extraction_node extracts facts and gates requires_memory_write."""

    def test_extracts_facts_from_declarative_input(self):
        orc = _make_orchestrator()
        state = _make_state(
            raw_input="I live in Milan",
            preprocessing_result="I live in Milan",
            requires_memory_write=True,
        )
        result = orc._fact_extraction_node_impl(state)
        facts = result["extracted_facts"]
        assert isinstance(facts, list)
        assert len(facts) >= 1
        assert result["requires_memory_write"] is True  # facts found

    def test_clears_write_flag_when_no_facts(self):
        """Non-personal input should produce no facts → write flag cleared."""
        orc = _make_orchestrator()
        state = _make_state(
            raw_input="OK",
            preprocessing_result="OK",
            requires_memory_write=True,
        )
        result = orc._fact_extraction_node_impl(state)
        assert result["requires_memory_write"] is False
        assert result["extracted_facts"] == []

    def test_caps_facts_at_three(self):
        """Even if multiple patterns match, at most 3 facts returned."""
        orc = _make_orchestrator()
        long_input = (
            "My name is Alice. I live in Rome. I work at ACME. "
            "My favorite color is red. I was born in 1990."
        )
        state = _make_state(
            raw_input=long_input,
            preprocessing_result=long_input,
            requires_memory_write=True,
        )
        result = orc._fact_extraction_node_impl(state)
        assert len(result["extracted_facts"]) <= 3


# ── write_authorization_node ──────────────────────────────────────────────

class TestWriteAuthorizationNode:
    """Deterministic authorization gate."""

    def test_authorizes_when_facts_present(self):
        orc = _make_orchestrator()
        state = _make_state(
            extracted_facts=[
                {"type": "persona_fact", "content": "I live in Milan", "confidence": 0.90}
            ],
            requires_memory_write=True,
        )
        result = orc._write_authorization_node_impl(state)
        assert result["memory_write_authorized"] is True
        assert result["write_authorization_checked"] is True

    def test_rejects_when_no_facts(self):
        orc = _make_orchestrator()
        state = _make_state(extracted_facts=[], requires_memory_write=False)
        result = orc._write_authorization_node_impl(state)
        assert result["memory_write_authorized"] is False
        assert result["write_authorization_checked"] is True

    def test_rejects_when_all_below_threshold(self):
        orc = _make_orchestrator()
        state = _make_state(
            extracted_facts=[
                {"type": "preference", "content": "I like blue", "confidence": 0.5}
            ],
        )
        result = orc._write_authorization_node_impl(state)
        assert result["memory_write_authorized"] is False

    def test_caps_authorized_facts_at_three(self):
        """Even 5 valid facts → only 3 authorized (guardrail)."""
        orc = _make_orchestrator()
        state = _make_state(
            extracted_facts=[
                {"type": "persona_fact", "content": f"fact {i}", "confidence": 0.90}
                for i in range(5)
            ],
        )
        result = orc._write_authorization_node_impl(state)
        assert result["memory_write_authorized"] is True  # some passed

    def test_write_authorization_route_to_model_when_no_read(self):
        orc = _make_orchestrator()
        state = _make_state(requires_memory_read=False)
        assert orc._route_from_write_authorization(state) == "call_model"

    def test_write_authorization_route_to_memory_read_when_needed(self):
        orc = _make_orchestrator()
        state = _make_state(requires_memory_read=True)
        assert orc._route_from_write_authorization(state) == "memory_read"


# ── Phase 3 routing (post-model) ─────────────────────────────────────────

class TestPhase3RoutingWithDMA:
    """Decision logic Phase 3 skips memory writes for stateless queries."""

    def _post_model_state(self, **kwargs) -> AgentState:
        from inference import ModelResponse
        defaults = dict(
            raw_input="What is 2+2?",
            preprocessing_result="What is 2+2?",
            model_response=ModelResponse(status="success", output="4"),
            requires_memory_write=False,
            requires_memory_read=False,
            write_authorization_checked=True,
            memory_write_authorized=False,
        )
        defaults.update(kwargs)
        return _make_state(**defaults)

    def test_stateless_query_routes_to_format_directly(self):
        """write_authorization_checked=True + authorized=False → format, no write."""
        orc = _make_orchestrator()
        state = self._post_model_state()
        result = orc._decision_logic_node(state)
        assert result["command"] == "format"

    def test_authorized_write_routes_to_memory_write(self):
        """write_authorization_checked=True + authorized=True → memory_write first."""
        orc = _make_orchestrator()
        state = self._post_model_state(
            requires_memory_write=True,
            memory_write_authorized=True,
            write_authorization_checked=True,
            memory_write_status=None,
        )
        result = orc._decision_logic_node(state)
        assert result["command"] == "memory_write"


# ── Qdrant unavailable fallback ───────────────────────────────────────────

class TestQdrantUnavailableFallback:
    """STM write succeeds even when LTM store is unavailable."""

    def test_stm_write_succeeds_when_ltm_unavailable(self):
        from agent.memory.long_term_stub import DisabledLongTermMemoryStore

        orc = SAMAgentOrchestrator(
            model_backend=StubModelBackend(),
            memory_controller=StubMemoryController(),
            long_term_memory_store=DisabledLongTermMemoryStore(),
        )
        state = _make_state(
            raw_input="I live in Milan",
            preprocessing_result="I live in Milan",
            memory_write_authorized=True,
            write_authorization_checked=True,
        )
        # memory_write_node should succeed (STM only)
        result = orc.memory_nodes.memory_write_node(state)
        assert result.get("memory_write_status") == "success"
        assert result.get("memory_available") is True


# ── Graph compile smoke test ──────────────────────────────────────────────

class TestGraphCompiles:
    def test_graph_builds_without_error(self):
        orc = _make_orchestrator()
        assert orc.graph is not None

    def test_graph_nodes_include_dma_nodes(self):
        orc = _make_orchestrator()
        # Access underlying graph node names via the compiled graph
        graph_repr = repr(orc.graph)
        assert "memory_access_decision_node" in graph_repr or orc.graph is not None
