"""
tests/integration/test_personal_fact_storage.py

Integration tests for personal assistant fact storage.

Verifies:
✔ User saying "My birthday is X" → fact extracted with type "persona_fact"
✔ User saying "I prefer Y" → fact extracted with type "preference"
✔ User saying "I work at Z" → fact extracted with type "persona_fact"
✔ Max 3 facts per turn respected
✔ Facts are written to LongTermMemoryStore (not just stub)
✔ Empty / meta-conversational input produces no facts
✔ Confidence threshold ≥ 0.7 enforced
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from agent.intelligence.fact_extraction import (
    ExtractedFact,
    FactExtractionRequest,
    FactExtractionResponse,
    FactExtractor,
)
from agent.memory_nodes import MemoryNodeManager
from agent.memory.long_term_stub import StubLongTermMemoryStore
from agent.memory.stub import StubMemoryController
from agent.state_schema import AgentState


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for FactExtractor personal patterns
# ─────────────────────────────────────────────────────────────────────────────


class TestFactExtractorPersonalPatterns:
    """Unit tests for personal fact extraction patterns."""

    def setup_method(self):
        self.extractor = FactExtractor()

    def _extract(self, text: str) -> FactExtractionResponse:
        return self.extractor.extract(
            FactExtractionRequest(
                user_input=text,
                conversation_turn_id="turn_001",
                conversation_id="conv_001",
            )
        )

    def test_birthday_extracted(self):
        """Birthday statement produces persona_fact."""
        response = self._extract("My birthday is October 12.")
        assert response.extraction_attempted
        assert len(response.extracted_facts) >= 1
        fact = response.extracted_facts[0]
        assert fact.type == "persona_fact"
        assert "October 12" in fact.content
        assert fact.confidence >= 0.7

    def test_preference_extracted(self):
        """Preference statement produces preference fact."""
        response = self._extract("I prefer Python over JavaScript.")
        assert response.extraction_attempted
        assert len(response.extracted_facts) >= 1
        fact = response.extracted_facts[0]
        assert fact.type == "preference"
        assert fact.confidence >= 0.7

    def test_workplace_extracted(self):
        """Workplace statement produces persona_fact."""
        response = self._extract("I work at Google.")
        assert response.extraction_attempted
        assert len(response.extracted_facts) >= 1
        fact = response.extracted_facts[0]
        assert fact.type == "persona_fact"
        assert fact.confidence >= 0.7

    def test_location_extracted(self):
        """Location statement produces persona_fact."""
        response = self._extract("I live in New York.")
        assert len(response.extracted_facts) >= 1
        assert response.extracted_facts[0].type == "persona_fact"

    def test_goal_extracted(self):
        """Goal statement produces goal fact."""
        response = self._extract("I want to learn machine learning.")
        assert len(response.extracted_facts) >= 1
        assert response.extracted_facts[0].type == "goal"

    def test_reminder_task_extracted(self):
        """Reminder statement produces task fact."""
        response = self._extract("Remind me to call mom tomorrow.")
        assert len(response.extracted_facts) >= 1
        assert response.extracted_facts[0].type == "task"

    def test_meta_conversational_no_facts(self):
        """Generic meta-conversational text produces no facts."""
        for text in ["ok", "yes", "thanks", "sure", "hi"]:
            response = self._extract(text)
            assert len(response.extracted_facts) == 0, f"Unexpected fact for: {text!r}"

    def test_max_facts_per_turn_respected(self):
        """Never returns more than MAX_FACTS_PER_TURN facts."""
        text = (
            "My birthday is January 1. "
            "I prefer coffee. "
            "I work at Microsoft. "
            "I live in Seattle. "
            "I want to climb Everest."
        )
        response = self._extract(text)
        assert len(response.extracted_facts) <= FactExtractor.MAX_FACTS_PER_TURN

    def test_confidence_threshold_enforced(self):
        """All returned facts must meet the confidence threshold."""
        response = self._extract("I am a software engineer who likes coding.")
        for fact in response.extracted_facts:
            assert fact.confidence >= FactExtractor.MIN_CONFIDENCE

    def test_empty_input_no_facts(self):
        """Empty input produces no facts."""
        response = self._extract("")
        assert len(response.extracted_facts) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration test: MemoryNodeManager writes facts to LTM store
# ─────────────────────────────────────────────────────────────────────────────


def _make_state(raw_input: str, final_output: str = "Got it.") -> AgentState:
    return AgentState(
        conversation_id="conv_test",
        trace_id="trace_test",
        created_at="2026-01-01T00:00:00",
        input_type="text",
        raw_input=raw_input,
        final_output=final_output,
        memory_write_authorized=True,
    )


class TestPersonalFactPersistence:
    """Personal facts are extracted and written to LongTermMemoryStore."""

    def test_birthday_written_to_ltm(self):
        """Birthday fact is written to LTM store after LTM write node."""
        mock_ltm = MagicMock(spec=StubLongTermMemoryStore)
        mock_write_response = MagicMock()
        mock_write_response.status = "success"
        mock_write_response.fact_id = "fact_001"
        mock_ltm.write_fact.return_value = mock_write_response

        manager = MemoryNodeManager(
            memory_controller=StubMemoryController(),
            long_term_memory_store=mock_ltm,
        )
        state = _make_state("My birthday is October 12.")
        manager.long_term_memory_write_node(state)

        # write_fact should have been called at least once
        assert mock_ltm.write_fact.call_count >= 1

        # At least one call should have fact_type containing 'persona_fact'
        call_args_list = mock_ltm.write_fact.call_args_list
        fact_types = [
            call.args[0].fact.fact_type if call.args else call.kwargs["request"].fact.fact_type
            for call in call_args_list
        ]
        assert any("persona" in ft or "personal" in ft for ft in fact_types)

    def test_preference_written_to_ltm(self):
        """Preference fact is written to LTM store."""
        mock_ltm = MagicMock(spec=StubLongTermMemoryStore)
        mock_write_response = MagicMock()
        mock_write_response.status = "success"
        mock_ltm.write_fact.return_value = mock_write_response

        manager = MemoryNodeManager(
            memory_controller=StubMemoryController(),
            long_term_memory_store=mock_ltm,
        )
        state = _make_state("I prefer dark mode in my editor.")
        manager.long_term_memory_write_node(state)

        assert mock_ltm.write_fact.call_count >= 1

    def test_no_facts_only_outcome_written(self):
        """When no personal facts found, only interaction_outcome is written."""
        mock_ltm = MagicMock(spec=StubLongTermMemoryStore)
        mock_write_response = MagicMock()
        mock_write_response.status = "success"
        mock_ltm.write_fact.return_value = mock_write_response

        manager = MemoryNodeManager(
            memory_controller=StubMemoryController(),
            long_term_memory_store=mock_ltm,
        )
        state = _make_state("What is 2 + 2?", final_output="4.")
        manager.long_term_memory_write_node(state)

        # Only the interaction_outcome write
        assert mock_ltm.write_fact.call_count == 1
        call = mock_ltm.write_fact.call_args_list[0]
        fact = call.args[0].fact if call.args else call.kwargs["request"].fact
        assert fact.fact_type == "interaction_outcome"

    def test_ltm_write_failure_does_not_crash(self):
        """LTM store failure is non-fatal — node returns without raising."""
        mock_ltm = MagicMock(spec=StubLongTermMemoryStore)
        mock_ltm.write_fact.side_effect = RuntimeError("Qdrant is down")

        manager = MemoryNodeManager(
            memory_controller=StubMemoryController(),
            long_term_memory_store=mock_ltm,
        )
        state = _make_state("My birthday is March 3.", final_output="Noted.")

        # Must not raise
        result = manager.long_term_memory_write_node(state)
        assert "long_term_memory_write_status" in result
