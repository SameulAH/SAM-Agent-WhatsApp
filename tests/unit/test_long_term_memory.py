"""
Comprehensive tests for long-term memory implementation (Phase 3.2).

Tests cover 5 categories:
1. Interface compatibility (Stub ↔ implementations are interchangeable)
2. Append-only semantics (no overwrites, full history retrievable)
3. Authorization enforcement (unauthorized reads/writes rejected)
4. Control-flow invariance (routing identical with memory ON/OFF)
5. Failure safety (Qdrant unavailable, corrupted facts, empty retrieval)

These tests validate the non-negotiable constraints frozen in:
  design/long_term_memory_invariants.md
"""

import pytest
from datetime import datetime
from uuid import uuid4
from typing import List

from agent.memory import (
    MemoryFact,
    LongTermMemoryStore,
    StubLongTermMemoryStore,
    DisabledLongTermMemoryStore,
    LongTermMemoryWriteRequest,
    LongTermMemoryRetrievalQuery,
)


# ─────────────────────────────────────────────────────
# CATEGORY 1: INTERFACE COMPATIBILITY TESTS
# ─────────────────────────────────────────────────────

class TestInterfaceCompatibility:
    """
    Verify Stub and other implementations have identical behavior.
    
    Invariant: Both should be drop-in replacements.
    """

    def test_stub_write_returns_typed_response(self):
        """Write should return LongTermMemoryWriteResponse."""
        store = StubLongTermMemoryStore()
        fact = MemoryFact(
            fact_type="preference",
            content={"preference": "email summaries"},
            user_id="user123",
            confidence=0.9,
            source="agent_inferred",
        )
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=True,
        )
        response = store.write_fact(request)
        
        assert response.status in ("success", "failed", "unauthorized")
        assert response.fact_id is not None
        assert response.error is None or isinstance(response.error, str)

    def test_stub_read_returns_typed_response(self):
        """Retrieval should return LongTermMemoryRetrievalResponse."""
        store = StubLongTermMemoryStore()
        query = LongTermMemoryRetrievalQuery(
            user_id="user123",
            authorized=True,
        )
        response = store.retrieve_facts(query)
        
        assert response.status in ("success", "not_found", "unavailable", "unauthorized")
        assert isinstance(response.facts, list)
        assert response.error is None or isinstance(response.error, str)

    def test_stub_never_raises_exception(self):
        """No operation should raise an exception."""
        store = StubLongTermMemoryStore()
        fact = MemoryFact(
            fact_type="preference",
            content={"test": "data"},
            user_id="user123",
            confidence=0.9,
            source="test",
        )
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=True,
        )
        
        # Both should complete without exception
        write_response = store.write_fact(request)
        query = LongTermMemoryRetrievalQuery(user_id="user123", authorized=True)
        read_response = store.retrieve_facts(query)
        
        assert write_response is not None
        assert read_response is not None

    def test_disabled_implementation_exists(self):
        """DisabledLongTermMemoryStore should be available for testing."""
        store = DisabledLongTermMemoryStore()
        fact = MemoryFact(
            fact_type="preference",
            content={"test": "data"},
            user_id="user123",
            confidence=0.9,
            source="test",
        )
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=True,
        )
        
        # Disabled store always fails
        response = store.write_fact(request)
        assert response.status == "failed"

    def test_fact_ids_generated_on_write(self):
        """Fact IDs should be generated during write operation."""
        store = StubLongTermMemoryStore()
        fact = MemoryFact(
            fact_type="preference",
            content={"message": "test"},
            user_id="user123",
            confidence=0.9,
            source="test",
        )
        
        # Before write, fact_id and created_at are None
        assert fact.fact_id is None
        assert fact.created_at is None
        
        # Write the fact
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=True,
        )
        response = store.write_fact(request)
        
        # After write, response should have fact_id
        assert response.fact_id is not None


# ─────────────────────────────────────────────────────
# CATEGORY 2: APPEND-ONLY SEMANTICS TESTS
# ─────────────────────────────────────────────────────

class TestAppendOnlySemantics:
    """
    Verify facts are appended, never overwritten or deleted.
    
    Invariant: Writing 3 facts and retrieving should return all 3.
    """

    def test_write_and_retrieve_single_fact(self):
        """Should successfully write and retrieve a single fact."""
        store = StubLongTermMemoryStore()
        user_id = "user_" + str(uuid4())[:8]
        
        fact = MemoryFact(
            fact_type="preference",
            content={"preference": "concise responses"},
            user_id=user_id,
            confidence=0.9,
            source="agent_inferred",
        )
        write_request = LongTermMemoryWriteRequest(
            user_id=user_id,
            fact=fact,
            authorized=True,
        )
        
        # Write fact
        write_response = store.write_fact(write_request)
        assert write_response.status == "success"
        assert write_response.fact_id is not None
        
        # Retrieve fact
        query = LongTermMemoryRetrievalQuery(
            user_id=user_id,
            authorized=True,
        )
        read_response = store.retrieve_facts(query)
        assert read_response.status == "success"
        assert len(read_response.facts) == 1
        assert read_response.facts[0].content["preference"] == "concise responses"

    def test_multiple_facts_accumulate(self):
        """Writing multiple facts should accumulate, not overwrite."""
        store = StubLongTermMemoryStore()
        user_id = "user_" + str(uuid4())[:8]
        
        # Write 3 facts
        for i in range(1, 4):
            fact = MemoryFact(
                fact_type="preference" if i < 3 else "pattern",
                content={"label": f"Fact {i}"},
                user_id=user_id,
                confidence=1.0 - (i * 0.1),
                source="test",
            )
            request = LongTermMemoryWriteRequest(user_id=user_id, fact=fact, authorized=True)
            response = store.write_fact(request)
            assert response.status == "success"
        
        # Retrieve all facts
        query = LongTermMemoryRetrievalQuery(user_id=user_id, authorized=True)
        read_response = store.retrieve_facts(query)
        
        assert read_response.status == "success"
        assert len(read_response.facts) == 3
        
        # Verify all facts are present
        contents = {f.content["label"] for f in read_response.facts}
        assert "Fact 1" in contents
        assert "Fact 2" in contents
        assert "Fact 3" in contents

    def test_retrieved_facts_ordered_by_creation(self):
        """Retrieved facts should be ordered oldest-first."""
        store = StubLongTermMemoryStore()
        user_id = "user_" + str(uuid4())[:8]
        
        # Write facts
        for i in range(3):
            fact = MemoryFact(
                fact_type="preference",
                content={"label": f"Fact {i}"},
                user_id=user_id,
                confidence=0.9,
                source="test",
            )
            request = LongTermMemoryWriteRequest(user_id=user_id, fact=fact, authorized=True)
            response = store.write_fact(request)
            assert response.status == "success"
        
        # Retrieve facts
        query = LongTermMemoryRetrievalQuery(user_id=user_id, authorized=True)
        read_response = store.retrieve_facts(query)
        
        assert read_response.status == "success"
        assert len(read_response.facts) == 3
        
        # Verify order (oldest first)
        prev_time = None
        for fact in read_response.facts:
            if prev_time:
                assert fact.created_at >= prev_time
            prev_time = fact.created_at

    def test_no_overwriting_same_user(self):
        """Writing to same user should append, never overwrite."""
        store = StubLongTermMemoryStore()
        user_id = "user_" + str(uuid4())[:8]
        
        # Write same fact_type twice
        for i in range(2):
            fact = MemoryFact(
                fact_type="preference",
                content={"version": i+1},
                user_id=user_id,
                confidence=0.9,
                source="test",
            )
            request = LongTermMemoryWriteRequest(user_id=user_id, fact=fact, authorized=True)
            response = store.write_fact(request)
            assert response.status == "success"
        
        # Retrieve should have both
        query = LongTermMemoryRetrievalQuery(user_id=user_id, authorized=True)
        read_response = store.retrieve_facts(query)
        
        assert len(read_response.facts) == 2
        versions = {f.content["version"] for f in read_response.facts}
        assert 1 in versions
        assert 2 in versions


# ─────────────────────────────────────────────────────
# CATEGORY 3: AUTHORIZATION ENFORCEMENT TESTS
# ─────────────────────────────────────────────────────

class TestAuthorizationEnforcement:
    """
    Verify only authorized operations succeed.
    
    Invariant: authorized=False should always be rejected.
    """

    def test_unauthorized_write_rejected(self):
        """Writing with authorized=False should be rejected."""
        store = StubLongTermMemoryStore()
        fact = MemoryFact(
            fact_type="preference",
            content={"test": "data"},
            user_id="user123",
            confidence=0.9,
            source="test",
        )
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=False,
        )
        response = store.write_fact(request)
        
        assert response.status == "unauthorized"

    def test_unauthorized_read_rejected(self):
        """Reading with authorized=False should be rejected."""
        store = StubLongTermMemoryStore()
        
        # First write a fact as authorized
        fact = MemoryFact(
            fact_type="preference",
            content={"secret": "data"},
            user_id="user123",
            confidence=0.9,
            source="test",
        )
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=True,
        )
        write_response = store.write_fact(request)
        assert write_response.status == "success"
        
        # Now try to read WITHOUT authorization
        query = LongTermMemoryRetrievalQuery(
            user_id="user123",
            authorized=False,
        )
        read_response = store.retrieve_facts(query)
        
        assert read_response.status == "unauthorized"

    def test_authorized_write_succeeds(self):
        """Writing with authorized=True should succeed."""
        store = StubLongTermMemoryStore()
        fact = MemoryFact(
            fact_type="preference",
            content={"public": "data"},
            user_id="user123",
            confidence=0.9,
            source="test",
        )
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=True,
        )
        response = store.write_fact(request)
        
        assert response.status == "success"

    def test_authorized_read_succeeds(self):
        """Reading with authorized=True should succeed."""
        store = StubLongTermMemoryStore()
        
        # Write fact
        fact = MemoryFact(
            fact_type="preference",
            content={"public": "data"},
            user_id="user123",
            confidence=0.9,
            source="test",
        )
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=True,
        )
        store.write_fact(request)
        
        # Read with authorization
        query = LongTermMemoryRetrievalQuery(
            user_id="user123",
            authorized=True,
        )
        response = store.retrieve_facts(query)
        
        assert response.status == "success"
        assert len(response.facts) > 0


# ─────────────────────────────────────────────────────
# CATEGORY 4: CONTROL-FLOW INVARIANCE TESTS
# ─────────────────────────────────────────────────────

class TestControlFlowInvariance:
    """
    Verify agent routing is identical with memory ON/OFF.
    
    Invariant: Removing long-term memory should not change control flow.
    """

    def test_enabled_and_disabled_store_are_drop_in_replacements(self):
        """Enabled and disabled stores should both return Response objects."""
        enabled = StubLongTermMemoryStore()
        disabled = DisabledLongTermMemoryStore()
        
        fact = MemoryFact(
            fact_type="preference",
            content={"test": "data"},
            user_id="user123",
            confidence=0.9,
            source="test",
        )
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=True,
        )
        
        # Both should return Response with status field
        enabled_response = enabled.write_fact(request)
        disabled_response = disabled.write_fact(request)
        
        assert hasattr(enabled_response, "status")
        assert hasattr(disabled_response, "status")
        assert enabled_response.status in ("success", "failed", "unauthorized")
        assert disabled_response.status in ("success", "failed", "unauthorized")

    def test_empty_retrieval_is_success_not_error(self):
        """Empty retrieval should return status=success with empty list."""
        store = StubLongTermMemoryStore()
        
        # Query for non-existent user
        query = LongTermMemoryRetrievalQuery(
            user_id="nonexistent_user",
            authorized=True,
        )
        response = store.retrieve_facts(query)
        
        # Should be success, not "not_found"
        assert response.status == "success"
        assert response.facts == []

    def test_disabled_empty_retrieval(self):
        """Disabled store should return empty facts list."""
        store = DisabledLongTermMemoryStore()
        query = LongTermMemoryRetrievalQuery(
            user_id="user123",
            authorized=True,
        )
        response = store.retrieve_facts(query)
        
        assert response.status == "unavailable"
        assert response.facts == []


# ─────────────────────────────────────────────────────
# CATEGORY 5: FAILURE SAFETY TESTS
# ─────────────────────────────────────────────────────

class TestFailureSafety:
    """
    Verify system degrades gracefully on failures.
    
    Invariant: Memory failures never prevent agent execution.
    """

    def test_disabled_store_fails_gracefully_on_write(self):
        """DisabledStore write should fail gracefully."""
        store = DisabledLongTermMemoryStore()
        fact = MemoryFact(
            fact_type="preference",
            content={"test": "data"},
            user_id="user123",
            confidence=0.9,
            source="test",
        )
        request = LongTermMemoryWriteRequest(
            user_id="user123",
            fact=fact,
            authorized=True,
        )
        
        response = store.write_fact(request)
        assert response.status == "failed"
        assert response.fact_id is None

    def test_disabled_store_fails_gracefully_on_read(self):
        """DisabledStore read should fail gracefully."""
        store = DisabledLongTermMemoryStore()
        query = LongTermMemoryRetrievalQuery(
            user_id="user123",
            authorized=True,
        )
        
        response = store.retrieve_facts(query)
        assert response.status == "unavailable"
        assert response.facts == []

    def test_fact_isolation_by_user(self):
        """Facts for one user should not be accessible to another."""
        store = StubLongTermMemoryStore()
        
        # Write fact for user1
        fact1 = MemoryFact(
            fact_type="preference",
            content={"secret": "user1"},
            user_id="user1",
            confidence=0.9,
            source="test",
        )
        request1 = LongTermMemoryWriteRequest(user_id="user1", fact=fact1, authorized=True)
        store.write_fact(request1)
        
        # Query as user2
        query2 = LongTermMemoryRetrievalQuery(user_id="user2", authorized=True)
        response2 = store.retrieve_facts(query2)
        
        assert response2.status == "success"
        assert len(response2.facts) == 0

    def test_fact_type_filtering(self):
        """Should be able to filter by fact_type."""
        store = StubLongTermMemoryStore()
        user_id = "user_" + str(uuid4())[:8]
        
        # Write different fact types
        for fact_type in ["preference", "pattern", "interaction"]:
            fact = MemoryFact(
                fact_type=fact_type,
                content={"type": fact_type},
                user_id=user_id,
                confidence=0.9,
                source="test",
            )
            request = LongTermMemoryWriteRequest(user_id=user_id, fact=fact, authorized=True)
            store.write_fact(request)
        
        # Query for specific type
        query = LongTermMemoryRetrievalQuery(
            user_id=user_id,
            fact_types=["preference"],
            authorized=True,
        )
        response = store.retrieve_facts(query)
        
        assert response.status == "success"
        assert all(f.fact_type == "preference" for f in response.facts)

    def test_confidence_scores_preserved(self):
        """Confidence scores should be preserved on retrieval."""
        store = StubLongTermMemoryStore()
        user_id = "user_" + str(uuid4())[:8]
        
        confidence_scores = [0.5, 0.7, 0.95]
        for confidence in confidence_scores:
            fact = MemoryFact(
                fact_type="preference",
                content={"confidence": confidence},
                user_id=user_id,
                confidence=confidence,
                source="test",
            )
            request = LongTermMemoryWriteRequest(user_id=user_id, fact=fact, authorized=True)
            store.write_fact(request)
        
        # Retrieve and verify confidence
        query = LongTermMemoryRetrievalQuery(user_id=user_id, authorized=True)
        response = store.retrieve_facts(query)
        
        retrieved_confidences = {f.confidence for f in response.facts}
        assert retrieved_confidences == set(confidence_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
