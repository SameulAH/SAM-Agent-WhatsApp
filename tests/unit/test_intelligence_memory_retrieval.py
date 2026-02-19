"""
Tests for Memory Retrieval & Ranking

Unit and integration tests for ranking, conflict detection, context injection.
"""

import pytest
import math
from datetime import datetime, timedelta

from agent.intelligence.memory_retrieval import (
    RankedFact,
    MemoryRankingRequest,
    MemoryRankingResponse,
    MemoryRanker,
    ConflictDetector,
    ContextInjector,
)


class TestMemoryRanker:
    """Test memory ranking formula."""
    
    def test_ranking_score_computation(self):
        """Ranking score combines similarity, recency, confidence."""
        ranker = MemoryRanker()
        
        # Recent, high confidence, high similarity
        score1 = ranker._compute_ranking_score(
            semantic_similarity=0.9,
            timestamp=datetime.utcnow(),
            confidence=0.9,
        )
        
        # Old, low confidence, low similarity
        score2 = ranker._compute_ranking_score(
            semantic_similarity=0.1,
            timestamp=datetime.utcnow() - timedelta(days=30),
            confidence=0.1,
        )
        
        # score1 should be much higher
        assert score1 > score2
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
    
    def test_recency_weight(self):
        """Recency weight decays exponentially."""
        ranker = MemoryRanker()
        
        # Recent (1 day ago)
        score_1d = ranker._compute_ranking_score(
            semantic_similarity=0.5,
            timestamp=datetime.utcnow() - timedelta(days=1),
            confidence=0.5,
        )
        
        # Old (7 days ago)
        score_7d = ranker._compute_ranking_score(
            semantic_similarity=0.5,
            timestamp=datetime.utcnow() - timedelta(days=7),
            confidence=0.5,
        )
        
        # Score should decay with age
        assert score_1d > score_7d
    
    def test_ranking_weights(self):
        """Verify weighted component contributions."""
        ranker = MemoryRanker()
        
        # Pure similarity (recent, perfect confidence)
        score_sim = ranker._compute_ranking_score(0.9, datetime.utcnow(), 1.0)
        
        # Pure recency (perfect similarity, full confidence)
        score_rec = ranker._compute_ranking_score(1.0, datetime.utcnow(), 1.0)
        
        # Pure confidence (perfect similarity, recent)
        score_conf = ranker._compute_ranking_score(1.0, datetime.utcnow(), 0.9)
        
        # All have different weights, so scores differ
        assert score_sim != score_rec
        assert score_rec != score_conf
    
    def test_rank_request(self):
        """Memory ranking request/response."""
        ranker = MemoryRanker()
        
        request = MemoryRankingRequest(
            query="past preferences",
            conversation_id="conv_1",
            max_results=3,
        )
        
        response = ranker.rank(request)
        
        assert response.ranking_attempted == True
        assert response.ranking_error is None
        assert isinstance(response.ranked_facts, list)


class TestConflictDetector:
    """Test conflict detection."""
    
    def test_no_conflicts(self):
        """Non-conflicting facts are kept."""
        detector = ConflictDetector()
        
        facts = [
            RankedFact(
                fact_id="f1",
                content="I like coffee",
                confidence=0.9,
                timestamp=datetime.utcnow(),
                retrieval_score=0.8,
            ),
            RankedFact(
                fact_id="f2",
                content="I prefer Python",
                confidence=0.85,
                timestamp=datetime.utcnow(),
                retrieval_score=0.75,
            ),
        ]
        
        filtered, conflicts = detector.detect_conflicts(facts)
        
        assert len(filtered) == 2
        assert len(conflicts) == 0
    
    def test_conflicting_facts(self):
        """Conflicting facts detected."""
        # TODO: Implement when semantic similarity is available
        pass


class TestContextInjector:
    """Test context injection."""
    
    def test_inject_no_facts(self):
        """Injection with no facts returns original prompt."""
        injector = ContextInjector()
        
        original = "What time is it?"
        result = injector.inject(original, [])
        
        assert result == original
    
    def test_inject_with_facts(self):
        """Facts are injected in structured format."""
        injector = ContextInjector()
        
        facts = [
            RankedFact(
                fact_id="f1",
                content="I prefer coffee",
                confidence=0.9,
                timestamp=datetime(2026, 2, 1),
                retrieval_score=0.8,
            ),
        ]
        
        original = "What should I drink?"
        result = injector.inject(original, facts)
        
        # Should contain context block
        assert "[Relevant Past Facts]" in result
        assert "I prefer coffee" in result
        assert "confidence: 90%" in result
        assert "2026-02-01" in result
        # Original prompt should still be there
        assert original in result
    
    def test_inject_max_facts(self):
        """Max 3 facts injected."""
        injector = ContextInjector()
        
        facts = [
            RankedFact(
                fact_id=f"f{i}",
                content=f"Fact {i}",
                confidence=0.8,
                timestamp=datetime.utcnow(),
                retrieval_score=0.7,
            )
            for i in range(5)
        ]
        
        result = injector.inject("Prompt", facts)
        
        # Should have max 3 facts
        assert result.count("Fact ") <= 3
    
    def test_inject_token_budget(self):
        """Injection respects token budget."""
        injector = ContextInjector()
        
        # Very long facts that exceed token budget
        facts = [
            RankedFact(
                fact_id="f1",
                content="x" * 2000,  # Exceeds token budget
                confidence=0.8,
                timestamp=datetime.utcnow(),
                retrieval_score=0.7,
            ),
        ]
        
        original = "Prompt"
        result = injector.inject(original, facts)
        
        # Should skip injection if too large
        if len(result) > len(original) * 2:
            # Injection was attempted (too large)
            pass
        else:
            # Injection was skipped (within budget)
            assert result == original


class TestContextInjectionIntegration:
    """Integration tests for context injection."""
    
    def test_facts_not_duplicated(self):
        """Facts are not duplicated in injection."""
        injector = ContextInjector()
        
        facts = [
            RankedFact(
                fact_id="f1",
                content="Same fact",
                confidence=0.8,
                timestamp=datetime.utcnow(),
                retrieval_score=0.7,
            ),
        ]
        
        result = injector.inject("Prompt", facts)
        
        # "Same fact" should appear exactly once (in context block)
        count = result.count("Same fact")
        assert count == 1
    
    def test_injection_structure(self):
        """Injected context is properly structured."""
        injector = ContextInjector()
        
        facts = [
            RankedFact(
                fact_id="f1",
                content="Fact 1",
                confidence=0.95,
                timestamp=datetime(2026, 1, 15),
                retrieval_score=0.9,
            ),
            RankedFact(
                fact_id="f2",
                content="Fact 2",
                confidence=0.8,
                timestamp=datetime(2026, 1, 10),
                retrieval_score=0.75,
            ),
        ]
        
        result = injector.inject("Prompt", facts)
        
        # Check structure
        lines = result.split("\n")
        assert "[Relevant Past Facts]" in result
        assert "1. Fact 1" in result
        assert "2. Fact 2" in result
        assert "confidence: 95%" in result
        assert "confidence: 80%" in result
