"""
Tests for Fact Extraction Layer

Unit and integration tests for ExtractedFact schema and FactExtractor.
"""

import pytest
from datetime import datetime

from agent.intelligence.fact_extraction import (
    ExtractedFact,
    FactExtractionRequest,
    FactExtractionResponse,
    FactExtractor,
)


class TestExtractedFactSchema:
    """Test ExtractedFact schema validation."""
    
    def test_valid_fact(self):
        """Valid fact is created."""
        fact = ExtractedFact(
            type="preference",
            content="I prefer coffee over tea",
            confidence=0.85,
            source_turn_id="turn_123",
        )
        
        assert fact.type == "preference"
        assert fact.content == "I prefer coffee over tea"
        assert fact.confidence == 0.85
        assert fact.fact_id is not None
    
    def test_confidence_range(self):
        """Confidence must be in [0, 1]."""
        # Valid
        ExtractedFact(
            type="goal",
            content="Learn Python",
            confidence=0.0,
            source_turn_id="turn_1",
        )
        
        ExtractedFact(
            type="goal",
            content="Learn Python",
            confidence=1.0,
            source_turn_id="turn_1",
        )
        
        # Invalid
        with pytest.raises(ValueError):
            ExtractedFact(
                type="goal",
                content="Learn Python",
                confidence=1.5,
                source_turn_id="turn_1",
            )
        
        with pytest.raises(ValueError):
            ExtractedFact(
                type="goal",
                content="Learn Python",
                confidence=-0.1,
                source_turn_id="turn_1",
            )
    
    def test_content_max_length(self):
        """Content must be <= 512 chars."""
        # Valid
        ExtractedFact(
            type="personal_fact",
            content="x" * 512,
            confidence=0.8,
            source_turn_id="turn_1",
        )
        
        # Invalid
        with pytest.raises(ValueError):
            ExtractedFact(
                type="personal_fact",
                content="x" * 513,
                confidence=0.8,
                source_turn_id="turn_1",
            )
    
    def test_content_not_empty(self):
        """Content cannot be empty or whitespace."""
        with pytest.raises(ValueError):
            ExtractedFact(
                type="task",
                content="",
                confidence=0.8,
                source_turn_id="turn_1",
            )
        
        with pytest.raises(ValueError):
            ExtractedFact(
                type="task",
                content="   ",
                confidence=0.8,
                source_turn_id="turn_1",
            )
    
    def test_fact_is_immutable(self):
        """ExtractedFact is frozen."""
        from pydantic import ValidationError
        
    def test_fact_is_immutable(self):
        """ExtractedFact is frozen."""
        from pydantic import ValidationError
        
        fact = ExtractedFact(
            type="preference",
            content="Test",
            confidence=0.8,
            source_turn_id="turn_1",
        )

        with pytest.raises(ValidationError):
            fact.content = "Modified"  # type: ignore
    
    def test_valid_types(self):
        """Fact types are constrained."""
        valid_types = ["preference", "personal_fact", "goal", "task"]
        
        for ftype in valid_types:
            fact = ExtractedFact(
                type=ftype,
                content="Test content",
                confidence=0.8,
                source_turn_id="turn_1",
            )
            assert fact.type == ftype
        
        # Invalid type
        with pytest.raises(ValueError):
            ExtractedFact(
                type="invalid_type",  # type: ignore
                content="Test",
                confidence=0.8,
                source_turn_id="turn_1",
            )


class TestFactExtractor:
    """Test FactExtractor.extract() method."""
    
    def test_extract_no_input(self):
        """Extraction with no input returns empty."""
        extractor = FactExtractor()
        
        request = FactExtractionRequest(
            user_input=None,
            model_response=None,
            conversation_turn_id="turn_1",
            conversation_id="conv_1",
        )
        
        response = extractor.extract(request)
        
        assert response.extraction_attempted == True
        assert len(response.extracted_facts) == 0
        assert response.extraction_error is None
    
    def test_extract_respects_confidence_threshold(self):
        """Facts below 0.7 confidence are rejected."""
        # TODO: Implement when extraction heuristics are added
        pass
    
    def test_extract_max_facts_per_turn(self):
        """Max 3 facts extracted per turn."""
        extractor = FactExtractor()
        
        # TODO: Implement when extraction heuristics are added
        # Should return at most 3 facts regardless of input
        pass
    
    def test_extract_handles_exception(self):
        """Extraction failure returns error, doesn't raise."""
        extractor = FactExtractor()
        
        # Pass invalid request type
        response = extractor.extract(None)  # type: ignore
        
        assert response.extraction_attempted == False
        assert len(response.extracted_facts) == 0
        assert response.extraction_error is not None


class TestFactExtractionIntegration:
    """Integration tests for fact extraction."""
    
    def test_fact_extraction_flow(self):
        """End-to-end fact extraction."""
        extractor = FactExtractor()
        
        request = FactExtractionRequest(
            user_input="I really like coffee in the morning",
            model_response="That's a great routine! Coffee helps with focus.",
            conversation_turn_id="turn_1",
            conversation_id="conv_1",
        )
        
        response = extractor.extract(request)
        
        # Should always succeed
        assert response.extraction_attempted == True
        assert response.extraction_error is None
        # May have facts or not (depends on implementation)
        assert isinstance(response.extracted_facts, list)
    
    def test_fact_not_stored_when_irrelevant(self):
        """No facts stored for irrelevant input."""
        extractor = FactExtractor()
        
        request = FactExtractionRequest(
            user_input="What is the weather today?",
            model_response="The weather is sunny and 72°F.",
            conversation_turn_id="turn_1",
            conversation_id="conv_1",
        )
        
        response = extractor.extract(request)
        
        # Weather facts typically not relevant to extract
        assert len(response.extracted_facts) == 0
