"""
Fact Extraction Layer

Extract structured facts from user input and model responses.
Store in memory with confidence scores and timestamps.

Enforces:
- Confidence threshold (>= 0.7)
- Max 3 facts per turn
- Deterministic schema
- Non-blocking failures
"""

from datetime import datetime
from typing import List, Optional, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class ExtractedFact(BaseModel):
    """
    Structured fact extracted from conversation.
    
    Invariants:
    - confidence ∈ [0.0, 1.0]
    - content length <= 512 chars
    - timestamp is UTC
    """
    
    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["preference", "personal_fact", "goal", "task"]
    content: str = Field(..., max_length=512)
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_turn_id: str = Field(..., description="Conversation turn ID")
    
    @validator("content")
    def content_not_empty(cls, v):
        """Content must be non-empty and not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()
    
    class Config:
        """Pydantic config."""
        frozen = True  # Immutable


class FactExtractionRequest(BaseModel):
    """Request for fact extraction."""
    
    user_input: Optional[str] = None
    model_response: Optional[str] = None
    conversation_turn_id: str
    conversation_id: str


class FactExtractionResponse(BaseModel):
    """Response from fact extraction."""
    
    extracted_facts: List[ExtractedFact] = Field(default_factory=list)
    extraction_attempted: bool = True
    extraction_error: Optional[str] = None


class FactExtractor:
    """
    Extract facts from conversation turns.
    
    Guardrails:
    - Only extract: preferences, personal facts, goals, tasks
    - Confidence threshold: >= 0.7
    - Max 3 facts per turn
    - Reject vague or empty content
    - Non-blocking failures (exceptions don't break flow)
    """
    
    MIN_CONFIDENCE = 0.7
    MAX_FACTS_PER_TURN = 3
    
    def extract(
        self,
        request: FactExtractionRequest,
    ) -> FactExtractionResponse:
        """
        Extract facts from user input and model response.
        
        Returns facts that pass confidence threshold.
        Never raises; exceptions are caught and logged.
        
        Args:
            request: FactExtractionRequest
        
        Returns:
            FactExtractionResponse with extracted facts (or empty if none)
        """
        
        try:
            facts: List[ExtractedFact] = []
            
            # Extract from user input
            if request.user_input:
                user_facts = self._extract_from_text(
                    request.user_input,
                    request.conversation_turn_id,
                    source="user_input"
                )
                facts.extend(user_facts)
            
            # Extract from model response
            if request.model_response:
                response_facts = self._extract_from_text(
                    request.model_response,
                    request.conversation_turn_id,
                    source="model_response"
                )
                facts.extend(response_facts)
            
            # Apply confidence threshold and limit
            filtered_facts = [
                f for f in facts
                if f.confidence >= self.MIN_CONFIDENCE
            ][:self.MAX_FACTS_PER_TURN]
            
            return FactExtractionResponse(
                extracted_facts=filtered_facts,
                extraction_attempted=True,
                extraction_error=None,
            )
        
        except Exception as e:
            # Non-blocking failure
            return FactExtractionResponse(
                extracted_facts=[],
                extraction_attempted=False,
                extraction_error=str(e),
            )
    
    def _extract_from_text(
        self,
        text: str,
        turn_id: str,
        source: str,
    ) -> List[ExtractedFact]:
        """
        Extract facts from a text block.
        
        Heuristic rules (deterministic):
        - Mentions of "I want" → goal (confidence 0.8)
        - Mentions of "I prefer" → preference (confidence 0.85)
        - Personal descriptors ("I am X") → personal_fact (confidence 0.75)
        - Action items ("I will", "I need to") → task (confidence 0.8)
        
        Args:
            text: Text to extract from
            turn_id: Conversation turn ID
            source: "user_input" or "model_response"
        
        Returns:
            List of ExtractedFact (may be empty)
        """
        
        facts: List[ExtractedFact] = []
        
        # TODO: Implement deterministic extraction heuristics
        # For now, return empty (no facts)
        # In production, use NLP or ML-based extraction
        
        return facts


def get_fact_extractor() -> FactExtractor:
    """Get singleton fact extractor instance."""
    global _fact_extractor
    if _fact_extractor is None:
        _fact_extractor = FactExtractor()
    return _fact_extractor


_fact_extractor: Optional[FactExtractor] = None
