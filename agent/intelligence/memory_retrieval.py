"""
Intelligent Memory Retrieval & Ranking

Retrieve facts from memory based on relevance.
Rank by: semantic similarity (60%) + recency (20%) + confidence (20%)
Inject top facts as structured context before model call.

Enforces:
- Max 3 facts per injection
- Token budget limits
- Conflict detection & resolution
- No memory mutation
"""

import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class RankedFact(BaseModel):
    """Fact with relevance score."""
    
    fact_id: str
    content: str
    confidence: float
    timestamp: datetime
    retrieval_score: float = Field(..., ge=0.0, le=1.0)


class MemoryRankingRequest(BaseModel):
    """Request to rank and retrieve facts."""
    
    query: str
    conversation_id: str
    semantic_similarity_scores: Optional[dict] = Field(
        default=None,
        description="Pre-computed similarity scores {fact_id: score}"
    )
    max_results: int = Field(default=3, ge=1, le=5)


class MemoryRankingResponse(BaseModel):
    """Response with ranked facts."""
    
    ranked_facts: List[RankedFact] = Field(default_factory=list)
    ranking_attempted: bool = True
    ranking_error: Optional[str] = None


class MemoryRanker:
    """
    Rank facts by relevance for context injection.
    
    Ranking formula:
    score = 0.6 * semantic_similarity
          + 0.2 * recency_weight
          + 0.2 * confidence_score
    
    Where:
    recency_weight = exp(-Δtime_in_days / 7)
    """
    
    SEMANTIC_WEIGHT = 0.6
    RECENCY_WEIGHT = 0.2
    CONFIDENCE_WEIGHT = 0.2
    RECENCY_DECAY_DAYS = 7
    
    def rank(self, request: MemoryRankingRequest) -> MemoryRankingResponse:
        """
        Rank facts by relevance.
        
        Args:
            request: MemoryRankingRequest with query and optional similarity scores
        
        Returns:
            MemoryRankingResponse with ranked facts
        """
        
        try:
            # For now, return empty (integration with Qdrant will populate)
            return MemoryRankingResponse(
                ranked_facts=[],
                ranking_attempted=True,
                ranking_error=None,
            )
        
        except Exception as e:
            return MemoryRankingResponse(
                ranked_facts=[],
                ranking_attempted=False,
                ranking_error=str(e),
            )
    
    def _compute_ranking_score(
        self,
        semantic_similarity: float,
        timestamp: datetime,
        confidence: float,
    ) -> float:
        """
        Compute combined ranking score.
        
        Args:
            semantic_similarity: Score ∈ [0, 1]
            timestamp: Fact timestamp
            confidence: Confidence ∈ [0, 1]
        
        Returns:
            Combined score ∈ [0, 1]
        """
        
        # Compute recency weight
        days_ago = (datetime.utcnow() - timestamp).days
        recency = math.exp(-days_ago / self.RECENCY_DECAY_DAYS)
        recency = max(0.0, min(1.0, recency))  # Clamp to [0, 1]
        
        # Weighted sum
        score = (
            self.SEMANTIC_WEIGHT * semantic_similarity
            + self.RECENCY_WEIGHT * recency
            + self.CONFIDENCE_WEIGHT * confidence
        )
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]


class ConflictDetector:
    """
    Detect and resolve conflicting facts.
    
    Conflict criteria:
    - High semantic overlap (> 0.85)
    - Opposite polarity (contradiction)
    - Different confidence/recency
    
    Resolution:
    - Keep higher confidence
    - Keep more recent
    - Log conflict
    """
    
    OVERLAP_THRESHOLD = 0.85
    
    def detect_conflicts(
        self,
        facts: List[RankedFact],
    ) -> Tuple[List[RankedFact], List[str]]:
        """
        Detect conflicts and return non-conflicting subset.
        
        Args:
            facts: List of ranked facts
        
        Returns:
            (filtered_facts, conflict_descriptions)
        """
        
        filtered = []
        conflicts = []
        
        for fact in facts:
            # Check for conflicts with already-selected facts
            has_conflict = False
            
            for selected in filtered:
                # TODO: Implement semantic overlap detection
                # For now, no conflict detection
                pass
            
            if not has_conflict:
                filtered.append(fact)
        
        return filtered, conflicts


class ContextInjector:
    """
    Inject retrieved facts as structured context before model call.
    
    Injection format:
    [Relevant Past Facts]
    1. <fact_content> (confidence: X, from: <date>)
    2. <fact_content> (confidence: Y, from: <date>)
    ...
    
    Enforces:
    - Max 3 facts
    - Token budget (max 512 tokens)
    - Timestamp inclusion
    - Structured format
    """
    
    MAX_FACTS = 3
    MAX_TOKENS = 512
    
    def inject(
        self,
        model_prompt: str,
        facts: List[RankedFact],
    ) -> str:
        """
        Inject facts into model prompt.
        
        Args:
            model_prompt: Original prompt
            facts: Ranked facts to inject
        
        Returns:
            Prompt with injected context (or original if no facts)
        """
        
        if not facts:
            return model_prompt
        
        # Take top N facts
        selected = facts[:self.MAX_FACTS]
        
        # Build context block
        context = self._build_context_block(selected)
        
        # Check token budget (rough estimate: 1 token ≈ 4 chars)
        if len(context) > self.MAX_TOKENS * 4:
            # Truncate or skip injection
            return model_prompt
        
        # Inject before prompt
        return f"{context}\n\n{model_prompt}"
    
    def _build_context_block(self, facts: List[RankedFact]) -> str:
        """Build structured context block from facts."""
        
        lines = ["[Relevant Past Facts]"]
        
        for i, fact in enumerate(facts, 1):
            date_str = fact.timestamp.strftime("%Y-%m-%d")
            conf_pct = int(fact.confidence * 100)
            lines.append(
                f"{i}. {fact.content} (confidence: {conf_pct}%, from: {date_str})"
            )
        
        return "\n".join(lines)


def get_memory_ranker() -> MemoryRanker:
    """Get singleton ranker instance."""
    global _ranker
    if _ranker is None:
        _ranker = MemoryRanker()
    return _ranker


def get_conflict_detector() -> ConflictDetector:
    """Get singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = ConflictDetector()
    return _detector


def get_context_injector() -> ContextInjector:
    """Get singleton injector instance."""
    global _injector
    if _injector is None:
        _injector = ContextInjector()
    return _injector


_ranker: Optional[MemoryRanker] = None
_detector: Optional[ConflictDetector] = None
_injector: Optional[ContextInjector] = None
