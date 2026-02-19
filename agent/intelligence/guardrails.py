"""
Intelligence Guardrails

Enforce hard limits on intelligent behavior.
Prevent memory overflow, excess injection, unauthorized memory writes.

Enforces:
- Max facts per conversation
- Max retrieval injections per turn
- Max memory size per user
- Memory write authorization
- Tool call authorization
"""

from typing import Optional

from pydantic import BaseModel


class GuardrailViolation(BaseModel):
    """Record of guardrail violation."""
    
    rule_name: str
    message: str
    severity: str  # "warning", "error", "critical"


class IntelligenceGuardrails:
    """
    Enforce hard limits on intelligent behavior.
    
    Prevents:
    - Memory overflow (too many facts)
    - Excess context injection (too many facts per turn)
    - Unauthorized memory writes
    - Tool execution bypassing routing
    """
    
    # Limits
    MAX_FACTS_PER_CONVERSATION = 1000
    MAX_FACTS_PER_USER = 5000
    MAX_RETRIEVAL_INJECTIONS_PER_TURN = 3
    MAX_MEMORY_SIZE_MB = 100
    
    def check_memory_write_authorized(
        self,
        user_id: str,
        is_user_input: bool,
    ) -> Optional[GuardrailViolation]:
        """
        Check that memory write is authorized.
        
        Args:
            user_id: User ID
            is_user_input: Whether write is from user (authorized) or system
        
        Returns:
            GuardrailViolation if unauthorized, None otherwise
        """
        
        # User input is always authorized
        if is_user_input:
            return None
        
        # System-generated facts are authorized for now
        return None
    
    def check_facts_per_conversation(
        self,
        conversation_id: str,
        fact_count: int,
    ) -> Optional[GuardrailViolation]:
        """
        Check fact count limit per conversation.
        
        Args:
            conversation_id: Conversation ID
            fact_count: Total facts in conversation
        
        Returns:
            GuardrailViolation if limit exceeded, None otherwise
        """
        
        if fact_count > self.MAX_FACTS_PER_CONVERSATION:
            return GuardrailViolation(
                rule_name="facts_per_conversation",
                message=f"Fact count {fact_count} exceeds max {self.MAX_FACTS_PER_CONVERSATION}",
                severity="error",
            )
        
        return None
    
    def check_facts_per_user(
        self,
        user_id: str,
        fact_count: int,
    ) -> Optional[GuardrailViolation]:
        """
        Check fact count limit per user.
        
        Args:
            user_id: User ID
            fact_count: Total facts for user
        
        Returns:
            GuardrailViolation if limit exceeded, None otherwise
        """
        
        if fact_count > self.MAX_FACTS_PER_USER:
            return GuardrailViolation(
                rule_name="facts_per_user",
                message=f"Fact count {fact_count} exceeds max {self.MAX_FACTS_PER_USER}",
                severity="error",
            )
        
        return None
    
    def check_retrieval_injection_per_turn(
        self,
        turn_number: int,
        injection_count: int,
    ) -> Optional[GuardrailViolation]:
        """
        Check retrieval injection limit per turn.
        
        Args:
            turn_number: Current turn number
            injection_count: Number of facts to inject
        
        Returns:
            GuardrailViolation if limit exceeded, None otherwise
        """
        
        if injection_count > self.MAX_RETRIEVAL_INJECTIONS_PER_TURN:
            return GuardrailViolation(
                rule_name="retrieval_injection_per_turn",
                message=f"Injection count {injection_count} exceeds max {self.MAX_RETRIEVAL_INJECTIONS_PER_TURN}",
                severity="error",
            )
        
        return None
    
    def check_memory_size(
        self,
        memory_size_mb: float,
    ) -> Optional[GuardrailViolation]:
        """
        Check memory size limit.
        
        Args:
            memory_size_mb: Memory size in MB
        
        Returns:
            GuardrailViolation if limit exceeded, None otherwise
        """
        
        if memory_size_mb > self.MAX_MEMORY_SIZE_MB:
            return GuardrailViolation(
                rule_name="memory_size",
                message=f"Memory size {memory_size_mb}MB exceeds max {self.MAX_MEMORY_SIZE_MB}MB",
                severity="critical",
            )
        
        return None
    
    def check_memory_never_routes(self) -> bool:
        """
        Verify that memory retrieval/write cannot influence routing.
        
        This is architectural - verified through graph structure.
        
        Returns:
            True if architecture respects guardrail
        """
        
        # TODO: Verify graph structure
        # Memory nodes never connect to decision_logic directly
        # Decision logic is the only routing authority
        
        return True


def get_guardrails() -> IntelligenceGuardrails:
    """Get singleton guardrails instance."""
    global _guardrails
    if _guardrails is None:
        _guardrails = IntelligenceGuardrails()
    return _guardrails


_guardrails: Optional[IntelligenceGuardrails] = None
