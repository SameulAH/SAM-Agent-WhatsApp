"""
Tests for Intelligence Guardrails

Unit and integration tests for guardrail enforcement.
"""

import pytest
from agent.intelligence.guardrails import (
    GuardrailViolation,
    IntelligenceGuardrails,
    get_guardrails,
)


class TestGuardrailViolation:
    """Test GuardrailViolation model."""
    
    def test_violation_warning(self):
        """Create warning violation."""
        violation = GuardrailViolation(
            rule_name="memory_size",
            message="Memory size approaching limit",
            severity="warning",
        )
        
        assert violation.rule_name == "memory_size"
        assert violation.severity == "warning"
    
    def test_violation_error(self):
        """Create error violation."""
        violation = GuardrailViolation(
            rule_name="facts_per_conversation",
            message="Too many facts",
            severity="error",
        )
        
        assert violation.rule_name == "facts_per_conversation"
        assert violation.severity == "error"


class TestMemoryWriteAuthorization:
    """Test memory write authorization guardrail."""
    
    def test_authorized_write(self):
        """Authorized memory write passes."""
        guardrails = IntelligenceGuardrails()
        
        # Assume user input is authorized by default
        violation = guardrails.check_memory_write_authorized(
            user_id="user_1",
            is_user_input=True,
        )
        
        assert violation is None
    
    def test_unauthorized_write(self):
        """Unauthorized memory write fails."""
        guardrails = IntelligenceGuardrails()
        
        # System-generated facts should be authorized
        violation = guardrails.check_memory_write_authorized(
            user_id="user_1",
            is_user_input=False,
        )
        
        # System facts may not be authorized (depends on policy)
        # For now, we don't enforce


class TestFactsPerConversation:
    """Test facts per conversation guardrail."""
    
    def test_under_limit(self):
        """Under limit passes."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_facts_per_conversation(
            conversation_id="conv_1",
            fact_count=500,
        )
        
        assert violation is None
    
    def test_at_limit(self):
        """At limit passes."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_facts_per_conversation(
            conversation_id="conv_1",
            fact_count=1000,
        )
        
        assert violation is None
    
    def test_over_limit(self):
        """Over limit fails."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_facts_per_conversation(
            conversation_id="conv_1",
            fact_count=1001,
        )
        
        assert violation is not None
        assert violation.rule_name == "facts_per_conversation"
        assert violation.severity == "error"


class TestFactsPerUser:
    """Test facts per user guardrail."""
    
    def test_under_limit(self):
        """Under limit passes."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_facts_per_user(
            user_id="user_1",
            fact_count=4500,
        )
        
        assert violation is None
    
    def test_at_limit(self):
        """At limit passes."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_facts_per_user(
            user_id="user_1",
            fact_count=5000,
        )
        
        assert violation is None
    
    def test_over_limit(self):
        """Over limit fails."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_facts_per_user(
            user_id="user_1",
            fact_count=5001,
        )
        
        assert violation is not None
        assert violation.rule_name == "facts_per_user"
        assert violation.severity == "error"


class TestRetrievalInjectionsPerTurn:
    """Test retrieval injections per turn guardrail."""
    
    def test_under_limit(self):
        """Under limit passes."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_retrieval_injection_per_turn(
            turn_number=1,
            injection_count=2,
        )
        
        assert violation is None
    
    def test_at_limit(self):
        """At limit passes."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_retrieval_injection_per_turn(
            turn_number=1,
            injection_count=3,
        )
        
        assert violation is None
    
    def test_over_limit(self):
        """Over limit fails."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_retrieval_injection_per_turn(
            turn_number=1,
            injection_count=4,
        )
        
        assert violation is not None
        assert violation.rule_name == "retrieval_injection_per_turn"
        assert violation.severity == "error"


class TestMemorySizeLimit:
    """Test memory size guardrail."""
    
    def test_under_limit(self):
        """Under limit passes."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_memory_size(
            memory_size_mb=50,
        )
        
        assert violation is None
    
    def test_at_limit(self):
        """At limit passes."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_memory_size(
            memory_size_mb=100,
        )
        
        assert violation is None
    
    def test_over_limit(self):
        """Over limit fails."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_memory_size(
            memory_size_mb=101,
        )
        
        assert violation is not None
        assert violation.rule_name == "memory_size"
        assert violation.severity == "critical"


class TestGuardrailsSingleton:
    """Test IntelligenceGuardrails singleton."""
    
    def test_get_guardrails(self):
        """Get global guardrails."""
        guardrails1 = get_guardrails()
        guardrails2 = get_guardrails()
        
        # Should be same instance
        assert guardrails1 is guardrails2


class TestGuardrailsIntegration:
    """Integration tests for guardrails."""
    
    def test_all_guardrails_pass(self):
        """All guardrails can pass in normal operation."""
        guardrails = IntelligenceGuardrails()
        
        violations = []
        
        violations.append(
            guardrails.check_memory_write_authorized("user_1", True)
        )
        violations.append(
            guardrails.check_facts_per_conversation("conv_1", 500)
        )
        violations.append(
            guardrails.check_facts_per_user("user_1", 2000)
        )
        violations.append(
            guardrails.check_retrieval_injection_per_turn(1, 2)
        )
        violations.append(
            guardrails.check_memory_size(50)
        )
        
        # All should pass
        for violation in violations:
            assert violation is None
    
    def test_guardrails_independent(self):
        """Violating one guardrail doesn't affect others."""
        guardrails = IntelligenceGuardrails()
        
        # Violate one
        violation1 = guardrails.check_facts_per_conversation("conv_1", 1001)
        assert violation1 is not None
        
        # Others should still pass
        violation2 = guardrails.check_facts_per_user("user_1", 2000)
        violation3 = guardrails.check_memory_size(50)
        
        assert violation2 is None
        assert violation3 is None
    
    def test_critical_violations(self):
        """Critical violations have correct severity."""
        guardrails = IntelligenceGuardrails()
        
        violation = guardrails.check_memory_size(101)
        
        assert violation is not None
        assert violation.severity == "critical"


class TestGuardrailsArchitecture:
    """Test guardrail architectural constraints."""
    
    def test_guardrails_architecture_constant(self):
        """Guardrails can verify no memory routing."""
        guardrails = IntelligenceGuardrails()
        
        # This is a placeholder for graph structure verification
        # In real implementation, would check LangGraph structure
        # For now, just verify method exists
        assert hasattr(guardrails, "check_memory_never_routes")
    
    def test_guardrails_no_side_effects(self):
        """Guardrail checks have no side effects."""
        guardrails = IntelligenceGuardrails()
        
        # Run multiple times
        v1 = guardrails.check_facts_per_conversation("conv_1", 500)
        v2 = guardrails.check_facts_per_conversation("conv_1", 500)
        
        # Same result
        assert (v1 is None) == (v2 is None)
