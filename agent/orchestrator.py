"""
SAM Agent Orchestrator

Main entry point for the agent control flow. This module is the public API
for invoking the agent.

The actual graph implementation is in langgraph_orchestrator.py, which
implements the exact structure defined in design/langgraph_skeleton.md.
"""

from typing import Optional, Dict, Any
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from inference import ModelBackend, StubModelBackend
from inference.ollama import OllamaModelBackend
from config import Config


def _default_model_backend() -> ModelBackend:
    """Return the configured model backend based on LLM_BACKEND env var."""
    if Config.LLM_BACKEND == "ollama":
        return OllamaModelBackend(
            model_name=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
        )
    return StubModelBackend()


class SAMOrchestrator:
    """
    Main orchestrator for the SAM agent.
    
    Public API for agent invocation. Delegates to LangGraph implementation.
    """
    
    def __init__(self, model_backend: Optional[ModelBackend] = None):
        """
        Initialize the agent orchestrator.
        
        Args:
            model_backend: ModelBackend instance (uses configured backend by default)
        """
        self.langgraph_orchestrator = SAMAgentOrchestrator(
            model_backend=model_backend or _default_model_backend()
        )
    
    async def invoke(self, raw_input: str, conversation_id: Optional[str] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a single agent invocation.
        
        Args:
            raw_input: Raw input to process (text, audio reference, or image reference)
            conversation_id: Optional conversation ID (generated if not provided)
            trace_id: Optional trace ID (generated if not provided)
            
        Returns:
            Response dict with conversation_id, trace_id, status, output, error_type, metadata
        """
        return await self.langgraph_orchestrator.invoke(
            raw_input=raw_input,
            conversation_id=conversation_id,
            trace_id=trace_id,
        )
