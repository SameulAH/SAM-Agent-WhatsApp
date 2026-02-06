"""
Model boundary layer for LLM inference.

This package provides a clean abstraction for model invocation,
allowing the agent to remain agnostic of the underlying backend.

Supported backends:
- StubModelBackend: Deterministic fake model (default for CI/tests)
- OllamaModelBackend: Local Ollama inference

Example usage:
    from inference import StubModelBackend, ModelRequest
    
    backend = StubModelBackend()
    request = ModelRequest(task="respond", prompt="Hello, world!")
    response = backend.generate(request)
"""

from .types import ModelRequest, ModelResponse, ModelStatus
from .base import ModelBackend
from .stub import StubModelBackend
from .ollama import OllamaModelBackend

__all__ = [
    "ModelRequest",
    "ModelResponse",
    "ModelStatus",
    "ModelBackend",
    "StubModelBackend",
    "OllamaModelBackend",
]
