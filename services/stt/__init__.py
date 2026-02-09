"""
Speech-to-Text service exports.

Clean interface for agent to import STT components.
"""

from .base import STTBackend, STTRequest, STTResponse, STTStatus
from .stub import StubSTTBackend, NoOpSTTBackend
from .whisper import WhisperLocalSTTBackend

__all__ = [
    "STTBackend",
    "STTRequest",
    "STTResponse",
    "STTStatus",
    "StubSTTBackend",
    "NoOpSTTBackend",
    "WhisperLocalSTTBackend",
]
