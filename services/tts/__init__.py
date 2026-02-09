"""
Text-to-Speech service exports.

Clean interface for agent to import TTS components.
"""

from .base import TTSBackend, TTSRequest, TTSResponse, TTSStatus
from .stub import StubTTSBackend, NoOpTTSBackend
from .coqui import CoquiTTSBackend

__all__ = [
    "TTSBackend",
    "TTSRequest",
    "TTSResponse",
    "TTSStatus",
    "StubTTSBackend",
    "NoOpTTSBackend",
    "CoquiTTSBackend",
]
