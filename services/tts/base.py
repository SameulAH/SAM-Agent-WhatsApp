"""
Text-to-Speech (TTS) abstract interface.

Role: Text → audio rendering only.

Rules:
- Output-only (no state mutation)
- Optional (failure → text response only)
- No memory access
- All failures are explicit and typed
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal


TTSStatus = Literal["success", "recoverable_error", "fatal_error"]


@dataclass
class TTSRequest:
    """Text-to-Speech request."""
    
    text: str
    language: str = "en"
    voice: str = "default"
    speed: float = 1.0  # 0.5-2.0
    timeout_s: Optional[int] = 30
    trace_id: Optional[str] = None


@dataclass
class TTSResponse:
    """Text-to-Speech response."""
    
    status: TTSStatus
    audio_data: Optional[bytes] = None  # Raw audio bytes
    audio_format: str = "wav"  # wav, mp3, ogg
    error_type: Optional[str] = None  # timeout | invalid_text | backend_unavailable
    metadata: Optional[Dict[str, Any]] = None


class TTSBackend(ABC):
    """
    Abstract TTS boundary.
    Agent code must depend ONLY on this interface.
    """

    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize text to audio.
        
        Args:
            request: TTSRequest with text and optional parameters
            
        Returns:
            TTSResponse with audio data or explicit error status
        """
        raise NotImplementedError
