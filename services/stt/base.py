"""
Speech-to-Text (STT) abstract interface.

Role: Audio → normalized text transformation only.

Rules:
- Pure transformation (no state mutation)
- No intent inference
- No memory access
- Failure → request user repetition (no crash)
- All failures are explicit and typed
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal


STTStatus = Literal["success", "recoverable_error", "fatal_error"]


@dataclass
class STTRequest:
    """Speech-to-Text request."""
    
    audio_data: bytes  # Raw audio bytes
    audio_format: str = "wav"  # wav, mp3, ogg, etc.
    language: str = "en"
    timeout_s: Optional[int] = 30
    trace_id: Optional[str] = None


@dataclass
class STTResponse:
    """Speech-to-Text response."""
    
    status: STTStatus
    text: Optional[str] = None  # Normalized text output
    confidence: Optional[float] = None  # 0.0-1.0
    error_type: Optional[str] = None  # timeout | invalid_audio | backend_unavailable
    metadata: Optional[Dict[str, Any]] = None


class STTBackend(ABC):
    """
    Abstract STT boundary.
    Agent code must depend ONLY on this interface.
    """

    @abstractmethod
    def transcribe(self, request: STTRequest) -> STTResponse:
        """
        Transcribe audio to text.
        
        Args:
            request: STTRequest with audio data and optional parameters
            
        Returns:
            STTResponse with text or explicit error status
        """
        raise NotImplementedError
