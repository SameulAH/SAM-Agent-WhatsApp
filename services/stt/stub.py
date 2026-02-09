"""
Stub STT backend for testing and offline development.

Deterministic, fast, and never fails silently.
"""

from .base import STTBackend, STTRequest, STTResponse


class StubSTTBackend(STTBackend):
    """
    Deterministic fake STT for testing and CI.
    
    Converts audio to a fixed response based on audio length.
    """

    def transcribe(self, request: STTRequest) -> STTResponse:
        """
        Transcribe audio to text (stubbed).
        
        Args:
            request: STTRequest with audio data
            
        Returns:
            STTResponse with deterministic output
        """
        # Stub behavior: return text based on audio length
        audio_len = len(request.audio_data)
        
        if audio_len == 0:
            return STTResponse(
                status="recoverable_error",
                error_type="invalid_audio",
                metadata={
                    "backend": "stub_stt",
                    "trace_id": request.trace_id
                }
            )
        
        # Deterministic response based on audio size
        word_count = max(1, audio_len // 100)
        words = [f"word_{i}" for i in range(word_count)]
        text = " ".join(words)
        
        return STTResponse(
            status="success",
            text=text,
            confidence=0.99,
            metadata={
                "backend": "stub_stt",
                "trace_id": request.trace_id,
                "audio_length": audio_len
            }
        )


class NoOpSTTBackend(STTBackend):
    """STT that always fails gracefully (for disabled mode)."""

    def transcribe(self, request: STTRequest) -> STTResponse:
        """Return disabled response."""
        return STTResponse(
            status="fatal_error",
            error_type="backend_unavailable",
            metadata={
                "backend": "noop_stt",
                "trace_id": request.trace_id,
                "reason": "STT disabled"
            }
        )
