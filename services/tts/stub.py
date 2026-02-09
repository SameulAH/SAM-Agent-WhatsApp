"""
Stub TTS backend for testing and offline development.

Deterministic, fast, and never fails silently.
"""

from .base import TTSBackend, TTSRequest, TTSResponse


class StubTTSBackend(TTSBackend):
    """
    Deterministic fake TTS for testing and CI.
    
    Converts text to deterministic audio bytes.
    """

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize text to audio (stubbed).
        
        Args:
            request: TTSRequest with text
            
        Returns:
            TTSResponse with deterministic audio bytes
        """
        if not request.text or len(request.text) == 0:
            return TTSResponse(
                status="recoverable_error",
                error_type="invalid_text",
                metadata={
                    "backend": "stub_tts",
                    "trace_id": request.trace_id
                }
            )
        
        # Stub audio: deterministic bytes based on text length
        audio_len = len(request.text) * 10  # 10 bytes per character
        audio_data = bytes([i % 256 for i in range(audio_len)])
        
        return TTSResponse(
            status="success",
            audio_data=audio_data,
            audio_format="wav",
            metadata={
                "backend": "stub_tts",
                "trace_id": request.trace_id,
                "text_length": len(request.text)
            }
        )


class NoOpTTSBackend(TTSBackend):
    """TTS that always fails gracefully (for disabled mode)."""

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Return disabled response."""
        return TTSResponse(
            status="fatal_error",
            error_type="backend_unavailable",
            metadata={
                "backend": "noop_tts",
                "trace_id": request.trace_id,
                "reason": "TTS disabled"
            }
        )
