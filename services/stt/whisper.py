"""
Local Whisper STT backend using OpenAI's whisper-cpp.

Requires: whisper-cpp or openai-whisper Python package.
Model: Downloaded on first use (requires internet once).
"""

import io
from typing import Optional
from .base import STTBackend, STTRequest, STTResponse

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class WhisperLocalSTTBackend(STTBackend):
    """
    Local Whisper STT backend using openai-whisper.
    
    Requires: pip install openai-whisper
    Model is downloaded on first use and cached locally.
    """

    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        Initialize Whisper backend.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: "cpu" or "cuda"
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "whisper not installed. Install with: pip install openai-whisper"
            )
        
        self.model_name = model_name
        self.device = device
        self.model = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self.model is None:
            self.model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(self, request: STTRequest) -> STTResponse:
        """
        Transcribe audio using Whisper.
        
        Args:
            request: STTRequest with audio data
            
        Returns:
            STTResponse with transcribed text or error
        """
        try:
            self._load_model()
            
            # Whisper expects file path or bytes-like object
            # Create in-memory file-like object
            audio_file = io.BytesIO(request.audio_data)
            
            result = self.model.transcribe(
                audio_file,
                language=request.language,
                verbose=False
            )
            
            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.95)
            
            if not text:
                return STTResponse(
                    status="recoverable_error",
                    error_type="invalid_audio",
                    metadata={
                        "backend": "whisper_local",
                        "model": self.model_name,
                        "trace_id": request.trace_id
                    }
                )
            
            return STTResponse(
                status="success",
                text=text,
                confidence=confidence,
                metadata={
                    "backend": "whisper_local",
                    "model": self.model_name,
                    "trace_id": request.trace_id
                }
            )
        
        except Exception as e:
            return STTResponse(
                status="fatal_error",
                error_type="backend_unavailable",
                metadata={
                    "backend": "whisper_local",
                    "model": self.model_name,
                    "error": str(e),
                    "trace_id": request.trace_id
                }
            )
