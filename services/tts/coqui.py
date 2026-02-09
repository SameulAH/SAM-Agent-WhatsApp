"""
Coqui TTS backend using open source TTS.

Requires: pip install TTS
Models are downloaded on first use (requires internet once).
Supports multiple languages and voices.
"""

import io
from typing import Optional
from .base import TTSBackend, TTSRequest, TTSResponse

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False


class CoquiTTSBackend(TTSBackend):
    """
    Coqui TTS backend using open-source TTS models.
    
    Requires: pip install TTS
    Models are cached locally after first download.
    """

    def __init__(self, model_name: str = "tts_models/en/ljspeech/glow-tts", device: str = "cpu"):
        """
        Initialize Coqui TTS backend.
        
        Args:
            model_name: TTS model ID (see TTS.list_models())
            device: "cpu" or "cuda"
        """
        if not COQUI_AVAILABLE:
            raise ImportError(
                "TTS not installed. Install with: pip install TTS"
            )
        
        self.model_name = model_name
        self.device = device
        self.model = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self.model is None:
            self.model = TTS(model_name=self.model_name, device=self.device, progress_bar=False)

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize text to audio using Coqui TTS.
        
        Args:
            request: TTSRequest with text
            
        Returns:
            TTSResponse with audio data or error
        """
        try:
            if not request.text or len(request.text) == 0:
                return TTSResponse(
                    status="recoverable_error",
                    error_type="invalid_text",
                    metadata={
                        "backend": "coqui_tts",
                        "model": self.model_name,
                        "trace_id": request.trace_id
                    }
                )
            
            self._load_model()
            
            # Synthesize to file in memory
            output_path = "/tmp/tts_output.wav"
            
            self.model.tts_to_file(
                text=request.text,
                file_path=output_path
            )
            
            # Read audio bytes
            with open(output_path, "rb") as f:
                audio_data = f.read()
            
            if not audio_data:
                return TTSResponse(
                    status="recoverable_error",
                    error_type="invalid_text",
                    metadata={
                        "backend": "coqui_tts",
                        "model": self.model_name,
                        "trace_id": request.trace_id
                    }
                )
            
            return TTSResponse(
                status="success",
                audio_data=audio_data,
                audio_format="wav",
                metadata={
                    "backend": "coqui_tts",
                    "model": self.model_name,
                    "trace_id": request.trace_id,
                    "text_length": len(request.text)
                }
            )
        
        except Exception as e:
            return TTSResponse(
                status="fatal_error",
                error_type="backend_unavailable",
                metadata={
                    "backend": "coqui_tts",
                    "model": self.model_name,
                    "error": str(e),
                    "trace_id": request.trace_id
                }
            )
