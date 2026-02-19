"""
Speech-to-Text (STT) Integration

Supports multiple STT backends:
  - Whisper (OpenAI) - Recommended, works offline
  - Coqui STT - Lightweight, open-source

Configuration via .env:
  STT_ENABLED=true
  STT_BACKEND=whisper|coqui
  WHISPER_MODEL=base|small|medium|large
  WHISPER_DEVICE=cpu|cuda
"""

import logging
from typing import Optional
from dataclasses import dataclass

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """STT processing result."""
    text: str  # Transcribed text
    language: str  # Detected language (e.g., "en")
    confidence: Optional[float] = None  # Confidence score (0-1)
    backend: str = "whisper"  # Which backend was used


class STTProcessor:
    """
    Speech-to-Text processor.
    
    Converts audio to text using configured backend.
    """
    
    # Model sizes for Whisper
    WHISPER_MODELS = {
        "tiny": 39,      # 39M params
        "base": 74,      # 74M params
        "small": 244,    # 244M params
        "medium": 769,   # 769M params
        "large": 1550,   # 1.5B params
    }
    
    def __init__(
        self,
        backend: str = "whisper",
        model_name: str = "base",
        device: str = "cpu",
        enabled: bool = True,
    ):
        """
        Initialize STT processor.
        
        Args:
            backend: STT backend ("whisper" or "coqui")
            model_name: Model to use (whisper: tiny/small/base/medium/large)
            device: Device to use ("cpu" or "cuda")
            enabled: Whether STT is enabled
        """
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.enabled = enabled
        self.model = None
        
        if not enabled:
            logger.info("STT disabled")
            return
        
        if backend == "whisper":
            self._init_whisper()
        else:
            logger.warning(f"Unknown STT backend: {backend}")
            self.enabled = False
    
    def _init_whisper(self):
        """Initialize Whisper model."""
        if not WHISPER_AVAILABLE:
            logger.error(
                "Whisper not available. Install with: pip install openai-whisper"
            )
            self.enabled = False
            return
        
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(
                self.model_name,
                device=self.device,
            )
            logger.info(f"Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            self.enabled = False
    
    def transcribe(self, audio_bytes: bytes) -> STTResult:
        """
        Transcribe audio to text.
        
        Args:
            audio_bytes: Audio data (WAV format)
            
        Returns:
            STTResult with transcribed text
            
        Raises:
            ValueError: If STT not enabled or transcription fails
        """
        if not self.enabled:
            raise ValueError("STT not enabled")
        
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Audio data is empty")
        
        if self.backend == "whisper":
            return self._transcribe_whisper(audio_bytes)
        
        raise ValueError(f"Unknown backend: {self.backend}")
    
    def _transcribe_whisper(self, audio_bytes: bytes) -> STTResult:
        """Transcribe using Whisper."""
        import io
        import tempfile
        
        try:
            # Create temporary file (Whisper needs file path)
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            try:
                logger.info(f"Transcribing with Whisper ({self.model_name})")
                
                # Transcribe
                result = self.model.transcribe(
                    tmp_path,
                    language="en",  # Can be None for auto-detection
                    fp16=self.device == "cuda",
                )
                
                text = result.get("text", "").strip()
                language = result.get("language", "en")
                
                logger.info(f"Transcribed: {text[:50]}...")
                
                return STTResult(
                    text=text,
                    language=language,
                    confidence=None,  # Whisper doesn't provide per-segment confidence
                    backend="whisper",
                )
            
            finally:
                # Clean up temp file
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}", exc_info=True)
            raise ValueError(f"Transcription failed: {str(e)}")


# Global STT processor (lazy initialized)
_stt_processor = None


def get_stt_processor(
    backend: str = "whisper",
    model_name: str = "base",
    device: str = "cpu",
    enabled: bool = True,
) -> STTProcessor:
    """
    Get or create STT processor.
    
    Args:
        backend: STT backend
        model_name: Model name
        device: Device to use
        enabled: Whether to enable STT
        
    Returns:
        STTProcessor instance
    """
    global _stt_processor
    
    # Allow reinitializing with different config
    if _stt_processor is None:
        _stt_processor = STTProcessor(
            backend=backend,
            model_name=model_name,
            device=device,
            enabled=enabled,
        )
    
    return _stt_processor


def create_stt_processor_from_config() -> STTProcessor:
    """
    Create STT processor from .env configuration.
    
    Environment variables:
      STT_ENABLED (bool): Enable STT
      STT_BACKEND (str): whisper or coqui
      WHISPER_MODEL (str): tiny/small/base/medium/large
      WHISPER_DEVICE (str): cpu or cuda
    
    Returns:
        STTProcessor instance
    """
    import os
    
    enabled = os.getenv("STT_ENABLED", "false").lower() == "true"
    backend = os.getenv("STT_BACKEND", "whisper")
    model_name = os.getenv("WHISPER_MODEL", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    
    logger.info(
        f"Creating STT processor: enabled={enabled}, backend={backend}, "
        f"model={model_name}, device={device}"
    )
    
    return get_stt_processor(
        backend=backend,
        model_name=model_name,
        device=device,
        enabled=enabled,
    )
