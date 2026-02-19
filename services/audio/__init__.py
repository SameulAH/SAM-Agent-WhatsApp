"""
Audio Services Module

Handles audio processing:
  - Format normalization
  - Speech-to-Text (STT)
  - Audio quality checking
"""

from .normalizer import (
    AudioNormalizer,
    NormalizedAudio,
    AudioMetadata,
    get_audio_normalizer,
)

from .stt import (
    STTProcessor,
    STTResult,
    get_stt_processor,
    create_stt_processor_from_config,
)

__all__ = [
    "AudioNormalizer",
    "NormalizedAudio",
    "AudioMetadata",
    "get_audio_normalizer",
    "STTProcessor",
    "STTResult",
    "get_stt_processor",
    "create_stt_processor_from_config",
]
