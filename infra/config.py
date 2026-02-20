"""
Infrastructure configuration system.

Environment-based backend selection with sensible defaults.
All components default to free, local-first stack.
"""

import os
from typing import Optional, Literal
from dataclasses import dataclass

from inference import ModelBackend, StubModelBackend, OllamaModelBackend
from services.stt import STTBackend, StubSTTBackend, WhisperLocalSTTBackend
from services.tts import TTSBackend, StubTTSBackend, CoquiTTSBackend
from agent.memory import LongTermMemoryStore, StubLongTermMemoryStore, QdrantLongTermMemoryStore


LLMBackendType = Literal["stub", "ollama", "groq"]
STTBackendType = Literal["stub", "whisper"]
TTSBackendType = Literal["stub", "coqui", "elevenlabs"]
LTMBackendType = Literal["stub", "qdrant"]


@dataclass
class InfraConfig:
    """Infrastructure configuration from environment."""
    
    # LLM
    llm_backend: LLMBackendType
    ollama_model: str
    ollama_base_url: str
    
    # STT
    stt_enabled: bool
    stt_backend: STTBackendType
    whisper_model: str
    whisper_device: str
    
    # TTS
    tts_enabled: bool
    tts_backend: TTSBackendType
    coqui_model: str
    coqui_device: str
    xtts_speaker_wav: Optional[str]   # path to WAV for voice cloning (optional)
    xtts_language: str                 # BCP-47 language code
    
    # LTM
    ltm_backend: LTMBackendType
    qdrant_url: str
    qdrant_api_key: Optional[str]
    
    @classmethod
    def from_env(cls) -> "InfraConfig":
        """
        Load configuration from environment variables.
        
        Defaults prioritize free, local-first stack:
        - LLM: ollama (phi by default)
        - STT: whisper (local)
        - TTS: coqui (local)
        - LTM: qdrant (local via Docker)
        """
        return cls(
            # LLM Configuration
            llm_backend=os.getenv("LLM_BACKEND", "ollama"),  # type: ignore
            ollama_model=os.getenv("OLLAMA_MODEL", "phi"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            
            # STT Configuration
            stt_enabled=os.getenv("STT_ENABLED", "true").lower() == "true",
            stt_backend=os.getenv("STT_BACKEND", "whisper"),  # type: ignore
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            
            # TTS Configuration
            tts_enabled=os.getenv("TTS_ENABLED", "true").lower() == "true",
            tts_backend=os.getenv("TTS_BACKEND", "coqui"),  # type: ignore
            coqui_model=os.getenv("COQUI_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
            coqui_device=os.getenv("COQUI_DEVICE", "cpu"),
            xtts_speaker_wav=os.getenv("XTTS_SPEAKER_WAV") or None,
            xtts_language=os.getenv("XTTS_LANGUAGE", "en"),
            
            # LTM Configuration
            ltm_backend=os.getenv("LTM_BACKEND", "qdrant"),  # type: ignore
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        )
    
    def create_llm_backend(self) -> ModelBackend:
        """Create LLM backend instance based on configuration."""
        if self.llm_backend == "ollama":
            return OllamaModelBackend(
                model_name=self.ollama_model,
                base_url=self.ollama_base_url
            )
        elif self.llm_backend == "stub":
            return StubModelBackend()
        else:
            # Default to ollama
            return OllamaModelBackend(
                model_name=self.ollama_model,
                base_url=self.ollama_base_url
            )
    
    def create_stt_backend(self) -> Optional[STTBackend]:
        """Create STT backend instance based on configuration."""
        if not self.stt_enabled:
            return None
        
        if self.stt_backend == "whisper":
            try:
                return WhisperLocalSTTBackend(
                    model_name=self.whisper_model,
                    device=self.whisper_device
                )
            except ImportError:
                # Whisper not installed, fall back to stub
                return StubSTTBackend()
        elif self.stt_backend == "stub":
            return StubSTTBackend()
        else:
            # Default to stub
            return StubSTTBackend()
    
    def create_tts_backend(self) -> Optional[TTSBackend]:
        """Create TTS backend instance based on configuration."""
        if not self.tts_enabled:
            return None
        
        if self.tts_backend == "coqui":
            try:
                return CoquiTTSBackend(
                    model_name=self.coqui_model,
                    device=self.coqui_device,
                    speaker_wav=self.xtts_speaker_wav,
                    language=self.xtts_language,
                )
            except ImportError:
                # Coqui not installed, fall back to stub
                return StubTTSBackend()
        elif self.tts_backend == "stub":
            return StubTTSBackend()
        else:
            # Default to stub
            return StubTTSBackend()
    
    def create_ltm_backend(self) -> LongTermMemoryStore:
        """Create LTM backend instance based on configuration."""
        if self.ltm_backend == "qdrant":
            return QdrantLongTermMemoryStore(
                qdrant_url=self.qdrant_url,
                collection_name="long_term_memory",
                vector_size=384
            )
        elif self.ltm_backend == "stub":
            return StubLongTermMemoryStore()
        else:
            # Default to qdrant
            return QdrantLongTermMemoryStore(
                qdrant_url=self.qdrant_url,
                collection_name="long_term_memory",
                vector_size=384
            )


def get_config() -> InfraConfig:
    """Get global infrastructure configuration."""
    return InfraConfig.from_env()
