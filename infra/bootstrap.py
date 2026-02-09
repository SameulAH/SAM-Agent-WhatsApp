"""
Infrastructure initialization and bootstrap.

Singleton pattern for creating all service backends from configuration.
"""

from typing import Optional

from inference import ModelBackend
from services.stt import STTBackend
from services.tts import TTSBackend
from agent.memory import LongTermMemoryStore

from .config import InfraConfig, get_config


class InfraBootstrap:
    """
    Bootstrap infrastructure based on configuration.
    
    Singleton pattern - single instance per process.
    """
    
    _instance: Optional["InfraBootstrap"] = None
    
    def __init__(self, config: Optional[InfraConfig] = None):
        """Initialize bootstrap with configuration."""
        self.config = config or get_config()
        self.llm_backend = self.config.create_llm_backend()
        self.stt_backend = self.config.create_stt_backend()
        self.tts_backend = self.config.create_tts_backend()
        self.ltm_backend = self.config.create_ltm_backend()
    
    @classmethod
    def get_instance(cls, config: Optional[InfraConfig] = None) -> "InfraBootstrap":
        """
        Get singleton instance.
        
        Args:
            config: Optional custom configuration (only used first time)
            
        Returns:
            Singleton InfraBootstrap instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None
    
    def get_llm_backend(self) -> ModelBackend:
        """Get LLM backend."""
        return self.llm_backend
    
    def get_stt_backend(self) -> Optional[STTBackend]:
        """Get STT backend (or None if disabled)."""
        return self.stt_backend
    
    def get_tts_backend(self) -> Optional[TTSBackend]:
        """Get TTS backend (or None if disabled)."""
        return self.tts_backend
    
    def get_ltm_backend(self) -> LongTermMemoryStore:
        """Get LTM backend."""
        return self.ltm_backend
    
    def __repr__(self) -> str:
        """String representation showing configured backends."""
        return (
            f"InfraBootstrap(llm={self.config.llm_backend}, "
            f"stt={self.config.stt_backend if self.config.stt_enabled else 'disabled'}, "
            f"tts={self.config.tts_backend if self.config.tts_enabled else 'disabled'}, "
            f"ltm={self.config.ltm_backend})"
        )


def bootstrap_infrastructure(config: Optional[InfraConfig] = None) -> InfraBootstrap:
    """
    Bootstrap all infrastructure backends.
    
    Args:
        config: Optional custom configuration
        
    Returns:
        InfraBootstrap instance with all backends initialized
    """
    return InfraBootstrap.get_instance(config)
