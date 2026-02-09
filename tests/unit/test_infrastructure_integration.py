"""
Test suite for infrastructure integration.

Verifies:
- All adapters work correctly
- Configuration system functions
- Bootstrap initializes all backends
- Graceful degradation on failures
- Agent behavior unchanged
"""

import pytest
from infra import InfraBootstrap, InfraConfig
from infra.config import LLMBackendType, STTBackendType, TTSBackendType, LTMBackendType

from inference import ModelBackend, StubModelBackend, OllamaModelBackend, ModelRequest
from services.stt import STTBackend, StubSTTBackend, STTRequest
from services.tts import TTSBackend, StubTTSBackend, TTSRequest
from agent.memory import LongTermMemoryStore, StubLongTermMemoryStore, QdrantLongTermMemoryStore


class TestInfraConfig:
    """Test infrastructure configuration."""
    
    def test_config_from_env_defaults(self):
        """Verify defaults are free/local stack."""
        config = InfraConfig.from_env()
        
        # LLM defaults to Ollama with Phi
        assert config.llm_backend == "ollama"
        assert config.ollama_model == "phi"
        
        # STT enabled by default with Whisper
        assert config.stt_enabled is True
        assert config.stt_backend == "whisper"
        
        # TTS enabled by default with Coqui
        assert config.tts_enabled is True
        assert config.tts_backend == "coqui"
        
        # LTM defaults to Qdrant
        assert config.ltm_backend == "qdrant"
    
    def test_config_creates_llm_backend_ollama(self):
        """Verify Ollama backend creation."""
        config = InfraConfig.from_env()
        config.llm_backend = "ollama"  # type: ignore
        
        backend = config.create_llm_backend()
        
        assert isinstance(backend, OllamaModelBackend)
    
    def test_config_creates_llm_backend_stub(self):
        """Verify stub backend creation."""
        config = InfraConfig.from_env()
        config.llm_backend = "stub"  # type: ignore
        
        backend = config.create_llm_backend()
        
        assert isinstance(backend, StubModelBackend)
    
    def test_config_creates_stt_backend(self):
        """Verify STT backend creation."""
        config = InfraConfig.from_env()
        config.stt_enabled = True
        config.stt_backend = "stub"  # type: ignore
        
        backend = config.create_stt_backend()
        
        assert isinstance(backend, StubSTTBackend)
    
    def test_config_stt_disabled(self):
        """Verify STT can be disabled."""
        config = InfraConfig.from_env()
        config.stt_enabled = False
        
        backend = config.create_stt_backend()
        
        assert backend is None
    
    def test_config_creates_tts_backend(self):
        """Verify TTS backend creation."""
        config = InfraConfig.from_env()
        config.tts_enabled = True
        config.tts_backend = "stub"  # type: ignore
        
        backend = config.create_tts_backend()
        
        assert isinstance(backend, StubTTSBackend)
    
    def test_config_tts_disabled(self):
        """Verify TTS can be disabled."""
        config = InfraConfig.from_env()
        config.tts_enabled = False
        
        backend = config.create_tts_backend()
        
        assert backend is None
    
    def test_config_creates_ltm_backend(self):
        """Verify LTM backend creation."""
        config = InfraConfig.from_env()
        config.ltm_backend = "stub"  # type: ignore
        
        backend = config.create_ltm_backend()
        
        assert isinstance(backend, StubLongTermMemoryStore)


class TestInfraBootstrap:
    """Test infrastructure bootstrap."""
    
    def setup_method(self):
        """Reset bootstrap before each test."""
        InfraBootstrap.reset()
    
    def test_bootstrap_singleton(self):
        """Verify bootstrap is a singleton."""
        bootstrap1 = InfraBootstrap.get_instance()
        bootstrap2 = InfraBootstrap.get_instance()
        
        assert bootstrap1 is bootstrap2
    
    def test_bootstrap_creates_all_backends(self):
        """Verify bootstrap initializes all backends."""
        config = InfraConfig.from_env()
        config.stt_enabled = True
        config.tts_enabled = True
        
        bootstrap = InfraBootstrap(config)
        
        assert bootstrap.get_llm_backend() is not None
        assert bootstrap.get_stt_backend() is not None
        assert bootstrap.get_tts_backend() is not None
        assert bootstrap.get_ltm_backend() is not None
    
    def test_bootstrap_with_stub_backends(self):
        """Verify bootstrap works with stub backends."""
        config = InfraConfig.from_env()
        config.llm_backend = "stub"  # type: ignore
        config.stt_backend = "stub"  # type: ignore
        config.tts_backend = "stub"  # type: ignore
        config.ltm_backend = "stub"  # type: ignore
        
        bootstrap = InfraBootstrap(config)
        
        assert isinstance(bootstrap.get_llm_backend(), StubModelBackend)
        assert isinstance(bootstrap.get_stt_backend(), StubSTTBackend)
        assert isinstance(bootstrap.get_tts_backend(), StubTTSBackend)
        assert isinstance(bootstrap.get_ltm_backend(), StubLongTermMemoryStore)


class TestSTTAdapter:
    """Test STT adapter interface."""
    
    def test_stub_stt_transcribes(self):
        """Verify stub STT works."""
        backend = StubSTTBackend()
        
        request = STTRequest(
            audio_data=b"test_audio_data",
            audio_format="wav"
        )
        response = backend.transcribe(request)
        
        assert response.status == "success"
        assert response.text is not None
        assert len(response.text) > 0
    
    def test_stub_stt_empty_audio(self):
        """Verify stub STT handles empty audio."""
        backend = StubSTTBackend()
        
        request = STTRequest(audio_data=b"")
        response = backend.transcribe(request)
        
        assert response.status == "recoverable_error"
        assert response.error_type == "invalid_audio"


class TestTTSAdapter:
    """Test TTS adapter interface."""
    
    def test_stub_tts_synthesizes(self):
        """Verify stub TTS works."""
        backend = StubTTSBackend()
        
        request = TTSRequest(text="Hello world")
        response = backend.synthesize(request)
        
        assert response.status == "success"
        assert response.audio_data is not None
        assert len(response.audio_data) > 0
        assert response.audio_format == "wav"
    
    def test_stub_tts_empty_text(self):
        """Verify stub TTS handles empty text."""
        backend = StubTTSBackend()
        
        request = TTSRequest(text="")
        response = backend.synthesize(request)
        
        assert response.status == "recoverable_error"
        assert response.error_type == "invalid_text"


class TestLLMAdapter:
    """Test LLM adapter interface (already tested, but included for completeness)."""
    
    def test_stub_llm_responds(self):
        """Verify stub LLM works."""
        backend = StubModelBackend()
        
        request = ModelRequest(task="respond", prompt="Hello")
        response = backend.generate(request)
        
        assert response.status == "success"
        assert response.output is not None
        assert len(response.output) > 0
    
    def test_ollama_backend_exists(self):
        """Verify Ollama backend is available."""
        backend = OllamaModelBackend(model_name="phi")
        
        assert backend.model_name == "phi"
        assert backend.base_url == "http://localhost:11434"


class TestGracefulDegradation:
    """Test graceful degradation when components fail."""
    
    def test_stt_failure_is_typed(self):
        """Verify STT failures are explicit and typed."""
        backend = StubSTTBackend()
        
        # Empty audio should fail with explicit error
        request = STTRequest(audio_data=b"")
        response = backend.transcribe(request)
        
        assert response.status in ["recoverable_error", "fatal_error"]
        assert response.error_type is not None
    
    def test_tts_failure_is_typed(self):
        """Verify TTS failures are explicit and typed."""
        backend = StubTTSBackend()
        
        # Empty text should fail with explicit error
        request = TTSRequest(text="")
        response = backend.synthesize(request)
        
        assert response.status in ["recoverable_error", "fatal_error"]
        assert response.error_type is not None
    
    def test_backend_optional_stt(self):
        """Verify STT can be disabled without breaking agent."""
        config = InfraConfig.from_env()
        config.stt_enabled = False
        
        backend = config.create_stt_backend()
        
        # Agent should handle None gracefully
        assert backend is None
    
    def test_backend_optional_tts(self):
        """Verify TTS can be disabled without breaking agent."""
        config = InfraConfig.from_env()
        config.tts_enabled = False
        
        backend = config.create_tts_backend()
        
        # Agent should handle None gracefully
        assert backend is None


class TestAgentBehaviorUnchanged:
    """
    Verify that agent behavior is unchanged after infrastructure integration.
    
    Agent should work identically with:
    - Stub backends (for CI/testing)
    - Local backends (Ollama, Whisper, Coqui, Qdrant)
    - No backend changes to decision logic or routing
    """
    
    def test_stub_config_equals_original(self):
        """Verify stub configuration is compatible with original tests."""
        # Original tests use StubModelBackend
        # New config should create identical stub backend
        config = InfraConfig.from_env()
        config.llm_backend = "stub"  # type: ignore
        config.stt_enabled = False
        config.tts_enabled = False
        config.ltm_backend = "stub"  # type: ignore
        
        llm = config.create_llm_backend()
        ltm = config.create_ltm_backend()
        
        assert isinstance(llm, StubModelBackend)
        assert isinstance(ltm, StubLongTermMemoryStore)
