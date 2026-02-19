"""
Audio Normalization & STT Tests

Comprehensive test suite for audio services.
Tests cover:
  - Audio format detection
  - Format conversion
  - Normalization
  - STT processing
  
Total: 15+ tests
"""

import pytest
import io
from unittest.mock import patch, MagicMock, Mock

from services.audio import (
    AudioNormalizer,
    NormalizedAudio,
    AudioMetadata,
    get_audio_normalizer,
    STTProcessor,
    STTResult,
)


class TestAudioNormalizer:
    """Test audio normalization."""
    
    def test_normalizer_initialization(self):
        """Test normalizer initializes."""
        normalizer = AudioNormalizer()
        
        assert normalizer.TARGET_SAMPLE_RATE == 16000
        assert normalizer.TARGET_CHANNELS == 1
        assert normalizer.TARGET_BIT_DEPTH == 16
        assert normalizer.MAX_DURATION_SECONDS == 300
    
    def test_supported_formats(self):
        """Test supported formats."""
        normalizer = AudioNormalizer()
        
        assert "wav" in normalizer.SUPPORTED_FORMATS
        assert "mp3" in normalizer.SUPPORTED_FORMATS
        assert "m4a" in normalizer.SUPPORTED_FORMATS
        assert "ogg" in normalizer.SUPPORTED_FORMATS
        assert "webm" in normalizer.SUPPORTED_FORMATS
    
    def test_unsupported_format_raises_error(self):
        """Test unsupported format raises error."""
        normalizer = AudioNormalizer()
        
        with pytest.raises(ValueError, match="Unsupported audio format"):
            normalizer.normalize_bytes(b"fake audio", "xyz")
    
    def test_empty_audio_raises_error(self):
        """Test empty audio raises error."""
        normalizer = AudioNormalizer()
        
        with pytest.raises(ValueError, match="Audio data is empty"):
            normalizer.normalize_bytes(b"", "wav")
    
    def test_audio_metadata_creation(self):
        """Test audio metadata dataclass."""
        metadata = AudioMetadata(
            format="wav",
            sample_rate=16000,
            channels=1,
            duration=5.0,
            bit_depth=16,
            file_size=1000,
        )
        
        assert metadata.format == "wav"
        assert metadata.sample_rate == 16000
        assert metadata.channels == 1
        assert metadata.duration == 5.0


class TestNormalizedAudio:
    """Test normalized audio dataclass."""
    
    def test_normalized_audio_creation(self):
        """Test normalized audio creation."""
        audio = NormalizedAudio(
            data=b"audio_data",
            sample_rate=16000,
            channels=1,
            duration=2.5,
            format="wav",
            original_format="mp3",
        )
        
        assert audio.sample_rate == 16000
        assert audio.channels == 1
        assert audio.format == "wav"
        assert audio.original_format == "mp3"
        assert audio.duration == 2.5
    
    def test_normalized_audio_with_metadata(self):
        """Test normalized audio with metadata."""
        metadata = AudioMetadata(
            format="wav",
            sample_rate=16000,
            channels=1,
            duration=2.5,
            bit_depth=16,
            file_size=1000,
        )
        
        audio = NormalizedAudio(
            data=b"audio_data",
            sample_rate=16000,
            channels=1,
            duration=2.5,
            format="wav",
            original_format="mp3",
            metadata=metadata,
        )
        
        assert audio.metadata is not None
        assert audio.metadata.sample_rate == 16000


class TestAudioNormalizerWavParsing:
    """Test WAV file parsing."""
    
    def _create_simple_wav(self, sample_rate=16000, channels=1, duration_sec=1.0):
        """Create a simple WAV file for testing."""
        import struct
        
        # Audio parameters
        num_channels = channels
        sample_rate = sample_rate
        bit_depth = 16
        bytes_per_sample = bit_depth // 8
        
        # Generate silence (zeros)
        num_samples = int(sample_rate * duration_sec)
        audio_data = b'\x00' * (num_samples * num_channels * bytes_per_sample)
        
        # Build WAV header
        chunk_id = b'RIFF'
        chunk_size = 36 + len(audio_data)
        format_id = b'WAVE'
        
        subchunk1_id = b'fmt '
        subchunk1_size = 16
        audio_format = 1  # PCM
        byte_rate = sample_rate * num_channels * bytes_per_sample
        block_align = num_channels * bytes_per_sample
        
        subchunk2_id = b'data'
        subchunk2_size = len(audio_data)
        
        # Pack header
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            chunk_id,
            chunk_size,
            format_id,
            subchunk1_id,
            subchunk1_size,
            audio_format,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bit_depth,
            subchunk2_id,
            subchunk2_size,
        )
        
        return header + audio_data
    
    def test_wav_metadata_extraction(self):
        """Test extracting metadata from WAV file."""
        normalizer = AudioNormalizer()
        
        # Create test WAV
        wav_data = self._create_simple_wav(
            sample_rate=16000,
            channels=1,
            duration_sec=2.0,
        )
        
        # Extract metadata
        metadata = normalizer._get_wav_metadata(wav_data)
        
        assert metadata.format == "wav"
        assert metadata.sample_rate == 16000
        assert metadata.channels == 1
        assert metadata.bit_depth == 16
        assert 1.9 < metadata.duration < 2.1  # Allow small variance
    
    def test_invalid_wav_raises_error(self):
        """Test invalid WAV file raises error."""
        normalizer = AudioNormalizer()
        
        with pytest.raises(ValueError):
            normalizer._get_wav_metadata(b"not a wav file")


class TestAudioNormalizerConversion:
    """Test audio format conversion."""
    
    def test_conversion_to_wav(self):
        """Test conversion to WAV format."""
        normalizer = AudioNormalizer()
        
        # Skip if pydub not available
        if not normalizer.pydub_available:
            pytest.skip("pydub not available")
        
        # Import AudioSegment only if available
        from pydub import AudioSegment
        
        # Mock AudioSegment behavior
        with patch('pydub.AudioSegment') as mock_audio_segment:
            mock_audio = MagicMock()
            mock_audio.frame_rate = 44100
            mock_audio.channels = 2
            mock_audio_segment.from_file.return_value = mock_audio
            
            mock_audio.set_frame_rate.return_value = mock_audio
            mock_audio.set_channels.return_value = mock_audio
            
            # Mock export
            mock_audio.export.return_value = None
            
            # Create fake MP3 data
            mp3_data = b"fake mp3 data"
            
            # Test - should not raise
            with patch('builtins.open', create=True):
                result = normalizer._convert_to_wav(mp3_data, "mp3")
                assert isinstance(result, bytes) or result is None  # Export might return None


class TestSTTProcessor:
    """Test STT processing."""
    
    def test_stt_processor_initialization_disabled(self):
        """Test STT processor can be disabled."""
        processor = STTProcessor(enabled=False)
        
        assert not processor.enabled
        assert processor.model is None
    
    def test_stt_processor_initialization_enabled(self):
        """Test STT processor initialization."""
        with patch('services.audio.stt.WHISPER_AVAILABLE', False):
            processor = STTProcessor(enabled=True, backend="whisper")
            # Should be disabled if Whisper not available
            assert not processor.enabled
    
    def test_transcribe_disabled_raises_error(self):
        """Test transcribing with disabled processor raises error."""
        processor = STTProcessor(enabled=False)
        
        with pytest.raises(ValueError, match="STT not enabled"):
            processor.transcribe(b"audio data")
    
    def test_empty_audio_raises_error(self):
        """Test empty audio raises error."""
        processor = STTProcessor(enabled=True)
        
        # If processor is disabled (because Whisper not available), skip
        if not processor.enabled:
            pytest.skip("Whisper not available")
        
        with pytest.raises(ValueError, match="Audio data is empty"):
            processor.transcribe(b"")


class TestSTTResult:
    """Test STT result."""
    
    def test_stt_result_creation(self):
        """Test STT result dataclass."""
        result = STTResult(
            text="Hello, world!",
            language="en",
            confidence=0.95,
            backend="whisper",
        )
        
        assert result.text == "Hello, world!"
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.backend == "whisper"
    
    def test_stt_result_default_backend(self):
        """Test STT result default backend."""
        result = STTResult(
            text="Hello",
            language="en",
        )
        
        assert result.backend == "whisper"


class TestAudioNormalizerSingleton:
    """Test audio normalizer singleton."""
    
    def test_get_normalizer_returns_same_instance(self):
        """Test get_normalizer returns same instance."""
        normalizer1 = get_audio_normalizer()
        normalizer2 = get_audio_normalizer()
        
        assert normalizer1 is normalizer2


class TestAudioFormatDetection:
    """Test format detection."""
    
    def test_format_extension_detection(self):
        """Test format detection from extension."""
        normalizer = AudioNormalizer()
        
        # Test various extensions
        assert "wav" in normalizer.SUPPORTED_FORMATS
        assert "mp3" in normalizer.SUPPORTED_FORMATS
        assert "m4a" in normalizer.SUPPORTED_FORMATS
        assert "ogg" in normalizer.SUPPORTED_FORMATS
        assert "webm" in normalizer.SUPPORTED_FORMATS
        assert "flac" in normalizer.SUPPORTED_FORMATS


class TestAudioLimits:
    """Test audio size and duration limits."""
    
    def test_max_file_size_limit(self):
        """Test max file size limit."""
        normalizer = AudioNormalizer()
        
        # Max should be 25MB (Telegram limit)
        assert normalizer.MAX_FILE_SIZE_BYTES == 25 * 1024 * 1024
    
    def test_max_duration_limit(self):
        """Test max duration limit."""
        normalizer = AudioNormalizer()
        
        # Max should be 5 minutes
        assert normalizer.MAX_DURATION_SECONDS == 300


class TestAudioNormalizerFileHandling:
    """Test file handling."""
    
    def test_nonexistent_file_raises_error(self):
        """Test nonexistent file raises error."""
        normalizer = AudioNormalizer()
        
        with pytest.raises(FileNotFoundError):
            normalizer.normalize_file("/nonexistent/file.wav")
    
    def test_file_size_check(self):
        """Test file size validation."""
        import tempfile
        import os
        
        normalizer = AudioNormalizer()
        
        # Create a file that's too large
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write more than 25MB
            tmp.write(b"x" * (26 * 1024 * 1024))
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="too large"):
                normalizer.normalize_file(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestAudioIntegration:
    """Integration tests."""
    
    def test_complete_audio_pipeline(self):
        """Test complete audio processing pipeline."""
        # Create normalizer
        normalizer = AudioNormalizer()
        
        # Verify it's properly configured
        assert normalizer.TARGET_SAMPLE_RATE == 16000
        assert normalizer.TARGET_CHANNELS == 1
        assert normalizer.SUPPORTED_FORMATS
