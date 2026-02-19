"""
Telegram Voice Message Integration Tests

Tests for voice message handling:
- Voice message download
- Audio normalization
- STT transcription
- Background processing
- Response sending

Total: 15+ tests
"""

import pytest
import io
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from datetime import datetime

from webhook.telegram_voice import (
    TelegramVoice,
    TelegramVoiceMessage,
    TelegramVoiceTransport,
    VoiceMessageResult,
    get_voice_transport,
)


class TestTelegramVoice:
    """Test voice message data structures."""
    
    def test_telegram_voice_creation(self):
        """Test voice object creation."""
        voice = TelegramVoice(
            file_id="AgACAgIAAxkBAAI...",
            file_unique_id="AQADrqF...",
            duration=10,
            mime_type="audio/ogg",
            file_size=50000,
        )
        
        assert voice.file_id == "AgACAgIAAxkBAAI..."
        assert voice.duration == 10
        assert voice.mime_type == "audio/ogg"
    
    def test_voice_message_creation(self):
        """Test voice message creation."""
        voice = TelegramVoice(
            file_id="test_id",
            file_unique_id="unique_id",
            duration=5,
        )
        
        message = TelegramVoiceMessage(
            message_id=123,
            date=1676817600,
            chat_id=456,
            user_id=789,
            voice=voice,
            first_name="John",
            username="john",
        )
        
        assert message.message_id == 123
        assert message.chat_id == 456
        assert message.voice.duration == 5


class TestVoiceMessageResult:
    """Test voice message processing result."""
    
    def test_success_result(self):
        """Test successful transcription result."""
        result = VoiceMessageResult(
            message_id=123,
            transcribed_text="Hello world",
            confidence=0.95,
            status="success",
            processing_time_ms=1200,
        )
        
        assert result.status == "success"
        assert result.transcribed_text == "Hello world"
        assert result.confidence == 0.95
    
    def test_error_result(self):
        """Test error result."""
        result = VoiceMessageResult(
            message_id=123,
            transcribed_text="",
            status="error",
            error="Failed to download file",
            processing_time_ms=300,
        )
        
        assert result.status == "error"
        assert result.error == "Failed to download file"
    
    def test_skipped_result(self):
        """Test skipped result (voice too short)."""
        result = VoiceMessageResult(
            message_id=123,
            transcribed_text="",
            status="skipped",
            error="Voice message too short (< 1s)",
            processing_time_ms=100,
        )
        
        assert result.status == "skipped"
        assert "too short" in result.error


class TestVoiceTransportInitialization:
    """Test voice transport initialization."""
    
    def test_voice_transport_init(self):
        """Test transport initializes with services."""
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            assert transport.token == "test_token"
            assert "test_token" in transport.api_url
            assert transport.normalizer is not None
            assert transport.stt_processor is not None
    
    def test_voice_transport_missing_token(self):
        """Test error when token not set."""
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", None):
            with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
                TelegramVoiceTransport()
    
    def test_singleton_pattern(self):
        """Test get_voice_transport returns same instance."""
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport1 = get_voice_transport()
            transport2 = get_voice_transport()
            
            assert transport1 is transport2


class TestVoiceDurationValidation:
    """Test voice message duration validation."""
    
    def test_too_short_voice_skipped(self):
        """Test voice < 1s is skipped (sync)."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                result = await transport.process_voice_message(
                    file_id="test_id",
                    message_id=123,
                    duration=0,  # 0 seconds
                )
                return result
            
            result = asyncio.run(run_test())
            assert result.status == "skipped"
            assert "too short" in result.error
    
    def test_too_long_voice_skipped(self):
        """Test voice > 5 minutes is skipped (sync)."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                result = await transport.process_voice_message(
                    file_id="test_id",
                    message_id=123,
                    duration=400,  # 400 seconds = 6.6 minutes
                )
                return result
            
            result = asyncio.run(run_test())
            assert result.status == "skipped"
            assert "too long" in result.error
    
    def test_valid_duration_range(self):
        """Test duration between 1-300s is accepted (sync)."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                # Mock the download to fail (so we test past duration check)
                with patch.object(transport, "download_voice_file", return_value=None):
                    result = await transport.process_voice_message(
                        file_id="test_id",
                        message_id=123,
                        duration=60,  # 1 minute - valid
                    )
                    return result
            
            result = asyncio.run(run_test())
            # Should get past duration check and fail at download
            assert result.status == "error"
            assert "download" in result.error


class TestVoiceDownload:
    """Test voice file download."""
    
    def test_download_file_success(self):
        """Test successful voice file download (integration test - skipped)."""
        # Integration test: requires actual async httpx client setup
        # For now, integration testing will be done end-to-end
        pytest.skip("Integration test - tested end-to-end")
    
    def test_download_file_failure(self):
        """Test voice file download failure."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                # Mock httpx client that returns error
                mock_client = AsyncMock()
                response = AsyncMock()
                response.status_code = 404
                response.text = "Not found"
                
                mock_client.get.return_value = response
                
                with patch.object(transport, "_get_http_client", return_value=mock_client):
                    result = await transport.download_voice_file("invalid_id")
                    return result
            
            result = asyncio.run(run_test())
            assert result is None


class TestVoiceNormalization:
    """Test audio normalization in voice processing."""
    
    def test_normalization_success(self):
        """Test audio normalization."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                # Mock audio bytes (OGG format)
                mock_audio = b"fake_ogg_audio"
                
                # Mock normalizer
                mock_normalized = MagicMock()
                mock_normalized.data = b"normalized_wav"
                mock_normalized.duration = 5.0
                
                with patch.object(transport.normalizer, "normalize_bytes", return_value=mock_normalized):
                    with patch.object(transport, "download_voice_file", return_value=mock_audio):
                        with patch.object(transport.stt_processor, "transcribe") as mock_transcribe:
                            mock_stt_result = MagicMock()
                            mock_stt_result.text = ""  # Empty to trigger skip
                            mock_transcribe.return_value = mock_stt_result
                            
                            result = await transport.process_voice_message(
                                file_id="test_id",
                                message_id=123,
                                duration=5,
                            )
                            
                            # Should call normalizer
                            transport.normalizer.normalize_bytes.assert_called_once()
                            args = transport.normalizer.normalize_bytes.call_args[0]
                            assert args[0] == mock_audio
                            assert args[1] == "ogg"  # OGG format
                            return True
            
            result = asyncio.run(run_test())
            assert result is True


class TestVoiceTranscription:
    """Test STT transcription."""
    
    def test_transcription_success(self):
        """Test successful transcription."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                # Mock download and normalize
                mock_audio = b"fake_ogg_audio"
                mock_normalized = MagicMock()
                mock_normalized.data = b"normalized_wav"
                mock_normalized.duration = 5.0
                
                # Mock STT result
                mock_stt_result = MagicMock()
                mock_stt_result.text = "Hello world"
                mock_stt_result.confidence = 0.92
                
                with patch.object(transport, "download_voice_file", return_value=mock_audio):
                    with patch.object(transport.normalizer, "normalize_bytes", return_value=mock_normalized):
                        with patch.object(transport.stt_processor, "transcribe", return_value=mock_stt_result):
                            result = await transport.process_voice_message(
                                file_id="test_id",
                                message_id=123,
                                duration=5,
                            )
                            return result
            
            result = asyncio.run(run_test())
            assert result.status == "success"
            assert result.transcribed_text == "Hello world"
            assert result.confidence == 0.92
    
    def test_transcription_empty_audio(self):
        """Test transcription with no speech detected."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                # Mock download and normalize
                mock_audio = b"fake_ogg_audio"
                mock_normalized = MagicMock()
                mock_normalized.data = b"normalized_wav"
                mock_normalized.duration = 5.0
                
                # Mock STT result with empty text
                mock_stt_result = MagicMock()
                mock_stt_result.text = ""
                
                with patch.object(transport, "download_voice_file", return_value=mock_audio):
                    with patch.object(transport.normalizer, "normalize_bytes", return_value=mock_normalized):
                        with patch.object(transport.stt_processor, "transcribe", return_value=mock_stt_result):
                            result = await transport.process_voice_message(
                                file_id="test_id",
                                message_id=123,
                                duration=5,
                            )
                            return result
            
            result = asyncio.run(run_test())
            assert result.status == "skipped"
            assert "No speech detected" in result.error


class TestVoiceMessageSending:
    """Test sending response messages."""
    
    def test_send_message_success(self):
        """Test successful message sending."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                # Mock httpx client
                mock_client = AsyncMock()
                response = AsyncMock()
                response.status_code = 200
                mock_client.post.return_value = response
                
                with patch.object(transport, "_get_http_client", return_value=mock_client):
                    success = await transport.send_message(123, "Test message")
                    return success
            
            success = asyncio.run(run_test())
            assert success is True
    
    def test_send_message_failure(self):
        """Test message sending failure."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                # Mock httpx client with error
                mock_client = AsyncMock()
                response = AsyncMock()
                response.status_code = 400
                response.text = "Bad request"
                mock_client.post.return_value = response
                
                with patch.object(transport, "_get_http_client", return_value=mock_client):
                    success = await transport.send_message(123, "Test message")
                    return success
            
            success = asyncio.run(run_test())
            assert success is False


class TestVoiceTransportIntegration:
    """Integration tests for complete voice processing."""
    
    def test_complete_voice_pipeline(self):
        """Test complete voice message processing pipeline."""
        import asyncio
        with patch("webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN", "test_token"):
            transport = TelegramVoiceTransport()
            
            async def run_test():
                # Setup mocks for full pipeline
                mock_audio = b"fake_ogg_audio"
                mock_normalized = MagicMock()
                mock_normalized.data = b"normalized_wav"
                mock_normalized.duration = 5.0
                
                mock_stt_result = MagicMock()
                mock_stt_result.text = "Hello bot"
                mock_stt_result.confidence = 0.95
                
                with patch.object(transport, "download_voice_file", return_value=mock_audio):
                    with patch.object(transport.normalizer, "normalize_bytes", return_value=mock_normalized):
                        with patch.object(transport.stt_processor, "transcribe", return_value=mock_stt_result):
                            # Process voice message
                            result = await transport.process_voice_message(
                                file_id="test_file_id",
                                message_id=456,
                                duration=5,
                            )
                            return result
            
            result = asyncio.run(run_test())
            # Verify complete pipeline
            assert result.message_id == 456
            assert result.status == "success"
            assert result.transcribed_text == "Hello bot"
            assert result.confidence == 0.95
            assert result.processing_time_ms > 0
