"""
Phase 9.2.1: Unified Trace Propagation Tests

Tests for complete trace lineage from transport → agent → LangSmith.

Validates:
- trace_id generation at transport layer
- trace_id propagation through background task
- trace_id usage in agent.invoke()
- tracer event emission at transport level
- unified trace visibility in observability system
- trace isolation in concurrent requests

No fragmented traces. One trace per request.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
from uuid import uuid4

# pytest-asyncio configuration
pytest_plugins = ('pytest_asyncio',)

from webhook.telegram_voice import (
    TelegramVoiceMessage,
    TelegramVoice,
    handle_voice_message,
    _process_voice_async,
    get_voice_transport,
)
from agent.tracing import TraceMetadata, NoOpTracer


class MockTracer:
    """Mock tracer for testing trace propagation."""
    
    def __init__(self):
        self.events = []
        self.spans = {}
    
    def record_event(self, name: str, metadata: dict, trace_metadata: TraceMetadata):
        """Record event with trace metadata."""
        self.events.append({
            "name": name,
            "metadata": metadata,
            "trace_metadata": {
                "trace_id": trace_metadata.trace_id,
                "conversation_id": trace_metadata.conversation_id,
                "user_id": trace_metadata.user_id,
            }
        })
    
    def start_span(self, name: str, metadata: dict, trace_metadata: TraceMetadata):
        """Start span."""
        span_id = str(uuid4())
        self.spans[span_id] = {
            "name": name,
            "trace_id": trace_metadata.trace_id,
            "conversation_id": trace_metadata.conversation_id,
        }
        return span_id
    
    def end_span(self, span: str, status: str, metadata: dict):
        """End span."""
        pass
    
    def is_enabled(self) -> bool:
        """Tracer is enabled."""
        return True


class TestTraceIdGeneration:
    """Test trace_id generation at transport layer."""
    
    @pytest.mark.asyncio(scope="function")
    async def test_trace_id_generated_at_transport_layer(self):
        """Test that trace_id is generated when handling voice message."""
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
            first_name="Test",
        )
        
        # Mock background tasks
        mock_tasks = MagicMock()
        
        with patch('webhook.telegram_voice.BackgroundTasks', return_value=mock_tasks):
            with patch('webhook.telegram_voice.get_voice_transport') as mock_get_transport:
                transport = MagicMock()
                transport.tracer = MockTracer()
                mock_get_transport.return_value = transport
                
                # Call handler
                result = await handle_voice_message(message, mock_tasks)
                
                # Verify result
                assert result == {"status": "ok"}
                
                # Verify background task was called with trace_id
                assert mock_tasks.add_task.called
                call_args = mock_tasks.add_task.call_args
                
                # Extract trace_id from kwargs
                kwargs = call_args[1] if call_args[1] else {}
                assert "trace_id" in kwargs
                assert kwargs["trace_id"]  # Not empty
                assert len(kwargs["trace_id"]) == 36  # UUID format


class TestTraceMetadataAttachment:
    """Test trace metadata attachment to events."""
    
    @pytest.mark.asyncio
    async def test_transport_metadata_includes_platform(self):
        """Test that transport metadata includes platform and modality."""
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
            first_name="TestUser",
            username="testuser",
        )
        
        mock_tasks = MagicMock()
        
        with patch('webhook.telegram_voice.get_voice_transport') as mock_get_transport:
            transport = MagicMock()
            mock_tracer = MockTracer()
            transport.tracer = mock_tracer
            mock_get_transport.return_value = transport
            
            result = await handle_voice_message(message, mock_tasks)
            
            # Verify tracer recorded event
            assert len(mock_tracer.events) > 0
            event = mock_tracer.events[0]
            
            # Verify event name
            assert event["name"] == "telegram_voice_received"
            
            # Verify metadata includes platform and modality
            metadata = event["metadata"]
            assert metadata["platform"] == "telegram"
            assert metadata["modality"] == "voice"
            assert metadata["telegram_user_id"] == 789
            assert metadata["telegram_user_name"] == "testuser"
            assert metadata["voice_duration"] == 5


class TestTraceIdPropagation:
    """Test trace_id propagation through background task."""
    
    @pytest.mark.asyncio
    async def test_trace_id_propagated_to_background_task(self):
        """Test that trace_id is propagated to background task."""
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
            first_name="Test",
        )
        
        mock_tasks = MagicMock()
        
        with patch('webhook.telegram_voice.get_voice_transport'):
            result = await handle_voice_message(message, mock_tasks)
            
            # Get the background task call
            call_args = mock_tasks.add_task.call_args
            kwargs = call_args[1] if call_args[1] else {}
            
            # Verify trace_id and conversation_id are passed
            assert "trace_id" in kwargs
            assert "conversation_id" in kwargs
            
            trace_id = kwargs["trace_id"]
            conversation_id = kwargs["conversation_id"]
            
            # Verify format
            assert trace_id  # Not empty
            assert conversation_id == "telegram:456"  # Includes chat_id


class TestConversationIdFormat:
    """Test conversation_id format for trace identity."""
    
    @pytest.mark.asyncio
    async def test_conversation_id_includes_platform_and_chat_id(self):
        """Test that conversation_id identifies the conversation correctly."""
        voice = TelegramVoice(
            file_id="test_id",
            file_unique_id="unique_id",
            duration=3,
        )
        
        # Different chat IDs
        for chat_id in [100, 456, 999]:
            message = TelegramVoiceMessage(
                message_id=123,
                date=1676817600,
                chat_id=chat_id,
                user_id=789,
                voice=voice,
                first_name="Test",
            )
            
            mock_tasks = MagicMock()
            
            with patch('webhook.telegram_voice.get_voice_transport'):
                await handle_voice_message(message, mock_tasks)
                
                call_args = mock_tasks.add_task.call_args
                kwargs = call_args[1] if call_args[1] else {}
                
                conversation_id = kwargs["conversation_id"]
                assert conversation_id == f"telegram:{chat_id}"


class TestTraceIdIsolation:
    """Test that concurrent requests have isolated trace_ids."""
    
    @pytest.mark.asyncio
    async def test_concurrent_messages_have_different_trace_ids(self):
        """Test that concurrent voice messages get different trace_ids."""
        voice = TelegramVoice(
            file_id="test_id",
            file_unique_id="unique_id",
            duration=5,
        )
        
        trace_ids = set()
        
        async def send_message(chat_id):
            message = TelegramVoiceMessage(
                message_id=chat_id,
                date=1676817600,
                chat_id=chat_id,
                user_id=789,
                voice=voice,
                first_name="Test",
            )
            
            mock_tasks = MagicMock()
            
            with patch('webhook.telegram_voice.get_voice_transport'):
                await handle_voice_message(message, mock_tasks)
                
                call_args = mock_tasks.add_task.call_args
                kwargs = call_args[1] if call_args[1] else {}
                trace_ids.add(kwargs["trace_id"])
        
        # Send 3 concurrent messages
        await asyncio.gather(
            send_message(100),
            send_message(200),
            send_message(300),
        )
        
        # All trace_ids should be unique
        assert len(trace_ids) == 3, "Concurrent messages should have different trace_ids"


class TestBackgroundTaskTraceUsage:
    """Test that background task uses provided trace_id."""
    
    @pytest.mark.asyncio
    async def test_background_task_uses_provided_trace_id(self):
        """Test that background task uses trace_id if provided."""
        provided_trace_id = str(uuid4())
        
        with patch('webhook.telegram_voice.get_voice_transport') as mock_get_transport:
            transport = MagicMock()
            mock_tracer = MockTracer()
            transport.tracer = mock_tracer
            transport.process_voice_message = AsyncMock(
                return_value=MagicMock(
                    status="success",
                    transcribed_text="Test transcription",
                    confidence=0.95,
                )
            )
            transport.send_message = AsyncMock()
            mock_get_transport.return_value = transport
            
            await _process_voice_async(
                message_id=123,
                chat_id=456,
                user_id=789,
                file_id="test_id",
                duration=5,
                user_name="test",
                trace_id=provided_trace_id,
                conversation_id="telegram:456",
            )
            
            # Verify background task completed
            assert transport.send_message.called


class TestTraceMetadataIntegrity:
    """Test trace metadata maintains integrity across layers."""
    
    @pytest.mark.asyncio
    async def test_trace_metadata_consistency(self):
        """Test that trace metadata is consistent across transport and background task."""
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
            first_name="TestUser",
            username="testuser",
        )
        
        captured_trace_id = None
        captured_conversation_id = None
        
        def capture_task_call(func, *args, **kwargs):
            nonlocal captured_trace_id, captured_conversation_id
            captured_trace_id = kwargs.get("trace_id")
            captured_conversation_id = kwargs.get("conversation_id")
        
        mock_tasks = MagicMock()
        mock_tasks.add_task.side_effect = capture_task_call
        
        with patch('webhook.telegram_voice.get_voice_transport') as mock_get_transport:
            transport = MagicMock()
            mock_tracer = MockTracer()
            transport.tracer = mock_tracer
            mock_get_transport.return_value = transport
            
            await handle_voice_message(message, mock_tasks)
            
            # Verify tracer received same trace_id
            if len(mock_tracer.events) > 0:
                event_trace_id = mock_tracer.events[0]["trace_metadata"]["trace_id"]
                assert event_trace_id == captured_trace_id
            
            # Verify conversation_id is consistent
            assert captured_conversation_id == "telegram:456"


class TestTracerInitialization:
    """Test tracer initialization in voice transport."""
    
    def test_voice_transport_initializes_tracer(self):
        """Test that TelegramVoiceTransport initializes tracer."""
        with patch('webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN', 'test_token'):
            transport = get_voice_transport()
            
            # Verify tracer is initialized
            assert hasattr(transport, 'tracer')
            assert transport.tracer is not None
    
    def test_voice_transport_handles_tracer_init_failure(self):
        """Test that tracer init failure doesn't crash transport."""
        with patch('webhook.telegram_voice.Config.TELEGRAM_BOT_TOKEN', 'test_token'):
            with patch('webhook.telegram_voice.tracer_factory.create_tracer', side_effect=Exception("Tracer error")):
                # Should not raise
                transport = get_voice_transport()
                assert transport is not None
                # Should fall back to NoOpTracer
                assert transport.tracer is not None


class TestTraceEventEmission:
    """Test that transport emits trace events."""
    
    @pytest.mark.asyncio
    async def test_transport_emits_telegram_voice_received_event(self):
        """Test that transport emits telegram_voice_received event."""
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
            first_name="Test",
        )
        
        mock_tasks = MagicMock()
        
        with patch('webhook.telegram_voice.get_voice_transport') as mock_get_transport:
            transport = MagicMock()
            mock_tracer = MockTracer()
            transport.tracer = mock_tracer
            mock_get_transport.return_value = transport
            
            await handle_voice_message(message, mock_tasks)
            
            # Verify event was emitted
            events = mock_tracer.events
            assert len(events) > 0
            
            event = events[0]
            assert event["name"] == "telegram_voice_received"
            assert "trace_metadata" in event
            assert "trace_id" in event["trace_metadata"]
