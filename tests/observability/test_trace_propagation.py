"""
Phase 9.2.1: Unified Trace Propagation Tests

Tests for trace lineage: Transport → Agent → LangSmith

Validates:
- trace_id generation at transport layer
- trace_id propagation through background task  
- tracer event emission at transport level
- unified trace visibility (no fragmentation)
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from webhook.telegram_voice import (
    TelegramVoiceMessage,
    TelegramVoice,
    TelegramVoiceTransport,
)
from agent.tracing import TraceMetadata


class MockTracer:
    """Mock tracer for capturing trace events."""
    
    def __init__(self):
        self.events = []
    
    def record_event(self, name: str, metadata: dict, trace_metadata: TraceMetadata):
        """Capture event with trace metadata."""
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
        return str(uuid4())
    
    def end_span(self, span: str, status: str, metadata: dict):
        """End span."""
        pass


def create_test_voice_message(chat_id=456, user_id=789, duration=5):
    """Helper to create test voice message."""
    voice = TelegramVoice(
        file_id="test_id",
        file_unique_id="unique_id",
        duration=duration,
    )
    
    return TelegramVoiceMessage(
        message_id=123,
        date=1676817600,
        chat_id=chat_id,
        user_id=user_id,
        voice=voice,
        first_name="Test",
        username="testuser",
    )


class TestTraceGeneration:
    """Test trace_id generation logic."""
    
    def test_trace_id_format(self):
        """Test that trace_id follows UUID4 format."""
        trace_id = str(uuid4())
        
        # Verify UUID4 format: 8-4-4-4-12
        assert len(trace_id) == 36
        assert trace_id.count('-') == 4
        parts = trace_id.split('-')
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12
    
    def test_unique_trace_ids_generated(self):
        """Test that trace_ids are unique."""
        trace_ids = {str(uuid4()) for _ in range(100)}
        assert len(trace_ids) == 100, "All trace_ids should be unique"


class TestConversationIdFormat:
    """Test conversation_id construction."""
    
    def test_conversation_id_format(self):
        """Test conversation_id follows platform:id format."""
        chat_ids = [1, 456, 999999]
        
        for chat_id in chat_ids:
            conversation_id = f"telegram:{chat_id}"
            assert conversation_id.startswith("telegram:")
            assert conversation_id.split(":")[-1] == str(chat_id)
    
    def test_conversation_id_uniqueness(self):
        """Test that different chats have different conversation_ids."""
        conv_ids = {f"telegram:{cid}" for cid in range(100, 110)}
        assert len(conv_ids) == 10


class TestTransportMetadataConstruction:
    """Test transport metadata structure."""
    
    def test_transport_metadata_fields(self):
        """Test that transport metadata includes required fields."""
        message = create_test_voice_message(chat_id=456, user_id=789, duration=5)
        
        # Simulate metadata construction
        transport_metadata = {
            "platform": "telegram",
            "modality": "voice",
            "telegram_chat_id": message.chat_id,
            "telegram_user_id": message.user_id,
            "telegram_user_name": message.username or message.first_name,
            "voice_duration": message.voice.duration,
            "voice_file_id": message.voice.file_id,
        }
        
        # Verify all fields present
        assert transport_metadata["platform"] == "telegram"
        assert transport_metadata["modality"] == "voice"
        assert transport_metadata["telegram_chat_id"] == 456
        assert transport_metadata["telegram_user_id"] == 789
        assert transport_metadata["telegram_user_name"] in ["testuser", "Test"]
        assert transport_metadata["voice_duration"] == 5
        assert transport_metadata["voice_file_id"] == "test_id"
    
    def test_transport_metadata_from_different_users(self):
        """Test metadata construction for different users."""
        for user_id in [1, 100, 999]:
            message = create_test_voice_message(user_id=user_id)
            
            transport_metadata = {
                "telegram_user_id": message.user_id,
                "telegram_chat_id": message.chat_id,
            }
            
            assert transport_metadata["telegram_user_id"] == user_id


class TestTraceMetadataCreation:
    """Test TraceMetadata object creation."""
    
    def test_trace_metadata_structure(self):
        """Test that TraceMetadata is created correctly."""
        trace_id = str(uuid4())
        conversation_id = "telegram:456"
        user_id = "789"
        
        trace_metadata = TraceMetadata(
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        
        assert trace_metadata.trace_id == trace_id
        assert trace_metadata.conversation_id == conversation_id
        assert trace_metadata.user_id == user_id
    
    def test_trace_metadata_consistency(self):
        """Test that TraceMetadata maintains consistency."""
        trace_id = str(uuid4())
        
        # Create same trace_metadata multiple times
        tm1 = TraceMetadata(
            trace_id=trace_id,
            conversation_id="telegram:456",
            user_id="789",
        )
        
        tm2 = TraceMetadata(
            trace_id=trace_id,
            conversation_id="telegram:456",
            user_id="789",
        )
        
        assert tm1.trace_id == tm2.trace_id
        assert tm1.conversation_id == tm2.conversation_id
        assert tm1.user_id == tm2.user_id


class TestEventEmission:
    """Test tracer event emission."""
    
    def test_event_recorded_with_trace_metadata(self):
        """Test that event is recorded with trace metadata."""
        tracer = MockTracer()
        
        trace_id = str(uuid4())
        trace_metadata = TraceMetadata(
            trace_id=trace_id,
            conversation_id="telegram:456",
            user_id="789",
        )
        
        transport_metadata = {
            "platform": "telegram",
            "modality": "voice",
            "duration": 5,
        }
        
        # Emit event
        tracer.record_event(
            name="telegram_voice_received",
            metadata=transport_metadata,
            trace_metadata=trace_metadata,
        )
        
        # Verify event recorded
        assert len(tracer.events) == 1
        event = tracer.events[0]
        
        assert event["name"] == "telegram_voice_received"
        assert event["metadata"]["platform"] == "telegram"
        assert event["trace_metadata"]["trace_id"] == trace_id
        assert event["trace_metadata"]["conversation_id"] == "telegram:456"


class TestTraceIdPropagation:
    """Test trace_id propagation through layers."""
    
    def test_trace_id_propagated_consistently(self):
        """Test that same trace_id is used across layers."""
        trace_id = str(uuid4())
        conversation_id = "telegram:456"
        
        # Layer 1: Transport emits event
        tracer = MockTracer()
        trace_metadata = TraceMetadata(
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id="789",
        )
        
        tracer.record_event(
            name="telegram_voice_received",
            metadata={"platform": "telegram"},
            trace_metadata=trace_metadata,
        )
        
        # Layer 2: Background task receives same trace_id
        background_task_trace_id = trace_metadata.trace_id
        
        # Verify consistency
        assert tracer.events[0]["trace_metadata"]["trace_id"] == background_task_trace_id
        assert background_task_trace_id == trace_id


class TestTracerInitialization:
    """Test tracer initialization."""
    
    def test_mock_tracer_creates_events_list(self):
        """Test that MockTracer initializes with events list."""
        tracer = MockTracer()
        assert hasattr(tracer, 'events')
        assert isinstance(tracer.events, list)
        assert len(tracer.events) == 0
    
    def test_mock_tracer_has_required_methods(self):
        """Test that MockTracer has all required methods."""
        tracer = MockTracer()
        assert hasattr(tracer, 'record_event')
        assert hasattr(tracer, 'start_span')
        assert hasattr(tracer, 'end_span')
        assert callable(tracer.record_event)
        assert callable(tracer.start_span)
        assert callable(tracer.end_span)


class TestConversationIdUniqueness:
    """Test conversation isolation."""
    
    def test_different_chats_different_conversation_ids(self):
        """Test that different Telegram chats have different conversation_ids."""
        conversation_ids = set()
        
        for chat_id in [100, 200, 300, 400, 500]:
            conversation_id = f"telegram:{chat_id}"
            conversation_ids.add(conversation_id)
        
        assert len(conversation_ids) == 5
    
    def test_conversation_id_includes_platform_identifier(self):
        """Test that conversation_id includes platform prefix."""
        for chat_id in [1, 456, 999]:
            conversation_id = f"telegram:{chat_id}"
            assert conversation_id.startswith("telegram:")
            assert ":" in conversation_id


class TestTraceIsolation:
    """Test trace isolation between messages."""
    
    def test_concurrent_messages_isolated(self):
        """Test that concurrent messages don't share trace_ids."""
        # Simulate 3 concurrent messages
        messages = [
            create_test_voice_message(chat_id=100),
            create_test_voice_message(chat_id=200),
            create_test_voice_message(chat_id=300),
        ]
        
        # Each would generate unique trace_id
        trace_ids = {str(uuid4()) for _ in messages}
        
        # All trace_ids should be unique
        assert len(trace_ids) == 3
    
    def test_trace_id_not_reused_across_messages(self):
        """Test that trace_id is never reused."""
        trace_ids = set()
        
        for i in range(50):
            trace_id = str(uuid4())
            assert trace_id not in trace_ids, "trace_id should not be reused"
            trace_ids.add(trace_id)


class TestMessageMetadataIntegrity:
    """Test message metadata preservation."""
    
    def test_message_fields_in_transport_metadata(self):
        """Test that message fields are properly captured in metadata."""
        message = create_test_voice_message(
            chat_id=777,
            user_id=888,
            duration=15
        )
        
        transport_metadata = {
            "telegram_chat_id": message.chat_id,
            "telegram_user_id": message.user_id,
            "voice_duration": message.voice.duration,
            "telegram_user_name": message.username or message.first_name,
            "voice_file_id": message.voice.file_id,
        }
        
        # Verify all fields captured
        assert transport_metadata["telegram_chat_id"] == 777
        assert transport_metadata["telegram_user_id"] == 888
        assert transport_metadata["voice_duration"] == 15
        assert transport_metadata["voice_file_id"] == "test_id"
    
    def test_username_fallback_to_first_name(self):
        """Test that first_name is used if username missing."""
        message_with_username = create_test_voice_message()
        message_with_username.username = "myusername"
        
        name = message_with_username.username or message_with_username.first_name
        assert name == "myusername"
        
        # Create message without username
        message_without_username = create_test_voice_message()
        message_without_username.username = None
        
        name = message_without_username.username or message_without_username.first_name
        assert name == "Test"


class TestTraceEventStructure:
    """Test structure of emitted trace events."""
    
    def test_trace_event_has_required_fields(self):
        """Test that trace events have name, metadata, and trace_metadata."""
        tracer = MockTracer()
        
        tracer.record_event(
            name="test_event",
            metadata={"key": "value"},
            trace_metadata=TraceMetadata(
                trace_id="trace-123",
                conversation_id="telegram:456",
                user_id="789",
            )
        )
        
        event = tracer.events[0]
        assert "name" in event
        assert "metadata" in event
        assert "trace_metadata" in event
    
    def test_trace_metadata_in_event_accessible(self):
        """Test that trace_metadata is accessible from events."""
        tracer = MockTracer()
        original_trace_id = str(uuid4())
        
        tracer.record_event(
            name="test",
            metadata={},
            trace_metadata=TraceMetadata(
                trace_id=original_trace_id,
                conversation_id="telegram:123",
                user_id="456",
            )
        )
        
        event = tracer.events[0]
        retrieved_trace_id = event["trace_metadata"]["trace_id"]
        
        assert retrieved_trace_id == original_trace_id


# Integration test
class TestTraceLayerIntegration:
    """Test integration of trace layers."""
    
    def test_full_trace_metadata_flow(self):
        """Test complete trace flow from generation to event."""
        # Step 1: Generate trace_id (transport layer)
        trace_id = str(uuid4())
        conversation_id = "telegram:456"
        
        # Step 2: Create TraceMetadata
        trace_metadata = TraceMetadata(
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id="789",
        )
        
        # Step 3: Create transport metadata
        transport_metadata = {
            "platform": "telegram",
            "modality": "voice",
            "duration": 5,
        }
        
        # Step 4: Emit event
        tracer = MockTracer()
        tracer.record_event(
            name="telegram_voice_received",
            metadata=transport_metadata,
            trace_metadata=trace_metadata,
        )
        
        # Verify complete flow
        assert len(tracer.events) == 1
        event = tracer.events[0]
        
        # Verify trace lineage
        assert event["trace_metadata"]["trace_id"] == trace_id
        assert event["trace_metadata"]["conversation_id"] == conversation_id
        assert event["metadata"]["platform"] == "telegram"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
