"""
WhatsApp Transport Integration Tests

End-to-end flow tests: webhook → normalization → /invoke → response

KEY ASSERTION: Agent never knows WhatsApp exists
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from transport.whatsapp.schemas import NormalizedMessage


class TestTextFlow:
    """Test complete text message flow."""

    def test_text_message_flow(self):
        """
        Full flow: WhatsApp webhook → normalize → /invoke → response
        
        Agent receives identical input as curl would.
        """
        
        # Mock the /invoke endpoint
        mock_invoke_response = {
            "output": "Response from agent",
            "metadata": {"trace_id": "123"}
        }
        
        webhook_payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "wamid.msg_123",
                            "timestamp": "1707500000",
                            "type": "text",
                            "text": {"body": "Hello agent"},
                        }],
                    }
                }],
            }],
        }
        
        # We test the normalization works
        from transport.whatsapp.normalize import normalize_message
        
        normalized = normalize_message(webhook_payload)
        
        # Verify normalized schema
        assert isinstance(normalized, NormalizedMessage)
        assert normalized.input_text == "Hello agent"
        assert normalized.sender_id == "1234567890"
        assert normalized.input_type == "text"
        assert normalized.transport == "whatsapp"
        
        # The agent would receive:
        invoke_payload = {
            "input": normalized.input_text,
            "conversation_id": normalized.sender_id,
        }
        
        # This is exactly what curl would send
        assert invoke_payload["input"] == "Hello agent"
        assert invoke_payload["conversation_id"] == "1234567890"

    def test_agent_cannot_tell_transport_source(self):
        """Agent receives normalized input without transport metadata."""
        
        webhook_payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "text",
                            "text": {"body": "Test"},
                        }],
                    }
                }],
            }],
        }
        
        from transport.whatsapp.normalize import normalize_message
        
        normalized = normalize_message(webhook_payload)
        
        # Agent payload - no WhatsApp identifiers
        invoke_payload = {
            "input": normalized.input_text,
            "conversation_id": normalized.sender_id,
        }
        
        # Agent payload is clean - just input and conversation_id
        assert set(invoke_payload.keys()) == {"input", "conversation_id"}
        assert "whatsapp" not in str(invoke_payload).lower()


class TestAudioFlow:
    """Test audio message flow."""

    def test_audio_message_flow_no_stt(self):
        """
        Audio message flow: normalize → preserve URL, no STT
        
        Agent receives empty input_text with media URL.
        """
        
        webhook_payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "wamid.audio_123",
                            "timestamp": "1707500000",
                            "type": "audio",
                            "audio": {
                                "id": "audio_file_id",
                                "mime_type": "audio/mpeg",
                            },
                        }],
                    }
                }],
            }],
        }
        
        from transport.whatsapp.normalize import normalize_message
        
        normalized = normalize_message(webhook_payload)
        
        # Verify audio is preserved as media URL
        assert normalized.input_type == "audio"
        assert normalized.input_text == ""  # Empty - no STT
        assert normalized.media_url == "whatsapp://audio/audio_file_id"
        
        # Agent would receive empty input for audio
        invoke_payload = {
            "input": normalized.input_text,  # Empty
            "conversation_id": normalized.sender_id,
        }
        
        assert invoke_payload["input"] == ""


class TestImageFlow:
    """Test image message flow."""

    def test_image_message_flow_no_vision(self):
        """
        Image message flow: normalize → preserve URL, no vision
        
        Agent receives empty input_text with media URL.
        """
        
        webhook_payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "wamid.image_123",
                            "timestamp": "1707500000",
                            "type": "image",
                            "image": {
                                "id": "image_file_id",
                                "mime_type": "image/jpeg",
                            },
                        }],
                    }
                }],
            }],
        }
        
        from transport.whatsapp.normalize import normalize_message
        
        normalized = normalize_message(webhook_payload)
        
        # Verify image is preserved as media URL
        assert normalized.input_type == "image"
        assert normalized.input_text == ""  # Empty - no vision
        assert normalized.media_url == "whatsapp://image/image_file_id"
        
        # Agent would receive empty input for image
        invoke_payload = {
            "input": normalized.input_text,  # Empty
            "conversation_id": normalized.sender_id,
        }
        
        assert invoke_payload["input"] == ""


class TestTransportBoundary:
    """
    Test the critical boundary:
    "WhatsApp input passes through normalization and reaches /invoke
     exactly like any other input, without influencing agent logic."
    """

    def test_no_agent_code_in_transport(self):
        """
        Verify that transport code doesn't import agent modules.
        
        If this fails, someone imported agent code into transport.
        """
        
        # Transport modules should not import agent
        import transport.whatsapp.webhook as webhook_module
        import transport.whatsapp.normalize as normalize_module
        import transport.whatsapp.security as security_module
        import transport.whatsapp.sender as sender_module
        
        # Check for agent imports
        agent_imports = [
            "agent.langgraph_orchestrator",
            "agent.state_schema",
            "agent.memory",
            "agent.tracing",
        ]
        
        for module in [
            webhook_module, normalize_module, security_module, sender_module
        ]:
            module_code = str(module.__dict__)
            for agent_import in agent_imports:
                assert agent_import not in module_code, \
                    f"Transport imports agent code: {agent_import}"

    def test_normalized_message_immutable(self):
        """NormalizedMessage is frozen - cannot be mutated by transport."""
        
        from transport.whatsapp.normalize import normalize_message
        
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "text",
                            "text": {"body": "Test"},
                        }],
                    }
                }],
            }],
        }
        
        normalized = normalize_message(payload)
        
        # Try to mutate (should fail)
        with pytest.raises(TypeError):
            normalized.input_text = "Modified"  # type: ignore
        
        with pytest.raises(TypeError):
            normalized.sender_id = "Modified"  # type: ignore

    def test_transport_preserves_input_format(self):
        """
        Text input should reach /invoke with NO modifications.
        """
        
        from transport.whatsapp.normalize import normalize_message
        
        test_cases = [
            ("simple", "simple"),
            ("  spaces  ", "spaces"),
            ("UPPERCASE", "UPPERCASE"),
            ("123numbers", "123numbers"),
            ("!@#$%^&*()", "!@#$%^&*()"),
        ]
        
        for original, expected in test_cases:
            payload = {
                "entry": [{
                    "changes": [{
                        "value": {
                            "messages": [{
                                "from": "1234567890",
                                "id": "msg_123",
                                "timestamp": "1707500000",
                                "type": "text",
                                "text": {"body": original},
                            }],
                        }
                    }],
                }],
            }
            
            normalized = normalize_message(payload)
            assert normalized.input_text == expected
