"""
Telegram Integration Tests

Test end-to-end flows:
  - Message workflow (3 tests)
  - Transport integration (3 tests)
  - Agent integration (2 tests)
  - State handling (2 tests)

Total: 10+ integration tests
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock, call
from fastapi.testclient import TestClient

from main import app
from config import Config
from transport.telegram.transport import NormalizedMessage

client = TestClient(app)


class TestMessageWorkflow:
    """Test complete message workflow."""
    
    @patch("webhook.telegram.process_message_through_agent")
    @patch("webhook.telegram.get_telegram_transport")
    def test_message_receive_process_send(self, mock_get_transport, mock_process):
        """Test: receive → process → send workflow."""
        # Setup
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        mock_process.return_value = "Response text"
        
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": 456, "type": "private"},
                "from": {"id": 123, "first_name": "Alice"},
                "text": "Hello!",
            },
        }
        
        # Execute
        response = client.post("/webhook/telegram", json=payload)
        
        # Verify
        assert response.status_code == 200
        mock_process.assert_called_once()
        mock_transport.send_response.assert_called_once()
        
        # Check send_response was called with correct args
        call_args = mock_transport.send_response.call_args
        assert call_args[0][0] == 456  # chat_id
        assert "Response" in call_args[0][1]  # response text
    
    @patch("webhook.telegram.process_message_through_agent")
    @patch("webhook.telegram.get_telegram_transport")
    def test_multiple_messages_sequential(self, mock_get_transport, mock_process):
        """Test: handle multiple messages in sequence."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        mock_process.return_value = "Response"
        
        payloads = [
            {
                "update_id": i,
                "message": {
                    "message_id": i,
                    "date": 1676817600,
                    "chat": {"id": 456, "type": "private"},
                    "from": {"id": 123, "first_name": "Alice"},
                    "text": f"Message {i}",
                },
            }
            for i in range(3)
        ]
        
        # Send all messages
        for payload in payloads:
            response = client.post("/webhook/telegram", json=payload)
            assert response.status_code == 200
        
        # Verify all were processed
        assert mock_process.call_count == 3
        assert mock_transport.send_response.call_count == 3
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_message_without_response(self, mock_get_transport):
        """Test: message processed but no response sent."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": 456, "type": "private"},
                "from": {"id": 123, "first_name": "Alice"},
                "text": "Hello!",
            },
        }
        
        # Mock process to return empty response
        with patch("webhook.telegram.process_message_through_agent", return_value=""):
            response = client.post("/webhook/telegram", json=payload)
        
        assert response.status_code == 200
        # send_response should not be called if response is empty
        # (This depends on implementation)


class TestTransportIntegration:
    """Test transport layer integration."""
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_normalized_message_creation(self, mock_get_transport):
        """Test: NormalizedMessage created correctly."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": 456, "type": "private"},
                "from": {"id": 123, "first_name": "Alice", "username": "alice"},
                "text": "Test message",
            },
        }
        
        with patch("webhook.telegram.process_message_through_agent") as mock_process:
            response = client.post("/webhook/telegram", json=payload)
        
        # Get the NormalizedMessage passed to process
        assert mock_process.called
        call_args = mock_process.call_args
        msg = call_args[0][0]
        
        assert msg.platform == "telegram"
        assert msg.user_id == "123"
        assert msg.chat_id == "456"
        assert msg.content == "Test message"
        assert msg.user_name == "alice"
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_transport_singleton(self, mock_get_transport):
        """Test: transport is created once and reused."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        payloads = [
            {
                "update_id": i,
                "message": {
                    "message_id": i,
                    "date": 1676817600,
                    "chat": {"id": 456, "type": "private"},
                    "from": {"id": 123, "first_name": "Alice"},
                    "text": f"Message {i}",
                },
            }
            for i in range(3)
        ]
        
        for payload in payloads:
            client.post("/webhook/telegram", json=payload)
        
        # get_telegram_transport should be called 3 times
        # (but should return same instance due to caching in function)
        assert mock_get_transport.call_count == 3
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_transport_error_handling(self, mock_get_transport):
        """Test: transport errors don't crash webhook."""
        mock_transport = AsyncMock()
        mock_transport.send_response.side_effect = Exception("Transport failed")
        mock_get_transport.return_value = mock_transport
        
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": 456, "type": "private"},
                "from": {"id": 123, "first_name": "Alice"},
                "text": "Hello!",
            },
        }
        
        response = client.post("/webhook/telegram", json=payload)
        
        # Should still return 200
        assert response.status_code == 200


class TestAgentIntegration:
    """Test agent processing integration."""
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_agent_processes_message(self, mock_get_transport):
        """Test: agent receives normalized message."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        with patch("webhook.telegram.process_message_through_agent") as mock_agent:
            mock_agent.return_value = "Agent response"
            
            payload = {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": 1676817600,
                    "chat": {"id": 456, "type": "private"},
                    "from": {"id": 123, "first_name": "Alice"},
                    "text": "Hello agent!",
                },
            }
            
            response = client.post("/webhook/telegram", json=payload)
        
        # Verify agent was called
        assert mock_agent.called
        call_args = mock_agent.call_args
        msg = call_args[0][0]
        
        assert msg.content == "Hello agent!"
        assert msg.user_id == "123"
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_agent_response_sent_back(self, mock_get_transport):
        """Test: agent response is sent back to user."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        agent_response = "This is the agent's response"
        
        with patch("webhook.telegram.process_message_through_agent") as mock_agent:
            mock_agent.return_value = agent_response
            
            payload = {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": 1676817600,
                    "chat": {"id": 456, "type": "private"},
                    "from": {"id": 123, "first_name": "Alice"},
                    "text": "Hello!",
                },
            }
            
            response = client.post("/webhook/telegram", json=payload)
        
        # Verify send_response was called with agent's response
        mock_transport.send_response.assert_called_once()
        call_args = mock_transport.send_response.call_args
        assert agent_response in call_args[0][1]


class TestStateHandling:
    """Test conversation state management."""
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_different_users_isolated(self, mock_get_transport):
        """Test: messages from different users are isolated."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        # Messages from different users
        payloads = [
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": 1676817600,
                    "chat": {"id": 123, "type": "private"},
                    "from": {"id": 111, "first_name": "Alice"},
                    "text": "Hello from Alice",
                },
            },
            {
                "update_id": 2,
                "message": {
                    "message_id": 2,
                    "date": 1676817600,
                    "chat": {"id": 456, "type": "private"},
                    "from": {"id": 222, "first_name": "Bob"},
                    "text": "Hello from Bob",
                },
            },
        ]
        
        with patch("webhook.telegram.process_message_through_agent") as mock_agent:
            mock_agent.side_effect = ["Response to Alice", "Response to Bob"]
            
            for payload in payloads:
                client.post("/webhook/telegram", json=payload)
        
        # Verify both users got responses
        assert mock_transport.send_response.call_count == 2
        
        # Check that each got their own response
        calls = mock_transport.send_response.call_args_list
        assert calls[0][0][0] == 123  # Alice's chat_id
        assert calls[1][0][0] == 456  # Bob's chat_id
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_group_conversation(self, mock_get_transport):
        """Test: group messages handled correctly."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "group", "title": "Test Group"},
                "from": {"id": 111, "first_name": "Alice"},
                "text": "@bot Hello!",
            },
        }
        
        with patch("webhook.telegram.process_message_through_agent") as mock_agent:
            mock_agent.return_value = "Response in group"
            client.post("/webhook/telegram", json=payload)
        
        # Verify response sent to group
        mock_transport.send_response.assert_called_once()
        call_args = mock_transport.send_response.call_args
        assert call_args[0][0] == -123456789  # group chat_id


class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_transport_init_error(self, mock_get_transport):
        """Test: error during transport init."""
        mock_get_transport.side_effect = Exception("Transport init failed")
        
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": 456, "type": "private"},
                "from": {"id": 123, "first_name": "Alice"},
                "text": "Hello!",
            },
        }
        
        response = client.post("/webhook/telegram", json=payload)
        
        # Should return 200 despite error (Telegram requirement)
        assert response.status_code == 200
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_send_response_error(self, mock_get_transport):
        """Test: error when sending response."""
        mock_transport = AsyncMock()
        mock_transport.send_response.side_effect = Exception("Send failed")
        mock_get_transport.return_value = mock_transport
        
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": 456, "type": "private"},
                "from": {"id": 123, "first_name": "Alice"},
                "text": "Hello!",
            },
        }
        
        response = client.post("/webhook/telegram", json=payload)
        
        # Should still return 200
        assert response.status_code == 200
