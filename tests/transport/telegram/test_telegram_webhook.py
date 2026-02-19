"""
Telegram Webhook Tests

Comprehensive test suite for webhook handler:
  - Message parsing (4 tests)
  - Update handling (4 tests)
  - Error cases (4 tests)
  
Total: 12+ unit tests for webhook handler
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from main import app
from webhook.telegram import (
    TelegramUpdate,
    TelegramMessage,
    TelegramChat,
    TelegramUser,
    process_message_through_agent,
    NormalizedMessage,
)

# Create test client
client = TestClient(app)


class TestWebhookParsing:
    """Test webhook message parsing."""
    
    def test_parse_valid_message(self):
        """Parse valid Telegram message."""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "private"},
                "from": {"id": 987654321, "first_name": "User"},
                "text": "Hello bot!",
            },
        }
        
        update = TelegramUpdate(**payload)
        
        assert update.update_id == 123456789
        assert update.message is not None
        assert update.message.text == "Hello bot!"
        assert update.message.from_.id == 987654321
    
    def test_parse_message_with_username(self):
        """Parse message with username."""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "private", "username": "testuser"},
                "from": {"id": 987654321, "first_name": "Test", "username": "testuser"},
                "text": "Hello!",
            },
        }
        
        update = TelegramUpdate(**payload)
        assert update.message.from_.username == "testuser"
    
    def test_parse_message_with_last_name(self):
        """Parse message with last name."""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "private"},
                "from": {"id": 987654321, "first_name": "John", "last_name": "Doe"},
                "text": "Hello!",
            },
        }
        
        update = TelegramUpdate(**payload)
        assert update.message.from_.last_name == "Doe"
    
    def test_parse_group_message(self):
        """Parse message from group."""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "group", "title": "Test Group"},
                "from": {"id": 987654321, "first_name": "User"},
                "text": "Hello group!",
            },
        }
        
        update = TelegramUpdate(**payload)
        assert update.message.chat.type == "group"
        assert update.message.chat.title == "Test Group"


class TestWebhookEndpoint:
    """Test webhook endpoint."""
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_webhook_receives_message(self, mock_get_transport):
        """Webhook receives and processes message."""
        # Setup mock
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "private"},
                "from": {"id": 987654321, "first_name": "User"},
                "text": "Hello bot!",
            },
        }
        
        # Send webhook
        response = client.post("/webhook/telegram", json=payload)
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_webhook_ignores_no_message(self):
        """Webhook gracefully ignores updates without message."""
        payload = {
            "update_id": 123456789,
            # No message
        }
        
        response = client.post("/webhook/telegram", json=payload)
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_webhook_ignores_no_text(self, mock_get_transport):
        """Webhook ignores messages without text."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "private"},
                "from": {"id": 987654321, "first_name": "User"},
                # No text
            },
        }
        
        response = client.post("/webhook/telegram", json=payload)
        
        assert response.status_code == 200
        # send_response should not be called
        mock_transport.send_response.assert_not_called()
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_webhook_handles_errors_gracefully(self, mock_get_transport):
        """Webhook returns 200 even on error (Telegram requires this)."""
        mock_get_transport.side_effect = Exception("Transport error")
        
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "private"},
                "from": {"id": 987654321, "first_name": "User"},
                "text": "Hello!",
            },
        }
        
        response = client.post("/webhook/telegram", json=payload)
        
        # Should still return 200 so Telegram doesn't retry
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @patch("webhook.telegram.get_telegram_transport")
    @patch("config.Config.TELEGRAM_BOT_USERNAME", "testbot")
    @patch("config.Config.TELEGRAM_BOT_TOKEN", "token123")
    def test_telegram_health_check(self, mock_get_transport):
        """Test Telegram health endpoint."""
        mock_transport = MagicMock()
        mock_get_transport.return_value = mock_transport
        
        response = client.get("/webhook/telegram/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["bot"] == "testbot"
        assert data["token_loaded"] is True
    
    def test_live_health_check(self):
        """Test live health check endpoint."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
    
    @patch("config.Config.validate")
    def test_ready_health_check_pass(self, mock_validate):
        """Test ready health check when config valid."""
        mock_validate.return_value = True
        
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
    
    @patch("config.Config.validate")
    def test_ready_health_check_fail(self, mock_validate):
        """Test ready health check when config invalid."""
        mock_validate.side_effect = Exception("Config missing")
        
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_ready"
        assert "Config missing" in data["reason"]


class TestMessageProcessing:
    """Test message processing through agent stub."""
    
    def test_process_message_echo(self):
        """Test agent stub echoes message."""
        message = NormalizedMessage(
            platform="telegram",
            user_id="123",
            chat_id="456",
            content="Test message",
            timestamp=datetime.now(),
            user_name="testuser",
        )
        
        response = process_message_through_agent(message)
        
        assert "Test message" in response
        assert "Echo:" in response


class TestRootEndpoints:
    """Test root and info endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "endpoints" in data
    
    @patch("config.Config.ENVIRONMENT", "development")
    @patch("config.Config.LLM_BACKEND", "ollama")
    @patch("config.Config.TELEGRAM_BOT_USERNAME", "testbot")
    @patch("config.Config.AGENT_PORT", 8000)
    def test_config_info_endpoint(self):
        """Test config info endpoint."""
        response = client.get("/config/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["environment"] == "development"
        assert data["llm_backend"] == "ollama"
        assert "@testbot" in data["telegram_bot"]


class TestConcurrentUpdates:
    """Test handling of concurrent updates."""
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_multiple_concurrent_updates(self, mock_get_transport):
        """Test handling multiple updates."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        payloads = [
            {
                "update_id": i,
                "message": {
                    "message_id": i,
                    "date": 1676817600,
                    "chat": {"id": -123456789, "type": "private"},
                    "from": {"id": 987654321, "first_name": "User"},
                    "text": f"Message {i}",
                },
            }
            for i in range(5)
        ]
        
        responses = [client.post("/webhook/telegram", json=p) for p in payloads]
        
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json() == {"status": "ok"} for r in responses)


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_very_long_message(self, mock_get_transport):
        """Test handling of very long messages."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        long_text = "x" * 10000  # 10k chars
        
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "private"},
                "from": {"id": 987654321, "first_name": "User"},
                "text": long_text,
            },
        }
        
        response = client.post("/webhook/telegram", json=payload)
        
        assert response.status_code == 200
    
    @patch("webhook.telegram.get_telegram_transport")
    def test_special_characters_in_message(self, mock_get_transport):
        """Test handling of special characters."""
        mock_transport = AsyncMock()
        mock_get_transport.return_value = mock_transport
        
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "date": 1676817600,
                "chat": {"id": -123456789, "type": "private"},
                "from": {"id": 987654321, "first_name": "User"},
                "text": "Hello! 你好! مرحبا! 🎉",
            },
        }
        
        response = client.post("/webhook/telegram", json=payload)
        
        assert response.status_code == 200
    
    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        response = client.post(
            "/webhook/telegram",
            content=b"invalid json",
            headers={"Content-Type": "application/json"},
        )
        
        # FastAPI returns 422 for validation errors
        assert response.status_code == 422
