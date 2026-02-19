"""
Telegram Transport Layer for SAM Agent

Pure I/O transport for Telegram messaging.
Handles:
- Message normalization (Telegram → NormalizedMessage)
- Response sending (NormalizedMessage → Telegram)
- Token management from .env
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config


class TelegramMessage(BaseModel):
    """Message from Telegram API."""
    message_id: int
    chat_id: int
    user_id: int
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None


class NormalizedMessage(BaseModel):
    """Normalized message contract (platform-agnostic)."""
    platform: str  # "telegram", "whatsapp", etc.
    user_id: str
    chat_id: str
    content: str
    timestamp: datetime
    user_name: Optional[str] = None


class TelegramTransport:
    """
    Telegram transport layer.
    
    Pure I/O: normalizes Telegram messages and sends responses.
    No agent logic, no routing, no state modification.
    """
    
    def __init__(self):
        """Initialize with token from .env."""
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.bot_username = Config.TELEGRAM_BOT_USERNAME
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
        
        # API endpoint (would use real requests in production)
        self.api_url = f"https://api.telegram.org/bot{self.token}"
    
    def normalize_message(self, telegram_msg: TelegramMessage) -> NormalizedMessage:
        """
        Convert Telegram message to normalized format.
        
        Args:
            telegram_msg: Message from Telegram API
        
        Returns:
            NormalizedMessage (platform-agnostic)
        """
        return NormalizedMessage(
            platform="telegram",
            user_id=str(telegram_msg.user_id),
            chat_id=str(telegram_msg.chat_id),
            content=telegram_msg.text,
            timestamp=telegram_msg.timestamp,
            user_name=telegram_msg.username or telegram_msg.first_name,
        )
    
    def send_message(self, chat_id: int, text: str) -> bool:
        """
        Send message via Telegram API.

        Args:
            chat_id: Telegram chat ID
            text: Message text to send

        Returns:
            True if successful
        """
        import requests

        try:
            # Telegram max message length is 4096 chars — truncate if needed
            if len(text) > 4096:
                text = text[:4090] + "…"

            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=15,
            )
            if response.status_code != 200:
                print(f"[Telegram] Failed to send to {chat_id}: {response.text}")
                return False
            print(f"[Telegram] Sent to {chat_id}: {text[:80]}")
            return True
        except Exception as e:
            print(f"[Telegram] Error sending to {chat_id}: {e}")
            return False
    
    def send_response(self, chat_id: int, response_text: str) -> bool:
        """Send agent response via Telegram."""
        return self.send_message(chat_id, response_text)


def create_telegram_transport() -> TelegramTransport:
    """Factory function to create Telegram transport."""
    return TelegramTransport()


if __name__ == "__main__":
    # Test Telegram transport
    print(f"Initializing Telegram Transport...")
    print(f"  Token: {'✓ Loaded' if Config.TELEGRAM_BOT_TOKEN else '✗ Missing'} (from .env)")
    print(f"  Bot: @{Config.TELEGRAM_BOT_USERNAME}")
    print(f"  API: [masked for security]")
    
    transport = create_telegram_transport()
    print(f"\n✓ Telegram Transport initialized successfully")
    
    # Test message normalization
    sample_msg = TelegramMessage(
        message_id=1,
        chat_id=123456789,
        user_id=987654321,
        text="Hello, SAM Agent!",
        first_name="Test",
        username="testuser"
    )
    
    normalized = transport.normalize_message(sample_msg)
    print(f"\nNormalized message:")
    print(f"  Platform: {normalized.platform}")
    print(f"  User ID: {normalized.user_id}")
    print(f"  Chat ID: {normalized.chat_id}")
    print(f"  Content: {normalized.content}")
    print(f"  User Name: {normalized.user_name}")
