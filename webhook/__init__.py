"""
Webhook module - FastAPI route handlers for different platforms.

Includes:
- telegram.py: Text message handler
- telegram_voice.py: Voice message handler
"""

from webhook.telegram import router as text_router
from webhook.telegram_voice import voice_router

__all__ = ["text_router", "voice_router"]
