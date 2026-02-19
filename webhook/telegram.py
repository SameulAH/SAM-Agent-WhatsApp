"""
Telegram Webhook Handler

Receives Telegram updates via webhook and processes them through the agent.

Security:
  - Verify Telegram signature (optional, add later if needed)
  - Rate limiting (can be added via middleware)
  - Error recovery (non-blocking failures)

Update Flow:
  webhook → parse_update → normalize → agent → send_response
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks

from config import Config
from transport.telegram.transport import (
    create_telegram_transport,
    NormalizedMessage,
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/webhook", tags=["webhook"])

# Telegram update structure
class TelegramUser(BaseModel):
    id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None


class TelegramChat(BaseModel):
    id: int
    type: str  # private, group, supergroup, channel
    title: Optional[str] = None
    username: Optional[str] = None


class TelegramVoice(BaseModel):
    """Voice object from Telegram."""
    file_id: str
    file_unique_id: str
    duration: int  # seconds
    mime_type: Optional[str] = None
    file_size: Optional[int] = None


class TelegramMessage(BaseModel):
    message_id: int
    date: int
    chat: TelegramChat
    from_: TelegramUser = Field(..., alias="from")
    text: Optional[str] = None
    voice: Optional[TelegramVoice] = None  # Support for voice messages
    
    class Config:
        populate_by_name = True


class TelegramUpdate(BaseModel):
    update_id: int
    message: Optional[TelegramMessage] = None


# Storage for transport (initialized once)
_transport = None


def get_telegram_transport():
    """Get or create Telegram transport (singleton)."""
    global _transport
    if _transport is None:
        _transport = create_telegram_transport()
    return _transport


@router.post("/telegram")
async def telegram_webhook(update: TelegramUpdate, background_tasks: BackgroundTasks):
    """
    Receive Telegram webhook updates.
    
    Handles both text and voice messages.
    Voice messages are forwarded to the voice handler.
    
    Expected payload:
    {
        "update_id": 123456789,
        "message": {
            "message_id": 1,
            "date": 1676817600,
            "chat": {"id": -123456789, "type": "private"},
            "from": {"id": 987654321, "first_name": "User"},
            "text": "Hello bot!"
        }
    }
    
    Returns:
        {"status": "ok"} if processed successfully
        Or error response if processing fails
    """
    try:
        # Validate we have a message
        if not update.message:
            logger.warning(f"Update {update.update_id} has no message")
            return {"status": "ok"}  # Still return 200 to Telegram
        
        msg = update.message
        
        # Check for voice message - forward to voice handler
        if msg.voice:
            from webhook.telegram_voice import handle_voice_message
            logger.info(f"Voice message detected, forwarding to voice handler")
            return await handle_voice_message(update, background_tasks)
        
        # Only handle text messages
        if not msg.text:
            logger.debug(f"Message {msg.message_id} has no text or voice, skipping")
            return {"status": "ok"}
        
        # Log received message
        logger.info(
            f"Received message from {msg.from_.username or msg.from_.first_name}: {msg.text[:50]}"
        )
        
        # Get transport
        transport = get_telegram_transport()
        
        # Create normalized message
        normalized = NormalizedMessage(
            platform="telegram",
            user_id=str(msg.from_.id),
            chat_id=str(msg.chat.id),
            content=msg.text,
            timestamp=datetime.fromtimestamp(msg.date),
            user_name=msg.from_.username or msg.from_.first_name,
        )
        
        # Process through agent
        response_text = await process_message_through_agent(normalized)

        # Clean chain-of-thought from response
        if response_text:
            from webhook.telegram_voice import clean_agent_response, text_to_ogg_voice
            response_text = clean_agent_response(response_text)

        # Send response back — voice if > 5 lines, else text
        if response_text:
            lines = [l for l in response_text.splitlines() if l.strip()]
            if len(lines) > 5:
                # Long reply → send as voice message
                from webhook.telegram_voice import get_voice_transport
                import asyncio
                loop = asyncio.get_event_loop()
                ogg_bytes = await loop.run_in_executor(None, text_to_ogg_voice, response_text)
                voice_transport = get_voice_transport()
                if ogg_bytes:
                    await voice_transport.send_voice_message(msg.chat.id, ogg_bytes)
                    logger.info(f"Sent long voice reply to chat {msg.chat.id}")
                else:
                    transport.send_response(msg.chat.id, response_text)
            else:
                transport.send_response(msg.chat.id, response_text)
                logger.info(f"Sent text reply to chat {msg.chat.id}")
        
        return {"status": "ok"}
    
    except Exception as e:
        logger.error(f"Error processing Telegram update: {str(e)}", exc_info=True)
        # Still return 200 to Telegram to acknowledge receipt
        # (Telegram will retry if we return error)
        return {"status": "ok"}


@router.get("/telegram/health")
async def telegram_health():
    """Health check for Telegram webhook."""
    try:
        transport = get_telegram_transport()
        return {
            "status": "ok",
            "bot": Config.TELEGRAM_BOT_USERNAME,
            "token_loaded": bool(Config.TELEGRAM_BOT_TOKEN),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_message_through_agent(message: NormalizedMessage) -> str:
    """
    Process a normalized message through the agent.
    
    Args:
        message: Normalized message from transport layer
        
    Returns:
        Response text to send back to user
    """
    try:
        from agent.orchestrator import SAMOrchestrator
        import uuid
        
        agent = SAMOrchestrator()
        
        # Generate conversation and trace IDs
        conversation_id = f"telegram_{message.chat_id}"
        trace_id = str(uuid.uuid4())
        
        # Invoke agent (already in async context)
        result = await agent.invoke(
            raw_input=message.content,
            conversation_id=conversation_id,
            trace_id=trace_id,
        )
        
        # Agent may return "output" or "final_output" depending on state serialization
        raw = result.get("output") or result.get("final_output") or ""
        from webhook.telegram_voice import clean_agent_response
        response = clean_agent_response(raw) or "Sorry, I couldn't process that."
        logger.info(f"Agent result keys: {list(result.keys())}, output: {response[:80] if response else None}")
        return response
    
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        return f"Echo: {message.content}"


@router.post("/telegram/set-webhook")
async def set_telegram_webhook(webhook_url: str):
    """
    Configure Telegram webhook (admin endpoint).
    
    Args:
        webhook_url: Full webhook URL (e.g., https://example.com/webhook/telegram)
        
    Returns:
        Result from Telegram API
        
    Note:
        This is an admin endpoint. In production, secure this with authentication.
    """
    try:
        transport = get_telegram_transport()
        
        # Call Telegram API to set webhook
        import requests
        
        response = requests.post(
            f"{transport.api_url}/setWebhook",
            json={"url": webhook_url},
            timeout=10,
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to set webhook: {response.text}")
            raise HTTPException(status_code=400, detail=response.text)
        
        result = response.json()
        logger.info(f"Webhook set successfully: {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error setting webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telegram/webhook-info")
async def get_webhook_info():
    """
    Get current webhook configuration.
    
    Returns:
        Webhook info from Telegram API
    """
    try:
        transport = get_telegram_transport()
        
        import requests
        
        response = requests.get(
            f"{transport.api_url}/getWebhookInfo",
            timeout=10,
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=response.text)
        
        return response.json()
    
    except Exception as e:
        logger.error(f"Error getting webhook info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
