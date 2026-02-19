"""
WhatsApp Response Sender

Sends agent output back to WhatsApp.
No formatting intelligence. No retries. No logic.
"""

import logging
from typing import Optional

import httpx

from .schemas import NormalizedMessage, WhatsAppMessageResponse

logger = logging.getLogger(__name__)


class WhatsAppSenderError(Exception):
    """Failed to send response to WhatsApp."""
    pass


async def send_response(
    message: NormalizedMessage,
    agent_response: str,
    phone_number_id: Optional[str] = None,
) -> WhatsAppMessageResponse:
    """
    Send agent response back to sender via WhatsApp Cloud API.
    
    No formatting, no branching, no retries.
    If WhatsApp fails → log and return error.
    
    Args:
        message: The original NormalizedMessage (for sender_id)
        agent_response: Text response from agent
        phone_number_id: WhatsApp Business Account phone_number_id
    
    Returns:
        WhatsAppMessageResponse from Meta API
    
    Raises:
        WhatsAppSenderError: If send fails
    """
    
    import os
    
    # Get configuration
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    if not access_token:
        raise WhatsAppSenderError("WHATSAPP_ACCESS_TOKEN not configured")
    
    if not phone_number_id:
        phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        if not phone_number_id:
            raise WhatsAppSenderError("WHATSAPP_PHONE_NUMBER_ID not configured")
    
    # Build API endpoint
    api_version = os.getenv("WHATSAPP_API_VERSION", "v18.0")
    endpoint = (
        f"https://graph.instagram.com/{api_version}/"
        f"{phone_number_id}/messages"
    )
    
    # Prepare payload
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": message.sender_id,
        "type": "text",
        "text": {
            "body": agent_response
        }
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Send (no retries per spec)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=30.0
            )
        
        if response.status_code != 200:
            error_text = response.text
            logger.error(
                f"WhatsApp API error: {response.status_code} - {error_text}",
                extra={
                    "status_code": response.status_code,
                    "error_body": error_text,
                }
            )
            raise WhatsAppSenderError(
                f"WhatsApp API returned {response.status_code}"
            )
        
        result = response.json()
        logger.info(
            f"Response sent to {message.sender_id}",
            extra={
                "sender_id": message.sender_id,
                "message_id": message.message_id,
                "response_id": result.get("messages", [{}])[0].get("id"),
            }
        )
        
        return WhatsAppMessageResponse(**result)
    
    except httpx.RequestError as e:
        logger.error(
            f"HTTP request failed: {e}",
            exc_info=True,
            extra={
                "sender_id": message.sender_id,
                "error": str(e),
            }
        )
        raise WhatsAppSenderError(f"HTTP request failed: {e}")
    
    except Exception as e:
        logger.error(
            f"Unexpected error sending response: {e}",
            exc_info=True,
            extra={
                "sender_id": message.sender_id,
                "error": str(e),
            }
        )
        raise WhatsAppSenderError(f"Unexpected error: {e}")
