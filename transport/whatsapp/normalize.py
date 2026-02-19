"""
WhatsApp Input Normalization

PURE CONVERSION - NO LOGIC, NO MODEL CALLS

Converts WhatsApp message formats into canonical NormalizedMessage.
- TEXT: Extract body, trim, no enrichment
- AUDIO: Preserve URL only, no STT
- IMAGE: Preserve URL only, no vision inference

The agent never knows the source was WhatsApp.
"""

from datetime import datetime
from typing import Optional

from .schemas import NormalizedMessage, WhatsAppWebhookPayload


class NormalizationError(Exception):
    """Input normalization failed."""
    pass


def normalize_message(
    payload: dict | WhatsAppWebhookPayload,
) -> NormalizedMessage:
    """
    Convert WhatsApp webhook message into NormalizedMessage.
    
    Handles:
    - Text messages
    - Audio messages (media_url only)
    - Image messages (media_url only)
    
    Args:
        payload: Raw WhatsApp webhook payload
    
    Returns:
        NormalizedMessage ready for /invoke
    
    Raises:
        NormalizationError: Invalid or unhandled message
    """
    
    # Ensure payload is dict
    if isinstance(payload, WhatsAppWebhookPayload):
        payload = payload.dict()
    
    # Extract the message entry
    try:
        entry = payload["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        messages = value.get("messages", [])
        
        if not messages:
            raise NormalizationError("No messages in payload")
        
        message = messages[0]
        sender_id = message["from"]
        message_id = message["id"]
        timestamp = int(message["timestamp"])
        
    except (KeyError, IndexError, ValueError) as e:
        raise NormalizationError(f"Invalid payload structure: {e}")
    
    # Extract message content based on type
    message_type = message.get("type")
    
    if message_type == "text":
        return _normalize_text_message(
            message, sender_id, message_id, timestamp
        )
    
    elif message_type == "audio":
        return _normalize_audio_message(
            message, sender_id, message_id, timestamp
        )
    
    elif message_type == "image":
        return _normalize_image_message(
            message, sender_id, message_id, timestamp
        )
    
    else:
        raise NormalizationError(f"Unsupported message type: {message_type}")


def _normalize_text_message(
    message: dict,
    sender_id: str,
    message_id: str,
    timestamp: int,
) -> NormalizedMessage:
    """
    Normalize text message.
    
    Rules:
    - Extract message body
    - Trim whitespace
    - No enrichment or corrections
    """
    
    try:
        text_body = message["text"]["body"]
    except KeyError:
        raise NormalizationError("Text message missing 'text.body'")
    
    # Trim whitespace - no other processing
    input_text = text_body.strip()
    
    return NormalizedMessage(
        input_text=input_text,
        sender_id=sender_id,
        message_id=message_id,
        timestamp=datetime.fromtimestamp(timestamp),
        transport="whatsapp",
        input_type="text",
        media_url=None,
    )


def _normalize_audio_message(
    message: dict,
    sender_id: str,
    message_id: str,
    timestamp: int,
) -> NormalizedMessage:
    """
    Normalize audio message.
    
    Rules:
    - Do NOT perform STT (speech-to-text)
    - Pass media URL only
    - input_text MUST be empty
    
    Why no STT:
    - STT is a model call and not allowed in transport
    - Future phase can add STT without refactoring this
    """
    
    try:
        audio = message["audio"]
        audio_id = audio.get("id")
        mime_type = audio.get("mime_type")
        
        if not audio_id:
            raise NormalizationError("Audio message missing ID")
    
    except KeyError:
        raise NormalizationError("Audio message missing 'audio' object")
    
    # Build media URL (Meta format)
    # In production, you'd use the WhatsApp API to fetch the media URL
    # For now, we preserve the ID as placeholder
    media_url = f"whatsapp://audio/{audio_id}"
    
    return NormalizedMessage(
        input_text="",  # MUST be empty - no STT here
        sender_id=sender_id,
        message_id=message_id,
        timestamp=datetime.fromtimestamp(timestamp),
        transport="whatsapp",
        input_type="audio",
        media_url=media_url,
    )


def _normalize_image_message(
    message: dict,
    sender_id: str,
    message_id: str,
    timestamp: int,
) -> NormalizedMessage:
    """
    Normalize image message.
    
    Rules:
    - Do NOT perform vision inference
    - Pass media URL only
    - input_text MUST be empty
    
    Why no vision:
    - Vision is a model call and not allowed in transport
    - Future phase can add vision without refactoring this
    """
    
    try:
        image = message["image"]
        image_id = image.get("id")
        mime_type = image.get("mime_type")
        
        if not image_id:
            raise NormalizationError("Image message missing ID")
    
    except KeyError:
        raise NormalizationError("Image message missing 'image' object")
    
    # Build media URL (Meta format)
    media_url = f"whatsapp://image/{image_id}"
    
    return NormalizedMessage(
        input_text="",  # MUST be empty - no vision here
        sender_id=sender_id,
        message_id=message_id,
        timestamp=datetime.fromtimestamp(timestamp),
        transport="whatsapp",
        input_type="image",
        media_url=media_url,
    )


def extract_sender_id(payload: dict) -> str:
    """
    Extract sender phone number from payload.
    
    Useful for routing/logging without full normalization.
    """
    try:
        return payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    except (KeyError, IndexError):
        raise NormalizationError("Cannot extract sender_id from payload")
