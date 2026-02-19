"""
WhatsApp Transport Layer - Pydantic Schemas

PURE DATA MODELS - NO LOGIC
Only defines the contract between WhatsApp and the normalized interface.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# NORMALIZED MESSAGE (THE CONTRACT)
# ============================================================================

class NormalizedMessage(BaseModel):
    """
    Canonical input format that the agent consumes.
    
    The agent never knows this came from WhatsApp.
    It's exactly what curl/Postman would pass.
    """
    
    input_text: str = Field(
        ...,
        description="Message content. Empty for audio/image (media only)."
    )
    sender_id: str = Field(..., description="WhatsApp phone number normalized")
    message_id: str = Field(..., description="Unique WhatsApp message ID")
    timestamp: datetime = Field(..., description="Message timestamp UTC")
    transport: Literal["whatsapp"] = Field(
        ...,
        description="Always 'whatsapp' - identifies transport layer"
    )
    input_type: Literal["text", "audio", "image"] = Field(
        ...,
        description="Content modality: text, audio, or image"
    )
    media_url: Optional[str] = Field(
        None,
        description="URL to media (audio/image). None for text."
    )
    
    class Config:
        """Pydantic config."""
        frozen = True  # Immutable - transport shouldn't mutate


# ============================================================================
# WHATSAPP WEBHOOK PAYLOAD SCHEMAS (INPUT)
# ============================================================================

class TextMessage(BaseModel):
    """Single text message from WhatsApp."""
    type: Literal["text"]
    text: dict[str, str]  # {"body": "..."}


class AudioMessage(BaseModel):
    """Audio message from WhatsApp."""
    type: Literal["audio"]
    audio: dict[str, str]  # {"id": "...", "mime_type": "audio/..."}


class ImageMessage(BaseModel):
    """Image message from WhatsApp."""
    type: Literal["image"]
    image: dict[str, str]  # {"id": "...", "mime_type": "image/..."}


class MessageObject(BaseModel):
    """A single WhatsApp message."""
    from_: str = Field(..., alias="from")
    id: str
    timestamp: str
    type: str
    
    text: Optional[dict[str, str]] = None
    audio: Optional[dict[str, str]] = None
    image: Optional[dict[str, str]] = None
    
    class Config:
        populate_by_name = True


class ContactObject(BaseModel):
    """Contact info."""
    wa_id: str
    profile: dict[str, str]


class MessageStatusChange(BaseModel):
    """Message status update (delivery, read, etc)."""
    id: str
    status: str
    timestamp: str
    recipient_id: str


class WhatsAppWebhookPayload(BaseModel):
    """
    Full WhatsApp webhook payload.
    
    ref: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-example
    """
    
    object: str = Field(..., description="Always 'whatsapp_business_account'")
    entry: list[dict] = Field(..., description="Webhook entries")
    
    class Config:
        extra = "allow"  # WhatsApp may add fields


# ============================================================================
# INVOKE API PAYLOAD (OUTPUT)
# ============================================================================

class InvokePayload(BaseModel):
    """
    Normalized payload sent to /invoke.
    
    The agent consumes this - it doesn't know it came from WhatsApp.
    """
    
    input: str = Field(..., description="User input text")
    conversation_id: str = Field(..., description="Sender phone number")
    
    class Config:
        extra = "allow"  # /invoke may use additional fields


# ============================================================================
# WHATSAPP API RESPONSE (OUTPUT)
# ============================================================================

class WhatsAppMessageResponse(BaseModel):
    """Response from WhatsApp Cloud API when sending a message."""
    
    messaging_product: str = Field(default="whatsapp")
    contacts: list[dict[str, str]]  # [{"input": "1234567890", "wa_id": "1234567890"}]
    messages: list[dict[str, str]]  # [{"id": "wamid.xxx", "message_status": "accepted"}]


@dataclass(frozen=True)
class TransportContext:
    """
    Context metadata about the transport.
    
    Used for logging/observability only.
    Never passed to agent.
    """
    
    sender_id: str
    message_id: str
    input_type: str  # text, audio, image
    timestamp: datetime
    
    # Observability only
    signature_verified: bool = True
    normalized_at: Optional[datetime] = None
    invoked_at: Optional[datetime] = None
