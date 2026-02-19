"""WhatsApp Transport Layer - Module Exports"""

from .normalize import (
    NormalizationError,
    extract_sender_id,
    normalize_message,
)
from .schemas import (
    AudioMessage,
    ImageMessage,
    InvokePayload,
    MessageObject,
    NormalizedMessage,
    TextMessage,
    TransportContext,
    WhatsAppMessageResponse,
    WhatsAppWebhookPayload,
)
from .security import SignatureVerificationError, verify_signature, verify_webhook_challenge
from .sender import WhatsAppSenderError, send_response
from .webhook import router

__all__ = [
    # Schemas
    "NormalizedMessage",
    "WhatsAppWebhookPayload",
    "TextMessage",
    "AudioMessage",
    "ImageMessage",
    "MessageObject",
    "InvokePayload",
    "WhatsAppMessageResponse",
    "TransportContext",
    # Normalization
    "normalize_message",
    "extract_sender_id",
    "NormalizationError",
    # Security
    "verify_signature",
    "verify_webhook_challenge",
    "SignatureVerificationError",
    # Sender
    "send_response",
    "WhatsAppSenderError",
    # Router
    "router",
]
