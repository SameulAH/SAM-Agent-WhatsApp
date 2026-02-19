"""
WhatsApp Signature Verification

SECURITY BOUNDARY - Verify Meta HMAC signature.
No agent imports. No retries. No logic.
"""

import hashlib
import hmac
from typing import Optional

from fastapi import HTTPException, Request, status


class SignatureVerificationError(Exception):
    """Signature verification failed."""
    pass


async def verify_signature(
    request: Request,
    body: bytes,
    verify_token: Optional[str] = None,
) -> None:
    """
    Verify Meta HMAC-SHA256 signature on WhatsApp webhook.
    
    WhatsApp sends:
    - X-Hub-Signature-256 header with HMAC
    - Request body
    
    We compute HMAC(body, app_secret) and compare.
    
    Raises:
        HTTPException(401): Missing signature
        HTTPException(403): Invalid signature
    
    Args:
        request: FastAPI Request object
        body: Raw request body bytes
        verify_token: Optional hub.verify_token (not used for HMAC, just logged)
    
    Returns:
        None (raises if invalid)
    """
    
    # Get signature from header
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Hub-Signature-256 header"
        )
    
    # Get app secret from environment
    import os
    app_secret = os.getenv("WHATSAPP_APP_SECRET")
    if not app_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="WHATSAPP_APP_SECRET not configured"
        )
    
    # Compute expected HMAC
    expected_signature = "sha256=" + hmac.new(
        key=app_secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256
    ).hexdigest()
    
    # Compare (constant-time to prevent timing attacks)
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid signature"
        )


def verify_webhook_challenge(
    hub_mode: str,
    hub_challenge: str,
    hub_verify_token: str,
) -> str:
    """
    Verify webhook subscription challenge from WhatsApp.
    
    WhatsApp calls GET /webhook with:
    - hub.mode=subscribe
    - hub.challenge=random_string
    - hub.verify_token=configured_token
    
    We verify the token and echo back the challenge.
    
    Args:
        hub_mode: Should be "subscribe"
        hub_challenge: Random string to echo back
        hub_verify_token: Token to verify
    
    Returns:
        The challenge string to echo back
    
    Raises:
        HTTPException(403): Invalid token
    """
    
    import os
    expected_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "default_token")
    
    if hub_mode != "subscribe":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid hub.mode"
        )
    
    if hub_verify_token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid hub.verify_token"
        )
    
    return hub_challenge
