"""
WhatsApp Webhook Receiver

FastAPI router that receives WhatsApp messages and calls /invoke.
No agent imports. No logic. Pure transport.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from .normalize import NormalizationError, normalize_message
from .security import SignatureVerificationError, verify_signature, verify_webhook_challenge
from .sender import WhatsAppSenderError, send_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["WhatsApp Transport"])


# ============================================================================
# WEBHOOK CHALLENGE (Setup only)
# ============================================================================

@router.get("/whatsapp")
async def whatsapp_webhook_challenge(
    hub_mode: str,
    hub_challenge: str,
    hub_verify_token: str,
) -> str:
    """
    Verify webhook subscription challenge from Meta.
    
    WhatsApp requires this GET endpoint to set up the webhook.
    We verify the token and echo back the challenge.
    
    Args:
        hub_mode: Should be "subscribe"
        hub_challenge: Random string to echo back
        hub_verify_token: Token configured in Meta dashboard
    
    Returns:
        The challenge string (plain text)
    
    Raises:
        HTTPException(403): Invalid token
        HTTPException(400): Invalid mode
    """
    
    try:
        return verify_webhook_challenge(hub_mode, hub_challenge, hub_verify_token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook challenge error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook challenge failed"
        )


# ============================================================================
# WEBHOOK RECEIVER (Message processing)
# ============================================================================

@router.post("/whatsapp")
async def whatsapp_webhook_receiver(request: Request) -> dict[str, str]:
    """
    Receive WhatsApp messages via webhook.
    
    Flow:
    1. Get raw payload
    2. Verify signature (403 if invalid, 401 if missing)
    3. Normalize to NormalizedMessage
    4. Call /invoke with normalized input
    5. Send response back to WhatsApp
    
    Rules:
    - No agent imports
    - No mutations
    - No retries
    - No memory access
    - No transport metadata in agent payload
    
    Args:
        request: FastAPI Request with WhatsApp payload
    
    Returns:
        {"status": "ok"} if successful
    
    Raises:
        HTTPException(401): Missing signature
        HTTPException(403): Invalid signature
        HTTPException(422): Invalid payload format
        HTTPException(400): Unsupported message type
    """
    
    # Step 1: Get raw body for signature verification
    try:
        body = await request.body()
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid JSON payload"
        )
    except Exception as e:
        logger.error(f"Failed to read request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read request"
        )
    
    # Step 2: Verify signature (security boundary)
    try:
        await verify_signature(request, body)
        logger.debug("Signature verified for WhatsApp webhook")
    except HTTPException as e:
        logger.warning(f"Signature verification failed: {e.detail}")
        raise
    except SignatureVerificationError as e:
        logger.error(f"Signature error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Signature verification failed"
        )
    
    # Step 3: Normalize to canonical schema
    try:
        normalized = normalize_message(payload)
        logger.info(
            f"Message normalized",
            extra={
                "sender_id": normalized.sender_id,
                "message_id": normalized.message_id,
                "input_type": normalized.input_type,
            }
        )
    except NormalizationError as e:
        logger.error(f"Normalization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Normalization failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected normalization error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Normalization failed"
        )
    
    # Step 4: Call /invoke (the agent)
    # Agent never knows this came from WhatsApp
    try:
        import httpx
        
        invoke_payload = {
            "input": normalized.input_text,
            "conversation_id": normalized.sender_id,
        }
        
        async with httpx.AsyncClient() as client:
            # NOTE: This assumes /invoke is on same server
            # In production, use full URL or service discovery
            invoke_response = await client.post(
                "http://localhost:8000/invoke",
                json=invoke_payload,
                timeout=30.0
            )
        
        if invoke_response.status_code != 200:
            error_text = invoke_response.text
            logger.error(
                f"Agent (/invoke) returned error: {invoke_response.status_code}",
                extra={
                    "status_code": invoke_response.status_code,
                    "error": error_text,
                }
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent request failed"
            )
        
        agent_result = invoke_response.json()
        agent_output = agent_result.get("output", "")
        
        logger.info(
            f"Agent invoked successfully",
            extra={
                "sender_id": normalized.sender_id,
                "output_length": len(agent_output),
            }
        )
    
    except httpx.RequestError as e:
        logger.error(f"Failed to call /invoke: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent request failed"
        )
    except Exception as e:
        logger.error(f"Unexpected error calling /invoke: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent request failed"
        )
    
    # Step 5: Send response back to WhatsApp
    # This happens asynchronously - we return 200 immediately
    try:
        phone_number_id = request.query_params.get("phone_number_id")
        await send_response(normalized, agent_output, phone_number_id)
        logger.info(
            f"Response sent to WhatsApp",
            extra={
                "sender_id": normalized.sender_id,
                "message_id": normalized.message_id,
            }
        )
    except WhatsAppSenderError as e:
        # Log error but don't fail the webhook response
        # (WhatsApp expects 200 immediately)
        logger.error(f"Failed to send response: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error sending response: {e}", exc_info=True)
    
    # Always return 200 to acknowledge the webhook
    # (Per WhatsApp requirements)
    return {"status": "ok"}
