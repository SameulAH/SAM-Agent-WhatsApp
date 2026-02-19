"""
WhatsApp Signature Verification Tests

Verify Meta HMAC-SHA256 signature validation.
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import HTTPException with fallback
try:
    from fastapi import HTTPException
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

from transport.whatsapp.security import (
    SignatureVerificationError,
    verify_signature,
    verify_webhook_challenge,
)


class TestSignatureVerification:
    """Test HMAC signature verification."""

    @pytest.mark.asyncio
    async def test_valid_signature(self):
        """Valid signature passes."""
        app_secret = "test_secret"
        payload = json.dumps({"test": "data"}).encode()
        
        expected_sig = "sha256=" + hmac.new(
            key=app_secret.encode(),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        request = MagicMock()
        request.headers = {"X-Hub-Signature-256": expected_sig}
        
        with patch.dict("os.environ", {"WHATSAPP_APP_SECRET": app_secret}):
            # Should not raise
            await verify_signature(request, payload)

    @pytest.mark.asyncio
    async def test_invalid_signature_returns_403(self):
        """Invalid signature returns 403."""
        request = MagicMock()
        request.headers = {"X-Hub-Signature-256": "sha256=invalid"}
        
        with patch.dict("os.environ", {"WHATSAPP_APP_SECRET": "secret"}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_signature(request, b"payload")
            
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_missing_signature_returns_401(self):
        """Missing signature header returns 401."""
        request = MagicMock()
        request.headers = {}
        
        with pytest.raises(HTTPException) as exc_info:
            await verify_signature(request, b"payload")
        
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_app_secret_returns_500(self):
        """Missing WHATSAPP_APP_SECRET returns 500."""
        request = MagicMock()
        request.headers = {"X-Hub-Signature-256": "sha256=sig"}
        
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                await verify_signature(request, b"payload")
            
            assert exc_info.value.status_code == 500

    def test_webhook_challenge_valid_token(self):
        """Valid webhook challenge succeeds."""
        token = "test_token"
        challenge = "challenge_string"
        
        with patch.dict("os.environ", {"WHATSAPP_VERIFY_TOKEN": token}):
            result = verify_webhook_challenge("subscribe", challenge, token)
            assert result == challenge

    def test_webhook_challenge_invalid_token_returns_403(self):
        """Invalid token returns 403."""
        with patch.dict("os.environ", {"WHATSAPP_VERIFY_TOKEN": "correct_token"}):
            with pytest.raises(HTTPException) as exc_info:
                verify_webhook_challenge("subscribe", "challenge", "wrong_token")
            
            assert exc_info.value.status_code == 403

    def test_webhook_challenge_invalid_mode_returns_400(self):
        """Invalid mode returns 400."""
        with patch.dict("os.environ", {"WHATSAPP_VERIFY_TOKEN": "token"}):
            with pytest.raises(HTTPException) as exc_info:
                verify_webhook_challenge("invalid_mode", "challenge", "token")
            
            assert exc_info.value.status_code == 400
