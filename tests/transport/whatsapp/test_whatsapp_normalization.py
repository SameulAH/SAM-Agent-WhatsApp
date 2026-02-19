"""
WhatsApp Input Normalization Tests

Test conversion of WhatsApp message types to NormalizedMessage.
"""

from datetime import datetime

import pytest

from transport.whatsapp.normalize import (
    NormalizationError,
    extract_sender_id,
    normalize_message,
)
from transport.whatsapp.schemas import NormalizedMessage


class TestNormalizationText:
    """Test text message normalization."""

    def test_normalize_text_message(self):
        """Text message normalizes correctly."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "wamid.msg_123",
                            "timestamp": "1707500000",
                            "type": "text",
                            "text": {"body": "Hello agent"},
                        }],
                    }
                }],
            }],
        }
        
        result = normalize_message(payload)
        
        assert isinstance(result, NormalizedMessage)
        assert result.input_text == "Hello agent"
        assert result.sender_id == "1234567890"
        assert result.message_id == "wamid.msg_123"
        assert result.transport == "whatsapp"
        assert result.input_type == "text"
        assert result.media_url is None

    def test_normalize_text_trims_whitespace(self):
        """Text is trimmed but not enriched."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "text",
                            "text": {"body": "  Hello  \n\t  "},
                        }],
                    }
                }],
            }],
        }
        
        result = normalize_message(payload)
        assert result.input_text == "Hello"

    def test_normalize_text_no_enrichment(self):
        """Text is not corrected or enriched."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "text",
                            "text": {"body": "hai im late"},  # Intentional typos
                        }],
                    }
                }],
            }],
        }
        
        result = normalize_message(payload)
        # Should NOT be corrected
        assert result.input_text == "hai im late"


class TestNormalizationAudio:
    """Test audio message normalization."""

    def test_normalize_audio_message(self):
        """Audio message normalizes with media URL."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "wamid.audio_123",
                            "timestamp": "1707500000",
                            "type": "audio",
                            "audio": {
                                "id": "audio_file_id",
                                "mime_type": "audio/mpeg",
                            },
                        }],
                    }
                }],
            }],
        }
        
        result = normalize_message(payload)
        
        assert isinstance(result, NormalizedMessage)
        assert result.input_type == "audio"
        assert result.input_text == ""  # MUST be empty
        assert result.media_url == "whatsapp://audio/audio_file_id"
        assert result.transport == "whatsapp"

    def test_normalize_audio_no_stt(self):
        """Audio message does NOT include STT."""
        # If this test fails because someone added STT, that's catching the bug
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "audio",
                            "audio": {
                                "id": "audio_id",
                                "mime_type": "audio/mpeg",
                            },
                        }],
                    }
                }],
            }],
        }
        
        result = normalize_message(payload)
        
        # input_text MUST be empty - no STT
        assert result.input_text == ""
        assert result.input_type == "audio"


class TestNormalizationImage:
    """Test image message normalization."""

    def test_normalize_image_message(self):
        """Image message normalizes with media URL."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "wamid.image_123",
                            "timestamp": "1707500000",
                            "type": "image",
                            "image": {
                                "id": "image_file_id",
                                "mime_type": "image/jpeg",
                            },
                        }],
                    }
                }],
            }],
        }
        
        result = normalize_message(payload)
        
        assert isinstance(result, NormalizedMessage)
        assert result.input_type == "image"
        assert result.input_text == ""  # MUST be empty
        assert result.media_url == "whatsapp://image/image_file_id"
        assert result.transport == "whatsapp"

    def test_normalize_image_no_vision(self):
        """Image message does NOT include vision inference."""
        # If this test fails because someone added vision, that's catching the bug
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "image",
                            "image": {
                                "id": "image_id",
                                "mime_type": "image/jpeg",
                            },
                        }],
                    }
                }],
            }],
        }
        
        result = normalize_message(payload)
        
        # input_text MUST be empty - no vision
        assert result.input_text == ""
        assert result.input_type == "image"


class TestNormalizationErrors:
    """Test error handling."""

    def test_invalid_payload_structure(self):
        """Invalid payload raises NormalizationError."""
        payload = {"invalid": "structure"}
        
        with pytest.raises(NormalizationError):
            normalize_message(payload)

    def test_missing_messages(self):
        """Missing messages raises NormalizationError."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [],
                    }
                }],
            }],
        }
        
        with pytest.raises(NormalizationError):
            normalize_message(payload)

    def test_unsupported_message_type(self):
        """Unsupported message type raises NormalizationError."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "video",  # Not supported yet
                        }],
                    }
                }],
            }],
        }
        
        with pytest.raises(NormalizationError):
            normalize_message(payload)

    def test_missing_text_body(self):
        """Missing text.body raises NormalizationError."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "text",
                            "text": {},  # Missing "body"
                        }],
                    }
                }],
            }],
        }
        
        with pytest.raises(NormalizationError):
            normalize_message(payload)

    def test_missing_audio_id(self):
        """Missing audio ID raises NormalizationError."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "audio",
                            "audio": {"mime_type": "audio/mpeg"},  # No ID
                        }],
                    }
                }],
            }],
        }
        
        with pytest.raises(NormalizationError):
            normalize_message(payload)


class TestExtractSenderId:
    """Test sender_id extraction."""

    def test_extract_sender_id(self):
        """Extract sender_id from payload."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "1234567890",
                            "id": "msg_123",
                            "timestamp": "1707500000",
                            "type": "text",
                            "text": {"body": "hello"},
                        }],
                    }
                }],
            }],
        }
        
        sender_id = extract_sender_id(payload)
        assert sender_id == "1234567890"

    def test_extract_sender_id_invalid_payload(self):
        """Invalid payload raises NormalizationError."""
        with pytest.raises(NormalizationError):
            extract_sender_id({"invalid": "structure"})
