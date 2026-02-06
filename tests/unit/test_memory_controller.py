"""
Test suite for memory controller boundary.

Verifies:
- Unauthorized read is blocked
- Unauthorized write is blocked
- Memory failure is non-fatal
- Memory is optional
"""

import pytest
from agent.memory import (
    MemoryController,
    StubMemoryController,
    DisabledMemoryController,
    MemoryReadRequest,
    MemoryWriteRequest,
)


class TestStubMemoryController:
    """Tests for StubMemoryController."""

    def test_read_success(self):
        """Verify authorized read retrieves data."""
        controller = StubMemoryController()
        
        # Write some data first
        write_req = MemoryWriteRequest(
            conversation_id="conv-123",
            key="user_prefs",
            data={"language": "en", "concise": True},
            authorized=True,
        )
        write_resp = controller.write(write_req)
        assert write_resp.status == "success"
        
        # Now read it
        read_req = MemoryReadRequest(
            conversation_id="conv-123",
            key="user_prefs",
            authorized=True,
        )
        read_resp = controller.read(read_req)
        
        assert read_resp.status == "success"
        assert read_resp.data == {"language": "en", "concise": True}

    def test_read_unauthorized_blocked(self):
        """Verify unauthorized read is blocked."""
        controller = StubMemoryController()
        
        # Write data
        write_req = MemoryWriteRequest(
            conversation_id="conv-123",
            key="user_prefs",
            data={"language": "en"},
            authorized=True,
        )
        controller.write(write_req)
        
        # Try to read without authorization
        read_req = MemoryReadRequest(
            conversation_id="conv-123",
            key="user_prefs",
            authorized=False,  # NOT authorized
        )
        read_resp = controller.read(read_req)
        
        assert read_resp.status == "unauthorized"
        assert read_resp.data is None

    def test_write_success(self):
        """Verify authorized write stores data."""
        controller = StubMemoryController()
        
        write_req = MemoryWriteRequest(
            conversation_id="conv-123",
            key="user_prefs",
            data={"language": "es"},
            authorized=True,
        )
        write_resp = controller.write(write_req)
        
        assert write_resp.status == "success"
        assert write_resp.error is None

    def test_write_unauthorized_blocked(self):
        """Verify unauthorized write is blocked."""
        controller = StubMemoryController()
        
        write_req = MemoryWriteRequest(
            conversation_id="conv-123",
            key="user_prefs",
            data={"language": "es"},
            authorized=False,  # NOT authorized
        )
        write_resp = controller.write(write_req)
        
        assert write_resp.status == "unauthorized"
        assert write_resp.error is not None

    def test_read_not_found(self):
        """Verify read returns not_found for missing data."""
        controller = StubMemoryController()
        
        read_req = MemoryReadRequest(
            conversation_id="nonexistent",
            key="user_prefs",
            authorized=True,
        )
        read_resp = controller.read(read_req)
        
        assert read_resp.status == "not_found"
        assert read_resp.data is None

    def test_multiple_conversations_isolated(self):
        """Verify memory is isolated per conversation."""
        controller = StubMemoryController()
        
        # Write to conv-1
        write_req1 = MemoryWriteRequest(
            conversation_id="conv-1",
            key="prefs",
            data={"lang": "en"},
            authorized=True,
        )
        controller.write(write_req1)
        
        # Write different data to conv-2
        write_req2 = MemoryWriteRequest(
            conversation_id="conv-2",
            key="prefs",
            data={"lang": "es"},
            authorized=True,
        )
        controller.write(write_req2)
        
        # Read from conv-1
        read_req1 = MemoryReadRequest(
            conversation_id="conv-1",
            key="prefs",
            authorized=True,
        )
        read_resp1 = controller.read(read_req1)
        assert read_resp1.data["lang"] == "en"
        
        # Read from conv-2
        read_req2 = MemoryReadRequest(
            conversation_id="conv-2",
            key="prefs",
            authorized=True,
        )
        read_resp2 = controller.read(read_req2)
        assert read_resp2.data["lang"] == "es"


class TestDisabledMemoryController:
    """Tests for DisabledMemoryController."""

    def test_read_always_unavailable(self):
        """Verify read always returns unavailable."""
        controller = DisabledMemoryController()
        
        read_req = MemoryReadRequest(
            conversation_id="conv-123",
            key="prefs",
            authorized=True,  # Even if authorized, still unavailable
        )
        read_resp = controller.read(read_req)
        
        assert read_resp.status == "unavailable"
        assert read_resp.data is None

    def test_write_always_fails(self):
        """Verify write always fails."""
        controller = DisabledMemoryController()
        
        write_req = MemoryWriteRequest(
            conversation_id="conv-123",
            key="prefs",
            data={"lang": "en"},
            authorized=True,  # Even if authorized, still fails
        )
        write_resp = controller.write(write_req)
        
        assert write_resp.status == "failed"
        assert write_resp.error is not None


class TestMemoryNeverRaises:
    """Verify memory operations never raise exceptions."""

    def test_stub_never_raises_on_read(self):
        """Memory read never raises, always returns response."""
        controller = StubMemoryController()
        
        # Try various malformed requests
        test_cases = [
            MemoryReadRequest(conversation_id="", key="", authorized=False),
            MemoryReadRequest(conversation_id="x" * 1000, key="y" * 1000, authorized=True),
            MemoryReadRequest(conversation_id=None, key=None, authorized=True),
        ]
        
        for req in test_cases:
            try:
                resp = controller.read(req)
                assert resp is not None  # Always returns response
                assert resp.status in ["success", "not_found", "unavailable", "unauthorized"]
            except Exception as e:
                pytest.fail(f"Memory read raised exception: {e}")

    def test_stub_never_raises_on_write(self):
        """Memory write never raises, always returns response."""
        controller = StubMemoryController()
        
        # Try various requests
        test_cases = [
            MemoryWriteRequest(conversation_id="", key="", data={}, authorized=False),
            MemoryWriteRequest(conversation_id="test", key="test", data={"nested": {"deep": "data"}}, authorized=True),
        ]
        
        for req in test_cases:
            try:
                resp = controller.write(req)
                assert resp is not None  # Always returns response
                assert resp.status in ["success", "failed", "unauthorized"]
            except Exception as e:
                pytest.fail(f"Memory write raised exception: {e}")


class TestMemoryDeterminism:
    """Verify memory is deterministic."""

    def test_stub_deterministic_read(self):
        """Same read twice returns same result."""
        controller = StubMemoryController()
        
        # Write data
        write_req = MemoryWriteRequest(
            conversation_id="conv-123",
            key="prefs",
            data={"lang": "en"},
            authorized=True,
        )
        controller.write(write_req)
        
        # Read twice
        read_req = MemoryReadRequest(
            conversation_id="conv-123",
            key="prefs",
            authorized=True,
        )
        resp1 = controller.read(read_req)
        resp2 = controller.read(read_req)
        
        assert resp1.status == resp2.status
        assert resp1.data == resp2.data

    def test_stub_deterministic_write(self):
        """Same write twice succeeds both times."""
        controller = StubMemoryController()
        
        write_req = MemoryWriteRequest(
            conversation_id="conv-123",
            key="prefs",
            data={"lang": "en"},
            authorized=True,
        )
        
        resp1 = controller.write(write_req)
        resp2 = controller.write(write_req)
        
        assert resp1.status == "success"
        assert resp2.status == "success"
