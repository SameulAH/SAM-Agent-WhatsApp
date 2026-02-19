"""
Unit tests for SQLite short-term memory adapter.

Tests persistence, commit semantics, and error handling.
"""

import pytest
import sqlite3
import tempfile
import os
import json
from pathlib import Path

from agent.memory.sqlite import SQLiteShortTermMemoryStore
from agent.memory.types import MemoryReadRequest, MemoryWriteRequest


class TestSqliteAdapter:
    """Unit tests for SQLiteShortTermMemoryStore."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for each test."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
        # Remove WAL and SHM files if they exist
        for ext in ["-wal", "-shm"]:
            wal_path = db_path + ext
            if os.path.exists(wal_path):
                os.remove(wal_path)

    @pytest.fixture
    def adapter(self, temp_db):
        """Create an adapter instance with temporary database."""
        return SQLiteShortTermMemoryStore(db_path=temp_db)

    def test_write_and_read_persistence(self, adapter):
        """Test that data written to SQLite persists across connections."""
        conversation_id = "test-conv-1"
        key = "test_key"
        data = {"message": "Hello, memory!"}
        
        # Write data
        write_request = MemoryWriteRequest(
            conversation_id=conversation_id,
            key=key,
            data=data,
            authorized=True,
            reason="test",
        )
        
        response = adapter.write(write_request)
        assert response.status == "success", f"Write failed: {response.error}"
        
        # Read data back
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key=key,
            authorized=True,
            reason="test",
        )
        
        response = adapter.read(read_request)
        assert response.status == "success"
        assert response.data == data

    def test_write_commit_semantics(self, adapter, temp_db):
        """Test that data is durably committed to disk."""
        conversation_id = "test-conv-2"
        key = "test_key"
        data = {"count": 42}
        
        # Write data
        write_request = MemoryWriteRequest(
            conversation_id=conversation_id,
            key=key,
            data=data,
            authorized=True,
            reason="test",
        )
        
        response = adapter.write(write_request)
        assert response.status == "success"
        
        # Create a NEW adapter instance to verify durability
        # (simulates process restart)
        new_adapter = SQLiteShortTermMemoryStore(db_path=temp_db)
        
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key=key,
            authorized=True,
            reason="test",
        )
        
        response = new_adapter.read(read_request)
        assert response.status == "success"
        assert response.data == data

    def test_unauthorized_write_rejected(self, adapter):
        """Test that unauthorized writes are rejected."""
        conversation_id = "test-conv-3"
        
        write_request = MemoryWriteRequest(
            conversation_id=conversation_id,
            key="test_key",
            data={"secret": "data"},
            authorized=False,  # Not authorized
            reason="test",
        )
        
        response = adapter.write(write_request)
        assert response.status == "unauthorized"
        
        # Verify data was NOT written
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key="test_key",
            authorized=True,
            reason="test",
        )
        
        response = adapter.read(read_request)
        assert response.status == "not_found"

    def test_unauthorized_read_rejected(self, adapter):
        """Test that unauthorized reads are rejected."""
        conversation_id = "test-conv-4"
        key = "test_key"
        
        # First write some data (authorized)
        write_request = MemoryWriteRequest(
            conversation_id=conversation_id,
            key=key,
            data={"data": "secret"},
            authorized=True,
            reason="test",
        )
        adapter.write(write_request)
        
        # Try to read without authorization
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key=key,
            authorized=False,  # Not authorized
            reason="test",
        )
        
        response = adapter.read(read_request)
        assert response.status == "unauthorized"

    def test_read_nonexistent_key(self, adapter):
        """Test reading a key that doesn't exist."""
        read_request = MemoryReadRequest(
            conversation_id="nonexistent-conv",
            key="nonexistent_key",
            authorized=True,
            reason="test",
        )
        
        response = adapter.read(read_request)
        assert response.status == "not_found"

    def test_upsert_overwrites_existing(self, adapter):
        """Test that writing to same key overwrites previous value."""
        conversation_id = "test-conv-5"
        key = "test_key"
        
        # Write first value
        write_request = MemoryWriteRequest(
            conversation_id=conversation_id,
            key=key,
            data={"version": 1},
            authorized=True,
            reason="test",
        )
        adapter.write(write_request)
        
        # Write new value (overwrite)
        write_request = MemoryWriteRequest(
            conversation_id=conversation_id,
            key=key,
            data={"version": 2},
            authorized=True,
            reason="test",
        )
        adapter.write(write_request)
        
        # Read and verify
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key=key,
            authorized=True,
            reason="test",
        )
        response = adapter.read(read_request)
        assert response.status == "success"
        assert response.data == {"version": 2}

    def test_multiple_conversations_isolated(self, adapter):
        """Test that memory is isolated per conversation."""
        conv1_id = "test-conv-6a"
        conv2_id = "test-conv-6b"
        key = "test_key"
        
        # Write to conversation 1
        write_request = MemoryWriteRequest(
            conversation_id=conv1_id,
            key=key,
            data={"conv": 1},
            authorized=True,
            reason="test",
        )
        adapter.write(write_request)
        
        # Write to conversation 2
        write_request = MemoryWriteRequest(
            conversation_id=conv2_id,
            key=key,
            data={"conv": 2},
            authorized=True,
            reason="test",
        )
        adapter.write(write_request)
        
        # Read from conversation 1
        read_request = MemoryReadRequest(
            conversation_id=conv1_id,
            key=key,
            authorized=True,
            reason="test",
        )
        response = adapter.read(read_request)
        assert response.data == {"conv": 1}
        
        # Read from conversation 2
        read_request = MemoryReadRequest(
            conversation_id=conv2_id,
            key=key,
            authorized=True,
            reason="test",
        )
        response = adapter.read(read_request)
        assert response.data == {"conv": 2}

    def test_json_serialization(self, adapter):
        """Test writing various JSON-serializable data types."""
        conversation_id = "test-conv-7"
        
        test_data = {
            "string": "value",
            "number": 123,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        
        write_request = MemoryWriteRequest(
            conversation_id=conversation_id,
            key="complex_data",
            data=test_data,
            authorized=True,
            reason="test",
        )
        
        response = adapter.write(write_request)
        assert response.status == "success"
        
        # Read back
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key="complex_data",
            authorized=True,
            reason="test",
        )
        response = adapter.read(read_request)
        assert response.data == test_data

    def test_non_serializable_data_fails(self, adapter):
        """Test that non-JSON-serializable data is rejected gracefully."""
        conversation_id = "test-conv-8"
        
        # Create non-serializable object
        import datetime
        bad_data = {"timestamp": datetime.datetime.now()}
        
        write_request = MemoryWriteRequest(
            conversation_id=conversation_id,
            key="bad_data",
            data=bad_data,
            authorized=True,
            reason="test",
        )
        
        response = adapter.write(write_request)
        assert response.status == "failed"
        assert "not JSON-serializable" in response.error

    def test_database_initialization_idempotent(self, temp_db):
        """Test that database initialization is safe to call multiple times."""
        adapter1 = SQLiteShortTermMemoryStore(db_path=temp_db)
        
        # Write something
        write_request = MemoryWriteRequest(
            conversation_id="test-conv-9",
            key="test_key",
            data={"data": 1},
            authorized=True,
            reason="test",
        )
        adapter1.write(write_request)
        
        # Create another adapter (should not corrupt database)
        adapter2 = SQLiteShortTermMemoryStore(db_path=temp_db)
        
        # Data should still be there
        read_request = MemoryReadRequest(
            conversation_id="test-conv-9",
            key="test_key",
            authorized=True,
            reason="test",
        )
        response = adapter2.read(read_request)
        assert response.status == "success"
        assert response.data == {"data": 1}

    def test_clear_conversation(self, adapter):
        """Test clearing all memory for a conversation."""
        conversation_id = "test-conv-10"
        
        # Write multiple keys
        for i in range(3):
            write_request = MemoryWriteRequest(
                conversation_id=conversation_id,
                key=f"key_{i}",
                data={"index": i},
                authorized=True,
                reason="test",
            )
            adapter.write(write_request)
        
        # Verify data exists
        read_request = MemoryReadRequest(
            conversation_id=conversation_id,
            key="key_0",
            authorized=True,
            reason="test",
        )
        response = adapter.read(read_request)
        assert response.status == "success"
        
        # Clear conversation
        result = adapter.clear_conversation(conversation_id)
        assert result is True
        
        # Verify data is gone
        response = adapter.read(read_request)
        assert response.status == "not_found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
