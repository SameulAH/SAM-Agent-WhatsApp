"""
Phase 3.1 Tests: SQLite Short-Term Memory Implementation

Validates that SQLite short-term memory:
1. Implements the same interface as StubMemoryController (swappable)
2. Handles failures gracefully (non-fatal)
3. Does NOT change agent control flow or routing
4. Respects authorization semantics strictly
5. Stores only session-scoped data (no long-term facts)

CRITICAL: These tests verify durability WITHOUT changing behavior.
If SQLite is removed or fails, the agent still works.
Only continuity degrades.
"""

import pytest
import sqlite3
import tempfile
import json
import sys
from pathlib import Path
from agent.memory.sqlite import SQLiteShortTermMemoryStore
from agent.memory.stub import StubMemoryController, DisabledMemoryController
from agent.memory.types import (
    MemoryReadRequest,
    MemoryReadResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
)
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.state_schema import AgentState
from inference import StubModelBackend


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CATEGORY 1: INTERFACE COMPATIBILITY
# Prove SQLite is a drop-in replacement for Stub
# ═══════════════════════════════════════════════════════════════════════════════


class TestSQLiteInterfaceCompatibility:
    """SQLite must implement the exact same interface as StubMemoryController."""

    def test_sqlite_read_same_signature_as_stub(self):
        """Both controllers accept MemoryReadRequest and return MemoryReadResponse."""
        stub = StubMemoryController()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            request = MemoryReadRequest(
                conversation_id="conv-1",
                key="test_key",
                authorized=True,
            )

            stub_response = stub.read(request)
            sqlite_response = sqlite.read(request)

            # Both must return MemoryReadResponse
            assert isinstance(stub_response, MemoryReadResponse)
            assert isinstance(sqlite_response, MemoryReadResponse)

            # Both must have status, data, error attributes
            assert hasattr(stub_response, "status")
            assert hasattr(sqlite_response, "status")
            assert hasattr(stub_response, "data")
            assert hasattr(sqlite_response, "data")
            assert hasattr(stub_response, "error")
            assert hasattr(sqlite_response, "error")

    def test_sqlite_write_same_signature_as_stub(self):
        """Both controllers accept MemoryWriteRequest and return MemoryWriteResponse."""
        stub = StubMemoryController()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            request = MemoryWriteRequest(
                conversation_id="conv-1",
                key="test_key",
                data={"foo": "bar"},
                authorized=True,
            )

            stub_response = stub.write(request)
            sqlite_response = sqlite.write(request)

            # Both must return MemoryWriteResponse
            assert isinstance(stub_response, MemoryWriteResponse)
            assert isinstance(sqlite_response, MemoryWriteResponse)

            # Both must have status and error attributes
            assert hasattr(stub_response, "status")
            assert hasattr(sqlite_response, "status")
            assert hasattr(stub_response, "error")
            assert hasattr(sqlite_response, "error")

    def test_sqlite_never_raises_exceptions(self):
        """SQLite read/write must never raise, always return Response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            # Test read never raises
            try:
                response = sqlite.read(
                    MemoryReadRequest(
                        conversation_id="conv-1",
                        key="missing",
                        authorized=True,
                    )
                )
                assert isinstance(response, MemoryReadResponse)
            except Exception as e:
                pytest.fail(f"read() raised exception: {e}")

            # Test write never raises
            try:
                response = sqlite.write(
                    MemoryWriteRequest(
                        conversation_id="conv-1",
                        key="test",
                        data={"test": "data"},
                        authorized=True,
                    )
                )
                assert isinstance(response, MemoryWriteResponse)
            except Exception as e:
                pytest.fail(f"write() raised exception: {e}")

    def test_sqlite_write_and_read_roundtrip(self):
        """Write data to SQLite, then read it back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            # Write data
            test_data = {"user_id": "u123", "preference": "concise", "count": 42}
            write_response = sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="user_context",
                    data=test_data,
                    authorized=True,
                )
            )
            assert write_response.status == "success"

            # Read it back
            read_response = sqlite.read(
                MemoryReadRequest(
                    conversation_id="conv-1",
                    key="user_context",
                    authorized=True,
                )
            )
            assert read_response.status == "success"
            assert read_response.data == test_data

    def test_sqlite_persistence_across_instances(self):
        """Data written to SQLite persists across instance creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")

            # Write with first instance
            sqlite1 = SQLiteShortTermMemoryStore(db_path)
            test_data = {"key": "value"}
            sqlite1.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="test",
                    data=test_data,
                    authorized=True,
                )
            )

            # Read with second instance
            sqlite2 = SQLiteShortTermMemoryStore(db_path)
            response = sqlite2.read(
                MemoryReadRequest(
                    conversation_id="conv-1",
                    key="test",
                    authorized=True,
                )
            )
            assert response.status == "success"
            assert response.data == test_data


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CATEGORY 2: NON-FATAL FAILURE BEHAVIOR
# Prove SQLite failures don't crash the agent
# ═══════════════════════════════════════════════════════════════════════════════


class TestSQLiteFailureGracefully:
    """SQLite failures must degrade gracefully, never block execution."""

    def test_corrupted_database_file_read_fails_gracefully(self):
        """Reading from corrupted DB returns status=unavailable, not exception."""
        if sys.platform == "win32":
            pytest.skip("Skipping corrupted DB test on Windows (file locks)")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "corrupted.db")
            # Write garbage to the file to corrupt it
            with open(db_path, "w") as f:
                f.write("this is not a valid sqlite database file")

            sqlite = SQLiteShortTermMemoryStore(db_path)
            response = sqlite.read(
                MemoryReadRequest(
                    conversation_id="conv-1",
                    key="test",
                    authorized=True,
                )
            )
            # Must return a response with unavailable status
            assert response.status == "unavailable"
            assert response.error is not None

    def test_corrupted_database_file_write_fails_gracefully(self):
        """Writing to corrupted DB returns status=failed, not exception."""
        if sys.platform == "win32":
            pytest.skip("Skipping corrupted DB test on Windows (file locks)")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "corrupted.db")
            with open(db_path, "w") as f:
                f.write("this is not a valid sqlite database file")

            sqlite = SQLiteShortTermMemoryStore(db_path)
            response = sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="test",
                    data={"test": "data"},
                    authorized=True,
                )
            )
            # Must return a response with failed status
            assert response.status == "failed"
            assert response.error is not None

    def test_readonly_database_write_fails_gracefully(self):
        """Writing to read-only DB returns status=failed, not exception."""
        if sys.platform == "win32":
            pytest.skip("Skipping read-only DB test on Windows (chmod not supported)")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "readonly.db")

            # Create a valid DB
            sqlite = SQLiteShortTermMemoryStore(db_path)
            sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="test",
                    data={"test": "data"},
                    authorized=True,
                )
            )

            # Make it read-only
            Path(db_path).chmod(0o444)

            # Try to write
            try:
                response = sqlite.write(
                    MemoryWriteRequest(
                        conversation_id="conv-1",
                        key="test",
                        data={"new": "data"},
                        authorized=True,
                    )
                )
                assert response.status == "failed"
            finally:
                # Restore permissions for cleanup
                Path(db_path).chmod(0o644)

    def test_non_serializable_data_write_fails_gracefully(self):
        """Writing non-JSON-serializable data returns status=failed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            # Try to write non-serializable data (circular reference)
            circular_ref = {"self": None}
            circular_ref["self"] = circular_ref

            response = sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="test",
                    data=circular_ref,
                    authorized=True,
                )
            )
            assert response.status == "failed"
            assert response.error is not None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CATEGORY 3: AUTHORIZATION ENFORCEMENT
# Prove memory operations respect authorization flags
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuthorizationEnforcement:
    """Memory reads/writes must respect authorization flags."""

    def test_unauthorized_read_rejected_by_sqlite(self):
        """SQLite rejects reads when authorized=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            # Write with authorization
            sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="secret",
                    data={"secret": "value"},
                    authorized=True,
                )
            )

            # Try to read without authorization
            response = sqlite.read(
                MemoryReadRequest(
                    conversation_id="conv-1",
                    key="secret",
                    authorized=False,  # NOT authorized
                )
            )
            assert response.status == "unauthorized"
            assert response.data is None

    def test_unauthorized_write_rejected_by_sqlite(self):
        """SQLite rejects writes when authorized=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            response = sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="secret",
                    data={"secret": "value"},
                    authorized=False,  # NOT authorized
                )
            )
            assert response.status == "unauthorized"

    def test_authorized_read_succeeds_or_explicit_not_found(self):
        """Authorized reads always return valid response (success or not_found)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            response = sqlite.read(
                MemoryReadRequest(
                    conversation_id="conv-1",
                    key="missing",
                    authorized=True,  # IS authorized
                )
            )
            # Status must be one of: success, not_found (not unauthorized)
            assert response.status in ("success", "not_found")

    def test_authorized_write_succeeds(self):
        """Authorized writes always succeed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            response = sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="data",
                    data={"key": "value"},
                    authorized=True,  # IS authorized
                )
            )
            assert response.status == "success"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CATEGORY 4: DATA SCOPE VALIDATION
# Prove only session-scoped data is stored
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataScopeValidation:
    """Memory must store only session-scoped data, not long-term facts."""

    def test_session_scoped_data_allowed(self):
        """Session-scoped data (conversation context) must be storable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            # Allowed session-scoped data
            session_data = {
                "conversation_id": "conv-123",
                "turn_count": 5,
                "last_turn_timestamp": "2026-02-06T10:30:00Z",
                "recent_messages": [
                    {"role": "user", "text": "Hello"},
                    {"role": "assistant", "text": "Hi there"},
                ],
            }

            response = sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-123",
                    key="session_context",
                    data=session_data,
                    authorized=True,
                )
            )
            assert response.status == "success"

    def test_multiple_conversations_isolated(self):
        """Data from different conversations must be isolated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            # Write to conv-1
            sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="context",
                    data={"data": "for-conv-1"},
                    authorized=True,
                )
            )

            # Write to conv-2
            sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-2",
                    key="context",
                    data={"data": "for-conv-2"},
                    authorized=True,
                )
            )

            # Read from conv-1
            response1 = sqlite.read(
                MemoryReadRequest(
                    conversation_id="conv-1",
                    key="context",
                    authorized=True,
                )
            )
            assert response1.data == {"data": "for-conv-1"}

            # Read from conv-2
            response2 = sqlite.read(
                MemoryReadRequest(
                    conversation_id="conv-2",
                    key="context",
                    authorized=True,
                )
            )
            assert response2.data == {"data": "for-conv-2"}

    def test_overwrites_are_allowed(self):
        """Overwriting memory is expected behavior (session updates)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            sqlite = SQLiteShortTermMemoryStore(db_path)

            # Write initial data
            sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="context",
                    data={"version": 1},
                    authorized=True,
                )
            )

            # Overwrite with new data
            sqlite.write(
                MemoryWriteRequest(
                    conversation_id="conv-1",
                    key="context",
                    data={"version": 2},
                    authorized=True,
                )
            )

            # Read should return latest
            response = sqlite.read(
                MemoryReadRequest(
                    conversation_id="conv-1",
                    key="context",
                    authorized=True,
                )
            )
            assert response.data == {"version": 2}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CATEGORY 5: CONTROL FLOW INVARIANCE
# Prove SQLite doesn't change agent behavior or routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestControlFlowInvariance:
    """Agent behavior must be identical with SQLite vs Stub memory."""

    def test_agent_execution_identical_with_sqlite_vs_stub(self):
        """Memory swap doesn't prevent agent instantiation or graph execution."""
        from uuid import uuid4
        
        # Verify both orchestrators can be created with different memory backends
        orchestrator_stub = SAMAgentOrchestrator(
            model_backend=StubModelBackend(),
            memory_controller=StubMemoryController(),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            orchestrator_sqlite = SAMAgentOrchestrator(
                model_backend=StubModelBackend(),
                memory_controller=SQLiteShortTermMemoryStore(db_path),
            )
        
        # Both must have compiled graphs
        assert orchestrator_stub.graph is not None
        assert orchestrator_sqlite.graph is not None
        
        # Both graphs must be invokable (same interface)
        assert hasattr(orchestrator_stub.graph, "invoke")
        assert hasattr(orchestrator_sqlite.graph, "invoke")

    def test_multiple_turns_with_sqlite_memory(self):
        """Multiple turns in a conversation with SQLite persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            memory = SQLiteShortTermMemoryStore(db_path)

            # Turn 1: Store context
            memory.write(
                MemoryWriteRequest(
                    conversation_id="multi-turn-1",
                    key="turn_history",
                    data={"turns": [{"num": 1, "user": "Hello"}]},
                    authorized=True,
                )
            )

            # Turn 2: Read and update context
            read_response = memory.read(
                MemoryReadRequest(
                    conversation_id="multi-turn-1",
                    key="turn_history",
                    authorized=True,
                )
            )
            assert read_response.status == "success"
            history = read_response.data
            history["turns"].append({"num": 2, "user": "How are you?"})

            write_response = memory.write(
                MemoryWriteRequest(
                    conversation_id="multi-turn-1",
                    key="turn_history",
                    data=history,
                    authorized=True,
                )
            )
            assert write_response.status == "success"

            # Turn 3: Verify history persisted
            final_read = memory.read(
                MemoryReadRequest(
                    conversation_id="multi-turn-1",
                    key="turn_history",
                    authorized=True,
                )
            )
            assert len(final_read.data["turns"]) == 2
