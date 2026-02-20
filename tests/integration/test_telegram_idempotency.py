"""
tests/integration/test_telegram_idempotency.py

Integration tests for idempotent Telegram webhook processing.

What is verified:
  • The same update_id is processed exactly once even when the webhook
    receives the identical payload twice (Telegram retry simulation).
  • send_message / send_response is called exactly once per unique update.
  • A second distinct update_id is processed independently (not blocked).
  • The handler always returns HTTP 200 so Telegram stops retrying.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# ── Bootstrap: stub out modules that are not installed in the local dev venv
# (they live in Docker). Must happen at MODULE LOAD TIME before any webhook
# import triggers the full import chain.
# ──────────────────────────────────────────────────────────────────────────────
def _stub_module(name: str) -> MagicMock:
    """Return a MagicMock that looks like a package."""
    m = MagicMock()
    m.__path__ = []          # satisfies subpackage checks
    m.__spec__ = MagicMock()
    sys.modules[name] = m
    return m


_HEAVY_DEPS = [
    # Telegram transport (local package, not installed in test venv)
    "transport",
    "transport.telegram",
    "transport.telegram.transport",
    # Audio / STT services pulled in by telegram_voice.py
    "services",
    "services.audio",
    "services.tts",
    "services.tts.coqui",
    "services.tts.base",
    # Agent / tracing (heavy LangGraph deps)
    "agent",
    "agent.tracing",
    "agent.orchestrator",
    "agent.intelligence",
    "agent.intelligence.tools",
    # LangGraph itself (not installed locally)
    "langgraph",
    "langchain",
]

for _dep in _HEAVY_DEPS:
    if _dep not in sys.modules:
        _stub_module(_dep)

# Provide concrete mock values for names imported BY NAME in the source files
# (e.g. "from transport.telegram.transport import create_telegram_transport")
_transport_mod = sys.modules["transport.telegram.transport"]
_transport_mod.create_telegram_transport = MagicMock(return_value=MagicMock())
_transport_mod.NormalizedMessage = MagicMock

_audio_mod = sys.modules["services.audio"]
_audio_mod.get_audio_normalizer = MagicMock(return_value=MagicMock())
_audio_mod.get_stt_processor = MagicMock(return_value=MagicMock())

_tracing_mod = sys.modules["agent.tracing"]
_tracing_mod.TraceMetadata = MagicMock
_tracing_mod.NoOpTracer = MagicMock
_tracing_mod.tracer_factory = MagicMock()
_tracing_mod.tracer_factory.create_tracer = MagicMock(return_value=MagicMock())

# ── Now it is safe to import from our webhook modules ─────────────────────────

import asyncio
from fastapi import BackgroundTasks
import pytest

# ── Reset the dedup cache between tests ──────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_dedup_cache():
    """Empty the TTLCache before every test to avoid cross-test pollution."""
    # Import here (not at module level) so the stub injection above has taken
    # effect first.
    from webhook.telegram import _processed_updates
    _processed_updates.clear()
    yield
    _processed_updates.clear()


# ── Minimal Telegram update fixtures ─────────────────────────────────────────

def _text_update(update_id: int, text: str = "Hello") -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 1700000000,
            "chat": {"id": 111, "type": "private"},
            "from": {"id": 999, "first_name": "Tester"},
            "text": text,
        },
    }


# ── Unit-level handler tests (no HTTP stack) ─────────────────────────────────

class TestDeduplicationUnit:
    """
    Call telegram_webhook directly — no HTTP overhead.
    BackgroundTasks are inspected without actually executing them so the
    agent / transport are never touched.
    """

    @pytest.mark.asyncio
    async def test_first_update_accepted(self):
        from webhook.telegram import telegram_webhook, TelegramUpdate

        bt = BackgroundTasks()
        update = TelegramUpdate(**_text_update(42))

        with patch("webhook.telegram._process_text_async", new_callable=AsyncMock) as mock_proc:
            result = await telegram_webhook(update, bt)

        assert result == {"status": "ok"}
        # Background task was enqueued (bt.tasks has one entry)
        assert len(bt.tasks) == 1

    @pytest.mark.asyncio
    async def test_duplicate_update_id_dropped(self):
        from webhook.telegram import telegram_webhook, TelegramUpdate

        bt1 = BackgroundTasks()
        bt2 = BackgroundTasks()
        update = TelegramUpdate(**_text_update(42))

        with patch("webhook.telegram._process_text_async", new_callable=AsyncMock):
            await telegram_webhook(update, bt1)
            result = await telegram_webhook(update, bt2)

        assert result == {"status": "ok"}
        # Second call must NOT enqueue another background task
        assert len(bt2.tasks) == 0, "Duplicate update must not enqueue a task"

    @pytest.mark.asyncio
    async def test_distinct_update_ids_both_processed(self):
        from webhook.telegram import telegram_webhook, TelegramUpdate

        bt1 = BackgroundTasks()
        bt2 = BackgroundTasks()

        update_a = TelegramUpdate(**_text_update(100))
        update_b = TelegramUpdate(**_text_update(101))

        with patch("webhook.telegram._process_text_async", new_callable=AsyncMock):
            await telegram_webhook(update_a, bt1)
            await telegram_webhook(update_b, bt2)

        assert len(bt1.tasks) == 1, "First distinct update must be enqueued"
        assert len(bt2.tasks) == 1, "Second distinct update must be enqueued"

    @pytest.mark.asyncio
    async def test_same_update_id_sent_three_times(self):
        """Simulate Telegram retrying a slow webhook 3 times."""
        from webhook.telegram import telegram_webhook, TelegramUpdate

        bts = [BackgroundTasks() for _ in range(3)]
        update = TelegramUpdate(**_text_update(77))

        with patch("webhook.telegram._process_text_async", new_callable=AsyncMock):
            for bt in bts:
                await telegram_webhook(update, bt)

        tasks_enqueued = sum(len(bt.tasks) for bt in bts)
        assert tasks_enqueued == 1, (
            f"Expected exactly 1 background task across 3 identical updates, got {tasks_enqueued}"
        )


# ── HTTP-level tests (TestClient) ────────────────────────────────────────────

class TestDeduplicationHTTP:
    """
    Exercise the full FastAPI routing stack via TestClient.
    The agent, transport, and background tasks are all patched out.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app, raise_server_exceptions=True)

    def test_webhook_returns_200_first_call(self, client):
        with patch("webhook.telegram._process_text_async", new_callable=AsyncMock):
            resp = client.post("/webhook/telegram", json=_text_update(200))
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_webhook_returns_200_on_duplicate(self, client):
        payload = _text_update(201)
        with patch("webhook.telegram._process_text_async", new_callable=AsyncMock):
            resp1 = client.post("/webhook/telegram", json=payload)
            resp2 = client.post("/webhook/telegram", json=payload)

        assert resp1.status_code == 200
        assert resp2.status_code == 200  # always 200 — never let Telegram retry


# ── Voice handler double-send regression test ────────────────────────────────

class TestVoiceHandlerNoDoubleSend:
    """
    Ensure _process_voice_async sends exactly ONE message when TTS fails.
    Before the fix, it would send text via the fallback line inside the
    success branch AND again via the outer send_message call — two sends.
    """

    @pytest.mark.asyncio
    async def test_tts_failure_sends_exactly_one_message(self):
        from webhook.telegram_voice import _process_voice_async

        send_calls: list = []

        # Fake transport that records every send_message call
        fake_transport = MagicMock()
        fake_transport.process_voice_message = AsyncMock(
            return_value=MagicMock(
                status="success",
                transcribed_text="hello world",
                confidence=0.95,
                error=None,
            )
        )
        fake_transport.send_message = AsyncMock(
            side_effect=lambda chat_id, text: send_calls.append(text)
        )
        # Voice send returns False → triggers text fallback path
        fake_transport.send_voice_message = AsyncMock(return_value=False)

        # Stub out the agent — returns a simple reply
        mock_agent = MagicMock()
        mock_agent.invoke = AsyncMock(return_value={"final_output": "Agent reply"})
        mock_orchestrator_cls = MagicMock(return_value=mock_agent)

        with (
            patch("webhook.telegram_voice.get_voice_transport", return_value=fake_transport),
            patch("webhook.telegram_voice.text_to_ogg_voice", return_value=None),
            patch(
                "webhook.telegram_voice.SAMOrchestrator",
                mock_orchestrator_cls,
                create=True,
            ),
        ):
            await _process_voice_async(
                message_id=1,
                chat_id=42,
                user_id=99,
                file_id="file123",
                duration=5,
                user_name="Tester",
                trace_id="test-trace",
                conversation_id="telegram:42",
            )

        assert len(send_calls) == 1, (
            f"Expected exactly 1 send_message call when TTS fails, "
            f"got {len(send_calls)}: {send_calls}"
        )
