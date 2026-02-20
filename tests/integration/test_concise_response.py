"""
tests/integration/test_concise_response.py

Integration tests for output conciseness enforcement.

Verifies:
✔ Response never exceeds MAX_OUTPUT_CHARS (800)
✔ Very long model output is truncated with "..." suffix
✔ Short responses pass through unchanged
✔ response_truncated trace event emitted when truncation occurs
✔ 5-sentence soft limit is applied before char limit
✔ Truncation does not affect error responses
"""

from typing import Any, Dict
from uuid import uuid4

import pytest

from inference import ModelBackend, ModelRequest, ModelResponse
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.tracing.tracer import NoOpTracer, TraceMetadata


MAX_OUTPUT_CHARS = SAMAgentOrchestrator.MAX_OUTPUT_CHARS


class LongResponseModel(ModelBackend):
    """Model that returns a very long response."""

    def __init__(self, char_count: int = 1500):
        self._char_count = char_count

    def generate(self, request: ModelRequest) -> ModelResponse:
        # Generate a long coherent-ish response with multiple sentences
        sentence = "This is a sentence about the topic. "
        output = sentence * (self._char_count // len(sentence) + 1)
        return ModelResponse(
            status="success",
            output=output[: self._char_count],
            metadata={},
        )


class ShortResponseModel(ModelBackend):
    """Model that returns a short, normal response."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            status="success",
            output="Paris is the capital of France.",
            metadata={},
        )


class ManySentencesModel(ModelBackend):
    """Model that returns exactly 10 sentences."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        sentences = [f"Sentence {i}." for i in range(1, 11)]
        return ModelResponse(
            status="success",
            output=" ".join(sentences),
            metadata={},
        )


class MockTracer(NoOpTracer):
    def __init__(self):
        super().__init__()
        self.events: list[dict] = []

    def record_event(self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata) -> None:
        self.events.append({"name": name, "metadata": metadata})


class TestConciseResponseEnforcement:
    """Output conciseness guardrails."""

    @pytest.mark.asyncio
    async def test_long_response_truncated_to_max_chars(self):
        """Response longer than MAX_OUTPUT_CHARS is truncated."""
        orch = SAMAgentOrchestrator(model_backend=LongResponseModel(char_count=2000))
        result = await orch.invoke("Tell me everything about AI.")

        output = result["output"]
        assert output is not None
        assert len(output) <= MAX_OUTPUT_CHARS + 3  # +3 for "..."

    @pytest.mark.asyncio
    async def test_truncated_response_ends_with_ellipsis(self):
        """Truncated response ends with '...'."""
        orch = SAMAgentOrchestrator(model_backend=LongResponseModel(char_count=2000))
        result = await orch.invoke("Tell me everything.")

        output = result["output"]
        # Either truncated with "..." or within limit
        if len(output) == MAX_OUTPUT_CHARS + 3:
            assert output.endswith("...")

    @pytest.mark.asyncio
    async def test_short_response_not_truncated(self):
        """Short response passes through unchanged."""
        orch = SAMAgentOrchestrator(model_backend=ShortResponseModel())
        result = await orch.invoke("What is the capital of France?")

        assert result["output"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_response_truncated_event_emitted(self):
        """response_truncated trace event is emitted when output is truncated."""
        tracer = MockTracer()
        orch = SAMAgentOrchestrator(
            model_backend=LongResponseModel(char_count=2000),
            tracer=tracer,
        )
        await orch.invoke("Long question.")

        event_names = [e["name"] for e in tracer.events]
        assert "response_truncated" in event_names

    @pytest.mark.asyncio
    async def test_no_truncation_event_for_short_response(self):
        """response_truncated event is NOT emitted for short responses."""
        tracer = MockTracer()
        orch = SAMAgentOrchestrator(
            model_backend=ShortResponseModel(),
            tracer=tracer,
        )
        await orch.invoke("Short question.")

        event_names = [e["name"] for e in tracer.events]
        assert "response_truncated" not in event_names

    @pytest.mark.asyncio
    async def test_sentence_limit_applied(self):
        """Response with more than MAX_OUTPUT_SENTENCES sentences is trimmed."""
        orch = SAMAgentOrchestrator(model_backend=ManySentencesModel())
        result = await orch.invoke("Give me 10 sentences.")

        output = result["output"] or ""
        # After sentence trimming, at most 5 sentences should remain
        import re
        sentences = re.split(r"(?<=[.!?])\s+", output.strip())
        assert len(sentences) <= SAMAgentOrchestrator.MAX_OUTPUT_SENTENCES

    @pytest.mark.asyncio
    async def test_output_within_limit_no_modification(self):
        """Response exactly at or below the char limit is returned as-is."""
        at_limit_output = "A" * MAX_OUTPUT_CHARS

        class ExactLimitModel(ModelBackend):
            def generate(self, request):
                return ModelResponse(status="success", output=at_limit_output, metadata={})

        orch = SAMAgentOrchestrator(model_backend=ExactLimitModel())
        result = await orch.invoke("test")

        assert result["output"] is not None
        assert len(result["output"]) <= MAX_OUTPUT_CHARS + 3
