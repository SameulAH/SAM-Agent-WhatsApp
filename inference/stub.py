from .base import ModelBackend
from .types import ModelRequest, ModelResponse


class StubModelBackend(ModelBackend):
    """
    Deterministic fake model for testing and CI.
    
    This backend is fast, deterministic, and never fails silently.
    Used as the default backend for all CI/test environments.
    """

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Generate a deterministic response based on task type.
        
        Args:
            request: ModelRequest with task, prompt, and optional parameters
            
        Returns:
            ModelResponse with deterministic output based on task
        """
        # Deterministic behavior based on task
        if request.task == "respond":
            return ModelResponse(
                status="success",
                output="This is a stubbed response.",
                metadata={"backend": "stub", "trace_id": request.trace_id}
            )

        if request.task == "fail":
            return ModelResponse(
                status="recoverable_error",
                error_type="invalid_output",
                metadata={"backend": "stub", "trace_id": request.trace_id}
            )

        # Default stub output for any other task
        return ModelResponse(
            status="success",
            output=f"Default stub output for task: {request.task}",
            metadata={"backend": "stub", "trace_id": request.trace_id}
        )
