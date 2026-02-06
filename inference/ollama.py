import requests
from .base import ModelBackend
from .types import ModelRequest, ModelResponse


class OllamaModelBackend(ModelBackend):
    """
    Ollama backend for local model inference.
    
    Used for local / exploratory runs only.
    Requires Ollama to be running at the specified base_url.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama backend.
        
        Args:
            model_name: Name of the model to use (e.g., "phi3:mini", "llama2")
            base_url: Base URL of Ollama service
        """
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Generate a response using Ollama backend.
        
        Args:
            request: ModelRequest with prompt and optional parameters
            
        Returns:
            ModelResponse with output from Ollama or explicit error status
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": request.prompt,
                "stream": False,
            }

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=request.timeout_s,
            )

            resp.raise_for_status()
            data = resp.json()

            return ModelResponse(
                status="success",
                output=data.get("response", ""),
                metadata={
                    "backend": "ollama",
                    "model": self.model_name,
                    "trace_id": request.trace_id
                },
            )

        except requests.Timeout:
            return ModelResponse(
                status="recoverable_error",
                error_type="timeout",
                metadata={
                    "backend": "ollama",
                    "model": self.model_name,
                    "trace_id": request.trace_id
                },
            )

        except Exception as e:
            return ModelResponse(
                status="fatal_error",
                error_type="backend_unavailable",
                metadata={
                    "backend": "ollama",
                    "model": self.model_name,
                    "error": str(e),
                    "trace_id": request.trace_id
                },
            )
