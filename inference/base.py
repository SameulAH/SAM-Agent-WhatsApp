from abc import ABC, abstractmethod
from .types import ModelRequest, ModelResponse


class ModelBackend(ABC):
    """
    Abstract model boundary.
    Agent code must depend ONLY on this interface.
    """

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response from the model."""
        raise NotImplementedError
