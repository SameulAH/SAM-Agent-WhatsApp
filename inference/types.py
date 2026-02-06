from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal

ModelStatus = Literal["success", "recoverable_error", "fatal_error"]


@dataclass
class ModelRequest:
    task: str                  # e.g. "respond", "summarize", "extract"
    prompt: str
    context: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    timeout_s: Optional[int] = 30
    trace_id: Optional[str] = None


@dataclass
class ModelResponse:
    status: ModelStatus
    output: Optional[str] = None
    error_type: Optional[str] = None   # timeout | invalid_output | backend_unavailable
    metadata: Optional[Dict[str, Any]] = None
