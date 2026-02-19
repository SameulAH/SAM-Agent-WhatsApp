"""
Health check endpoints for deployment readiness.

Provides:
- /health/live: Liveness probe (agent is running)
- /health/ready: Readiness probe (agent is ready to serve)

Both endpoints reflect agent state WITHOUT external service dependencies.
"""

from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class HealthStatus:
    """Health status response."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    agent_ready: bool
    uptime_seconds: float
    mode: str  # "stub", "local", "prod"
    config_profile: str  # "dev", "prod-cpu", "prod-gpu"
    optional_services: Dict[str, bool]
    message: str
    metadata: Dict[str, Any]


class HealthChecker:
    """
    Health checker for agent readiness.
    
    Invariant: Health checks do NOT verify external services.
    Only verify that the agent itself is ready to run.
    """
    
    def __init__(self, start_time: float):
        """Initialize health checker."""
        self.start_time = start_time
    
    def get_mode(self) -> str:
        """Determine operating mode from configuration."""
        llm_backend = os.getenv("LLM_BACKEND", "stub")
        if llm_backend == "stub":
            return "stub"
        elif llm_backend == "ollama":
            return "local"
        else:
            return "prod"
    
    def get_profile(self) -> str:
        """Determine build profile from environment."""
        # Check if CUDA/GPU is available
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False
        
        if cuda_available:
            return "prod-gpu"
        else:
            return "prod-cpu"
    
    def check_optional_services(self) -> Dict[str, bool]:
        """Check if optional services are configured."""
        return {
            "stt": os.getenv("STT_ENABLED", "false").lower() == "true",
            "tts": os.getenv("TTS_ENABLED", "false").lower() == "true",
            "lte_qdrant": os.getenv("LTM_BACKEND", "stub") == "qdrant",
        }
    
    def check_live(self) -> HealthStatus:
        """
        Liveness probe: Is the agent process running?
        
        Always returns healthy if this endpoint responds.
        """
        import time
        uptime = time.time() - self.start_time
        
        return HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            agent_ready=True,
            uptime_seconds=uptime,
            mode=self.get_mode(),
            config_profile=self.get_profile(),
            optional_services=self.check_optional_services(),
            message="Agent process is running",
            metadata={
                "llm_backend": os.getenv("LLM_BACKEND", "stub"),
                "stt_enabled": os.getenv("STT_ENABLED", "false"),
                "tts_enabled": os.getenv("TTS_ENABLED", "false"),
                "ltm_backend": os.getenv("LTM_BACKEND", "stub"),
            }
        )
    
    def check_ready(self) -> HealthStatus:
        """
        Readiness probe: Is the agent ready to serve requests?
        
        Agent is ready if:
        - Core agent logic can execute
        - Memory interfaces are initialized
        - Stub backends are available
        
        Agent is NOT blocked by:
        - Ollama unavailability
        - Qdrant unavailability
        - Whisper/Coqui unavailability
        """
        import time
        uptime = time.time() - self.start_time
        
        # Check if we can import core agent modules
        try:
            from agent.orchestrator import SAMOrchestrator  # noqa: F401
            from agent.state_schema import AgentState  # noqa: F401
            from agent.memory import StubMemoryController  # noqa: F401
            from agent.memory import StubLongTermMemoryStore  # noqa: F401
            from inference import StubModelBackend  # noqa: F401
            
            agent_logic_ok = True
            agent_message = "Agent logic initialized"
        except ImportError as e:
            agent_logic_ok = False
            agent_message = f"Agent logic initialization failed: {str(e)}"
        
        # Check if we can bootstrap infrastructure
        try:
            from infra import bootstrap_infrastructure  # noqa: F401
            bootstrap_ok = True
            bootstrap_message = "Infrastructure bootstrap available"
        except ImportError as e:
            bootstrap_ok = False
            bootstrap_message = f"Infrastructure bootstrap failed: {str(e)}"
        
        # Determine overall readiness
        ready = agent_logic_ok and bootstrap_ok
        status = "healthy" if ready else "unhealthy"
        message = f"{agent_message}; {bootstrap_message}"
        
        return HealthStatus(
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            agent_ready=ready,
            uptime_seconds=uptime,
            mode=self.get_mode(),
            config_profile=self.get_profile(),
            optional_services=self.check_optional_services(),
            message=message,
            metadata={
                "agent_logic_ok": agent_logic_ok,
                "bootstrap_ok": bootstrap_ok,
                "llm_backend": os.getenv("LLM_BACKEND", "stub"),
                "ltm_backend": os.getenv("LTM_BACKEND", "stub"),
            }
        )
    
    def to_dict(self, status: HealthStatus) -> Dict[str, Any]:
        """Convert HealthStatus to dict for JSON serialization."""
        return {
            "status": status.status,
            "timestamp": status.timestamp,
            "agent_ready": status.agent_ready,
            "uptime_seconds": status.uptime_seconds,
            "mode": status.mode,
            "profile": status.config_profile,
            "optional_services": status.optional_services,
            "message": status.message,
            "metadata": status.metadata,
        }


# Global health checker instance
_health_checker: HealthChecker = None  # type: ignore


def initialize_health_checker():
    """Initialize global health checker."""
    global _health_checker
    import time
    _health_checker = HealthChecker(start_time=time.time())


def get_health_checker() -> HealthChecker:
    """Get or initialize health checker."""
    global _health_checker
    if _health_checker is None:
        initialize_health_checker()
    return _health_checker  # type: ignore


async def health_live() -> Dict[str, Any]:
    """GET /health/live endpoint."""
    checker = get_health_checker()
    status = checker.check_live()
    return checker.to_dict(status)


async def health_ready() -> Dict[str, Any]:
    """GET /health/ready endpoint."""
    checker = get_health_checker()
    status = checker.check_ready()
    return checker.to_dict(status)
