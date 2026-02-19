"""
Local observability interface.

Aggregates signals from tracer and memory for read-only inspection.
"""

from typing import Optional, Dict, Any, List
from agent.observability.store import ObservabilityStore


class LocalObservabilityInterface:
    """
    Read-only interface for local observability.
    
    Aggregates:
    - Tracer spans and traces
    - Memory event metadata
    - Health status
    - Configuration
    
    Does NOT expose:
    - Prompts or user inputs
    - Model outputs
    - Memory contents
    - Decision logic
    """

    def __init__(self, store: Optional[ObservabilityStore] = None):
        """
        Initialize observability interface.
        
        Args:
            store: Optional ObservabilityStore instance
        """
        self.store = store or ObservabilityStore()

    def get_health_info(self, agent) -> Dict[str, Any]:
        """Get agent health and configuration info."""
        try:
            return {
                "status": "healthy",
                "graph_compiled": agent.graph is not None,
                "tracer_enabled": agent.tracer is not None,
                "memory_backend": type(agent.memory_controller).__name__,
                "model_backend": type(agent.model_backend).__name__,
                "long_term_memory_backend": type(agent.long_term_memory_store).__name__,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_graph_structure(self, agent) -> Dict[str, Any]:
        """Get static graph structure (no execution state)."""
        try:
            if not agent.graph:
                return {"error": "Graph not compiled"}

            graph = agent.graph.graph
            
            return {
                "nodes": list(graph.nodes()) if hasattr(graph, 'nodes') else [],
                "edges": [
                    {"from": u, "to": v}
                    for u, v in (graph.edges() if hasattr(graph, 'edges') else [])
                ],
                "compiled": True,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trace metadata (no content)."""
        return self.store.get_recent_traces(limit) if self.store else []

    def get_recent_spans(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent span metadata (no content)."""
        return self.store.get_recent_spans(limit) if self.store else []

    def get_memory_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent memory operation metadata (no content)."""
        return self.store.get_memory_events(limit) if self.store else []

    def get_active_traces(self) -> List[Dict[str, Any]]:
        """Get currently active traces."""
        return self.store.get_active_traces() if self.store else []

    def get_store_stats(self) -> Dict[str, int]:
        """Get observability store statistics."""
        return self.store.get_stats() if self.store else {}

    def clear_store(self) -> None:
        """Clear observability store (dev-only)."""
        if self.store:
            self.store.clear()


# Global instance (created in API if enabled)
_global_observability: Optional[LocalObservabilityInterface] = None


def get_observability() -> Optional[LocalObservabilityInterface]:
    """Get global observability interface."""
    return _global_observability


def set_observability(obs: Optional[LocalObservabilityInterface]) -> None:
    """Set global observability interface."""
    global _global_observability
    _global_observability = obs
