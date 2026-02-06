# SAM: Stateful Agent Model

A production-oriented AI agent system designed for WhatsApp integration and conversational interactions.

## Overview

SAM is a modular, stateful agent framework that orchestrates multiple AI and speech services to provide intelligent, context-aware interactions. The system is built with production requirements in mind: scalability, observability, testability, and deployment automation.

## Key Characteristics

- **Stateful Architecture**: Maintains conversation context and agent state across interactions
- **Graph-Based Orchestration**: Defines execution flow through a directed graph
- **Service Abstraction**: Pluggable interfaces for LLM, STT, and TTS providers
- **Memory Management**: Separate short-term and long-term memory subsystems
- **Interface Agnostic**: WhatsApp integration is replaceable; core logic is channel-agnostic
- **Production-Ready**: CI/CD, observability, and containerization from the ground up

## Architecture

```
sam-core/
├── agent/          # Core orchestration, state, memory, graph
├── services/       # Abstracted LLM, STT, TTS providers
├── api/            # External interface (WhatsApp webhook)
├── infra/          # Docker, CI/CD, observability
└── tests/          # Unit, integration, contract tests
```

## Project Goals

1. **Clarity**: Clear separation of concerns with well-defined responsibilities
2. **Testability**: Deterministic execution with comprehensive test coverage
3. **Maintainability**: Minimal abstractions, domain-driven design
4. **Scalability**: Distributed, stateless where possible, with efficient resource management
5. **Observability**: Rich logging, metrics, and tracing for production visibility

## Getting Started

(Documentation to be added during implementation)

## Development

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
ruff check .
black --check .

# Run the agent
python -m sam.agent.orchestrator
```

## Contributing

(Contributing guidelines to be added)

## License

(License information to be added)
