# Integration Tests

## Responsibility

This directory contains integration tests that validate the full agent graph execution with mocked external services. These tests verify that components work together correctly.

## Test Scope

- **Full Graph Execution**: End-to-end execution paths through the agent
- **Service Integration**: Interaction between agent and service boundaries
- **State Transitions**: Verify correct state changes across the graph
- **Error Handling**: Test error propagation and recovery

## Responsibilities

- Test complete workflows through the agent
- Mock external services (LLM, STT, TTS, WhatsApp)
- Verify component interactions
- Test state persistence and retrieval
- Validate error handling and fallbacks
- Document integration scenarios
- Support performance testing
