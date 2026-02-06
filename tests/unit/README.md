# Unit Tests

## Responsibility

This directory contains deterministic unit tests that validate individual components in isolation. Unit tests should have no external dependencies and execute quickly.

## Test Categories

- **State Tests**: Validate state schema, mutations, and lifecycle
- **Routing Tests**: Test graph routing logic and decision paths
- **Contract Tests**: Verify component interfaces and contracts
- **Tool Tests**: Test tool execution and parameter validation

## Responsibilities

- Test individual functions and classes in isolation
- Mock all external dependencies
- Provide high code coverage
- Execute in parallel for speed
- Be deterministic and repeatable
- Document expected behavior through test names
- Test both happy paths and error conditions
