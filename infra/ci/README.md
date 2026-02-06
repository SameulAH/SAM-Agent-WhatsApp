# CI/CD Pipeline

## Responsibility

This directory documents the CI/CD pipeline stages and their responsibilities. It defines the automated processes for code quality, testing, and deployment.

## Pipeline Stages

1. **Lint & Format Check**: Code style enforcement
2. **Unit Tests**: Fast, deterministic tests with no external dependencies
3. **Integration Tests**: Full graph execution with mocked services
4. **Contract Tests**: Validation of interfaces between components
5. **Build**: Create container image
6. **Deploy**: Release to staging/production environments

## Responsibilities

- Enforce code quality standards
- Run comprehensive test suite
- Perform security scanning
- Generate test coverage reports
- Build optimized container images
- Manage deployment artifacts
- Provide rollback mechanisms
- Monitor deployment health
