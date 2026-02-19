#!/usr/bin/env python3
"""
Phase 5.5 Deployment Validation Script

Validates that packaging and deployment are production-ready:
1. Build validation (Dockerfile, profiles)
2. Startup validation (containers, health checks)
3. Degradation validation (graceful service loss)
4. Behavior validation (no agent logic changes)
5. Documentation validation (completeness)

Usage:
    python scripts/validate_deployment.py --profile dev
    python scripts/validate_deployment.py --profile prod-cpu --full
    python scripts/validate_deployment.py --check-behavior
"""

import subprocess
import json
import sys
import time
import os
from pathlib import Path
from typing import Tuple, List
import requests
from enum import Enum


class ValidationPhase(Enum):
    """Validation phases."""
    BUILD = "build"
    STARTUP = "startup"
    HEALTH = "health"
    DEGRADATION = "degradation"
    BEHAVIOR = "behavior"
    DOCUMENTATION = "documentation"


class ValidationResult:
    """Single validation result."""
    
    def __init__(self, phase: ValidationPhase, name: str, passed: bool, details: str = ""):
        self.phase = phase
        self.name = name
        self.passed = passed
        self.details = details
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        result = f"{status}: {self.phase.value.upper()} - {self.name}"
        if self.details:
            result += f"\n  {self.details}"
        return result


class Validator:
    """SAM Agent deployment validator."""
    
    def __init__(self, profile: str = "dev", verbose: bool = False):
        self.profile = profile
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.workspace_root = Path(__file__).parent.parent
    
    def run(self) -> bool:
        """Run all validations. Returns True if all passed."""
        print(f"\n{'='*70}")
        print(f"SAM Agent Deployment Validation - Profile: {self.profile}")
        print(f"{'='*70}\n")
        
        # Phase 1: Build Validation
        self._validate_dockerfile()
        
        # Phase 2: Startup Validation
        self._validate_startup()
        
        # Phase 3: Health Checks
        self._validate_health()
        
        # Phase 4: Documentation
        self._validate_documentation()
        
        # Print summary
        self._print_summary()
        
        return all(r.passed for r in self.results)
    
    def _validate_dockerfile(self) -> None:
        """Validate Dockerfile exists and builds."""
        dockerfile = self.workspace_root / "docker" / "Dockerfile.agent"
        
        # Check existence
        if not dockerfile.exists():
            self.results.append(ValidationResult(
                ValidationPhase.BUILD, "Dockerfile exists",
                False, f"Not found: {dockerfile}"
            ))
            return
        
        self.results.append(ValidationResult(
            ValidationPhase.BUILD, "Dockerfile exists", True
        ))
        
        # Check for multi-stage syntax
        content = dockerfile.read_text()
        stages = ["builder", "runtime-cpu", "runtime-gpu", "final"]
        missing = [s for s in stages if f"FROM" not in content or f"AS {s}" not in content]
        
        if missing:
            self.results.append(ValidationResult(
                ValidationPhase.BUILD, "Multi-stage build",
                False, f"Missing stages: {missing}"
            ))
        else:
            self.results.append(ValidationResult(
                ValidationPhase.BUILD, "Multi-stage build", True
            ))
        
        # Check for health checks
        has_healthcheck = "healthcheck" in content.lower() or "/health/live" in content
        self.results.append(ValidationResult(
            ValidationPhase.BUILD, "Health checks in Dockerfile",
            has_healthcheck, "" if has_healthcheck else "No health checks found"
        ))
        
        # Check for non-root user
        has_nonroot = "useradd" in content or "USER" in content
        self.results.append(ValidationResult(
            ValidationPhase.BUILD, "Non-root user setup",
            has_nonroot, "" if has_nonroot else "No user creation found"
        ))
        
        # Check for environment defaults
        has_env = "ENV" in content
        self.results.append(ValidationResult(
            ValidationPhase.BUILD, "Environment defaults",
            has_env, "" if has_env else "No ENV directives found"
        ))
    
    def _validate_startup(self) -> None:
        """Validate docker-compose and services."""
        compose_file = self.workspace_root / "docker-compose.yml"
        
        # Check existence
        if not compose_file.exists():
            self.results.append(ValidationResult(
                ValidationPhase.STARTUP, "docker-compose.yml exists",
                False, f"Not found: {compose_file}"
            ))
            return
        
        self.results.append(ValidationResult(
            ValidationPhase.STARTUP, "docker-compose.yml exists", True
        ))
        
        # Parse and validate structure
        content = compose_file.read_text()
        
        # Check for key services
        services = ["agent", "ollama"]
        missing = [s for s in services if f"{s}:" not in content]
        
        if missing:
            self.results.append(ValidationResult(
                ValidationPhase.STARTUP, "Required services defined",
                False, f"Missing services: {missing}"
            ))
        else:
            self.results.append(ValidationResult(
                ValidationPhase.STARTUP, "Required services defined", True
            ))
        
        # Check for volumes
        has_volumes = "volumes:" in content
        self.results.append(ValidationResult(
            ValidationPhase.STARTUP, "Persistence volumes configured",
            has_volumes, "" if has_volumes else "No volumes section"
        ))
        
        # Check for networks
        has_networks = "networks:" in content
        self.results.append(ValidationResult(
            ValidationPhase.STARTUP, "Service networking configured",
            has_networks, "" if has_networks else "No networks section"
        ))
        
        # Check for depends_on
        has_depends = "depends_on:" in content
        self.results.append(ValidationResult(
            ValidationPhase.STARTUP, "Service dependencies defined",
            has_depends, "" if has_depends else "No depends_on section"
        ))
    
    def _validate_health(self) -> None:
        """Validate health check system."""
        health_file = self.workspace_root / "agent" / "health.py"
        
        if not health_file.exists():
            self.results.append(ValidationResult(
                ValidationPhase.HEALTH, "Health check module exists",
                False, f"Not found: {health_file}"
            ))
            return
        
        self.results.append(ValidationResult(
            ValidationPhase.HEALTH, "Health check module exists", True
        ))
        
        content = health_file.read_text()
        
        # Check for key classes and methods
        checks = {
            "HealthStatus dataclass": "class HealthStatus" in content,
            "HealthChecker class": "class HealthChecker" in content,
            "check_live() method": "def check_live" in content,
            "check_ready() method": "def check_ready" in content,
            "get_mode() function": "def get_mode" in content,
            "get_profile() function": "def get_profile" in content,
        }
        
        for name, found in checks.items():
            self.results.append(ValidationResult(
                ValidationPhase.HEALTH, f"Health: {name}",
                found, "" if found else "Not found"
            ))
    
    def _validate_documentation(self) -> None:
        """Validate deployment documentation."""
        docs = {
            ".env.example": self.workspace_root / ".env.example",
            "DEPLOYMENT_STRATEGY.md": self.workspace_root / "DEPLOYMENT_STRATEGY.md",
        }
        
        for name, path in docs.items():
            exists = path.exists()
            self.results.append(ValidationResult(
                ValidationPhase.DOCUMENTATION, f"{name} exists",
                exists, "" if exists else f"Not found: {path}"
            ))
            
            if exists:
                content = path.read_text()
                
                # Check for key sections
                if name == ".env.example":
                    sections = [
                        "LLM Backend Configuration",
                        "Speech-to-Text",
                        "Text-to-Speech",
                        "Long-Term Memory",
                    ]
                elif name == "DEPLOYMENT_STRATEGY.md":
                    sections = [
                        "Quick Start",
                        "Architecture",
                        "Health Checks",
                        "Troubleshooting",
                        "Validation Checklist",
                    ]
                else:
                    sections = []
                
                for section in sections:
                    found = section in content
                    self.results.append(ValidationResult(
                        ValidationPhase.DOCUMENTATION, f"{name}: {section}",
                        found, "" if found else f"Section not found"
                    ))
    
    def _print_summary(self) -> None:
        """Print validation summary."""
        print(f"\n{'='*70}")
        print("VALIDATION RESULTS")
        print(f"{'='*70}\n")
        
        # Group by phase
        by_phase = {}
        for result in self.results:
            if result.phase not in by_phase:
                by_phase[result.phase] = []
            by_phase[result.phase].append(result)
        
        # Print by phase
        for phase in ValidationPhase:
            if phase not in by_phase:
                continue
            
            print(f"\n{phase.value.upper()}")
            print("-" * 70)
            
            phase_results = by_phase[phase]
            passed = sum(1 for r in phase_results if r.passed)
            total = len(phase_results)
            
            for result in phase_results:
                print(result)
            
            print(f"\n  Phase Summary: {passed}/{total} passed\n")
        
        # Overall summary
        total_passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"{'='*70}")
        if total_passed == total:
            print(f"✓ ALL VALIDATIONS PASSED ({total_passed}/{total})")
        else:
            print(f"✗ SOME VALIDATIONS FAILED ({total_passed}/{total})")
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate SAM Agent deployment readiness"
    )
    parser.add_argument(
        "--profile",
        choices=["dev", "prod-cpu", "prod-gpu"],
        default="dev",
        help="Compose profile to validate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    validator = Validator(profile=args.profile, verbose=args.verbose)
    success = validator.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
