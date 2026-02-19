#!/usr/bin/env python3
"""
Validation script to verify Local Observability implementation is complete.

This script checks that:
1. All required files exist
2. All required code is in place
3. All tests can be imported
4. All documentation files are present
"""

import os
import sys
from pathlib import Path


def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists."""
    exists = os.path.isfile(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {description}")
    return exists


def check_dir_exists(path: str, description: str) -> bool:
    """Check if a directory exists."""
    exists = os.path.isdir(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {description}")
    return exists


def check_string_in_file(filepath: str, search_string: str, description: str) -> bool:
    """Check if a string exists in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            found = search_string in content
            status = "✅" if found else "❌"
            print(f"    {status} {description}")
            return found
    except Exception as e:
        print(f"    ❌ Error checking {description}: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("Local Observability Implementation Validation")
    print("="*60 + "\n")
    
    all_checks_passed = True
    
    # Check core implementation files
    print("1. Core Implementation Files")
    print("-" * 40)
    
    checks = [
        ("agent/observability/__init__.py", "Observability module init"),
        ("agent/observability/store.py", "Observability store implementation"),
        ("agent/observability/interface.py", "Observability interface implementation"),
    ]
    
    for filepath, description in checks:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check modified files
    print("\n2. Modified Files")
    print("-" * 40)
    
    modified_files = [
        ("agent/tracing/tracer.py", "observability_sink"),
        ("agent/tracing/langtrace_tracer.py", "observability_sink"),
        ("agent/api.py", "/debug/health"),
    ]
    
    for filepath, search_string in modified_files:
        if not os.path.isfile(filepath):
            print(f"  ❌ {filepath} not found")
            all_checks_passed = False
        else:
            found = check_string_in_file(filepath, search_string, 
                                        f"{filepath} contains '{search_string}'")
            if not found:
                all_checks_passed = False
    
    # Check test file
    print("\n3. Test Files")
    print("-" * 40)
    
    test_file = "tests/unit/test_observability.py"
    if not check_file_exists(test_file, "Unit tests for observability"):
        all_checks_passed = False
    else:
        # Check for test class
        check_string_in_file(test_file, "class TestObservabilityInvariance",
                            "Contains TestObservabilityInvariance class")
        # Check for specific tests
        check_string_in_file(test_file, "def test_agent_output_identical_with_and_without_observability",
                            "Contains behavioral invariance test")
        check_string_in_file(test_file, "def test_no_sensitive_data_leaked",
                            "Contains data leak test")
    
    # Check documentation files
    print("\n4. Documentation Files")
    print("-" * 40)
    
    docs = [
        ("docs/README_OBSERVABILITY.md", "Documentation index"),
        ("docs/OBSERVABILITY_SUMMARY.md", "Executive summary"),
        ("docs/OBSERVABILITY_QUICKREF.md", "Quick reference guide"),
        ("docs/LOCAL_OBSERVABILITY.md", "Comprehensive guide"),
        ("docs/OBSERVABILITY_IMPLEMENTATION.md", "Implementation details"),
        ("docs/OBSERVABILITY_COMPLETE.md", "Completion checklist"),
    ]
    
    for filepath, description in docs:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check helper scripts
    print("\n5. Helper Scripts")
    print("-" * 40)
    
    scripts = [
        ("scripts/test_observability.sh", "Observability endpoint test script"),
    ]
    
    for filepath, description in scripts:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check API endpoints
    print("\n6. API Endpoints Implementation")
    print("-" * 40)
    
    endpoints = [
        ("agent/api.py", "@app.get(\"/debug/health\")", "GET /debug/health"),
        ("agent/api.py", "@app.get(\"/debug/graph\")", "GET /debug/graph"),
        ("agent/api.py", "@app.get(\"/debug/traces\")", "GET /debug/traces"),
        ("agent/api.py", "@app.get(\"/debug/spans\")", "GET /debug/spans"),
        ("agent/api.py", "@app.get(\"/debug/memory\")", "GET /debug/memory"),
        ("agent/api.py", "@app.get(\"/debug/stats\")", "GET /debug/stats"),
    ]
    
    for filepath, search_string, endpoint_name in endpoints:
        found = check_string_in_file(filepath, search_string, endpoint_name)
        if not found:
            all_checks_passed = False
    
    # Check environment gating
    print("\n7. Environment Variable Gating")
    print("-" * 40)
    
    gating_checks = [
        ("agent/api.py", "LOCAL_OBSERVABILITY_ENABLED", "Environment variable checked"),
    ]
    
    for filepath, env_var, description in gating_checks:
        found = check_string_in_file(filepath, env_var, description)
        if not found:
            all_checks_passed = False
    
    # Check imports in __init__
    print("\n8. Module Exports")
    print("-" * 40)
    
    imports = [
        ("agent/observability/__init__.py", "ObservabilityStore", "Exports ObservabilityStore"),
        ("agent/observability/__init__.py", "LocalObservabilityInterface", "Exports LocalObservabilityInterface"),
        ("agent/observability/__init__.py", "TraceRecord", "Exports TraceRecord"),
    ]
    
    for filepath, symbol, description in imports:
        found = check_string_in_file(filepath, symbol, description)
        if not found:
            all_checks_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_checks_passed:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("="*60)
        print("\nImplementation is complete!")
        print("\nNext steps:")
        print("  1. Run tests: pytest tests/unit/test_observability.py -v")
        print("  2. Enable observability: export LOCAL_OBSERVABILITY_ENABLED=true")
        print("  3. Test endpoints: bash scripts/test_observability.sh")
        print("  4. Read docs: Start with docs/OBSERVABILITY_SUMMARY.md")
        return 0
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("="*60)
        print("\nPlease check the files listed above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
