#!/usr/bin/env python3
"""
Validation script for LangSmith observability implementation.

Verifies:
1. All required files exist
2. All required code is in place
3. All imports are correct
4. All exports are available
"""

import os
import sys
from pathlib import Path


def check_file(path: str, description: str) -> bool:
    """Check if file exists."""
    exists = os.path.isfile(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {description}")
    return exists


def check_import(module_path: str, symbol: str, description: str) -> bool:
    """Check if symbol can be imported."""
    try:
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            module_name, _ = parts
            exec(f"from {module_path} import {symbol}")
        else:
            exec(f"from {module_path} import {symbol}")
        status = "✅"
        result = True
    except Exception as e:
        status = "❌"
        result = False
        description += f" ({str(e)[:50]})"
    
    print(f"    {status} {description}")
    return result


def check_string_in_file(filepath: str, search_string: str, description: str) -> bool:
    """Check if string exists in file."""
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
    print("\n" + "="*70)
    print("LangSmith Observability Implementation Validation")
    print("="*70 + "\n")
    
    all_passed = True
    
    # 1. Check files exist
    print("1. Core Implementation Files")
    print("-" * 70)
    
    files = [
        ("agent/tracing/langsmith_tracer.py", "LangSmith tracer implementation"),
        ("agent/tracing/langtrace_tracer_placeholder.py", "Langtrace placeholder"),
        ("agent/tracing/tracer_factory.py", "Tracer factory"),
    ]
    
    for filepath, description in files:
        if not check_file(filepath, description):
            all_passed = False
    
    # 2. Check tests
    print("\n2. Test Files")
    print("-" * 70)
    
    if not check_file("tests/unit/test_langsmith_observability.py", "LangSmith observability tests"):
        all_passed = False
    
    # 3. Check modifications
    print("\n3. Modified Files (Backward Compatibility)")
    print("-" * 70)
    
    modifications = [
        ("agent/tracing/__init__.py", "LangSmithTracer", "Exports LangSmithTracer"),
        ("agent/tracing/__init__.py", "create_tracer", "Exports create_tracer"),
        ("agent/tracing/__init__.py", "get_tracer_backend", "Exports get_tracer_backend"),
        ("agent/api.py", "/health/trace", "Added /health/trace endpoint"),
        (".env.example", "TRACER_BACKEND", "Added TRACER_BACKEND config"),
    ]
    
    for filepath, search_string, description in modifications:
        if not check_string_in_file(filepath, search_string, description):
            all_passed = False
    
    # 4. Check documentation
    print("\n4. Documentation Files")
    print("-" * 70)
    
    docs = [
        ("docs/LANGSMITH_OBSERVABILITY.md", "Comprehensive guide"),
        ("docs/LANGSMITH_QUICK_REFERENCE.md", "Quick reference"),
        ("LANGSMITH_IMPLEMENTATION_REPORT.md", "Implementation report"),
    ]
    
    for filepath, description in docs:
        if not check_file(filepath, description):
            all_passed = False
    
    # 5. Check imports work
    print("\n5. Import Validation")
    print("-" * 70)
    
    try:
        print("    Checking imports...")
        from agent.tracing import (
            Tracer,
            NoOpTracer,
            LangSmithTracer,
            create_tracer,
            get_tracer_backend,
            get_tracer_config,
        )
        print("    ✅ All imports successful")
    except ImportError as e:
        print(f"    ❌ Import failed: {e}")
        all_passed = False
    
    # 6. Check factory works
    print("\n6. Factory Functionality")
    print("-" * 70)
    
    try:
        from agent.tracing import create_tracer, get_tracer_backend, get_tracer_config
        
        # Test factory
        tracer = create_tracer()
        print(f"    ✅ create_tracer() returns {type(tracer).__name__}")
        
        # Test backend getter
        backend = get_tracer_backend()
        print(f"    ✅ get_tracer_backend() returns '{backend}'")
        
        # Test config getter
        config = get_tracer_config()
        if "tracer_backend" in config:
            print(f"    ✅ get_tracer_config() returns valid config")
        else:
            print(f"    ❌ get_tracer_config() missing tracer_backend")
            all_passed = False
    
    except Exception as e:
        print(f"    ❌ Factory check failed: {e}")
        all_passed = False
    
    # 7. Check tracer interface
    print("\n7. Tracer Interface Compliance")
    print("-" * 70)
    
    try:
        from agent.tracing import LangSmithTracer, TraceMetadata
        
        tracer = LangSmithTracer(enabled=False)
        
        # Check methods exist
        methods = ["start_span", "end_span", "record_event", "is_enabled", "_filter_safe_metadata"]
        for method in methods:
            if hasattr(tracer, method):
                print(f"    ✅ LangSmithTracer.{method} exists")
            else:
                print(f"    ❌ LangSmithTracer.{method} missing")
                all_passed = False
        
        # Test basic operation
        trace_metadata = TraceMetadata(trace_id="test")
        span = tracer.start_span("test", {}, trace_metadata)
        tracer.end_span(span, "success", {})
        tracer.record_event("test", {}, trace_metadata)
        print(f"    ✅ Basic operations work")
        
    except Exception as e:
        print(f"    ❌ Interface check failed: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("="*70)
        print("\nImplementation is complete and ready!\n")
        print("Next steps:")
        print("  1. Run tests: pytest tests/unit/test_langsmith_observability.py -v")
        print("  2. Check health: curl http://localhost:8000/health/trace")
        print("  3. Read docs: docs/LANGSMITH_QUICK_REFERENCE.md")
        print("  4. Deploy with TRACER_BACKEND=noop (default, zero overhead)")
        return 0
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("="*70)
        print("\nPlease check the items marked with ❌ above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
