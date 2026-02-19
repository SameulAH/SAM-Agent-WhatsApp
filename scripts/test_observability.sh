#!/usr/bin/env bash
# Quick script to test observability endpoints locally

set -e

BASE_URL="http://localhost:8000"

echo "Testing Observability Endpoints..."
echo "===================================="
echo ""

# Test 1: Health endpoint
echo "1. Testing /debug/health endpoint..."
curl -s "${BASE_URL}/debug/health" | python -m json.tool || echo "FAILED (observability may be disabled)"
echo ""

# Test 2: Graph endpoint
echo "2. Testing /debug/graph endpoint..."
curl -s "${BASE_URL}/debug/graph" | python -m json.tool || echo "FAILED (observability may be disabled)"
echo ""

# Test 3: Traces endpoint
echo "3. Testing /debug/traces endpoint..."
curl -s "${BASE_URL}/debug/traces?limit=5" | python -m json.tool || echo "FAILED (observability may be disabled)"
echo ""

# Test 4: Spans endpoint
echo "4. Testing /debug/spans endpoint..."
curl -s "${BASE_URL}/debug/spans?limit=10" | python -m json.tool || echo "FAILED (observability may be disabled)"
echo ""

# Test 5: Memory endpoint
echo "5. Testing /debug/memory endpoint..."
curl -s "${BASE_URL}/debug/memory?limit=10" | python -m json.tool || echo "FAILED (observability may be disabled)"
echo ""

# Test 6: Stats endpoint
echo "6. Testing /debug/stats endpoint..."
curl -s "${BASE_URL}/debug/stats" | python -m json.tool || echo "FAILED (observability may be disabled)"
echo ""

echo "===================================="
echo "Testing complete!"
echo ""
echo "To enable observability, set:"
echo "  export LOCAL_OBSERVABILITY_ENABLED=true"
echo ""
echo "Then restart the agent."
