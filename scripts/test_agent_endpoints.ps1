#!/usr/bin/env powershell
<#
.SYNOPSIS
    Test SAM Agent API endpoints
    
.DESCRIPTION
    Verifies all agent HTTP endpoints are working correctly:
    - GET /
    - GET /health/live
    - GET /health/ready
    - POST /invoke
    
.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File test_agent_endpoints.ps1
#>

param(
    [string]$Endpoint = "http://localhost:8000"
)

Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "SAM Agent Endpoint Test Suite" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Test 1: Root endpoint
Write-Host "Test 1: GET /" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$Endpoint/" -UseBasicParsing
    $data = $response.Content | ConvertFrom-Json
    Write-Host "✓ Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Endpoints available: $(($data.endpoints -join ', '))"
    Write-Host ""
} catch {
    Write-Host "✗ Failed: $_" -ForegroundColor Red
    Write-Host ""
}

# Test 2: Health liveness
Write-Host "Test 2: GET /health/live" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$Endpoint/health/live" -UseBasicParsing
    $data = $response.Content | ConvertFrom-Json
    Write-Host "✓ Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Agent Status: $($data.status)" -ForegroundColor Green
    Write-Host "  Uptime: $([math]::Round($data.uptime_seconds, 2))s"
    Write-Host "  Mode: $($data.mode)"
    Write-Host ""
} catch {
    Write-Host "✗ Failed: $_" -ForegroundColor Red
    Write-Host ""
}

# Test 3: Health readiness
Write-Host "Test 3: GET /health/ready" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$Endpoint/health/ready" -UseBasicParsing
    $data = $response.Content | ConvertFrom-Json
    Write-Host "✓ Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Agent Ready: $($data.agent_ready)" -ForegroundColor Green
    Write-Host "  LTM Backend: $($data.metadata.ltm_backend)"
    Write-Host "  Bootstrap OK: $($data.metadata.bootstrap_ok)"
    Write-Host ""
} catch {
    Write-Host "✗ Failed: $_" -ForegroundColor Red
    Write-Host ""
}

# Test 4: Agent invocation
Write-Host "Test 4: POST /invoke" -ForegroundColor Yellow
try {
    $body = @{
        "input" = "Hello! Are you running locally with Phi?"
    } | ConvertTo-Json
    
    $response = Invoke-WebRequest -Uri "$Endpoint/invoke" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body `
        -UseBasicParsing
    
    $data = $response.Content | ConvertFrom-Json
    Write-Host "✓ Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Invocation Status: $($data.status)" -ForegroundColor Green
    Write-Host "  Input: $($data.input)"
    Write-Host "  Output: $($data.output -replace '^$', '[empty - using stub model]')"
    Write-Host "  Conversation ID: $($data.conversation_id)"
    Write-Host "  Trace ID: $($data.trace_id)"
    Write-Host ""
} catch {
    Write-Host "✗ Failed: $_" -ForegroundColor Red
    Write-Host ""
}

# Test 5: Multiple invocations
Write-Host "Test 5: Multiple Invocations" -ForegroundColor Yellow
$messages = @(
    "What is 2 + 2?",
    "Tell me a joke",
    "How are you today?"
)

foreach ($msg in $messages) {
    try {
        $body = @{"input" = $msg} | ConvertTo-Json
        $response = Invoke-WebRequest -Uri "$Endpoint/invoke" `
            -Method POST `
            -ContentType "application/json" `
            -Body $body `
            -UseBasicParsing
        
        $data = $response.Content | ConvertFrom-Json
        if ($data.status -eq "success") {
            Write-Host "  ✓ '$msg'" -ForegroundColor Green
        } else {
            Write-Host "  ✗ '$msg'" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ✗ '$msg': $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Test Suite Complete" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
