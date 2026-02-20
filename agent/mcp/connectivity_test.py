"""
agent/mcp/connectivity_test.py

Connectivity probe for MCP web-search providers via Smithery.

Tests three providers in priority order: Exa → Brave → Linkup

Usage:
    python agent/mcp/connectivity_test.py

Required (.env) — at least one of:
  EXA_API_KEY      → https://dashboard.exa.ai/api-keys       (free: 1k req/mo)
  BRAVE_API_KEY    → https://brave.com/search/api/            (free: 2k req/mo)
  LINKUP_API_KEY   → https://app.linkup.so/home               (free tier available)

Optional (Smithery Connect proxy — credentials stored server-side):
  SMITHERY_API_KEY       → https://smithery.ai/account/api-keys
  SMITHERY_NAMESPACE     → your namespace
  EXA_CONNECTION_ID      → auto-created by this script
  BRAVE_CONNECTION_ID    → auto-created by this script
  LINKUP_CONNECTION_ID   → auto-created by this script

Exit codes:
  0 — at least one provider is fully working
  1 — all providers failed
"""

import asyncio
import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(repo_root / ".env", override=False)
except ImportError:
    pass

import httpx

PASS = ""
FAIL = ""
SKIP = "○"

# ── Provider metadata ─────────────────────────────────────────

_PROVIDERS = {
    "exa": {
        "label":       "Exa Search",
        "env_key":     "EXA_API_KEY",
        "env_conn":    "EXA_CONNECTION_ID",
        "direct_url":  "https://exa.run.tools",
        "query_param": "exaApiKey",
        "tool_name":   "web_search_exa",
        "tool_args":   {"query": "connectivity test", "numResults": 1},
        "run_tools_id": "exa",
        "key_url":     "https://dashboard.exa.ai/api-keys",
    },
    "brave": {
        "label":       "Brave Search",
        "env_key":     "BRAVE_API_KEY",
        "env_conn":    "BRAVE_CONNECTION_ID",
        "direct_url":  "https://brave.run.tools",
        "query_param": "braveApiKey",
        "tool_name":   "brave_web_search",
        "tool_args":   {"query": "connectivity test", "count": 1},
        "run_tools_id": "brave",
        "key_url":     "https://brave.com/search/api/",
    },
    "linkup": {
        "label":       "Linkup",
        "env_key":     "LINKUP_API_KEY",
        "env_conn":    "LINKUP_CONNECTION_ID",
        "direct_url":  "https://linkup-mcp-server--linkupplatform.run.tools",
        "query_param": "apiKey",
        "tool_name":   "linkup-search",
        "tool_args":   {"query": "connectivity test", "outputType": "sourcedAnswer", "depth": "standard"},
        "run_tools_id": "LinkupPlatform/linkup-mcp-server",
        "key_url":     "https://app.linkup.so/home",
    },
}


def _get_creds():
    return {
        "smithery_key": os.getenv("SMITHERY_API_KEY", ""),
        "namespace":    os.getenv("SMITHERY_NAMESPACE", ""),
        "providers": {
            pid: {
                "api_key": os.getenv(cfg["env_key"], ""),
                "conn_id": os.getenv(cfg["env_conn"], ""),
            }
            for pid, cfg in _PROVIDERS.items()
        },
    }


def _provider_url(pid: str, creds: dict) -> str:
    cfg     = _PROVIDERS[pid]
    p       = creds["providers"][pid]
    conn_id = p["conn_id"]
    api_key = p["api_key"]
    ns      = creds["namespace"]
    sk      = creds["smithery_key"]

    if conn_id and ns and sk:
        return f"https://api.smithery.ai/connect/{ns}/{conn_id}/mcp"
    if api_key:
        from urllib.parse import urlencode
        return f"{cfg['direct_url']}?{urlencode({cfg['query_param']: api_key})}"
    return ""


def _auth_headers(smithery_key: str) -> dict:
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if smithery_key:
        h["Authorization"] = f"Bearer {smithery_key}"
    return h


async def run_checks() -> bool:
    creds = _get_creds()
    ph    = "your_"

    def ok(v: str) -> bool:
        return bool(v) and not v.startswith(ph)

    print("\n")
    print("  MCP Web-Search Connectivity Test")
    print("  Providers: Exa → Brave → Linkup  (priority order)")
    print("\n")

    # ── Check 1: Which providers have credentials? ──────────
    available = [
        pid for pid, p in creds["providers"].items()
        if ok(p["api_key"]) or (ok(creds["smithery_key"]) and ok(p["conn_id"]))
    ]

    if not available:
        print(f"{FAIL} [1] No provider credentials found.")
        print()
        print("  Set at least one of these in .env:")
        for pid, cfg in _PROVIDERS.items():
            print(f"    {cfg['env_key']:25s} → {cfg['key_url']}")
        return False

    print(f"{PASS} [1] Credentials found for: {', '.join(available)}")
    for pid in available:
        p   = creds["providers"][pid]
        key = p["api_key"]
        if ok(key):
            print(f"      {_PROVIDERS[pid]['label']:20s} direct key ***{key[-4:]}")
        elif ok(p["conn_id"]):
            print(f"      {_PROVIDERS[pid]['label']:20s} Smithery Connect  conn={p['conn_id']}")
    print()

    # ── Check 2: Smithery API key (optional) ───────────────
    sk = creds["smithery_key"]
    if ok(sk):
        try:
            async with httpx.AsyncClient(timeout=8.0) as c:
                r = await c.get("https://api.smithery.ai/namespaces",
                                headers={"Authorization": f"Bearer {sk}"})
            if r.status_code == 200:
                ns_list = [n["name"] for n in r.json().get("namespaces", [])]
                print(f"{PASS} [2] Smithery API key valid  namespaces: {ns_list}")
                if not creds["namespace"] and ns_list:
                    creds["namespace"] = ns_list[0]
            else:
                print(f"{FAIL} [2] Smithery API key rejected (HTTP {r.status_code})")
        except Exception as e:
            print(f"{FAIL} [2] Smithery check failed: {e}")
        print()
    else:
        print(f"{SKIP} [2] No SMITHERY_API_KEY — skipping proxy check (direct mode only)")
        print()

    # ── Checks 3–5: Test each available provider ───────────
    any_working = False
    primary_provider = None
    check_num = 3

    for pid in ["exa", "brave", "linkup"]:
        if pid not in available:
            print(f"{SKIP} [{check_num}] {_PROVIDERS[pid]['label']:20s} — no credentials, skipping")
            check_num += 1
            continue

        cfg    = _PROVIDERS[pid]
        p      = creds["providers"][pid]
        url    = _provider_url(pid, creds)
        label  = cfg["label"]

        if not url:
            print(f"{FAIL} [{check_num}] {label:20s} — could not build URL")
            check_num += 1
            continue

        # Optionally create Smithery Connect connection if key present but no conn_id
        if ok(p["api_key"]) and ok(creds["smithery_key"]) and ok(creds["namespace"]) and not ok(p["conn_id"]):
            from urllib.parse import urlencode
            mc_url = f"{cfg['direct_url']}?{urlencode({cfg['query_param']: p['api_key']})}"
            try:
                async with httpx.AsyncClient(timeout=10.0) as c:
                    r = await c.post(
                        f"https://api.smithery.ai/connect/{creds['namespace']}",
                        json={"mcpUrl": mc_url, "name": f"sam-agent-{pid}"},
                        headers={"Authorization": f"Bearer {creds['smithery_key']}",
                                 "Content-Type": "application/json"},
                    )
                data   = r.json()
                new_id = data.get("connectionId", "")
                state  = data.get("status", {}).get("state", "unknown")
                if state == "connected" and new_id:
                    p["conn_id"] = new_id
                    url = f"https://api.smithery.ai/connect/{creds['namespace']}/{new_id}/mcp"
                    print(f"  ✦ [{check_num}] {label} connection created: {new_id}")
                    print(f"       Add to .env: {cfg['env_conn']}={new_id}")
            except Exception:
                pass  # non-fatal — fall through to direct mode

        # Run a test search call
        payload = {
            "jsonrpc": "2.0",
            "method":  "tools/call",
            "params":  {"name": cfg["tool_name"], "arguments": cfg["tool_args"]},
            "id":      1,
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.post(url, json=payload,
                                 headers=_auth_headers(creds["smithery_key"]))
            data = _parse_response(r)

            if "error" in data and "result" not in data:
                msg = data["error"].get("message", str(data["error"])) \
                      if isinstance(data["error"], dict) else str(data["error"])
                print(f"{FAIL} [{check_num}] {label:20s} — RPC error: {msg[:120]}")
            else:
                result   = data.get("result", {})
                is_error = result.get("isError", False)
                content  = next((i.get("text", "")
                                  for i in result.get("content", [])
                                  if i.get("type") == "text"), "")

                if is_error and content:
                    print(f"{FAIL} [{check_num}] {label:20s} — provider error: {content[:120]}")
                else:
                    # Truncate content preview
                    preview = content[:80].replace("\n", " ") if content else "(empty)"
                    print(f"{PASS} [{check_num}] {label:20s} — OK  response: {preview!r}")
                    any_working = True
                    if primary_provider is None:
                        primary_provider = pid

        except httpx.TimeoutException:
            print(f"{FAIL} [{check_num}] {label:20s} — timed out after 15 s")
        except Exception as e:
            print(f"{FAIL} [{check_num}] {label:20s} — {type(e).__name__}: {e}")

        check_num += 1

    # ── Summary ────────────────────────────────────────────
    print()
    if any_working:
        print(f"  Active provider: {_PROVIDERS[primary_provider]['label']}")
        print(f"  All checks passed  web search ready.")
    else:
        print(f"  All providers failed — check credentials above.")
    print()
    return any_working


def _parse_response(r: httpx.Response) -> dict:
    ct = r.headers.get("content-type", "")
    if "text/event-stream" in ct:
        last: dict = {}
        for line in r.text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        last = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
        return last if last else {"error": {"message": "empty SSE", "code": r.status_code}}
    try:
        parsed = r.json()
        return parsed if isinstance(parsed, dict) else {"error": {"message": str(parsed)}}
    except Exception:
        return {"error": {"message": r.text[:200], "code": r.status_code}}


if __name__ == "__main__":
    ok = asyncio.run(run_checks())
    sys.exit(0 if ok else 1)
