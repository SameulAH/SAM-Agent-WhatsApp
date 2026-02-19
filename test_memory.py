#!/usr/bin/env python3
"""Test memory system with conversation context"""

import httpx
import json
import time

def test_memory():
    """Test conversation memory by exchanging multiple messages"""
    
    api_url = "http://localhost:8000/invoke"
    
    print("=" * 70)
    print("Testing Memory System with Conversation Context")
    print("=" * 70)
    print()
    
    # First message - establish context
    print("[1] Sending first message (context setup)...")
    response1 = httpx.post(
        api_url,
        json={"input": "My name is Alice and I like programming."},
        timeout=30
    )
    
    if response1.status_code == 200:
        data1 = response1.json()
        conversation_id = data1.get("conversation_id", "N/A")
        print(f"    Status: {response1.status_code}")
        print(f"    Conversation ID: {conversation_id}")
        print(f"    Output: {data1.get('output', '')[:100]}...")
        print()
        
        # Small delay
        time.sleep(1)
        
        # Second message - should recall context if memory works
        print("[2] Sending follow-up message (memory test)...")
        response2 = httpx.post(
            api_url,
            json={
                "input": "What's my name and what do I like?",
                "conversation_id": conversation_id
            },
            timeout=30
        )
        
        if response2.status_code == 200:
            data2 = response2.json()
            output = data2.get('output', '')
            print(f"    Status: {response2.status_code}")
            print(f"    Output: {output[:150]}...")
            print()
            
            # Check if memory worked
            if 'alice' in output.lower() and 'program' in output.lower():
                print("[OK] Memory system is working!")
                print("    Agent correctly recalled context from previous message")
            else:
                print("[!] Memory may not be working")
                print("    Agent did not recall context (but this is normal if STM is disabled)")
        else:
            print(f"    Error: {response2.status_code}")
    else:
        print(f"    Error: {response1.status_code}")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    test_memory()
