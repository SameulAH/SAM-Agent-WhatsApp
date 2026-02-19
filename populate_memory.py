#!/usr/bin/env python3
"""Test memory system with multiple requests"""

import httpx
import json
import time

def test_memory_with_data():
    """Send multiple requests to populate the database"""
    
    api_url = "http://localhost:8000/invoke"
    
    print("=" * 70)
    print("Testing Memory System with Data Population")
    print("=" * 70)
    print()
    
    # Send 3 requests
    for i in range(1, 4):
        print(f"[{i}] Sending request {i}...")
        response = httpx.post(
            api_url,
            json={"input": f"This is test message number {i}. Remember this."},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"    Status: Success")
            print(f"    Conversation ID: {data.get('conversation_id')[:8]}...")
        else:
            print(f"    Status: Error ({response.status_code})")
        
        time.sleep(0.5)
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    test_memory_with_data()
