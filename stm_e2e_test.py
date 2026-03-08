import asyncio
import sys
import json
import sqlite3

sys.path.insert(0, '/app')
from agent.orchestrator import SAMOrchestrator

async def test():
    agent = SAMOrchestrator()
    print("Running test invocation...")
    result = await agent.invoke(
        raw_input="What is 2 + 2?",
        conversation_id="telegram_test_99999",
        trace_id="trace-stm-test-001",
    )
    print(f"Full result dict: {result}")
    print(f"Status : {result.get('status')}")
    print(f"Output : {result.get('output') or result.get('final_output')}")
    print()

    # Check STM
    conn = sqlite3.connect('/app/data/memory.db')
    cur = conn.cursor()
    cur.execute("SELECT conversation_id, key, data FROM short_term_memory WHERE conversation_id='telegram_test_99999' ORDER BY rowid DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()

    if row:
        print("STM entry found:")
        print(f"  conversation_id : {row[0]}")
        print(f"  key             : {row[1]}")
        data = json.loads(row[2]) if row[2] else {}
        print(f"  raw_input       : {data.get('raw_input')}")
        print(f"  final_output    : {data.get('final_output')}")
        print(f"  timestamp       : {data.get('interaction_timestamp')}")
    else:
        print("❌ No STM entry found for telegram_test_99999!")

asyncio.run(test())
