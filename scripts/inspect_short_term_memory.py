#!/usr/bin/env python3
"""
Helper script to inspect short-term memory database.

Safe utility for debugging and CI verification.
Prints last N rows from the short_term_memory table.

Usage:
    python scripts/inspect_short_term_memory.py [--db PATH] [--limit N]
"""

import sqlite3
import argparse
import json
from pathlib import Path
from datetime import datetime


def inspect_database(db_path: str, limit: int = 10):
    """
    Inspect and display contents of short-term memory database.
    
    Args:
        db_path: Path to SQLite database file
        limit: Maximum number of rows to display
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='short_term_memory'"
        )
        
        if not cursor.fetchone():
            print(f"✗ Table 'short_term_memory' does not exist in {db_path}")
            return
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM short_term_memory")
        total_rows = cursor.fetchone()[0]
        
        print(f"Short-Term Memory Database: {db_path}")
        print(f"Total rows: {total_rows}")
        print()
        
        if total_rows == 0:
            print("(Empty - no conversation data stored yet)")
            return
        
        # Fetch last N rows
        cursor.execute(
            f"""
            SELECT conversation_id, key, data, created_at, updated_at
            FROM short_term_memory
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        
        rows = cursor.fetchall()
        
        print(f"Last {min(limit, total_rows)} entries:")
        print("-" * 100)
        
        for i, row in enumerate(rows, 1):
            conversation_id, key, data_json, created_at, updated_at = row
            
            print(f"\n[{i}] Conversation: {conversation_id}")
            print(f"    Key: {key}")
            print(f"    Created: {created_at}")
            print(f"    Updated: {updated_at}")
            
            try:
                data = json.loads(data_json)
                print(f"    Data: {json.dumps(data, indent=6)}")
            except json.JSONDecodeError:
                print(f"    Data: (corrupted JSON: {data_json[:50]}...)")
        
        print()
        print("-" * 100)
        print(f"✓ Database healthy ({total_rows} total entries)")
        
        conn.close()
        
    except sqlite3.OperationalError as e:
        print(f"✗ Database error: {str(e)}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect short-term memory database"
    )
    parser.add_argument(
        "--db",
        default="/app/data/memory.db",
        help="Path to SQLite database (default: /app/data/memory.db)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of rows to display (default: 10)",
    )
    
    args = parser.parse_args()
    
    if not Path(args.db).exists():
        print(f"✗ Database file not found: {args.db}")
        return
    
    inspect_database(args.db, limit=args.limit)


if __name__ == "__main__":
    main()
