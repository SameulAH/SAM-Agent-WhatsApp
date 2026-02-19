#!/usr/bin/env python3
"""Check SQLite memory database status"""

import sqlite3
import os
import sys

db_file = '/app/data/memory.db'

print("=" * 70)
print("STM Database Status Check")
print("=" * 70)
print()

# Check if database exists
if os.path.exists(db_file):
    print(f"Database file exists: {db_file}")
    print(f"File size: {os.path.getsize(db_file)} bytes")
    print()
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"Tables found: {len(tables)}")
        for table in tables:
            print(f"  - {table[0]}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"    Rows: {count}")
        
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")
else:
    print(f"Database file does not exist: {db_file}")
    print()
    print("Status: No conversation data stored yet")
    print("The database will be created on first memory write operation")

print()
print("=" * 70)
