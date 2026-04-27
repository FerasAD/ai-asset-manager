import sqlite3
from pathlib import Path

# Local SQLite database file location
DB_PATH = Path("database/assets.db")


def get_connection():
    # Returns a connection to the database — used by every function in asset_repository.py
    return sqlite3.connect(DB_PATH)


def init_db():
    # Called once at startup from main.py — creates all tables if they don't exist yet
    conn = get_connection()
    cursor = conn.cursor()

    # One row per imported audio file — filepath is UNIQUE to prevent duplicates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            filetype TEXT NOT NULL,
            duration REAL,
            file_size_mb REAL,
            imported_at TEXT NOT NULL
        )
    """)

    # Tags linked to assets — UNIQUE(asset_id, tag) prevents duplicate tags per asset
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            source TEXT NOT NULL,
            UNIQUE(asset_id, tag),
            FOREIGN KEY (asset_id) REFERENCES assets(id)
        )
    """)

    # Logs every search query for tracking purposes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            search_mode TEXT NOT NULL,
            results_count INTEGER NOT NULL,
            searched_at TEXT NOT NULL
        )
    """)

    # CLAP audio embeddings stored as binary blobs — one per asset, computed once
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audio_embeddings (
            asset_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            FOREIGN KEY (asset_id) REFERENCES assets(id)
        )
    """)

    conn.commit()

    # Adds missing columns if running against an older database version
    cursor.execute("PRAGMA table_info(assets)")
    columns = [row[1] for row in cursor.fetchall()]

    if "duration" not in columns:
        cursor.execute("ALTER TABLE assets ADD COLUMN duration REAL")

    if "file_size_mb" not in columns:
        cursor.execute("ALTER TABLE assets ADD COLUMN file_size_mb REAL")

    conn.commit()
    conn.close()