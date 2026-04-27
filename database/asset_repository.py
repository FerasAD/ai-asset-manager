from datetime import datetime
import numpy as np
from database.db import get_connection

#each file insterted into the assets table
def insert_asset(filename, filepath, filetype, duration=0.0, file_size_mb=0.0):
    conn = get_connection()
    cursor = conn.cursor()

    # INSERT OR IGNORE skips the insert if this filepath already exists — prevents duplicates on rescan
    cursor.execute("""
        INSERT OR IGNORE INTO assets (
            filename, filepath, filetype, duration, file_size_mb, imported_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, filepath, filetype, duration, file_size_mb, datetime.now().isoformat()))

    conn.commit()

    # Fetch the asset's id whether it was just inserted or already existed
    cursor.execute("SELECT id FROM assets WHERE filepath = ?", (filepath,))
    asset = cursor.fetchone()
    conn.close()

    return asset[0] if asset else None


def insert_tag(asset_id, tag, source="auto"):
    # INSERT OR IGNORE prevents the same tag being added twice to the same asset
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO tags (asset_id, tag, source) VALUES (?, ?, ?)
    """, (asset_id, tag, source))
    conn.commit()
    conn.close()


def get_tags_for_asset(asset_id):
    # Returns a plain list of tag strings for a given asset — used in the details panel
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT tag FROM tags WHERE asset_id = ? ORDER BY tag ASC", (asset_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def get_all_assets():
    # Returns every asset sorted alphabetically — used to populate the list on startup
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, filepath, filetype, duration, file_size_mb, imported_at
        FROM assets ORDER BY filename ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def search_assets(keyword):
    # Keyword search — matches against filename, filepath, filetype, or any tag
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT a.id, a.filename, a.filepath, a.filetype, a.duration, a.file_size_mb, a.imported_at
        FROM assets a
        LEFT JOIN tags t ON a.id = t.asset_id
        WHERE a.filename LIKE ? OR a.filepath LIKE ? OR a.filetype LIKE ? OR t.tag LIKE ?
        ORDER BY a.filename ASC
    """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_assets_with_tags():
    # Returns every asset with its tags bundled — used by the text-based semantic fallback
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, filepath, filetype, duration, file_size_mb, imported_at
        FROM assets ORDER BY filename ASC
    """)
    assets = cursor.fetchall()
    results = []
    for asset in assets:
        asset_id = asset[0]
        cursor.execute("SELECT tag FROM tags WHERE asset_id = ? ORDER BY tag ASC", (asset_id,))
        tags = [row[0] for row in cursor.fetchall()]
        results.append({"asset": asset, "tags": tags})
    conn.close()
    return results


def log_search(query_text, search_mode, results_count):
    # Records every search query to the search_logs table for evaluation purposes
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO search_logs (query_text, search_mode, results_count, searched_at)
        VALUES (?, ?, ?, ?)
    """, (query_text, search_mode, results_count, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def has_audio_embedding(asset_id: int) -> bool:
    # Returns True if this asset already has a CLAP embedding stored — used to skip re-embedding
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM audio_embeddings WHERE asset_id = ?", (asset_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def store_audio_embedding(asset_id: int, embedding: np.ndarray):
    # Converts the numpy vector to raw bytes and saves it — INSERT OR REPLACE overwrites if exists
    conn = get_connection()
    cursor = conn.cursor()
    blob = embedding.astype(np.float32).tobytes()
    cursor.execute("""
        INSERT OR REPLACE INTO audio_embeddings (asset_id, embedding) VALUES (?, ?)
    """, (asset_id, blob))
    conn.commit()
    conn.close()


def get_all_audio_embeddings() -> list:
    # Loads all stored embeddings from the database, converts blobs back to numpy arrays
    # Returns a list of dicts with 'asset' tuple and 'embedding' array — fed into semantic search
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ae.asset_id, ae.embedding,
               a.filename, a.filepath, a.filetype, a.duration, a.file_size_mb, a.imported_at
        FROM audio_embeddings ae
        JOIN assets a ON ae.asset_id = a.id
    """)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        asset_id, blob, filename, filepath, filetype, duration, file_size_mb, imported_at = row
        embedding = np.frombuffer(blob, dtype=np.float32).copy()
        results.append({
            "asset": (asset_id, filename, filepath, filetype, duration, file_size_mb, imported_at),
            "embedding": embedding
        })
    return results