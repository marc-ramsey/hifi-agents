import json
import logging
import os
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)


class MemoryStore:
    """SQLite + numpy cosine similarity. Zero external deps beyond numpy."""

    def __init__(self, db_path: str, dimensions: int = 384):
        self.db_path = db_path
        self.dimensions = dimensions
        parent = os.path.dirname(db_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                title TEXT,
                metadata TEXT,
                vector BLOB NOT NULL
            )
        """)
        # FTS5 for text search fallback
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(text, url, title, content='chunks', content_rowid='id')
        """)
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks
            BEGIN
                INSERT INTO chunks_fts(rowid, text, url, title)
                VALUES(new.id, new.text, new.url, new.title);
            END
        """)
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks
            BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, url, title)
                VALUES('delete', old.id, old.text, old.url, old.title);
            END
        """)
        self.conn.commit()

    @staticmethod
    def _serialize(vector: np.ndarray) -> bytes:
        """Pack float32 vector into bytes."""
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def _deserialize(blob: bytes, dimensions: int) -> np.ndarray:
        """Unpack bytes back to float32 vector."""
        return np.frombuffer(blob, dtype=np.float32).reshape(1, -1)

    def upsert(self, text: str, vector: np.ndarray, url: str,
               title: str = "", metadata: dict | None = None) -> int:
        """Add or update a chunk. Returns chunk ID."""
        vec_blob = self._serialize(vector)
        meta_json = json.dumps(metadata or {})

        # Try insert first; if URL conflict, update
        try:
            cur = self.conn.execute(
                """INSERT INTO chunks (text, url, title, metadata, vector)
                   VALUES (?, ?, ?, ?, ?)""",
                (text, url, title, meta_json, vec_blob)
            )
            chunk_id = cur.lastrowid
        except sqlite3.IntegrityError:
            self.conn.execute(
                """UPDATE chunks SET text=?, title=?, metadata=?, vector=?
                   WHERE url=?""",
                (text, title, meta_json, vec_blob, url)
            )
            chunk_id = self.conn.execute(
                "SELECT id FROM chunks WHERE url = ?", (url,)
            ).fetchone()[0]

        self.conn.commit()
        return chunk_id

    def query(self, vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Cosine similarity search. Returns top-k results."""
        # Normalize query
        q = vector.flatten().astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        # Fetch all vectors (for small-medium datasets this is fine)
        rows = self.conn.execute(
            "SELECT id, text, url, title, metadata, vector FROM chunks"
        ).fetchall()

        if not rows:
            return []

        scores = []
        for row in rows:
            vec = self._deserialize(row[5], self.dimensions).flatten()
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            score = float(np.dot(q_norm, vec_norm))
            if score > 0:
                scores.append((score, row))

        # Sort by score descending, take top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, row in scores[:top_k]:
            results.append({
                "text": row[1],
                "url": row[2],
                "title": row[3] or "",
                "score": round(score, 4),
                "metadata": json.loads(row[4]) if row[4] else {}
            })
        return results

    def text_search(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25-style text search via FTS5."""
        rows = self.conn.execute("""
            SELECT c.text, c.url, c.title, c.metadata,
                   rank as search_rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, top_k)).fetchall()

        results = []
        for row in rows:
            results.append({
                "text": row[0],
                "url": row[1],
                "title": row[2] or "",
                "score": round(1.0 / (1.0 + row[4]), 4),
                "metadata": json.loads(row[3]) if row[3] else {}
            })
        return results

    def prune_oldest(self, keep: int = 10000):
        """Remove oldest chunks, keeping the most recent."""
        n = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if n <= keep:
            return
        to_delete = n - keep
        ids = self.conn.execute(
            "SELECT id FROM chunks ORDER BY id ASC LIMIT ?", (to_delete,)
        ).fetchall()
        id_list = [r[0] for r in ids]
        if id_list:
            placeholders = ",".join("?" * len(id_list))
            self.conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", id_list)
            self.conn.commit()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __del__(self):
        try:
            self.close()
        except sqlite3.Error:
            logger.exception("Error closing hifi_search database during cleanup")
