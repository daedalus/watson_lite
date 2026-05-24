import json
import os
import sqlite3
import time
from typing import Any

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "watson_lite_cache.sqlite3",
)


class Cache:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created_at REAL)"
        )

    def get(self, key: str) -> Any | None:  # noqa: ANN401
        row = self.con.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, key: str, value: Any) -> None:  # noqa: ANN401
        self.con.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), time.time()),
        )
        self.con.commit()

    def clear(self) -> None:
        self.con.execute("DELETE FROM cache")
        self.con.commit()

    def close(self) -> None:
        self.con.close()


_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache
