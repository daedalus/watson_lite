import atexit
import json
import logging
import os
import pathlib
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)

# Store the cache database in a platform-appropriate user cache directory so
# it does not end up inside the installed package tree.
_DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "watson_lite"

# Sentinel used to distinguish a cached ``None`` value from a cache miss.
_SENTINEL = object()


def _default_db_path() -> str:
    _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(_DEFAULT_CACHE_DIR / "watson_lite_cache.sqlite3")


class Cache:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path if db_path is not None else _default_db_path()
        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS cache"
            " (key TEXT PRIMARY KEY, value TEXT, created_at REAL)"
        )

    def get(self, key: str) -> Any:  # noqa: ANN401
        """Return the cached value for *key*, or ``None`` on a miss.

        A value of ``None`` that was explicitly cached is returned as ``None``
        (indistinguishable from a miss at the call-site).  If you need to
        distinguish the two cases use :meth:`get_or_sentinel`.
        """
        row = self.con.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            wrapped = json.loads(row[0])
            return wrapped["v"]
        return None

    def get_or_sentinel(self, key: str) -> Any:  # noqa: ANN401
        """Return the cached value, or :data:`_SENTINEL` on a miss.

        Unlike :meth:`get`, this lets callers distinguish a cached ``None``
        from an absent key.
        """
        row = self.con.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            wrapped = json.loads(row[0])
            return wrapped["v"]
        return _SENTINEL

    def set(self, key: str, value: Any) -> None:  # noqa: ANN401
        # Wrap the value so that ``None`` is stored distinctly from a miss.
        self.con.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps({"v": value}, default=str), time.time()),
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
        atexit.register(_cache.close)
    return _cache
