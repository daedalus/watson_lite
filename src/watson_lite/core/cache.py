import atexit
import json
import logging
import pathlib
import sqlite3
import time
from copy import deepcopy
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# Store the cache database in a platform-appropriate user cache directory so
# it does not end up inside the installed package tree.
_DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "watson_lite"

# Sentinel used to distinguish a cached ``None`` value from a cache miss.
SENTINEL = object()
_SENTINEL = SENTINEL  # Backward-compat alias for internal imports.


class CacheMetrics(TypedDict):
    hits: int
    misses: int
    hits_by_namespace: dict[str, int]
    misses_by_namespace: dict[str, int]


_cache_metrics: CacheMetrics = {
    "hits": 0,
    "misses": 0,
    "hits_by_namespace": {},
    "misses_by_namespace": {},
}


def _namespace_for_key(key: str) -> str:
    prefix = key.split(":", 1)[0].strip().lower()
    return prefix or "other"


def _bump_metric(bucket: str, key: str) -> None:
    namespace = _namespace_for_key(key)
    if bucket == "hits_by_namespace":
        _cache_metrics["hits_by_namespace"][namespace] = (
            _cache_metrics["hits_by_namespace"].get(namespace, 0) + 1
        )
    elif bucket == "misses_by_namespace":
        _cache_metrics["misses_by_namespace"][namespace] = (
            _cache_metrics["misses_by_namespace"].get(namespace, 0) + 1
        )


def get_cache_metrics_snapshot() -> CacheMetrics:
    """Return a deep copy of cache hit/miss counters for KPI reporting."""
    return deepcopy(_cache_metrics)


def reset_cache_metrics() -> None:
    """Reset cache hit/miss counters used by KPI diagnostics."""
    _cache_metrics["hits"] = 0
    _cache_metrics["misses"] = 0
    _cache_metrics["hits_by_namespace"] = {}
    _cache_metrics["misses_by_namespace"] = {}


def is_cache_miss(value: object) -> bool:
    """Return ``True`` when *value* is the sentinel, i.e. a cache miss.

    This is the preferred way to check for a miss; prefer it over an
    identity check against :data:`SENTINEL` so that callers are not
    coupled to the sentinel object itself.
    """
    return value is SENTINEL


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

    @staticmethod
    def _unwrap(raw: str) -> object:
        wrapped = json.loads(raw)
        if isinstance(wrapped, dict) and "v" in wrapped:
            return wrapped["v"]
        return wrapped

    def get(self, key: str) -> Any:  # noqa: ANN401
        """Return the cached value for *key*, or ``None`` on a miss.

        A value of ``None`` that was explicitly cached is returned as ``None``
        (indistinguishable from a miss at the call-site).  If you need to
        distinguish the two cases use :meth:`get_or_sentinel`.

        .. note::

           Legacy cache rows (stored before the ``{"v": …}`` wrapping was
           introduced) are handled transparently via :meth:`_unwrap`.
        """
        row = self.con.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return self._unwrap(row[0])
        return None

    def get_or_sentinel(self, key: str) -> Any:  # noqa: ANN401
        """Return the cached value, or :data:`SENTINEL` on a miss.

        Unlike :meth:`get`, this lets callers distinguish a cached ``None``
        from an absent key.
        """
        row = self.con.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            _cache_metrics["hits"] = int(_cache_metrics["hits"]) + 1
            _bump_metric("hits_by_namespace", key)
            return self._unwrap(row[0])
        _cache_metrics["misses"] = int(_cache_metrics["misses"]) + 1
        _bump_metric("misses_by_namespace", key)
        return SENTINEL

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
