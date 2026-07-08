import atexit
import hashlib
import json
import logging
import math
import os
import pathlib
import sqlite3
import time
from copy import deepcopy
from threading import Lock
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# Store the cache database in a platform-appropriate user cache directory so
# it does not end up inside the installed package tree.
_DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "watson_lite"
_DEFAULT_MAX_ENTRIES = 5000
_MAINTENANCE_INTERVAL_WRITES = 25

# Sentinel used to distinguish a cached ``None`` value from a cache miss.
SENTINEL = object()
_SENTINEL = SENTINEL  # Backward-compat alias for internal imports.

_BLOOM_ERROR_RATE = 0.01


class BloomFilter:
    """Bloom filter backed by a :class:`bytearray`.

    Uses a single SHA-256 hash with bit-variable slicing to produce *k*
    independent index positions from the 256-bit digest.  The filter size *m*
    is rounded up to a power of two so that fast bitwise masking (``& (m-1)``)
    can replace modulo.

    Each position *p* is mapped to a byte in the backing array and a bit
    within that byte::

        byte_idx = (p & mask) >> 3
        bit_idx  = (p & mask) & 7
    """

    def __init__(self, capacity: int, error_rate: float = _BLOOM_ERROR_RATE) -> None:
        n = max(capacity, 1)
        m_ideal = -n * math.log(error_rate) / (math.log(2) ** 2)
        self.m = 1 << max(1, int(m_ideal).bit_length())
        self._mask = self.m - 1
        self._bits_per_slice = self.m.bit_length() - 1
        self._k = max(1, 256 // self._bits_per_slice)

        self._byte_len = (self.m + 7) // 8
        self._bits = bytearray(self._byte_len)

    @staticmethod
    def _digest(key: str) -> int:
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")

    def _check(self, value: int) -> bool:
        v = value
        for _ in range(self._k):
            pos = v & self._mask
            byte_idx = pos >> 3
            bit_idx = pos & 7
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
            v >>= self._bits_per_slice
        return True

    def _set(self, value: int) -> None:
        v = value
        for _ in range(self._k):
            pos = v & self._mask
            byte_idx = pos >> 3
            bit_idx = pos & 7
            self._bits[byte_idx] |= 1 << bit_idx
            v >>= self._bits_per_slice

    def add(self, key: str) -> None:
        self._set(self._digest(key))

    def query(self, key: str) -> bool:
        return self._check(self._digest(key))

    def update(self, key: str) -> bool:
        """Check membership then add.  Returns ``True`` if the key was already present."""
        value = self._digest(key)
        if self._check(value):
            return True
        self._set(value)
        return False

    @property
    def load_factor(self) -> float:
        """Fraction of bits set to 1.  Used to decide when to resize."""
        bits_set = sum(b.bit_count() for b in self._bits)
        return bits_set / self.m

    def clear(self) -> None:
        self._bits = bytearray(self._byte_len)


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
_cache_metrics_lock = Lock()


def _namespace_for_key(key: str) -> str:
    prefix = key.split(":", 1)[0].strip().lower()
    return prefix or "other"


def _record_cache_hit(key: str) -> None:
    namespace = _namespace_for_key(key)
    with _cache_metrics_lock:
        _cache_metrics["hits"] = int(_cache_metrics["hits"]) + 1
        _cache_metrics["hits_by_namespace"][namespace] = (
            _cache_metrics["hits_by_namespace"].get(namespace, 0) + 1
        )


def _record_cache_miss(key: str) -> None:
    namespace = _namespace_for_key(key)
    with _cache_metrics_lock:
        _cache_metrics["misses"] = int(_cache_metrics["misses"]) + 1
        _cache_metrics["misses_by_namespace"][namespace] = (
            _cache_metrics["misses_by_namespace"].get(namespace, 0) + 1
        )


def get_cache_metrics_snapshot() -> CacheMetrics:
    """Return a deep copy of cache hit/miss counters for KPI reporting."""
    with _cache_metrics_lock:
        return deepcopy(_cache_metrics)


def reset_cache_metrics() -> None:
    """Reset cache hit/miss counters used by KPI diagnostics."""
    with _cache_metrics_lock:
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
    def __init__(
        self,
        db_path: str | None = None,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.db_path = db_path if db_path is not None else _default_db_path()
        self.max_entries = max_entries
        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("PRAGMA busy_timeout=5000")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS cache"
            " (key TEXT PRIMARY KEY, value TEXT, created_at REAL, expires_at REAL)"
        )
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_created_at ON cache(created_at)"
        )
        self._ensure_expires_column()
        self._entry_count = self._count_entries()
        self._writes_since_maintenance = 0
        self._bloom_check_counter = 0
        self._delete_expired()
        self._init_bloom()

    def _init_bloom(self) -> None:
        capacity = max(1, self._entry_count) * 10
        self._bloom = BloomFilter(capacity)
        rows = self.con.execute("SELECT key FROM cache").fetchall()
        for (key,) in rows:
            self._bloom.add(key)

    def _maybe_grow_bloom(self) -> None:
        self._bloom_check_counter += 1
        interval = max(1, int(self._entry_count * 0.05))
        if self._bloom_check_counter < interval:
            return
        self._bloom_check_counter = 0
        if self._bloom.load_factor > 0.8:
            self._init_bloom()

    def _maybe_shrink_bloom(self, deleted: int) -> None:
        """Rebuild bloom filter after heavy pruning to clear stale bits.

        When many entries are deleted, the bloom filter retains bits for keys
        that no longer exist in the database, causing unnecessary SQLite lookups.
        Rebuild when load factor drops below 0.3 after deleting ≥10% of entries.
        """
        if deleted <= 0 or self._entry_count <= 0:
            return
        ratio = deleted / (self._entry_count + deleted)
        if ratio >= 0.1 and self._bloom.load_factor < 0.3:
            self._init_bloom()

    def _count_entries(self) -> int:
        row = self.con.execute("SELECT COUNT(*) FROM cache").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def canonicalize_key(key: str) -> str:
        parts = key.strip().split(":", 1)
        namespace = parts[0].strip().lower()
        if len(parts) == 1:
            return namespace
        suffix = " ".join(parts[1].split())
        return f"{namespace}:{suffix}"

    def _ensure_expires_column(self) -> None:
        columns = {
            str(row[1])
            for row in self.con.execute("PRAGMA table_info(cache)").fetchall()
        }
        if "expires_at" not in columns:
            self.con.execute("ALTER TABLE cache ADD COLUMN expires_at REAL")
            self.con.commit()

    def _delete_expired(self) -> int:
        now = time.time()
        cur = self.con.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        deleted = int(cur.rowcount or 0)
        if deleted:
            self._entry_count = max(0, self._entry_count - deleted)
            self.con.commit()
        return deleted

    def _prune_if_needed(self) -> int:
        if self.max_entries <= 0:
            return 0
        overflow = self._entry_count - self.max_entries
        if overflow <= 0:
            return 0
        cur = self.con.execute(
            "DELETE FROM cache WHERE key IN ("
            "SELECT key FROM cache ORDER BY created_at ASC LIMIT ?"
            ")",
            (overflow,),
        )
        deleted = int(cur.rowcount or 0)
        if deleted:
            self._entry_count = max(0, self._entry_count - deleted)
            self.con.commit()
            self._maybe_shrink_bloom(deleted)
        return deleted

    def _delete_key(self, canonical_key: str) -> None:
        cur = self.con.execute("DELETE FROM cache WHERE key = ?", (canonical_key,))
        deleted = int(cur.rowcount or 0)
        if deleted:
            self._entry_count = max(0, self._entry_count - deleted)
            self.con.commit()

    @staticmethod
    def _unwrap(raw: str) -> Any:  # noqa: ANN401
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
        canonical_key = self.canonicalize_key(key)
        if not self._bloom.query(canonical_key):
            return None
        row = self.con.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (canonical_key,)
        ).fetchone()
        if row:
            expires_at = row[1]
            if expires_at is not None and float(expires_at) <= time.time():
                self._delete_key(canonical_key)
                return None
            return self._unwrap(row[0])
        return None

    def get_or_sentinel(self, key: str) -> Any:  # noqa: ANN401
        """Return the cached value, or :data:`SENTINEL` on a miss.

        Unlike :meth:`get`, this lets callers distinguish a cached ``None``
        from an absent key.
        """
        canonical_key = self.canonicalize_key(key)
        if not self._bloom.query(canonical_key):
            _record_cache_miss(canonical_key)
            return SENTINEL
        row = self.con.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (canonical_key,)
        ).fetchone()
        if row:
            expires_at = row[1]
            if expires_at is not None and float(expires_at) <= time.time():
                self._delete_key(canonical_key)
                _record_cache_miss(canonical_key)
                return SENTINEL
            _record_cache_hit(canonical_key)
            return self._unwrap(row[0])
        _record_cache_miss(canonical_key)
        return SENTINEL

    def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:  # noqa: ANN401
        canonical_key = self.canonicalize_key(key)
        self._bloom.add(canonical_key)
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive (greater than zero)")
        expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None
        existing = self.con.execute(
            "SELECT 1 FROM cache WHERE key = ?",
            (canonical_key,),
        ).fetchone()
        # Wrap the value so that ``None`` is stored distinctly from a miss.
        self.con.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (
                canonical_key,
                json.dumps({"v": value}, default=str),
                time.time(),
                expires_at,
            ),
        )
        self.con.commit()
        if existing is None:
            self._entry_count += 1
        self._writes_since_maintenance += 1
        needs_maintenance = (
            self._writes_since_maintenance >= _MAINTENANCE_INTERVAL_WRITES
            or self._entry_count > self.max_entries
        )
        if needs_maintenance:
            self._delete_expired()
            self._prune_if_needed()
            self._writes_since_maintenance = 0
        self._maybe_grow_bloom()

    def clear(self) -> None:
        self.con.execute("DELETE FROM cache")
        self.con.commit()
        self._entry_count = 0
        self._writes_since_maintenance = 0
        self._bloom_check_counter = 0
        self._init_bloom()

    def close(self) -> None:
        self.con.close()


_cache: Cache | None = None


def _max_entries_from_env() -> int:
    raw = os.getenv("WATSON_LITE_CACHE_MAX_ENTRIES", str(_DEFAULT_MAX_ENTRIES))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"WATSON_LITE_CACHE_MAX_ENTRIES must be an integer; received {raw!r}"
        ) from exc


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        max_entries = _max_entries_from_env()
        _cache = Cache(max_entries=max_entries)
        atexit.register(_cache.close)
    return _cache
