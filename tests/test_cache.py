import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from watson_lite.core.cache import (
    Cache,
    SENTINEL,
    _record_cache_hit,
    _record_cache_miss,
    get_cache_metrics_snapshot,
    reset_cache_metrics,
)


class TestCache:
    def setup_method(self) -> None:
        fd, self.tmp = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        self.cache = Cache(self.tmp)

    def teardown_method(self) -> None:
        self.cache.close()
        os.unlink(self.tmp)
        reset_cache_metrics()

    def test_get_miss(self) -> None:
        assert self.cache.get("nonexistent") is None

    def test_get_or_sentinel_miss(self) -> None:
        assert self.cache.get_or_sentinel("nonexistent") is SENTINEL

    def test_set_and_get(self) -> None:
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

    def test_overwrite(self) -> None:
        self.cache.set("k", "v1")
        self.cache.set("k", "v2")
        assert self.cache.get("k") == "v2"

    def test_complex_value(self) -> None:
        data = {"a": [1, 2, 3], "b": "hello"}
        self.cache.set("complex", data)
        assert self.cache.get("complex") == data

    def test_clear(self) -> None:
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        self.cache.clear()
        assert self.cache.get("a") is None
        assert self.cache.get("b") is None

    def test_none_value_is_cached(self) -> None:
        """Caching None must be distinguishable from a miss via get_or_sentinel."""
        self.cache.set("null", None)
        # get() returns None for both a miss and a cached None — that's documented.
        assert self.cache.get("null") is None
        # get_or_sentinel() distinguishes the two cases.
        assert self.cache.get_or_sentinel("null") is None
        assert self.cache.get_or_sentinel("absent") is SENTINEL

    def test_cache_metrics_updates_are_thread_safe(self) -> None:
        reset_cache_metrics()

        def record_metrics() -> None:
            for _ in range(200):
                _record_cache_hit("wiki:paris")
                _record_cache_miss("graph:eiffel")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(record_metrics) for _ in range(8)]
            for future in futures:
                future.result()

        metrics = get_cache_metrics_snapshot()
        assert metrics["hits"] == 1600
        assert metrics["misses"] == 1600
        assert metrics["hits_by_namespace"] == {"wiki": 1600}
        assert metrics["misses_by_namespace"] == {"graph": 1600}
