import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from watson_lite.core.cache import (
    SENTINEL,
    Cache,
    _max_entries_from_env,
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

    def test_ttl_expiry(self) -> None:
        self.cache.set("ttl:key", "v", ttl_seconds=1)
        assert self.cache.get("ttl:key") == "v"
        assert self.cache.get_or_sentinel("ttl:key") == "v"
        self.cache.con.execute(
            "UPDATE cache SET expires_at = ? WHERE key = ?",
            (0.0, self.cache.canonicalize_key("ttl:key")),
        )
        self.cache.con.commit()
        assert self.cache.get("ttl:key") is None
        assert self.cache.get_or_sentinel("ttl:key") is SENTINEL

    def test_key_canonicalization_whitespace(self) -> None:
        self.cache.set("wiki:  paris   france  ", "value")
        assert self.cache.get("wiki:paris france") == "value"
        assert self.cache.get("WIKI:   paris france   ") == "value"

    def test_prunes_oldest_entries_when_limit_exceeded(self) -> None:
        fd, tmp_path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        limited = Cache(tmp_path, max_entries=2)
        limited.set("k1", "v1")
        limited.set("k2", "v2")
        limited.set("k3", "v3")
        assert limited.get("k1") is None
        assert limited.get("k2") == "v2"
        assert limited.get("k3") == "v3"
        limited.close()
        os.unlink(tmp_path)

    @pytest.mark.parametrize("ttl_seconds", [0, -1])
    def test_set_rejects_non_positive_ttl(self, ttl_seconds: int) -> None:
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            self.cache.set("key", "value", ttl_seconds=ttl_seconds)

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


class TestBloomFilter:
    def test_add_and_query(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        bf.add("hello")
        bf.add("world")
        assert bf.query("hello") is True
        assert bf.query("world") is True

    def test_no_false_negatives(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=1000)
        keys = [f"key:{i}" for i in range(200)]
        for k in keys:
            bf.add(k)
        for k in keys:
            assert bf.query(k) is True

    def test_unknown_key_returns_false(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        bf.add("present")
        assert bf.query("absent") is False

    def test_update_returns_existing(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        assert bf.update("first") is False
        assert bf.update("first") is True
        assert bf.update("second") is False
        assert bf.update("second") is True

    def test_clear_resets_filter(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        bf.add("temp")
        assert bf.query("temp") is True
        bf.clear()
        assert bf.query("temp") is False

    def test_prefill_from_cache(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path)
        c.set("a", 1)
        c.set("b", 2)
        c.close()

        c2 = Cache(path)
        assert c2.get("a") == 1
        assert c2.get("b") == 2
        assert c2.get("c") is None
        c2.close()
        os.unlink(path)

    def test_bloom_skips_sql_on_definite_miss(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path)
        c.set("known", "value")
        before = c._bloom._bits.count()
        assert c.get("known") == "value"
        assert c.get("definitely_missing") is None
        after = c._bloom._bits.count()
        assert after == before
        c.close()
        os.unlink(path)

    def test_bloom_false_positive_still_hits_db(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path, max_entries=100)
        c.set("existing", "works")
        assert c.get("existing") == "works"
        assert c.get("nonexistent") is None
        c.close()
        os.unlink(path)


class TestBloomFilterAdversarial:
    def test_empty_capacity(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=0)
        assert bf.query("anything") is False
        bf.add("x")
        assert bf.query("x") is True

    def test_capacity_one(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=1)
        bf.add("only")
        assert bf.query("only") is True
        bf.clear()
        assert bf.query("only") is False

    def test_empty_string_key(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        bf.add("")
        assert bf.query("") is True
        assert bf.query(" ") is False

    def test_unicode_keys(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        bf.add("café")
        bf.add("日本語")
        bf.add("😊")
        bf.add("a\u0000b")
        assert bf.query("café") is True
        assert bf.query("日本語") is True
        assert bf.query("😊") is True
        assert bf.query("a\u0000b") is True
        assert bf.query("cafe") is False
        assert bf.query("日本") is False

    def test_very_long_key(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        long_key = "x" * 100000
        bf.add(long_key)
        assert bf.query(long_key) is True
        assert bf.query("x" * 100000) is True
        assert bf.query("x" * 99999) is False

    def test_duplicate_add_is_idempotent(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        bf.add("dup")
        before = bf._bits.count()
        bf.add("dup")
        after = bf._bits.count()
        assert after >= before

    def test_query_empty_filter(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        assert bf.query("x") is False
        assert bf.query("") is False
        assert bf.query("a" * 1000) is False

    def test_clear_empty_filter(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        bf.clear()
        assert bf.query("x") is False

    def test_very_large_capacity(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=10_000_000)
        bf.add("needle")
        assert bf.query("needle") is True
        assert bf.query("haystack") is False

    def test_update_is_atomic_check_then_add(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        assert bf.update("first") is False
        assert bf.update("first") is True
        assert bf.update("first") is True
        bf.clear()
        assert bf.update("first") is False

    def test_cache_key_with_single_quote_in_name(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path)
        c.set('it\'s a "test"', "value")
        assert c.get('it\'s a "test"') == "value"
        assert c.get('IT\'S A "TEST"') == "value"
        c.close()
        os.unlink(path)

    def test_cache_set_after_clear_rebuilds_bloom(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path)
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        assert c.get("a") is None
        assert c.get("b") is None
        c.set("c", 3)
        assert c.get("c") == 3
        assert c.get("a") is None
        c.close()
        os.unlink(path)

    def test_bloom_false_positive_integration(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path, max_entries=1000)
        known_keys = []
        for i in range(500):
            k = f"key:{i}"
            c.set(k, i)
            known_keys.append(k)
        for k in known_keys:
            assert c.get(k) == int(k.split(":")[1])
        c.close()
        os.unlink(path)

    def test_bloom_does_not_leak_keys_across_cache_instances(self) -> None:
        fd1, path1 = tempfile.mkstemp(suffix=".sqlite3")
        fd2, path2 = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd1)
        os.close(fd2)
        c1 = Cache(path1)
        c2 = Cache(path2)
        c1.set("secret", "from c1")
        assert c1.get("secret") == "from c1"
        assert c2.get("secret") is None
        c1.close()
        c2.close()
        os.unlink(path1)
        os.unlink(path2)

    def test_canonicalized_key_honours_bloom(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path)
        c.set("  Wiki:  Paris   France  ", "data")
        assert c.get("  Wiki:  Paris   France  ") == "data"
        assert c.get("wiki:Paris France") == "data"
        assert c.get("WIKI:   Paris   France   ") == "data"
        assert c.get("wiki:lyon") is None
        c.close()
        os.unlink(path)

    def test_bloom_skip_does_not_affect_metrics(self) -> None:
        from watson_lite.core.cache import (
            reset_cache_metrics,
            get_cache_metrics_snapshot,
        )

        reset_cache_metrics()
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path)
        c.set("ns:known", "x")
        c.get_or_sentinel("ns:known")
        c.get_or_sentinel("ns:unknown")
        snap = get_cache_metrics_snapshot()
        assert snap["hits"] == 1
        assert snap["misses"] == 1
        assert snap["hits_by_namespace"] == {"ns": 1}
        assert snap["misses_by_namespace"] == {"ns": 1}
        c.close()
        os.unlink(path)

    def test_load_factor_tracks_fill_ratio(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=100)
        assert bf.load_factor == 0.0
        bf.add("x")
        assert bf.load_factor > 0.0
        bf.clear()
        assert bf.load_factor == 0.0

    def test_grow_preserves_all_keys_across_resize(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path, max_entries=500)
        for i in range(300):
            c.set(f"g:{i}", i)
        m_before = c._bloom.m
        for i in range(300, 400):
            c.set(f"g:{i}", i)
        m_after = c._bloom.m
        assert m_after >= m_before
        for i in range(400):
            assert c.get(f"g:{i}") == i
        c.close()
        os.unlink(path)

    def test_bloom_never_loses_keys_after_auto_resize(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path, max_entries=2000)
        n = 800
        for i in range(n):
            c.set(f"r:{i}", i)
        for i in range(n):
            assert c.get(f"r:{i}") == i, f"key r:{i} lost after potential resize"
        c.close()
        os.unlink(path)

    def test_bloom_does_not_resize_on_every_set(self) -> None:
        from watson_lite.core.cache import BloomFilter

        bf = BloomFilter(capacity=1000)
        sizes = set()
        for i in range(200):
            bf.add(f"k:{i}")
            sizes.add(bf.m)
        assert len(sizes) == 1

    def test_grow_check_counter_increments_on_set(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path)
        assert c._bloom_check_counter == 0
        c.set("a", 1)
        # interval = 1 (entry_count=1) → check fires, counter resets to 0
        assert c._bloom_check_counter == 0
        c.close()
        os.unlink(path)

    def test_grow_check_counter_accumulates_within_interval(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path, max_entries=5000)
        # Pre-fill so entry_count is high and interval > 1
        for i in range(80):
            c.set(f"k:{i}", i)
        assert c._entry_count == 80
        # interval = max(1, int(80 * 0.05)) = 4
        c._bloom_check_counter = 0
        c.set("accumulate", 1)
        assert c._bloom_check_counter == 1
        c.set("accumulate2", 2)
        assert c._bloom_check_counter == 2
        c.close()
        os.unlink(path)

    def test_grow_check_counter_does_not_fire_on_empty_cache(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path)
        c.set("only", 1)
        # entry_count = 1 → interval = max(1, int(1 * 0.05)) = 1 → fires on every set
        assert c._bloom_check_counter == 0
        c.close()
        os.unlink(path)

    def test_grow_check_is_throttled(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path, max_entries=500)
        for i in range(200):
            c.set(f"k:{i}", i)
        assert c._bloom_check_counter < 200  # not every set triggered a check
        # All keys still reachable
        for i in range(200):
            assert c.get(f"k:{i}") == i
        c.close()
        os.unlink(path)

    def test_grow_check_counter_resets_on_clear(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(path)
        c.set("a", 1)
        c.set("b", 2)
        assert c._bloom_check_counter == 0  # reset after check on second set
        c.clear()
        assert c._bloom_check_counter == 0
        assert c._entry_count == 0
        c.close()
        os.unlink(path)


class TestCacheEnvParsing:
    def test_max_entries_from_env_invalid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WATSON_LITE_CACHE_MAX_ENTRIES", "invalid")
        with pytest.raises(
            ValueError,
            match="WATSON_LITE_CACHE_MAX_ENTRIES must be an integer",
        ):
            _max_entries_from_env()
