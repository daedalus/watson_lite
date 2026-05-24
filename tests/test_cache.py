import os
import tempfile

from watson_lite.core.cache import Cache


class TestCache:
    def setup_method(self) -> None:
        self.tmp = tempfile.mktemp(suffix=".sqlite3")
        self.cache = Cache(self.tmp)

    def teardown_method(self) -> None:
        self.cache.close()
        os.unlink(self.tmp)

    def test_get_miss(self) -> None:
        assert self.cache.get("nonexistent") is None

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

    def test_none_value(self) -> None:
        self.cache.set("null", None)
        assert self.cache.get("null") is None
