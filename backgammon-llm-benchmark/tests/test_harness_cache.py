"""Tests: content-addressed response cache round-trip (PLAN.md §5.2)."""

from harness.cache import ResponseCache, cache_key


def test_cache_key_stable_and_sensitive():
    base = dict(model="a/b", prompt_version="v1", position_id="p1", track="text",
                sampling={"temperature": 0, "n": 1})
    k1 = cache_key(**base)
    k2 = cache_key(**base)
    assert k1 == k2
    # order of sampling keys must not matter
    base2 = dict(base, sampling={"n": 1, "temperature": 0})
    assert cache_key(**base2) == k1
    # any identifying change flips the key
    assert cache_key(**dict(base, model="c/d")) != k1
    assert cache_key(**dict(base, track="image")) != k1
    assert cache_key(**dict(base, sampling={"temperature": 0.7, "n": 1})) != k1


def test_cache_miss_then_hit(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    key = cache.key(model="a/b", prompt_version="v1", position_id="p1", track="text",
                    sampling={"temperature": 0})
    assert cache.get(key) is None
    assert not cache.has(key)
    value = {"text": "MOVE: 8/5 6/5", "cost_usd": 0.001, "prompt_tokens": 10}
    cache.put(key, value)
    assert cache.has(key)
    assert cache.get(key) == value


def test_get_or_none(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    kw = dict(model="a/b", prompt_version="v1", position_id="p2", track="text",
              sampling={"temperature": 0})
    k, hit = cache.get_or_none(**kw)
    assert hit is None
    cache.put(k, {"text": "x"})
    k2, hit2 = cache.get_or_none(**kw)
    assert k2 == k and hit2 == {"text": "x"}


def test_corrupt_file_returns_none(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    key = cache.key(model="a", prompt_version="v", position_id="p", track="text", sampling={})
    path = cache._path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert cache.get(key) is None
