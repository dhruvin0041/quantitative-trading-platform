import asyncio
import time


class SimpleCache:
    def __init__(self, default_ttl=300):
        self._cache = {}
        self._lock = asyncio.Lock()
        self.default_ttl = default_ttl

    async def set(self, key, value, ttl=None):
        ttl = ttl or self.default_ttl
        async with self._lock:
            self._cache[key] = {"value": value, "expiry": time.time() + ttl}

    async def get(self, key):
        async with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            if time.time() > item["expiry"]:
                del self._cache[key]
                return None
            return item["value"]
