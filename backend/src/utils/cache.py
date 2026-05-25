import asyncio

class SimpleCache:
    def __init__(self):
        self._cache = {}
        self._lock = asyncio.Lock()

    async def set(self, key, value):
        async with self._lock:
            self._cache[key] = value

    async def get(self, key):
        async with self._lock:
            return self._cache.get(key)
