import sqlite3
import json
import time
import threading

class SQLiteCache:
    def __init__(self, db_path="data/cache.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS api_cache (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        timestamp REAL
                    )
                ''')

    def get(self, key, max_age_seconds=300):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value, timestamp FROM api_cache WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    value, timestamp = row
                    if time.time() - timestamp <= max_age_seconds:
                        return json.loads(value)
        return None

    def set(self, key, value):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                v = json.dumps(value)
                t = time.time()
                conn.execute(
                    "INSERT OR REPLACE INTO api_cache (key, value, timestamp) VALUES (?, ?, ?)",
                    (key, v, t)
                )

# Global Cache Instance
api_cache = SQLiteCache()
