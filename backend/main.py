"""
Hydra Institutional Trading Platform - Main Application Entrypoint.

Re-exports FastAPI application from api module for uvicorn compatibility:
    uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from api import app
except ImportError:
    from backend.api import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
