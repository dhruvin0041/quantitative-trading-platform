import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from execution.broker_interface import BaseBrokerAdapter, MockPaperBroker
except ImportError:
    from .broker_interface import BaseBrokerAdapter, MockPaperBroker

__all__ = ["BaseBrokerAdapter", "MockPaperBroker"]
