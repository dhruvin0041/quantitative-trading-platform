import importlib.util
from pathlib import Path

# Explicitly load from backend/execution/broker_interface.py to prevent circular sys.path collision
_impl_path = Path(__file__).resolve().parent.parent.parent / "execution" / "broker_interface.py"
_spec = importlib.util.spec_from_file_location("_broker_interface_impl", str(_impl_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BaseBrokerAdapter = _mod.BaseBrokerAdapter
MockPaperBroker = _mod.MockPaperBroker

__all__ = ["BaseBrokerAdapter", "MockPaperBroker"]
