# backend/src/models/classical/__init__.py
from src.models.classical.rf_agent import RandomForestAgent, train_rf_agent

__all__ = ["RandomForestAgent", "train_rf_agent"]
