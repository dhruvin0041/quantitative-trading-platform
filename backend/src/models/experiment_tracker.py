import json
import os
from datetime import datetime

class ExperimentTracker:
    """
    Lightweight, institutional-grade experiment tracking (similar to MLflow/W&B).
    Records hyperparameters, metrics, and model artifacts for reproducibility.
    """
    def __init__(self, db_path="logs/experiments.json"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.experiments = self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.experiments, f, indent=4)

    def log_experiment(self, run_name: str, config: dict, metrics: dict, artifacts: list = None):
        exp = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "artifacts": artifacts or []
        }
        self.experiments.append(exp)
        self._save()
        return exp["run_id"]
