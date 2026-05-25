import os
import json
from datetime import datetime
import mlflow

class ExperimentTracker:
    """
    Institutional-grade experiment tracking using MLflow.
    Records hyperparameters, metrics, and model artifacts for reproducibility
    and maintains a model registry.
    """
    def __init__(self, tracking_uri="file:./mlruns", db_path="data/experiments.json"):
        self.tracking_uri = tracking_uri
        self.db_path = db_path # Keep JSON for simple backward compatibility
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
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
        temp_path = f"{self.db_path}.tmp"
        with open(temp_path, "w") as f:
            json.dump(self.experiments, f, indent=4)
        os.replace(temp_path, self.db_path)

    def log_experiment(self, run_name: str, config: dict, metrics: dict, artifacts: list = None, model=None, model_type=None):
        """
        Logs the experiment to both MLflow and the local JSON file.
        """
        run_id_json = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Log to JSON (legacy/fallback)
        exp = {
            "run_id": run_id_json,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "artifacts": artifacts or []
        }
        self.experiments.append(exp)
        self._save()
        
        # 2. Log to MLflow
        mlflow.set_experiment(run_name.split("_")[0] if "_" in run_name else "StockIndicator")
        
        with mlflow.start_run(run_name=run_name):
            # Log flattened params
            flat_config = self._flatten_dict(config)
            mlflow.log_params(flat_config)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model if provided
            if model is not None and model_type is not None:
                if model_type == "keras":
                    mlflow.keras.log_model(model, artifact_path="model", registered_model_name=f"{run_name}_keras")
                elif model_type == "lightgbm":
                    mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name=f"{run_name}_lgb")
                elif model_type == "xgboost":
                    mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=f"{run_name}_xgb")
                elif model_type == "pytorch":
                    mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=f"{run_name}_pt")
                elif model_type == "sklearn":
                    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=f"{run_name}_skl")
                    
            # Log artifacts (files)
            if artifacts:
                for path in artifacts:
                    if os.path.exists(path):
                        mlflow.log_artifact(path)
                        
        return run_id_json

    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flattens a nested dictionary for MLflow param logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
