import json
import os
import unittest

import numpy as np
import pandas as pd

from src.execution.alerts import AlertSystem
from src.execution.factor_model import FactorModel
from src.execution.portfolio_optimizer import PortfolioOptimizer
from src.models.monitoring.drift_monitor import DriftMonitor
from src.models.monitoring.experiment_tracker import ExperimentTracker
from src.models.regime.calibration import ModelCalibrator


class TestInstitutionalExcellence(unittest.TestCase):
    def test_experiment_tracker(self):
        db_path = "logs/test_experiments.json"
        tracker = ExperimentTracker(db_path=db_path)
        run_id = tracker.log_experiment("Test_Run", {"lr": 0.01}, {"acc": 0.95})
        self.assertTrue(run_id is not None)
        self.assertTrue(os.path.exists(db_path))
        with open(db_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data[-1]["run_name"], "Test_Run")
        os.remove(db_path)

    def test_portfolio_optimizer(self):
        optimizer = PortfolioOptimizer()
        expected_returns = np.array([0.1, 0.12, 0.15])
        cov_matrix = np.array(
            [[0.05, 0.01, 0.02], [0.01, 0.06, 0.03], [0.02, 0.03, 0.07]]
        )
        weights = optimizer.mean_variance_optimization(expected_returns, cov_matrix)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))

    def test_drift_monitor(self):
        monitor = DriftMonitor()
        train = np.random.normal(0, 1, (100, 5))
        live_no_drift = np.random.normal(0, 1, (100, 5))
        live_drift = np.random.normal(5, 1, (100, 5))

        res_no = monitor.check_covariate_drift(train, live_no_drift)
        res_drift = monitor.check_covariate_drift(train, live_drift)

        self.assertFalse(res_no["is_drifting"])
        self.assertTrue(res_drift["is_drifting"])

    def test_model_calibrator(self):
        calibrator = ModelCalibrator()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        calibrator.fit("test_model", y_true, y_prob)
        calibrated = calibrator.calibrate("test_model", np.array([0.15, 0.85]))
        self.assertTrue(calibrated[0] < calibrated[1])

    def test_factor_model(self):
        fm = FactorModel(n_factors=2)
        returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 10)))
        factor_returns = fm.fit_statistical_factors(returns)
        self.assertEqual(factor_returns.shape, (100, 2))

        ticker_returns = returns.iloc[:, 0]
        idio_risk = fm.calculate_idiosyncratic_risk(
            "T1", ticker_returns, factor_returns
        )
        self.assertTrue(idio_risk > 0)

    def test_alert_system(self):
        alert_system = AlertSystem()
        perf = {"max_drawdown": -20.0, "sharpe": 0.2, "win_rate": 30.0}
        alerts = alert_system.check_performance(perf)
        self.assertEqual(len(alerts), 3)
        self.assertEqual(alerts[0]["type"], "PERFORMANCE_DEGRADATION")

    def test_atomic_writes(self):
        tracker = ExperimentTracker(db_path="logs/atomic_test.json")
        tracker.log_experiment("Atomic", {}, {})
        self.assertTrue(os.path.exists("logs/atomic_test.json"))
        # Check that temp file is cleaned up
        self.assertFalse(os.path.exists("logs/atomic_test.json.tmp"))
        os.remove("logs/atomic_test.json")


if __name__ == "__main__":
    unittest.main()
