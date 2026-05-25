import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Institutional Alerting System.
    Monitors performance degradation, model drift, and system health.
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {
            "max_drawdown": -15.0,  # 15% Max DD
            "sharpe_min": 0.5,
            "drift_p_value": 0.05,
            "win_rate_min": 40.0,
        }
        self.alerts_history = []

    def check_performance(self, performance_summary):
        alerts = []
        if performance_summary.get("max_drawdown", 0) < self.thresholds["max_drawdown"]:
            alerts.append(
                {
                    "type": "PERFORMANCE_DEGRADATION",
                    "severity": "CRITICAL",
                    "message": f"Max Drawdown ({performance_summary['max_drawdown']:.2f}%) exceeded threshold.",
                }
            )

        if performance_summary.get("sharpe", 0) < self.thresholds["sharpe_min"]:
            alerts.append(
                {
                    "type": "PERFORMANCE_WARNING",
                    "severity": "HIGH",
                    "message": f"Sharpe Ratio ({performance_summary['sharpe']:.2f}) is below acceptable institutional limits.",
                }
            )

        if performance_summary.get("win_rate", 0) < self.thresholds["win_rate_min"]:
            alerts.append(
                {
                    "type": "MODEL_VALIDATION_FAILURE",
                    "severity": "MEDIUM",
                    "message": f"Win Rate ({performance_summary['win_rate']:.2f}%) dropped below threshold.",
                }
            )

        self._log_alerts(alerts)
        return alerts

    def check_drift(self, drift_results):
        alerts = []
        if drift_results.get("is_drifting", False):
            alerts.append(
                {
                    "type": "MODEL_DRIFT_DETECTED",
                    "severity": "HIGH",
                    "message": f"Statistical drift detected in {drift_results.get('drift_ratio', 0) * 100:.1f}% of features.",
                }
            )
        self._log_alerts(alerts)
        return alerts

    def _log_alerts(self, alerts):
        for alert in alerts:
            alert["timestamp"] = datetime.now().isoformat()
            logger.warning(f"ALERT: {json.dumps(alert)}")
            self.alerts_history.append(alert)
            # Trim history
            if len(self.alerts_history) > 100:
                self.alerts_history = self.alerts_history[-100:]

    def get_recent_alerts(self, limit=10):
        return self.alerts_history[-limit:]
