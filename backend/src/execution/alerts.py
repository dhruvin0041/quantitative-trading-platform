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

        def get_val(key):
            val = performance_summary.get(key, 0)
            if isinstance(val, str):
                try:
                    # Strip symbols like % if present
                    clean_val = val.replace("%", "").split(" ")[0]
                    return float(clean_val)
                except (ValueError, IndexError):
                    return None
            return float(val)

        max_dd = get_val("max_drawdown")
        if max_dd is not None and max_dd < self.thresholds["max_drawdown"]:
            alerts.append(
                {
                    "type": "PERFORMANCE_DEGRADATION",
                    "severity": "CRITICAL",
                    "message": f"Max Drawdown ({max_dd:.2f}%) exceeded threshold.",
                }
            )

        sharpe = get_val("sharpe")
        if sharpe is not None and sharpe < self.thresholds["sharpe_min"]:
            alerts.append(
                {
                    "type": "PERFORMANCE_WARNING",
                    "severity": "HIGH",
                    "message": f"Sharpe Ratio ({sharpe:.2f}) is below acceptable institutional limits.",
                }
            )

        win_rate = get_val("win_rate")
        if win_rate is not None and win_rate < self.thresholds["win_rate_min"]:
            alerts.append(
                {
                    "type": "MODEL_VALIDATION_FAILURE",
                    "severity": "MEDIUM",
                    "message": f"Win Rate ({win_rate:.2f}%) dropped below threshold.",
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
