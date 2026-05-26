import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src.execution.signal_journal import SignalJournal
from src.execution.signal_learning import SignalLearningEngine


def analyze_signals():
    journal = SignalJournal()
    engine = SignalLearningEngine(journal)

    report = engine.discover_optimal_conditions()

    print("=" * 60)
    print("HYDRA SIGNAL LEARNING REPORT")
    print("=" * 60)

    if report["status"] == "INSUFFICIENT_DATA":
        print(
            f"Status: Waiting for more trade outcomes (Current: {report.get('count', 0)})"
        )
    else:
        print(f"Total Signals Analyzed: {report['total_signals_analyzed']}")
        print(f"Overall Win Rate: {report['win_rate']}%")
        print(f"Golden Condition: {report['golden_condition']}")

        print("\nPerformance by Quality Grade:")
        for grade, stats in report["grade_performance"].items():
            print(
                f"  {grade:15} | Win Rate: {stats['mean'] * 100:>.1f}% | Count: {stats['count']}"
            )

        print("\nPerformance by Market Regime:")
        for regime, stats in report["regime_performance"].items():
            print(
                f"  {regime:15} | Win Rate: {stats['mean'] * 100:>.1f}% | Count: {stats['count']}"
            )

    print("=" * 60)


if __name__ == "__main__":
    analyze_signals()
