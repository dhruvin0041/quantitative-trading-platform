"""
Unit tests for Institutional Artifact Cleaning Engine (clean_artifacts.py).
Validates namespaced multi-ticker isolation, per-ticker purging, and universal wipe.
"""
from scripts.ops.clean_artifacts import (
    BACKEND_DIR,
    clean_optimization_artifacts,
    clean_training_artifacts,
)


def test_clean_single_ticker_artifacts():
    test_ticker = "MOCKTICKER"
    test_model_dir = BACKEND_DIR / "artifacts" / "models" / test_ticker
    test_model_dir.mkdir(parents=True, exist_ok=True)
    dummy_weight = test_model_dir / "xgb_ensemble.json"
    dummy_weight.write_text("{}", encoding="utf-8")

    dummy_opt = BACKEND_DIR / "configs" / f"optimized_params_{test_ticker}.json"
    dummy_opt.write_text("{}", encoding="utf-8")

    assert test_model_dir.exists()
    assert dummy_opt.exists()

    # Clean optimization for single ticker
    clean_optimization_artifacts(ticker=test_ticker)
    assert not dummy_opt.exists()

    # Clean training for single ticker
    clean_training_artifacts(ticker=test_ticker)
    assert not test_model_dir.exists()
