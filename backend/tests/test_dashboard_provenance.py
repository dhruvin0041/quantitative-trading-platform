import hashlib
from pathlib import Path

import pytest

from api import _enforce_dashboard_provenance, inference_service
from src.execution.signal_ledger import SignalLedger


@pytest.fixture
def mock_ledger(tmp_path):
    db_path = str(tmp_path / "test_signal_ledger.db")
    ledger = SignalLedger(db_path=db_path)

    # 1. Legacy signal
    ledger.record_signal(
        symbol="AAPL",
        bar_timestamp="2026-06-15",
        signal="BUY",
        confidence=0.85,
        execution_target_bar="2026-06-16",
        execution_price=300.0,
        model_version="V1_LEGACY",
        raw_features_hash="hash_legacy",
        dataset="PRELIMINARY_HISTORICAL_EVIDENCE"
    )

    # 2. Prospective signal with wrong manifest hash
    ledger.record_prospective_signal(
        symbol="AAPL",
        source_candle_timestamp="2026-07-01",
        signal_generation_timestamp="2026-07-01T16:00:00Z",
        signal="BUY",
        probability=0.75,
        confidence=0.75,
        feature_hash="feat_old",
        model_hash="model_old",
        execution_target_timestamp="2026-07-02",
        signal_reference_price=310.0,
        strategy_version="HYDRA_PROSPECTIVE_V1",
        manifest_hash="old_invalid_hash"
    )

    # 3. Prospective signal with NO manifest hash
    ledger.record_prospective_signal(
        symbol="AAPL",
        source_candle_timestamp="2026-07-05",
        signal_generation_timestamp="2026-07-05T16:00:00Z",
        signal="SELL",
        probability=0.75,
        confidence=0.75,
        feature_hash="feat_old2",
        model_hash="model_old2",
        execution_target_timestamp="2026-07-06",
        signal_reference_price=310.0,
        strategy_version="HYDRA_PROSPECTIVE_V1",
        manifest_hash=None
    )

    return ledger

def test_provenance_filter_success(mock_ledger):
    """Test that valid markers are kept and invalid ones are excluded."""
    manifest_path = Path(__file__).resolve().parent.parent / "artifacts" / "frozen_strategy_manifest_v2.3.json"
    active_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    # 4. Valid prospective signal
    mock_ledger.record_prospective_signal(
        symbol="AAPL",
        source_candle_timestamp="2026-07-10",
        signal_generation_timestamp="2026-07-10T16:00:00Z",
        signal="SELL",
        probability=0.95,
        confidence=0.95,
        feature_hash="feat_new",
        model_hash="model_new",
        execution_target_timestamp="2026-07-11",
        signal_reference_price=320.0,
        strategy_version="HYDRA_PROSPECTIVE_V2.3",
        manifest_hash=active_hash
    )

    inference_service.signal_ledger = mock_ledger

    response_data = {"historical_markers": [{"action": "OLD_STUFF"}]}
    result = _enforce_dashboard_provenance("AAPL", response_data)

    markers = result["historical_markers"]
    assert len(markers) == 1, "Should exactly include only the valid manifest hash marker"
    assert markers[0]["manifest_hash"] == active_hash
    assert markers[0]["provenance"] == "MANIFEST_MATCHED"


def test_provenance_filter_fail_closed(mock_ledger, monkeypatch):
    """Test that if manifest is unreadable, it fails closed and raises RuntimeError."""
    inference_service.signal_ledger = mock_ledger

    original_read_bytes = Path.read_bytes
    def mock_read_bytes(self):
        if "frozen_strategy_manifest" in str(self):
            raise FileNotFoundError("Manifest missing")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", mock_read_bytes)

    response_data = {"historical_markers": [{"action": "OLD_STUFF"}]}

    with pytest.raises(RuntimeError, match="Cannot verify Stage 3 provenance"):
        _enforce_dashboard_provenance("AAPL", response_data)


def test_api_predict_cache_hit_and_miss(mock_ledger, monkeypatch):
    """Test that /predict applies provenance filtering on both cache hits and misses."""
    from fastapi.testclient import TestClient

    from api import app, inference_service

    # Use the test ledger
    inference_service.signal_ledger = mock_ledger

    manifest_path = Path(__file__).resolve().parent.parent / "artifacts" / "frozen_strategy_manifest_v2.3.json"
    active_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    # Insert one valid marker and one invalid marker
    mock_ledger.record_prospective_signal(
        symbol="AAPL",
        source_candle_timestamp="2026-07-15",
        signal_generation_timestamp="2026-07-15T16:00:00Z",
        signal="BUY",
        probability=0.9,
        confidence=0.9,
        feature_hash="feat_valid",
        model_hash="model_valid",
        execution_target_timestamp="2026-07-16",
        signal_reference_price=330.0,
        strategy_version="HYDRA_PROSPECTIVE_V2.3",
        manifest_hash=active_hash
    )

    mock_ledger.record_prospective_signal(
        symbol="AAPL",
        source_candle_timestamp="2026-07-16",
        signal_generation_timestamp="2026-07-16T16:00:00Z",
        signal="SELL",
        probability=0.9,
        confidence=0.9,
        feature_hash="feat_invalid",
        model_hash="model_invalid",
        execution_target_timestamp="2026-07-17",
        signal_reference_price=340.0,
        strategy_version="HYDRA_PROSPECTIVE_V2.3",
        manifest_hash="bad_hash"
    )

    # 1. Mock get_prediction to return a response with both valid and invalid markers
    async def mock_get_prediction(*args, **kwargs):
        return {
            "ticker": "AAPL",
            "current_price": 330.0,
            "signal": "BUY",
            "confidence_score": 0.9,
            "uncertainty_score": 0.1,
            "market_regime": "BULL",
            "volatility_state": "LOW",
            "volume_ratio": 1.0,
            "models": {"ensemble": {"signal": "BUY", "probability": 0.6}},
            "projections": {"confidence": 0.9, "reliability": "HIGH", "next_session": 331.0},
            "historical_markers": [
                {
                    "manifest_hash": active_hash,
                    "action": "BUY"
                },
                {
                    "manifest_hash": "bad_hash",
                    "action": "SELL"
                }
            ]
        }

    monkeypatch.setattr(inference_service, "get_prediction", mock_get_prediction)

    # 2. Mock api_cache
    class MockCache:
        def __init__(self):
            self.store = {}
        async def get(self, key):
            return self.store.get(key)
        async def set(self, key, value):
            self.store[key] = value

    mock_cache = MockCache()
    monkeypatch.setattr("api.api_cache", mock_cache)

    # 3. Mock paper_engine
    class MockPaperEngine:
        def get_portfolio_summary(self, _):
            return {"cash": 1000.0, "equity": 1000.0, "return_pct": 0.0, "positions": {}}
    monkeypatch.setattr("api.paper_engine", MockPaperEngine())

    # 4. Mock verify_api_key
    from api import verify_api_key
    app.dependency_overrides[verify_api_key] = lambda: "test-api-key"

    client = TestClient(app)

    # 5. Cache miss (get_prediction should be called)
    res_miss = client.get("/predict?ticker=AAPL")
    assert res_miss.status_code == 200
    data_miss = res_miss.json()

    markers_miss = data_miss.get("historical_markers", [])
    assert len(markers_miss) == 1
    assert markers_miss[0]["manifest_hash"] == active_hash

    # 6. Cache hit
    # Let's forcefully put an unfiltered response in the cache to verify the cache hit filters it
    mock_cache.store["predict_AAPL"] = {
        "ticker": "AAPL",
        "current_price": 330.0,
        "signal": "BUY",
        "confidence_score": 0.9,
        "uncertainty_score": 0.1,
        "market_regime": "BULL",
        "volatility_state": "LOW",
        "volume_ratio": 1.0,
        "models": {"ensemble": {"signal": "BUY", "probability": 0.6}},
        "projections": {"confidence": 0.9, "reliability": "HIGH", "next_session": 331.0},
        "historical_markers": [
            {"manifest_hash": active_hash, "action": "BUY"},
            {"manifest_hash": "stale_hash", "action": "SELL"}
        ]
    }

    res_hit = client.get("/predict?ticker=AAPL")
    assert res_hit.status_code == 200
    data_hit = res_hit.json()

    markers_hit = data_hit.get("historical_markers", [])
    assert len(markers_hit) == 1
    assert markers_hit[0]["manifest_hash"] == active_hash


def test_api_predict_corrupted_hash_rejection(mock_ledger, monkeypatch):
    """Test that /predict explicitly rejects corrupted observations despite valid manifest_hash."""
    from fastapi.testclient import TestClient

    from api import app, inference_service

    inference_service.signal_ledger = mock_ledger

    manifest_path = Path(__file__).resolve().parent.parent / "artifacts" / "frozen_strategy_manifest_v2.3.json"
    active_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    # Instantiate ProspectiveValidationManager to create and manage the authoritative table
    from scripts.ops.run_prospective_validation import BACKEND_DIR, ProspectiveValidationManager
    manager = ProspectiveValidationManager(
        backend_dir=BACKEND_DIR,
        db_path=mock_ledger.db_path,
        reports_dir=Path("/tmp")
    )

    # Insert legacy record to satisfy reconciliation
    sig_id = mock_ledger.record_prospective_signal(
        symbol="AAPL",
        source_candle_timestamp="2026-07-20",
        signal_generation_timestamp="2026-07-20T16:00:00Z",
        signal="BUY",
        probability=0.9,
        confidence=0.9,
        feature_hash="feat_valid",
        model_hash="model_valid",
        execution_target_timestamp="2026-07-21",
        signal_reference_price=330.0,
        strategy_version="HYDRA_PROSPECTIVE_V2.3",
        manifest_hash=active_hash
    )

    # Insert authoritative record with deliberately corrupted observation_hash
    with manager._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO prospective_observations (
                signal_id, strategy_version, source_asset, source_candle_date, signal_date,
                candle_finalization_timestamp, data_ingestion_timestamp, feature_computation_timestamp,
                signal_generation_timestamp, order_submission_timestamp,
                model_prediction_probabilities, individual_model_predictions, primary_model_prediction,
                veto_result, macro_filter_result, final_trading_decision, signal_reference_price,
                execution_status, manifest_hash, canonical_signal_hash, prev_observation_hash,
                observation_hash, market_data_vendor, source_timestamp, feature_computation_start,
                feature_computation_end, inference_start, inference_end, signal_finalization_timestamp,
                veto_audit_json, unambiguous_model_outputs, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sig_id, "HYDRA_PROSPECTIVE_V2.3", "AAPL", "2026-07-20", "2026-07-20",
                "2026-07-20T16:00:00Z", "2026-07-20T16:00:01Z", "2026-07-20T16:00:02Z",
                "2026-07-20T16:00:05Z", "2026-07-20T16:00:06Z",
                "{}", "{}", "0.9", "PASS", "PASS", "BUY", 330.0,
                "SIMULATED_FILLED", active_hash, "valid_canonical_hash", "NONE",
                "corrupted_hash", "yfinance", "NOT_PROVIDED", "start", "end", "start", "end", "final",
                "{}", "{}", "2026-07-20T16:00:10Z"
            )
        )
        conn.commit()

    # Mock get_prediction to return the corrupted marker
    async def mock_get_prediction(*args, **kwargs):
        return {
            "ticker": "AAPL",
            "historical_markers": [
                {
                    "manifest_hash": active_hash,
                    "action": "BUY"
                }
            ]
        }
    monkeypatch.setattr(inference_service, "get_prediction", mock_get_prediction)

    class MockCache:
        async def get(self, key):
            return None
        async def set(self, key, value):
            pass
    monkeypatch.setattr("api.api_cache", MockCache())

    class MockPaperEngine:
        def get_portfolio_summary(self, _):
            return {"cash": 1000.0, "equity": 1000.0, "return_pct": 0.0, "positions": {}}
    monkeypatch.setattr("api.paper_engine", MockPaperEngine())

    from api import verify_api_key
    app.dependency_overrides[verify_api_key] = lambda: "test-api-key"

    client = TestClient(app)

    # Calling predict will hit _enforce_dashboard_provenance, which now runs verify_hash_chain
    # The corrupted row will cause verified=False, raising RuntimeError
    with pytest.raises(RuntimeError, match="Cannot verify Stage 3 provenance"):
        client.get("/predict?ticker=AAPL")


