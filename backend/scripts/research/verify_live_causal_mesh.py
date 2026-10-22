"""
Verification Audit: Live Signals, Zero Lookahead / Anti-Repainting, and Adaptive Mesh Parity.
Audits:
1. Live Signals: Live inference via InferenceService.get_prediction on real-time market data.
2. Zero Lookahead & Anti-Repainting: Causal execution timing (T -> T+1 Open), forming candle isolation, immutable ledger constraints.
3. Adaptive Mesh Alignment: Validates that signals generated across all 8 universe tickers (META, TSLA, MSFT, AMZN, NVDA, AAPL, GOOGL, AMD) match their designated Adaptive Mesh regimes.
"""
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agents.orchestrator import InstitutionalOrchestrator
from src.data_ingestion.alternative_data import PhysicalEdgeAnalyzer
from src.data_ingestion.nlp_processor import GeminiAnalyzer
from src.data_ingestion.supply_chain_graph import SupplyChainGraph
from src.execution.asset_intelligence import ADAPTIVE_MESH_MAP
from src.execution.fx_engine import FXEngine
from src.execution.inference_service import InferenceService
from src.execution.live_inference import check_bar_forming_status, load_config
from src.execution.paper_trading import PaperTradingEngine
from src.execution.performance_analyzer import PerformanceAnalyzer
from src.execution.reporting import ReportGenerator
from src.execution.signal_journal import SignalJournal
from src.execution.smart_router import PredictiveSmartRouter
from src.models.model_loader import ModelManager


async def run_full_verification():
    print("=" * 90)
    print("  INSTITUTIONAL AUDIT: LIVE SIGNALS, ZERO LOOKAHEAD/REPAINTING & ADAPTIVE MESH")
    print("=" * 90)

    # ---------------------------------------------------------
    # 1. Initialize Components
    # ---------------------------------------------------------
    backend_dir = Path(__file__).resolve().parent.parent.parent
    config = load_config()
    kf_path = backend_dir / "configs" / "kept_features.json"
    with open(kf_path) as f:
        kept_features = json.load(f)

    config["data"]["num_features"] = len(kept_features)
    mm = ModelManager(config, kept_features)
    mm.load_all_models()

    fx = FXEngine()
    paper = PaperTradingEngine(fx_engine=fx)
    report_gen = ReportGenerator(kept_features)
    perf = PerformanceAnalyzer()
    journal = SignalJournal()
    gemini = GeminiAnalyzer()
    edge = PhysicalEdgeAnalyzer()
    graph = SupplyChainGraph()
    orch = InstitutionalOrchestrator()
    router = PredictiveSmartRouter()

    service = InferenceService(
        mm, gemini, edge, graph, orch, router, report_gen, paper, perf, journal
    )

    # ---------------------------------------------------------
    # 2. Audit Adaptive Mesh Routing Definitions
    # ---------------------------------------------------------
    print("\n[AUDIT 1] Adaptive Mesh Matrix Definition:")
    expected_mesh = {
        "META": "oversold_rebound",
        "TSLA": "oversold_rebound",
        "MSFT": "oversold_rebound",
        "AMZN": "oversold_rebound",
        "NVDA": "oversold_rebound",
        "AAPL": "core_tactical",
        "GOOGL": "core_tactical",
        "AMD": "core_tactical",
    }
    for sym, mode in expected_mesh.items():
        actual_mode = ADAPTIVE_MESH_MAP.get(sym)
        status = "PASSED" if actual_mode == mode else "FAILED"
        print(f"   Ticker: {sym:<6} | Expected Mode: {mode:<16} | Configured: {actual_mode:<16} | {status}")
        assert actual_mode == mode, f"Mismatch in ADAPTIVE_MESH_MAP for {sym}"

    # ---------------------------------------------------------
    # 3. Audit Zero Lookahead & Anti-Repainting Timing Invariants
    # ---------------------------------------------------------
    print("\n[AUDIT 2] Zero Lookahead & Anti-Repainting Architecture Verification:")

    # Invariant A: Forming candle isolation
    dummy_past = pd.DataFrame(
        {"Close": [100.0, 101.0], "Open": [99.0, 100.0], "High": [102.0, 102.0], "Low": [98.0, 99.0]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )
    is_forming, bar_state = check_bar_forming_status(dummy_past)
    print(f"   Invariant 2A (Past Bar Finality): is_forming={is_forming}, status={bar_state} -> {'PASSED' if not is_forming else 'FAILED'}")
    assert not is_forming

    # Invariant B: Ledger Schema Immutability
    db_path = backend_dir / "artifacts" / "signal_ledger.db"
    if db_path.exists():
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='signal_ledger';")
            tbl_sql = cursor.fetchone()
            has_unique = tbl_sql and "UNIQUE(symbol, bar_timestamp)" in tbl_sql[0]
            print(f"   Invariant 2B (Signal Ledger Table): UNIQUE(symbol, bar_timestamp) constraint -> {'PASSED' if has_unique else 'FAILED'}")
            assert has_unique

            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='prospective_signals';")
            pros_sql = cursor.fetchone()
            has_pros_unique = pros_sql and "UNIQUE(symbol, source_candle_timestamp, strategy_version)" in pros_sql[0]
            print(f"   Invariant 2C (Prospective Signals Table): UNIQUE(symbol, source_candle_timestamp, strategy_version) -> {'PASSED' if has_pros_unique else 'FAILED'}")
            assert has_pros_unique

    # ---------------------------------------------------------
    # 4. Audit Live Signal Generation (Real-Time Ingestion)
    # ---------------------------------------------------------
    print("\n[AUDIT 3] Live Signal Generation via get_prediction across Representative Tickers:")
    test_tickers = ["META", "TSLA", "MSFT", "AAPL", "NVDA"]

    for sym in test_tickers:
        metadata = {
            "ticker": sym,
            "market": "US_EQUITY",
            "exchange": "NASDAQ" if sym in ["META", "AAPL", "MSFT", "NVDA"] else "NASDAQ",
            "currency": "USD",
            "timezone": "America/New_York",
        }
        res = await service.get_prediction(sym, config, metadata)
        sub_mode = ADAPTIVE_MESH_MAP.get(sym)
        expected_strat = "OVERSOLD_REBOUND" if sub_mode == "oversold_rebound" else "CORE_TACTICAL_ALPHA"

        print(f"\n   [{sym}] Live Inference Telemetry:")
        print(f"      Signal:             {res.get('signal')}")
        print(f"      Current Price:      ${res.get('current_price'):.2f}")
        print(f"      Strategy Mode:      {res.get('strategy_mode')} (Expected: {expected_strat})")
        print(f"      Tactical Action:    {res.get('tactical_action')}")
        print(f"      Target Exposure:    {res.get('target_exposure'):.2f}x")
        print(f"      Calibrated Conf:    {res.get('confidence_score')}%")
        print(f"      Tactical Rationale: {res.get('tactical_rationale')[:85]}...")

        assert res.get("strategy_mode") == expected_strat, f"Strategy mode mismatch for {sym}: {res.get('strategy_mode')} != {expected_strat}"
        assert res.get("target_exposure") <= 1.00, f"Leverage violation for {sym}: target_exposure {res.get('target_exposure')} > 1.00x"
        assert res.get("current_price") > 0, f"Invalid live price for {sym}"
        assert res.get("signal") in ["BUY", "SELL", "HOLD", "VETOED"], f"Invalid signal: {res.get('signal')}"

    # ---------------------------------------------------------
    # 5. Audit Historical Causal Markers: Timing & Execution Parity
    # ---------------------------------------------------------
    print("\n[AUDIT 4] Historical Causal Replay & Next-Session Execution Timing Audit:")
    spy_df = yf.download("SPY", period="1y", interval="1d", progress=False)
    if hasattr(spy_df.columns, "droplevel"):
        try:
            spy_df.columns = spy_df.columns.droplevel(1)
        except Exception:
            pass

    for sym in ["META", "TSLA", "MSFT", "AAPL"]:
        df = yf.download(sym, period="1y", interval="1d", progress=False)
        if hasattr(df.columns, "droplevel"):
            try:
                df.columns = df.columns.droplevel(1)
            except Exception:
                pass

        markers = service.get_causal_chart_markers(sym, df, spy_df, force_recompute=True)
        print(f"\n   [{sym}] Verified {len(markers)} Causal Markers:")

        for i, m in enumerate(markers):
            bar_date = m.get("bar_timestamp")
            target_date = m.get("execution_target_bar")
            action = m.get("signal")
            exec_px = m.get("execution_price")

            # Check that execution_target_bar is strictly after bar_timestamp
            if target_date != "NEXT_SESSION_OPEN":
                b_dt = pd.to_datetime(bar_date)
                t_dt = pd.to_datetime(target_date)
                assert t_dt > b_dt, f"Lookahead violation! Target {target_date} is not after bar {bar_date}"

            # Check 5 bps slippage model
            if target_date != "NEXT_SESSION_OPEN" and target_date in df.index:
                actual_open = float(df.loc[target_date, "Open"])
                expected_px = round(actual_open * 1.0005 if action == "BUY" else actual_open * 0.9995, 2)
                diff = abs(exec_px - expected_px)
                assert diff <= 0.05, f"Execution price {exec_px} deviates from Open +/- 5bps {expected_px}"

            if i < 3 or i == len(markers) - 1:
                print(f"      Bar {bar_date} -> Fill: {target_date} @ ${exec_px:.2f} ({action}) | Rule: Open[t+1] +/- 5bps | Conf: {m.get('confidence'):.2f}")

    print("\n" + "=" * 90)
    print("  ALL AUDIT CHECKS PASSED: SIGNALS LIVE, ZERO LOOKAHEAD/REPAINTING, ADAPTIVE MESH VERIFIED")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(run_full_verification())
