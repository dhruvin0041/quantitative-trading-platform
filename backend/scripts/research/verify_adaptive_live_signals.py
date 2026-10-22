import json
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

import yfinance as yf

from src.agents.orchestrator import InstitutionalOrchestrator
from src.data_ingestion.alternative_data import PhysicalEdgeAnalyzer
from src.data_ingestion.nlp_processor import GeminiAnalyzer
from src.data_ingestion.supply_chain_graph import SupplyChainGraph
from src.execution.fx_engine import FXEngine
from src.execution.inference_service import InferenceService
from src.execution.live_inference import load_config
from src.execution.paper_trading import PaperTradingEngine
from src.execution.performance_analyzer import PerformanceAnalyzer
from src.execution.reporting import ReportGenerator
from src.execution.signal_journal import SignalJournal
from src.execution.smart_router import PredictiveSmartRouter
from src.models.model_loader import ModelManager

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

spy_df = yf.download("SPY", period="1y", interval="1d", progress=False)
if hasattr(spy_df.columns, "droplevel"):
    try:
        spy_df.columns = spy_df.columns.droplevel(1)
    except Exception:
        pass

print("=" * 80)
print("  TESTING LIVE INFERENCE SERVICE REPLAY WITH ADAPTIVE MESH")
print("=" * 80)

for sym in ["META", "TSLA", "MSFT", "AAPL"]:
    df = yf.download(sym, period="1y", interval="1d", progress=False)
    if hasattr(df.columns, "droplevel"):
        try:
            df.columns = df.columns.droplevel(1)
        except Exception:
            pass

    markers = service.get_causal_chart_markers(sym, df, spy_df, force_recompute=True)
    print(f"\n[{sym}] Generated {len(markers)} Causal Markers under Adaptive Mesh:")
    for m in markers[:6]:
        print(f"   Bar: {m.get('bar_timestamp')} | Action: {m.get('signal'):<4} | Target: {m.get('execution_target_bar')} @ ${m.get('execution_price'):.2f} | Conf: {m.get('confidence'):.2f}")
    if len(markers) > 6:
        print(f"   ... and {len(markers) - 6} more markers")
