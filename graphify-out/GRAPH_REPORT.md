# Graph Report - Stock_Indicator  (2026-05-20)

## Corpus Check
- 57 files · ~38,556 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 390 nodes · 520 edges · 72 communities detected
- Extraction: 64% EXTRACTED · 36% INFERRED · 0% AMBIGUOUS · INFERRED: 189 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]

## God Nodes (most connected - your core abstractions)
1. `get_prediction()` - 21 edges
2. `SimpleCache` - 19 edges
3. `JSONFormatter` - 17 edges
4. `Institutional Sanitization: Only allow 1-5 alphanumeric chars or hyphens.` - 16 edges
5. `DQNAgent` - 16 edges
6. `PaperTradingEngine` - 14 edges
7. `TestInstitutionalExcellence` - 14 edges
8. `NewsTokenizer` - 13 edges
9. `run_backtest()` - 12 edges
10. `AlertSystem` - 12 edges

## Surprising Connections (you probably didn't know these)
- `JSONFormatter` --uses--> `DQNAgent`  [INFERRED]
  backend\api.py → backend\src\models\dqn_agent.py
- `JSONFormatter` --uses--> `NewsTokenizer`  [INFERRED]
  backend\api.py → backend\src\data_ingestion\nlp_processor.py
- `JSONFormatter` --uses--> `PhysicalEdgeAnalyzer`  [INFERRED]
  backend\api.py → backend\src\data_ingestion\alternative_data.py
- `JSONFormatter` --uses--> `SupplyChainGraph`  [INFERRED]
  backend\api.py → backend\src\data_ingestion\supply_chain_graph.py
- `JSONFormatter` --uses--> `InstitutionalOrchestrator`  [INFERRED]
  backend\api.py → backend\src\agents\orchestrator.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (17): AlertSystem, Institutional Alerting System.     Monitors performance degradation, model drif, get_alerts(), get_performance(), JSONFormatter, Institutional Sanitization: Only allow 1-5 alphanumeric chars or hyphens., sanitize_ticker(), SimpleCache (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (23): get_prediction(), Implements institutional Walk-Forward Analysis.     Splits the last 2 years int, Implements institutional Walk-Forward Analysis.     Splits the last 2 years int, Implements institutional Walk-Forward Analysis.     Splits the last 2 years int, run_backtest(), run_walk_forward(), DQNAgent, fetch_live_data() (+15 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (10): DriftMonitor, Uses Kolmogorov-Smirnov test to detect data drift on a per-feature basis., Monitors data and model drift in production.     Alerts if live inference distr, ExperimentTracker, Lightweight, institutional-grade experiment tracking (similar to MLflow/W&B)., FactorModel, Fits PCA on a panel of asset returns to extract statistical factors.         re, Calculates the specific risk (volatility not explained by factors). (+2 more)

### Community 3 - "Community 3"
Cohesion: 0.1
Nodes (13): ModelCalibrator, Fits an isotonic regression model to calibrate probabilities.         y_prob sh, Calculates confidence intervals using ensemble variance., Calibrates model output probabilities to reflect true confidence using Isotonic, apply_dynamic_triple_barrier(), PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier, objective(), objective() (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (7): generate_performance_report(), Generates a periodic performance report in Markdown format., PaperTradingEngine, Value at Risk (Historical), Expected Shortfall (CVaR), PerformanceAnalyzer, Institutional Performance Analysis Engine.     Computes Sharpe, Sortino, Calmar

### Community 5 - "Community 5"
Cohesion: 0.14
Nodes (9): SQLiteCache, Parses raw model outputs into the specified JSON schema., SignalFormatter, build_panel_dataset(), get_sp500_tickers(), load_universal_params(), Loads the universal AI-discovered parameters., Scrapes the S&P 500 and returns a subset. (+1 more)

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (9): generate_daily_signals(), Formats data for the Next.js institutional dashboard., Handles generation of historical markers, chart data, and AI reports     to dec, ReportGenerator, add_advanced_features(), clean_multiindex_columns(), Adds advanced, trend-strength, volatility-aware, and macro features., Standardizes yfinance MultiIndex columns to a single level. (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.21
Nodes (8): AlphaAgent, ExecutionAgent, InstitutionalOrchestrator, Final Arbiter: Validates signals against VaR and crowding metrics.     Has Veto, Optimizes for best fill and predictive liquidity., SOTA 2026 Agentic Mesh Orchestrator.     Coordinates the collaborative intellig, Maximizes expected return using hybrid ensemble signals., RiskAgent

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (8): build_cnn_branch(), CNN branch for pattern recognition in the recent price/indicator matrix.     In, build_finbert_branch(), get_finbert(), build_fusion_model(), build_lstm_branch(), build_transformer_branch(), Transformer branch for capturing complex long-range dependencies in time-series

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (11): Backend (FastAPI), CNN, DQN, FinBERT, Frontend (Next.js), LSTM, 5-Model Hybrid Ensemble, Optuna (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.18
Nodes (6): rel_type can be 'supplier' or 'customer'.         Direction: supplier -> custom, Builds a 2-tier graph around a ticker using Yahoo Finance data.         In a fu, Institutional Metric: Centrality-based risk.         If nodes connected to our, Builds an institutional N-tier dependency graph using open-source proxies., Returns all dependencies up to N-tiers deep., SupplyChainGraph

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (10): BaseModel, AIReport, ContextReport, ModelConfidence, ModelsReport, PortfolioSummary, Position, PredictResponse (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.2
Nodes (10): calculate_beta(), calculate_full_kelly(), calculate_jensens_alpha(), detect_stampede_risk(), get_position_sizing(), Measures 'Skill' (Alpha) by subtracting expected market returns from actual retu, Elite Metric: Detects 'Crowding' or 'Stampede Risk'.     If retail excitement i, Calculates the Beta of a ticker relative to SPY.     Beta > 1: More volatile th (+2 more)

### Community 13 - "Community 13"
Cohesion: 0.22
Nodes (5): PhysicalEdgeAnalyzer, Fetches current weather data for major ports to estimate supply chain delay risk, Uses search intensity/volume logic to proxy 'Retail Foot Traffic'.         For, Returns a dictionary of physical/macro alpha features., Simulates institutional geospatial intelligence using free public proxies.

### Community 14 - "Community 14"
Cohesion: 0.24
Nodes (4): ABC, AlpacaBroker, BrokerIntegration, Stub for Alpaca Trade API Integration.

### Community 15 - "Community 15"
Cohesion: 0.27
Nodes (4): MarketTimeGAN, Runs the DQN agent through 10,000 synthetic paths to find Max Drawdown., Runs the DQN agent through 10,000 synthetic paths to find Max Drawdown., SOTA 2026 Generative Adversarial Network for synthetic market simulation.     C

### Community 16 - "Community 16"
Cohesion: 0.28
Nodes (3): PortfolioOptimizer, Maximizes Sharpe ratio or minimizes variance for a target return., Institutional Portfolio Optimization using Mean-Variance (Markowitz)      and R

### Community 17 - "Community 17"
Cohesion: 0.38
Nodes (6): clean_cache(), clean_optimization_artifacts(), clean_training_artifacts(), Deletes models and scalers but KEEPS optimized parameters., Deletes models and specific Optuna databases/configs to force a fresh search., Clears python cache folders recursively.

### Community 18 - "Community 18"
Cohesion: 0.5
Nodes (1): DQNetwork

### Community 20 - "Community 20"
Cohesion: 0.67
Nodes (3): GMM Regime Detection, Macro Kill-Switch, VIX

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Hydra Terminal

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Optimizes for best fill and predictive liquidity.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): SOTA 2026 Agentic Mesh Orchestrator.     Coordinates the collaborative intellig

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Formats data for the Next.js institutional dashboard.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Determines the suggested capital allocation using a blend of confidence and Kell

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Simulates institutional LOB imbalance calculation at hardware speeds (Numba).

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): SOTA 2026 Smart Order Router.     Predicts optimal venue liquidity using volume

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Deterministic execution logic simulation.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Handles generation of historical markers, chart data, and AI reports     to dec

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Detects swing highs and lows over a 5-year period to provide context.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Formats data for the Next.js institutional dashboard.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Deletes models and scalers but KEEPS optimized parameters.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Deletes models and specific Optuna databases/configs to force a fresh search.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Scrapes the S&P 500 and returns a subset.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Maximizes expected return using hybrid ensemble signals.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Final Arbiter: Validates signals against VaR and crowding metrics.     Has Veto

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Optimizes for best fill and predictive liquidity.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): SOTA 2026 Agentic Mesh Orchestrator.     Coordinates the collaborative intellig

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Simulates institutional geospatial intelligence using free public proxies.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Uses search intensity/volume logic to proxy 'Retail Foot Traffic'.         For

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Returns a dictionary of physical/macro alpha features.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Fetches real historical daily data from Yahoo Finance.

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Converts news + SEC filings into BERT input IDs and Attention Masks.

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Builds an institutional N-tier dependency graph using open-source proxies.

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Institutional Metric: Centrality-based risk.         If nodes connected to our

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Returns all dependencies up to N-tiers deep.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Standardizes yfinance MultiIndex columns to a single level.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Adds advanced, trend-strength, volatility-aware, and macro features.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): PATH 4: Drops highly correlated features to prevent multicollinearity (network c

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Calculates the Beta of a ticker relative to SPY.     Beta > 1: More volatile th

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Elite Metric: Detects 'Crowding' or 'Stampede Risk'.     If retail excitement i

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Determines the suggested capital allocation using a blend of confidence and Kell

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): Parses raw model outputs into the specified JSON schema.

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): Simulates institutional LOB imbalance calculation at hardware speeds (Numba).

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): SOTA 2026 Smart Order Router.     Predicts optimal venue liquidity using volume

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Deterministic execution logic simulation.

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): CNN branch for pattern recognition in the recent price/indicator matrix.     In

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): SOTA 2026 Generative Adversarial Network for synthetic market simulation.     C

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): Generates 'What-If' scenarios by adding noise to the latent space.

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): Transformer branch for capturing complex long-range dependencies in time-series

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (1): Determines the suggested capital allocation using a blend of confidence and Kell

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Converts news + SEC filings into BERT input IDs and Attention Masks.

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (1): Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (1): Determines the suggested capital allocation using a blend of confidence and Kell

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (1): Converts a string of daily news into BERT input IDs and Attention Masks.

## Knowledge Gaps
- **113 isolated node(s):** `Clears python cache folders recursively.`, `Deletes models and scalers but KEEPS optimized parameters.`, `Deletes models and specific Optuna databases/configs to force a fresh search.`, `Loads the universal AI-discovered parameters.`, `Scrapes the S&P 500 and returns a subset.` (+108 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 18`** (5 nodes): `dqn_agent.py`, `.__init__()`, `DQNetwork`, `.forward()`, `.__init__()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (2 nodes): `README.md`, `Hydra Terminal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Optimizes for best fill and predictive liquidity.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `SOTA 2026 Agentic Mesh Orchestrator.     Coordinates the collaborative intellig`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Formats data for the Next.js institutional dashboard.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Determines the suggested capital allocation using a blend of confidence and Kell`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Simulates institutional LOB imbalance calculation at hardware speeds (Numba).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `SOTA 2026 Smart Order Router.     Predicts optimal venue liquidity using volume`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Deterministic execution logic simulation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Handles generation of historical markers, chart data, and AI reports     to dec`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Detects swing highs and lows over a 5-year period to provide context.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Formats data for the Next.js institutional dashboard.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Deletes models and scalers but KEEPS optimized parameters.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Deletes models and specific Optuna databases/configs to force a fresh search.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Scrapes the S&P 500 and returns a subset.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Maximizes expected return using hybrid ensemble signals.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Final Arbiter: Validates signals against VaR and crowding metrics.     Has Veto`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Optimizes for best fill and predictive liquidity.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `SOTA 2026 Agentic Mesh Orchestrator.     Coordinates the collaborative intellig`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Simulates institutional geospatial intelligence using free public proxies.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Uses search intensity/volume logic to proxy 'Retail Foot Traffic'.         For`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Returns a dictionary of physical/macro alpha features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Fetches real historical daily data from Yahoo Finance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Converts news + SEC filings into BERT input IDs and Attention Masks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Builds an institutional N-tier dependency graph using open-source proxies.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Institutional Metric: Centrality-based risk.         If nodes connected to our`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Returns all dependencies up to N-tiers deep.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Standardizes yfinance MultiIndex columns to a single level.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Adds advanced, trend-strength, volatility-aware, and macro features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `PATH 4: Drops highly correlated features to prevent multicollinearity (network c`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Calculates the Beta of a ticker relative to SPY.     Beta > 1: More volatile th`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Elite Metric: Detects 'Crowding' or 'Stampede Risk'.     If retail excitement i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Determines the suggested capital allocation using a blend of confidence and Kell`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `Parses raw model outputs into the specified JSON schema.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `Simulates institutional LOB imbalance calculation at hardware speeds (Numba).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `SOTA 2026 Smart Order Router.     Predicts optimal venue liquidity using volume`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Deterministic execution logic simulation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `CNN branch for pattern recognition in the recent price/indicator matrix.     In`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `SOTA 2026 Generative Adversarial Network for synthetic market simulation.     C`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `Generates 'What-If' scenarios by adding noise to the latent space.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `Transformer branch for capturing complex long-range dependencies in time-series`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `Determines the suggested capital allocation using a blend of confidence and Kell`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Converts news + SEC filings into BERT input IDs and Attention Masks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `Determines the suggested capital allocation using a blend of confidence and Kell`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `Converts a string of daily news into BERT input IDs and Attention Masks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestInstitutionalExcellence` connect `Community 2` to `Community 16`, `Community 0`, `Community 3`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Why does `SimpleCache` connect `Community 0` to `Community 1`, `Community 2`, `Community 4`, `Community 6`, `Community 7`, `Community 10`, `Community 11`, `Community 13`, `Community 14`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `JSONFormatter` connect `Community 0` to `Community 1`, `Community 2`, `Community 4`, `Community 6`, `Community 7`, `Community 10`, `Community 11`, `Community 13`, `Community 14`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `get_prediction()` (e.g. with `fetch_live_data()` and `calculate_beta()`) actually correct?**
  _`get_prediction()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `SimpleCache` (e.g. with `DQNAgent` and `NewsTokenizer`) actually correct?**
  _`SimpleCache` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `JSONFormatter` (e.g. with `DQNAgent` and `NewsTokenizer`) actually correct?**
  _`JSONFormatter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Institutional Sanitization: Only allow 1-5 alphanumeric chars or hyphens.` (e.g. with `DQNAgent` and `NewsTokenizer`) actually correct?**
  _`Institutional Sanitization: Only allow 1-5 alphanumeric chars or hyphens.` has 15 INFERRED edges - model-reasoned connections that need verification._