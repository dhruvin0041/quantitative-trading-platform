# Graph Report - Stock_Indicator  (2026-05-13)

## Corpus Check
- 40 files · ~25,003 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 207 nodes · 258 edges · 23 communities detected
- Extraction: 72% EXTRACTED · 28% INFERRED · 0% AMBIGUOUS · INFERRED: 71 edges (avg confidence: 0.79)
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
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]

## God Nodes (most connected - your core abstractions)
1. `get_prediction()` - 18 edges
2. `run_backtest()` - 12 edges
3. `DQNAgent` - 11 edges
4. `prepare_data()` - 10 edges
5. `build_fusion_model()` - 10 edges
6. `add_advanced_features()` - 9 edges
7. `main()` - 8 edges
8. `NewsTokenizer` - 8 edges
9. `fetch_live_data()` - 7 edges
10. `train_dqn()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `get_prediction()` --calls--> `NewsTokenizer`  [INFERRED]
  backend\api.py → backend\src\data_ingestion\nlp_processor.py
- `get_prediction()` --calls--> `add_advanced_features()`  [INFERRED]
  backend\api.py → backend\src\data_ingestion\technical_indicators.py
- `run_backtest()` --calls--> `build_fusion_model()`  [INFERRED]
  backend\backtester.py → backend\src\models\fusion_network.py
- `run_backtest()` --calls--> `DQNAgent`  [INFERRED]
  backend\backtester.py → backend\src\models\dqn_agent.py
- `run_backtest()` --calls--> `fetch_historical_data()`  [INFERRED]
  backend\backtester.py → backend\src\data_ingestion\market_data.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.14
Nodes (17): get_prediction(), fetch_live_data(), fetch_live_news(), load_config(), main(), fetch_historical_data(), Fetches real historical daily data from Yahoo Finance., calculate_beta() (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (14): apply_dynamic_triple_barrier(), get_sector_peer(), PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier, PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier, objective(), objective(), create_time_series_sequences(), Converts a 2D Pandas DataFrame into a 3D NumPy array for LSTM input. (+6 more)

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (8): Implements institutional Walk-Forward Analysis.     Splits the last 2 years int, run_backtest(), run_walk_forward(), GeminiAnalyzer, NewsTokenizer, Uses Gemini to perform qualitative alpha extraction.         Specifically looks, Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed., Converts news + SEC filings into BERT input IDs and Attention Masks.

### Community 3 - "Community 3"
Cohesion: 0.21
Nodes (8): AlphaAgent, ExecutionAgent, InstitutionalOrchestrator, Final Arbiter: Validates signals against VaR and crowding metrics.     Has Veto, Optimizes for best fill and predictive liquidity., SOTA 2026 Agentic Mesh Orchestrator.     Coordinates the collaborative intellig, Maximizes expected return using hybrid ensemble signals., RiskAgent

### Community 4 - "Community 4"
Cohesion: 0.17
Nodes (8): build_cnn_branch(), CNN branch for pattern recognition in the recent price/indicator matrix.     In, build_finbert_branch(), get_finbert(), build_fusion_model(), build_lstm_branch(), build_transformer_branch(), Transformer branch for capturing complex long-range dependencies in time-series

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (11): Backend (FastAPI), CNN, DQN, FinBERT, Frontend (Next.js), LSTM, 5-Model Hybrid Ensemble, Optuna (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.18
Nodes (6): rel_type can be 'supplier' or 'customer'.         Direction: supplier -> custom, Builds a 2-tier graph around a ticker using Yahoo Finance data.         In a fu, Institutional Metric: Centrality-based risk.         If nodes connected to our, Returns all dependencies up to N-tiers deep., Builds an institutional N-tier dependency graph using open-source proxies., SupplyChainGraph

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (3): DQNAgent, DQNetwork, train_dqn()

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (5): PhysicalEdgeAnalyzer, Fetches current weather data for major ports to estimate supply chain delay risk, Uses search intensity/volume logic to proxy 'Retail Foot Traffic'.         For, Returns a dictionary of physical/macro alpha features., Simulates institutional geospatial intelligence using free public proxies.

### Community 9 - "Community 9"
Cohesion: 0.22
Nodes (5): generate_daily_signals(), add_advanced_features(), clean_multiindex_columns(), Adds advanced, trend-strength, volatility-aware, and macro features., Standardizes yfinance MultiIndex columns to a single level.

### Community 10 - "Community 10"
Cohesion: 0.27
Nodes (4): MarketTimeGAN, Generates 'What-If' scenarios by adding noise to the latent space., Runs the DQN agent through 10,000 synthetic paths to find Max Drawdown., SOTA 2026 Generative Adversarial Network for synthetic market simulation.     C

### Community 11 - "Community 11"
Cohesion: 0.31
Nodes (7): train_macro_regime_model(), build_panel_dataset(), get_sp500_tickers(), load_universal_params(), Loads the universal AI-discovered parameters., Scrapes the S&P 500 and returns a subset., train_universal_engine()

### Community 12 - "Community 12"
Cohesion: 0.25
Nodes (5): calculate_micro_imbalance(), PredictiveSmartRouter, SOTA 2026 Smart Order Router.     Predicts optimal venue liquidity using volume, Deterministic execution logic simulation., Simulates institutional LOB imbalance calculation at hardware speeds (Numba).

### Community 13 - "Community 13"
Cohesion: 0.38
Nodes (6): clean_cache(), clean_optimization_artifacts(), clean_training_artifacts(), Deletes models and scalers but KEEPS optimized parameters., Deletes models and specific Optuna databases/configs to force a fresh search., Clears python cache folders recursively.

### Community 14 - "Community 14"
Cohesion: 0.4
Nodes (2): Parses raw model outputs into the specified JSON schema., SignalFormatter

### Community 16 - "Community 16"
Cohesion: 0.67
Nodes (3): GMM Regime Detection, Macro Kill-Switch, VIX

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Hydra Terminal

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Determines the suggested capital allocation using a blend of confidence and Kell

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Converts news + SEC filings into BERT input IDs and Attention Masks.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Determines the suggested capital allocation using a blend of confidence and Kell

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Converts a string of daily news into BERT input IDs and Attention Masks.

## Knowledge Gaps
- **48 isolated node(s):** `Clears python cache folders recursively.`, `Deletes models and scalers but KEEPS optimized parameters.`, `Deletes models and specific Optuna databases/configs to force a fresh search.`, `Loads the universal AI-discovered parameters.`, `Scrapes the S&P 500 and returns a subset.` (+43 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 14`** (5 nodes): `signal_generator.py`, `Parses raw model outputs into the specified JSON schema.`, `SignalFormatter`, `.format_output()`, `.__init__()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (2 nodes): `README.md`, `Hydra Terminal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Determines the suggested capital allocation using a blend of confidence and Kell`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Converts news + SEC filings into BERT input IDs and Attention Masks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Determines the suggested capital allocation using a blend of confidence and Kell`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Converts a string of daily news into BERT input IDs and Attention Masks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_prediction()` connect `Community 0` to `Community 2`, `Community 3`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 12`?**
  _High betweenness centrality (0.296) - this node is a cross-community bridge._
- **Why does `build_fusion_model()` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `run_backtest()` connect `Community 2` to `Community 0`, `Community 1`, `Community 4`, `Community 7`, `Community 9`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `get_prediction()` (e.g. with `fetch_live_data()` and `calculate_beta()`) actually correct?**
  _`get_prediction()` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `run_backtest()` (e.g. with `.load()` and `build_fusion_model()`) actually correct?**
  _`run_backtest()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `DQNAgent` (e.g. with `Implements institutional Walk-Forward Analysis.     Splits the last 2 years int` and `run_backtest()`) actually correct?**
  _`DQNAgent` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `prepare_data()` (e.g. with `fetch_historical_data()` and `add_advanced_features()`) actually correct?**
  _`prepare_data()` has 8 INFERRED edges - model-reasoned connections that need verification._