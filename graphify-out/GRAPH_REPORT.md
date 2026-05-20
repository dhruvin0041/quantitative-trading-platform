# Graph Report - Stock_Indicator  (2026-05-20)

## Corpus Check
- 42 files · ~34,978 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 255 nodes · 274 edges · 60 communities detected
- Extraction: 70% EXTRACTED · 30% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.78)
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

## God Nodes (most connected - your core abstractions)
1. `get_prediction()` - 20 edges
2. `DQNAgent` - 13 edges
3. `run_backtest()` - 12 edges
4. `add_advanced_features()` - 11 edges
5. `prepare_data()` - 10 edges
6. `NewsTokenizer` - 10 edges
7. `build_fusion_model()` - 10 edges
8. `main()` - 8 edges
9. `fetch_live_data()` - 7 edges
10. `train_dqn()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `get_prediction()` --calls--> `NewsTokenizer`  [INFERRED]
  backend\api.py → backend\src\data_ingestion\nlp_processor.py
- `get_prediction()` --calls--> `add_advanced_features()`  [INFERRED]
  backend\api.py → backend\src\data_ingestion\technical_indicators.py
- `run_backtest()` --calls--> `build_fusion_model()`  [INFERRED]
  backend\backtester.py → backend\src\models\fusion_network.py
- `run_backtest()` --calls--> `fetch_historical_data()`  [INFERRED]
  backend\backtester.py → backend\src\data_ingestion\market_data.py
- `run_backtest()` --calls--> `add_advanced_features()`  [INFERRED]
  backend\backtester.py → backend\src\data_ingestion\technical_indicators.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.13
Nodes (18): get_prediction(), fetch_live_data(), fetch_live_news(), load_config(), main(), fetch_historical_data(), Fetches real historical daily data from Yahoo Finance., Detects swing highs and lows over a 5-year period to provide context. (+10 more)

### Community 1 - "Community 1"
Cohesion: 0.14
Nodes (11): Implements institutional Walk-Forward Analysis.     Splits the last 2 years int, Implements institutional Walk-Forward Analysis.     Splits the last 2 years int, Implements institutional Walk-Forward Analysis.     Splits the last 2 years int, run_backtest(), run_walk_forward(), DQNAgent, DQNetwork, NewsTokenizer (+3 more)

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (13): apply_dynamic_triple_barrier(), get_sector_peer(), Dynamically identifies a high-correlation peer in the same sector., PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier, objective(), objective(), create_time_series_sequences(), Converts a 2D Pandas DataFrame into a 3D NumPy array for LSTM input. (+5 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (9): generate_daily_signals(), Formats data for the Next.js institutional dashboard., Handles generation of historical markers, chart data, and AI reports     to dec, ReportGenerator, add_advanced_features(), clean_multiindex_columns(), Adds advanced, trend-strength, volatility-aware, and macro features., Standardizes yfinance MultiIndex columns to a single level. (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.21
Nodes (8): AlphaAgent, ExecutionAgent, InstitutionalOrchestrator, Final Arbiter: Validates signals against VaR and crowding metrics.     Has Veto, Optimizes for best fill and predictive liquidity., SOTA 2026 Agentic Mesh Orchestrator.     Coordinates the collaborative intellig, Maximizes expected return using hybrid ensemble signals., RiskAgent

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (8): build_cnn_branch(), CNN branch for pattern recognition in the recent price/indicator matrix.     In, build_finbert_branch(), get_finbert(), build_fusion_model(), build_lstm_branch(), build_transformer_branch(), Transformer branch for capturing complex long-range dependencies in time-series

### Community 6 - "Community 6"
Cohesion: 0.15
Nodes (11): Backend (FastAPI), CNN, DQN, FinBERT, Frontend (Next.js), LSTM, 5-Model Hybrid Ensemble, Optuna (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.18
Nodes (6): rel_type can be 'supplier' or 'customer'.         Direction: supplier -> custom, Builds a 2-tier graph around a ticker using Yahoo Finance data.         In a fu, Institutional Metric: Centrality-based risk.         If nodes connected to our, Builds an institutional N-tier dependency graph using open-source proxies., Returns all dependencies up to N-tiers deep., SupplyChainGraph

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (5): PhysicalEdgeAnalyzer, Fetches current weather data for major ports to estimate supply chain delay risk, Uses search intensity/volume logic to proxy 'Retail Foot Traffic'.         For, Returns a dictionary of physical/macro alpha features., Simulates institutional geospatial intelligence using free public proxies.

### Community 9 - "Community 9"
Cohesion: 0.27
Nodes (4): MarketTimeGAN, Runs the DQN agent through 10,000 synthetic paths to find Max Drawdown., Runs the DQN agent through 10,000 synthetic paths to find Max Drawdown., SOTA 2026 Generative Adversarial Network for synthetic market simulation.     C

### Community 10 - "Community 10"
Cohesion: 0.31
Nodes (7): train_macro_regime_model(), build_panel_dataset(), get_sp500_tickers(), load_universal_params(), Loads the universal AI-discovered parameters., Scrapes the S&P 500 and returns a subset., train_universal_engine()

### Community 11 - "Community 11"
Cohesion: 0.25
Nodes (5): calculate_micro_imbalance(), PredictiveSmartRouter, SOTA 2026 Smart Order Router.     Predicts optimal venue liquidity using volume, Deterministic execution logic simulation., Simulates institutional LOB imbalance calculation at hardware speeds (Numba).

### Community 12 - "Community 12"
Cohesion: 0.38
Nodes (6): clean_cache(), clean_optimization_artifacts(), clean_training_artifacts(), Deletes models and scalers but KEEPS optimized parameters., Deletes models and specific Optuna databases/configs to force a fresh search., Clears python cache folders recursively.

### Community 13 - "Community 13"
Cohesion: 0.4
Nodes (2): GeminiAnalyzer, Uses Gemini to perform qualitative alpha extraction.         Specifically looks

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
Nodes (1): Handles generation of historical markers, chart data, and AI reports     to dec

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Detects swing highs and lows over a 5-year period to provide context.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Formats data for the Next.js institutional dashboard.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Deletes models and scalers but KEEPS optimized parameters.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Deletes models and specific Optuna databases/configs to force a fresh search.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Scrapes the S&P 500 and returns a subset.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Maximizes expected return using hybrid ensemble signals.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Final Arbiter: Validates signals against VaR and crowding metrics.     Has Veto

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Optimizes for best fill and predictive liquidity.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): SOTA 2026 Agentic Mesh Orchestrator.     Coordinates the collaborative intellig

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Simulates institutional geospatial intelligence using free public proxies.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Uses search intensity/volume logic to proxy 'Retail Foot Traffic'.         For

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Returns a dictionary of physical/macro alpha features.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Fetches real historical daily data from Yahoo Finance.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Converts news + SEC filings into BERT input IDs and Attention Masks.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Builds an institutional N-tier dependency graph using open-source proxies.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Institutional Metric: Centrality-based risk.         If nodes connected to our

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Returns all dependencies up to N-tiers deep.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Standardizes yfinance MultiIndex columns to a single level.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Adds advanced, trend-strength, volatility-aware, and macro features.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): PATH 4: Drops highly correlated features to prevent multicollinearity (network c

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Calculates the Beta of a ticker relative to SPY.     Beta > 1: More volatile th

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Elite Metric: Detects 'Crowding' or 'Stampede Risk'.     If retail excitement i

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Determines the suggested capital allocation using a blend of confidence and Kell

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Parses raw model outputs into the specified JSON schema.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Simulates institutional LOB imbalance calculation at hardware speeds (Numba).

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): SOTA 2026 Smart Order Router.     Predicts optimal venue liquidity using volume

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Deterministic execution logic simulation.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): CNN branch for pattern recognition in the recent price/indicator matrix.     In

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): SOTA 2026 Generative Adversarial Network for synthetic market simulation.     C

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Generates 'What-If' scenarios by adding noise to the latent space.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Transformer branch for capturing complex long-range dependencies in time-series

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Determines the suggested capital allocation using a blend of confidence and Kell

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Converts news + SEC filings into BERT input IDs and Attention Masks.

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Determines the suggested capital allocation using a blend of confidence and Kell

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): Converts a string of daily news into BERT input IDs and Attention Masks.

## Knowledge Gaps
- **88 isolated node(s):** `Clears python cache folders recursively.`, `Deletes models and scalers but KEEPS optimized parameters.`, `Deletes models and specific Optuna databases/configs to force a fresh search.`, `Loads the universal AI-discovered parameters.`, `Scrapes the S&P 500 and returns a subset.` (+83 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 13`** (5 nodes): `nlp_processor.py`, `GeminiAnalyzer`, `.analyze_fundamental_alpha()`, `.__init__()`, `Uses Gemini to perform qualitative alpha extraction.         Specifically looks`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (5 nodes): `signal_generator.py`, `Parses raw model outputs into the specified JSON schema.`, `SignalFormatter`, `.format_output()`, `.__init__()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (2 nodes): `README.md`, `Hydra Terminal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Handles generation of historical markers, chart data, and AI reports     to dec`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Detects swing highs and lows over a 5-year period to provide context.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Formats data for the Next.js institutional dashboard.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Deletes models and scalers but KEEPS optimized parameters.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Deletes models and specific Optuna databases/configs to force a fresh search.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Scrapes the S&P 500 and returns a subset.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Maximizes expected return using hybrid ensemble signals.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Final Arbiter: Validates signals against VaR and crowding metrics.     Has Veto`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Optimizes for best fill and predictive liquidity.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `SOTA 2026 Agentic Mesh Orchestrator.     Coordinates the collaborative intellig`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Simulates institutional geospatial intelligence using free public proxies.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Uses search intensity/volume logic to proxy 'Retail Foot Traffic'.         For`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Returns a dictionary of physical/macro alpha features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Fetches real historical daily data from Yahoo Finance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `PATH 1: Dynamic Barriers based on current volatility (ATR).     - Upper Barrier`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Converts news + SEC filings into BERT input IDs and Attention Masks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Builds an institutional N-tier dependency graph using open-source proxies.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Institutional Metric: Centrality-based risk.         If nodes connected to our`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Returns all dependencies up to N-tiers deep.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Standardizes yfinance MultiIndex columns to a single level.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Adds advanced, trend-strength, volatility-aware, and macro features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `PATH 4: Drops highly correlated features to prevent multicollinearity (network c`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Calculates the Beta of a ticker relative to SPY.     Beta > 1: More volatile th`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Elite Metric: Detects 'Crowding' or 'Stampede Risk'.     If retail excitement i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Determines the suggested capital allocation using a blend of confidence and Kell`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Parses raw model outputs into the specified JSON schema.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Simulates institutional LOB imbalance calculation at hardware speeds (Numba).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `SOTA 2026 Smart Order Router.     Predicts optimal venue liquidity using volume`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Deterministic execution logic simulation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `CNN branch for pattern recognition in the recent price/indicator matrix.     In`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `SOTA 2026 Generative Adversarial Network for synthetic market simulation.     C`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Generates 'What-If' scenarios by adding noise to the latent space.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Transformer branch for capturing complex long-range dependencies in time-series`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Determines the suggested capital allocation using a blend of confidence and Kell`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Converts news + SEC filings into BERT input IDs and Attention Masks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Calculates the Full Kelly Criterion for optimal position sizing.     Formula: K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Determines the suggested capital allocation using a blend of confidence and Kell`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `Converts a string of daily news into BERT input IDs and Attention Masks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_prediction()` connect `Community 0` to `Community 1`, `Community 3`, `Community 4`, `Community 7`, `Community 8`, `Community 11`, `Community 13`?**
  _High betweenness centrality (0.223) - this node is a cross-community bridge._
- **Why does `build_fusion_model()` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `run_backtest()` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 5`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `get_prediction()` (e.g. with `fetch_live_data()` and `calculate_beta()`) actually correct?**
  _`get_prediction()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `DQNAgent` (e.g. with `Implements institutional Walk-Forward Analysis.     Splits the last 2 years int` and `run_backtest()`) actually correct?**
  _`DQNAgent` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `run_backtest()` (e.g. with `.load()` and `build_fusion_model()`) actually correct?**
  _`run_backtest()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `add_advanced_features()` (e.g. with `generate_daily_signals()` and `run_backtest()`) actually correct?**
  _`add_advanced_features()` has 8 INFERRED edges - model-reasoned connections that need verification._