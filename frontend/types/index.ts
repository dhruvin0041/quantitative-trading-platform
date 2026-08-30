export interface AssetMetadata {
  ticker: string;
  name?: string;
  market: string;
  exchange: string;
  currency: string;
  timezone: string;
}

export interface UniverseStock {
  ticker: string;
  name: string;
  price: number;
  pct_change: number;
  market: string;
  metadata?: AssetMetadata;
}

export interface ModelPrediction {
  signal: string;
  probability: number;
}

export interface TechnicalSnapshot {
  RSI: number;
  MACD: number;
  ATR: number;
  BB_Position: number;
  ADX: number;
  Volume_Ratio: number;
  ATR_Regime_Ratio?: number;
}

export interface AIReport {
  Models: {
    Primary_Deep_Learning: {
      Suggested_Action: string;
      Confidence: string;
    };
    Secondary_XGBoost: {
      Suggested_Action: string;
      Confidence: string;
    };
  };
  Risk_Management: {
    Meta_Model_Status: string;
    Dynamic_10_Day_Range: {
      Low: number;
      High: number;
    };
  };
  Context: {
    Top_Headline_Processed: string;
  };
}

export interface Position {
  shares: number;
  avg_price: number;
  currency: string;
  market: string;
}

export interface Trade {
  ticker: string;
  entry_time: string;
  exit_time: string;
  entry_price?: number;
  exit_price?: number;
  realized_pnl: number;
  pnl_pct?: number;
  outcome?: 'WIN' | 'LOSS';
}

export interface BacktestSignal {
  date: string;
  ticker: string;
  signal: string;
  confidence: number;
  actual_return: number;
}

export interface BacktestSummary {
  ticker: string;
  period: string;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  vetoed_rate: number;
  coverage: number;
  total_trades: number;
  signal_coverage?: number; // Support for alternate field name
  best_signal?: BacktestSignal;
  worst_signal?: BacktestSignal;
  monthly_win_rates: { month: string; win_rate: number }[];
}

export interface Portfolio {
  cash: number;
  equity: number;
  return_pct: number;
  base_currency: string;
  today_pnl: number;
  mtd_pnl: number;
  ytd_pnl: number;
  inception_pnl: number;
  realized_pnl: number;
  unrealized_pnl: number;
  positions: Record<string, Position>;
  fx_rates: Record<string, number>;
  
  // Audit Fields
  initial_capital: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  profit_factor: number;
  recent_closed_trades?: Trade[];
}

export interface SignalQuality {
  score: number;
  grade: 'INSTITUTIONAL' | 'WATCHLIST' | 'NO_TRADE';
  explanation: string;
  layers_passed: string[];
  layers_failed: string[];
}

export interface ExpectedValueMetrics {
  ev_pct: number;
  win_prob: number;
  avg_gain_pct: number;
  avg_loss_pct: number;
}

export interface ModelWeight {
  weight: number;
  reason: string;
  recent_accuracy: number;
}

export interface ChartData {
  ticker: string;
  current_price: number;
  price: number; // Legacy support
  signal: string;
  confidence_score: number;
  uncertainty_score?: number;
  sentiment_score?: number;
  signal_note?: string | null;
  
  // Phase 3: Semantic Separation
  structural_regime: string;
  signal_bias: string;
  execution_state: string;
  execution_reasoning: string;
  execution_authority?: {
    structural_regime?: string;
  };
  
  // Institutional Interpretations
  forecast_interpretation: string;
  forecast_explanation: string;
  consensus_intelligence: string;
  decision_tree?: { node: string; status: string; detail: string }[];
  
  // Phase 10: Institutional Explainability
  signal_reasoning?: string;
  veto_reason?: string;
  timing_reason?: string;
  forecast_reason?: string;
  rr_reason?: string;
  timing_intelligence?: Record<string, unknown>;
  confidence_breakdown?: Record<string, number>;
  explainable_confidence?: number;
  trade_parameters?: Record<string, unknown>;
  
  // Governance
  governance?: {
    veto_rate: number;
    approval_rate: number;
    total_signals: number;
    governance_status: string;
    signal_starvation: boolean;
    throughput_coherence: string;
  };
  
  // Signal V2.0 Fields
  market_regime: string;
  market_regime_v2?: string;
  volatility_state: string;
  volume_ratio: number;
  is_point_forecast: boolean;
  model_agreement: number;
  bullish_models?: number;
  bearish_models?: number;
  neutral_models?: number;
  
  quality?: SignalQuality;
  calibration?: {
    brier_score: number;
    ece: number;
    reliability_diagram: Record<string, unknown>[];
  };
  expected_value?: ExpectedValueMetrics;
  multi_timeframe_consensus?: Record<string, string>;
  asset_class?: string;
  asset_context?: Record<string, unknown>;

  risk?: {
    var_95: number;
    cvar: number;
    beta: number;
    kelly_fraction: number;
    target_size: number;
    max_drawdown: number;
    
    // Phase 2: Institutional Risk Index
    institutional_risk_index: number;
    risk_regime: string;

    // Transparency Fields
    win_probability: number;
    expected_value: number;
    risk_reward_ratio: number;
    peak_equity: number;
    peak_date: string;
    trough_equity: number;
    trough_date: string;
  };
  xai?: {
    top_drivers: { feature: string; impact: number; direction: string; stability: number; confidence: number; }[];
    explanation: string;
  };
  
  // Chart related
  candles: { time: string; open: number; high: number; low: number; close: number; volume?: number }[];
  clouds: { time: string; ribbon_upper: number; ribbon_lower: number; bb_upper: number; bb_lower: number }[];
  forecast_fan?: { time: string; p10: number; p50: number; p90: number }[];
  ai_report: AIReport; // Keep for legacy component compatibility
  historical_markers: { time: string; action: string; probability: number; label?: string }[];
  portfolio: Portfolio;
  timestamp: string;
  metadata: AssetMetadata;
  models: Record<string, ModelPrediction>;
  model_weights?: Record<string, ModelWeight>;
  projections: {
    floor: number;
    median?: number;
    ceiling: number;
    confidence?: number;
    reliability?: string;
    drift?: number;
    expected_move?: number;
  };
  technical_snapshot: TechnicalSnapshot;
  qualitative_alpha?: string | null;
  qualitative_citations?: {
    source: string;
    url?: string;
    sentiment: number;
    impact: string;
    snippet: string;
  }[];
}
