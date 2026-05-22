export interface UniverseStock {
  ticker: string;
  name: string;
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
}

export interface Portfolio {
  cash: number;
  equity: number;
  return_pct: number;
  positions: Record<string, Position>;
}

export interface ChartData {
  ticker: string;
  current_price: number;
  price: number; // Legacy support
  signal: string;
  confidence_score: number;
  signal_note?: string | null;
  market_regime: string;
  volatility_state: string;
  volume_ratio: number;
  models: Record<string, ModelPrediction>;
  projections: {
    floor: number;
    ceiling: number;
  };
  technical_snapshot: TechnicalSnapshot;
  qualitative_alpha?: string | null;
  
  // Chart related
  candles: { time: string; open: number; high: number; low: number; close: number }[];
  clouds: { time: string; ribbon_upper: number; ribbon_lower: number; bb_upper: number; bb_lower: number }[];
  ai_report: AIReport; // Keep for legacy component compatibility
  historical_markers: { time: string; action: string; probability: number; label?: string }[];
  portfolio?: Portfolio;
}
