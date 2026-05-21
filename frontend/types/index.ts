export interface UniverseStock {
  ticker: string;
  name: string;
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
  price: number;
  candles: { time: string; open: number; high: number; low: number; close: number }[];
  clouds: { time: string; ribbon_upper: number; ribbon_lower: number; bb_upper: number; bb_lower: number }[];
  ai_report: AIReport;
  historical_markers: { time: string; action: string; probability: number; label?: string }[];
  portfolio?: Portfolio;
}
