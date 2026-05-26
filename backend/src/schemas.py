from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class ModelPrediction(BaseModel):
    signal: str
    probability: float


class TechnicalSnapshot(BaseModel):
    RSI: float
    MACD: float
    ATR: float
    BB_Position: float
    ADX: float
    Volume_Ratio: float


class Position(BaseModel):
    shares: int
    avg_price: float
    currency: str = "USD"
    market: str = "USA"


class FXRate(BaseModel):
    pair: str
    rate: float
    timestamp: str


class PortfolioSummary(BaseModel):
    cash: float
    equity: float
    return_pct: float
    base_currency: str = "USD"
    today_pnl: float = 0.0
    mtd_pnl: float = 0.0
    ytd_pnl: float = 0.0
    inception_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    positions: Dict[str, Position]
    fx_rates: Dict[str, float] = {}

    # Audit Fields
    initial_capital: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0


class XAIDriver(BaseModel):
    feature: str
    impact: float
    direction: str


class XAIBlock(BaseModel):
    top_drivers: List[XAIDriver]
    explanation: str


class Projections(BaseModel):
    floor: float
    median: float
    ceiling: float


class RiskMetrics(BaseModel):
    var_95: float
    cvar: float
    beta: float
    kelly_fraction: float
    target_size: float
    max_drawdown: float

    # Transparency Fields
    win_probability: float = 0.0
    expected_value: float = 0.0
    risk_reward_ratio: float = 0.0
    peak_equity: float = 0.0
    peak_date: str = ""
    trough_equity: float = 0.0
    trough_date: str = ""


class PredictResponse(BaseModel):
    ticker: str
    current_price: float
    signal: str
    confidence_score: float
    uncertainty_score: float
    signal_note: Optional[str] = None
    market_regime: str
    volatility_state: str
    volume_ratio: float
    is_point_forecast: bool = False
    models: Dict[str, ModelPrediction]
    projections: Projections
    qualitative_alpha: Optional[str] = None
    xai: Optional[XAIBlock] = None
    sentiment_score: Optional[float] = None
    risk: Optional[RiskMetrics] = None

    # Transparency fields
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_agreement: float = 0.0
    bullish_models: int = 0
    bearish_models: int = 0
    neutral_models: int = 0

    # Chart & Portfolio
    portfolio: Optional[PortfolioSummary] = None
    historical_markers: Optional[List[Dict[str, Any]]] = None
    candles: Optional[List[Dict[str, Any]]] = None
    clouds: Optional[List[Dict[str, Any]]] = None


class AssetMetadata(BaseModel):
    ticker: str
    market: str
    exchange: str
    currency: str
    timezone: str


class UniverseStockItem(BaseModel):
    ticker: str
    name: str
    price: float
    pct_change: float
    market: str
    metadata: Optional[AssetMetadata] = None


class UniverseResponse(BaseModel):
    universe: List[UniverseStockItem]


class BacktestSignal(BaseModel):
    date: str
    ticker: str
    signal: str
    confidence: float
    actual_return: float


class BacktestSummary(BaseModel):
    ticker: str
    period: str
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    vetoed_rate: float
    coverage: float
    best_signal: Optional[BacktestSignal] = None
    worst_signal: Optional[BacktestSignal] = None
    monthly_win_rates: List[Dict[str, Any]] = []
