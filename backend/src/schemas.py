from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class ModelPrediction(BaseModel):
    signal: str
    probability: float

class Projections(BaseModel):
    floor: float
    ceiling: float

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

class PortfolioSummary(BaseModel):
    cash: float
    equity: float
    return_pct: float
    positions: Dict[str, Position]

class PredictResponse(BaseModel):
    ticker: str
    current_price: float
    signal: str
    confidence_score: float
    signal_note: Optional[str] = None
    market_regime: str
    volatility_state: str
    volume_ratio: float
    models: Dict[str, ModelPrediction]
    projections: Projections
    technical_snapshot: TechnicalSnapshot
    qualitative_alpha: Optional[str] = None
    
    # Keep some legacy fields for backward compatibility if needed, 
    # but the instructions specify a strict new schema.
    # I'll include the ones used by the current frontend if possible.
    portfolio: Optional[PortfolioSummary] = None
    historical_markers: Optional[List[Dict[str, Any]]] = None
    candles: Optional[List[Dict[str, Any]]] = None
    clouds: Optional[List[Dict[str, Any]]] = None
