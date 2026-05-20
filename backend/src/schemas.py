from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class ModelConfidence(BaseModel):
    Suggested_Action: str
    Confidence: str

class ModelsReport(BaseModel):
    Primary_Deep_Learning: ModelConfidence
    Secondary_XGBoost: ModelConfidence

class RangeReport(BaseModel):
    Low: float
    High: float

class RiskManagementReport(BaseModel):
    Meta_Model_Status: str
    Dynamic_10_Day_Range: RangeReport

class ContextReport(BaseModel):
    Top_Headline_Processed: str

class AIReport(BaseModel):
    Models: ModelsReport
    Risk_Management: RiskManagementReport
    Context: ContextReport

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
    action: str
    confidence: str
    price: float
    ai_report: AIReport
    portfolio: Optional[PortfolioSummary] = None
    historical_markers: Optional[List[Dict[str, Any]]] = None
    candles: Optional[List[Dict[str, Any]]] = None
    clouds: Optional[List[Dict[str, Any]]] = None
    agent_consensus: Optional[Dict[str, Any]] = None
    institutional_metrics: Optional[Dict[str, Any]] = None
    physical_edge: Optional[Dict[str, Any]] = None
    smart_routing: Optional[Dict[str, Any]] = None
    xai_reasoning: Optional[str] = None
    news: Optional[str] = None
    paper_trade: Optional[Dict[str, Any]] = None
