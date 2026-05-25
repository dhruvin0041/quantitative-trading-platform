# API REFERENCE

Hydra Terminal API (v2.1.0) is a high-performance REST interface for quantitative inference and portfolio analytics.

## 🔓 Authentication
All endpoints require an `X-API-Key` header.
```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/health
```

---

## 📊 Market Intelligence Endpoints

### 1. `GET /universe`
Returns the tracked stock universe across all supported markets (USA, India) with live price snapshots and institutional metadata.

**Response Schema**:
```json
{
  "universe": [
    {
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "price": 182.45,
      "pct_change": 1.25,
      "market": "us",
      "metadata": {
        "exchange": "NASDAQ",
        "currency": "USD",
        "timezone": "America/New_York"
      }
    }
  ]
}
```

### 2. `GET /predict`
Executes the full agentic inference pipeline for a specific ticker. This includes multi-modal data fetching, model ensemble fusion, and risk agent validation.

**Parameters**:
- `ticker` (string, required): The stock ticker symbol (e.g., `AAPL`, `RELIANCE.NS`).

**Response Highlights**:
- `signal`: Final action (`BUY`, `SELL`, `HOLD`, `VETOED`).
- `confidence_score`: 0-100 probability of the selected direction.
- `xai`: SHAP-based feature importance drivers.
- `risk`: Kelly sizing, Beta, and VaR metrics.
- `projections`: Multi-horizon price forecasts (Floor, Median, Ceiling).

---

### 3. `GET /backtest`
Retrieves aggregated historical performance for a ticker over a specified period.

**Parameters**:
- `ticker` (string, required): Stock ticker symbol.
- `period` (string): Options include `3m`, `6m`, `1y` (default), `2y`.

---

## 💼 Portfolio & Execution Endpoints

### 4. `GET /performance`
Returns comprehensive institutional performance analytics for the paper trading engine.
- **Summary**: Total Return, Sharpe, Sortino, Max Drawdown.
- **PnL**: Today, MTD, YTD, Inception.
- **Attribution**: PnL breakdown by Market Regime and Sector.

### 5. `POST /portfolio/base_currency`
Sets the global base currency for all portfolio accounting.
**Body**: `{ "currency": "INR" }`
**Supported**: `USD`, `INR`, `EUR`, `GBP`.

---

## 🛠 System & Observability

### 6. `GET /health`
Verifies system integrity and model readiness status.

### 7. `GET /metrics`
Exposes Prometheus-formatted metrics for API monitoring.
- `api_requests_total`: Request counts by endpoint and status code.
- `api_request_latency_seconds`: Histogram of inference latency.

### 8. `GET /fx_rates`
Returns a summary of current exchange rates and synchronization timestamps from the `FXEngine`.
