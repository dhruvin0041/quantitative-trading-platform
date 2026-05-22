<div align="center">

# HYDRA TERMINAL

> Institutional-Grade AI Signal Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-black?style=flat&logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/TailwindCSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white" alt="TailwindCSS" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  <img src="https://img.shields.io/badge/Status-Active_Development-orange.svg" alt="Status" />
  <img src="https://img.shields.io/badge/Platform-Web-lightgrey.svg" alt="Platform" />
  <img src="https://img.shields.io/badge/Models-LSTM_+_XGBoost-purple.svg" alt="Models" />
</p>

<table align="center">
  <tr>
    <td align="center">
      <img src="file:///C:/Users/heer0/OneDrive/Pictures/Screenshots/Screenshot%202026-05-22%20124158.png" width="100%" alt="Light Mode" />
      <br />
      <em>Light Mode</em>
    </td>
    <td align="center">
      <img src="file:///C:/Users/heer0/OneDrive/Pictures/Screenshots/Screenshot%202026-05-22%20125555.png" width="100%" alt="Dark Mode" />
      <br />
      <em>Dark Mode</em>
    </td>
  </tr>
</table>

</div>

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [System Architecture](#-system-architecture)
3. [Project Structure](#-project-structure)
4. [Signal System Explained](#-signal-system-explained)
5. [Risk Metrics Explained](#-risk-metrics-explained)
6. [Getting Started](#-getting-started)
7. [Environment Variables](#-environment-variables)
8. [API Endpoints](#-api-endpoints)
9. [Theme System](#-theme-system)
10. [Known Limitations](#-known-limitations)
11. [Roadmap](#-roadmap)
12. [Contributing](#-contributing)
13. [License](#-license)

---

## 🔭 Overview

Hydra Terminal is a full-stack stock signal system that utilizes advanced machine learning models to generate real-time BUY/SELL signals on S&P 500 stocks. Designed for the modern quantitative trader, it provides an institutional-style dashboard integrating live market data, AI-driven analysis, and comprehensive risk management.

**Key Features:**
- **ML-Driven Signals:** Generates real-time BUY/SELL signals using dual ML model consensus (`DL_LSTM_V4` + `XGB_AGENT`) with transparent confidence scores.
- **Institutional Dashboard:** Seamless TradingView chart integration overlaid with predictive markers, moving averages, and support/resistance zones.
- **AI Qualitative Analysis:** Uses the Google Gemini API to analyze SEC EDGAR filings, news, and price action to generate human-readable "Qualitative Alpha".
- **Risk & Validation:** Real-time portfolio tracking paired with advanced risk management (VAR, CVaR, Kelly Fraction) and system validation metrics (Sharpe, Sortino, Calmar).
- **Extensive Coverage:** Full S&P 500 coverage universe available in an instantly searchable, interactive sidebar.
- **Premium Themes:** Support for an ultra-dark mode and a premium off-white/orange light theme.

---

## 🏗️ System Architecture

Hydra Terminal is built on a decoupled, high-performance architecture.

### Frontend (Next.js / React / TailwindCSS)
- **Navbar:** Features a global stock search bar, Terminal Command palette (`⌘K`), theme toggle, and a live "System Online" indicator.
- **Ticker Bar:** A scrolling, real-time ticker displaying percentage changes for the entire market.
- **Sidebar:** The full S&P 500 coverage universe, offering instant 1-click loading for any stock.
- **Chart Panel:** A TradingView widget displaying OHLCV candles, BUY/SELL signal overlays, moving averages, support/resistance levels, and dynamic price labels.
- **Bottom Panels:**
  - **Model Consensus:** Displays individual signals and confidence % from `DL_LSTM_V4` and `XGB_AGENT`.
  - **10-Day Projections:** Shows dynamic Floor/Ceiling price targets and an XAI (Explainable AI) explanation block.
  - **NLP Analysis:** Provides a plain-English summary of recent SEC EDGAR filings and news sentiment.
- **Right Panel:**
  - **Live Portfolio:** Tracks Total Equity, Available Cash, Net Return, and Active Positions.
  - **Risk Management:** Displays VAR (95%), CVaR, Beta, Kelly Fraction, and Max Drawdown.
  - **System Validation:** Shows historical Sharpe Ratio, Sortino Ratio, Calmar Ratio, Profit Factor, and Win Rate.

### Backend (Python / FastAPI)
- **`api.py`:** The main FastAPI application serving the `/predict` and `/universe` endpoints.
- **`live_inference.py`:** Fetches live OHLCV market data using `yfinance` and orchestrates the inference pipeline.
- **Feature Engineering:** Calculates features on-the-fly including Returns, Volume Change, High-Low spreads, MA20, and MA50.
- **`latest_scaler.joblib`:** A `StandardScaler` artifact fitted on historical data, used to normalize live features before inference.
- **Gemini API Integration:** Generates qualitative alpha using the `gemini-2.0-flash` model.
- **CORS:** Configured to allow seamless communication with the Next.js frontend.

### ML Models
- **`DL_LSTM_V4`:** A Deep Learning LSTM (Long Short-Term Memory) model optimized for sequential, temporal pattern recognition in price data.
- **`XGB_AGENT`:** An XGBoost gradient boosting model designed for rapid feature-based classification.
- **Consensus Logic:** Both models vote independently. A signal is only issued if there is a strong consensus; otherwise, it is VETOED.
- **Scaler:** Standardizes incoming data to zero mean and unit variance before it hits the models.

---

## 🗂️ Project Structure

```text
hydra-terminal/
├── frontend/
│   ├── components/
│   │   ├── ThemeToggle.tsx
│   │   ├── SearchBar.tsx
│   │   ├── Chart.tsx
│   │   └── ...
│   ├── pages/ (or app/)
│   ├── styles/
│   └── package.json
├── backend/
│   ├── api.py
│   ├── live_inference.py
│   ├── latest_scaler.joblib
│   ├── models/
│   └── .env
├── screenshots/
└── README.md
```

---

## 🚦 Signal System Explained

Hydra Terminal removes human bias by relying entirely on quantitative model consensus.

- **BUY Signal:** Both models predict upward movement with a probability > 0.55.
- **SELL Signal:** Both models predict downward movement with a probability < 0.45.
- **VETOED / HOLD:** The models conflict (e.g., one is bullish, the other bearish) or lack strong conviction.
- **Confidence %:** Represents how strongly both models agree on the direction (scaled to 100%).
- **Visuals:** Signals are plotted directly on the TradingView chart as green upward arrows (`BUY`) and red downward arrows (`SELL`).

---

## 🛡️ Risk Metrics Explained

The right panel features institutional-grade risk metrics designed to protect capital:

- **VAR (95%):** *Value at Risk.* The maximum expected loss in dollars over a specific timeframe with 95% confidence.
- **CVaR:** *Conditional Value at Risk (Expected Shortfall).* The expected average loss *if* the VAR threshold is breached (the worst 5% of scenarios).
- **Beta:** Correlation to the broader market. `< 1` means the stock is historically less volatile than the S&P 500; `> 1` means it is more volatile.
- **Kelly Fraction:** Based on the Kelly Criterion, this dictates the mathematically optimal fraction of your portfolio to risk on this specific trade.
- **Max Drawdown:** The largest historical peak-to-trough decline in the asset's value.
- **Sharpe Ratio:** Risk-adjusted return. A measure of return earned per unit of total risk.
- **Sortino Ratio:** A variation of the Sharpe ratio that only penalizes downside volatility.
- **Calmar Ratio:** Annualized return divided by the maximum drawdown.
- **Profit Factor:** Gross profit divided by gross loss across historical signals.
- **Win Rate:** The percentage of historical trades that resulted in a profit.

---

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.11+
- Google AI Studio API key (available on the free tier)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/hydra-terminal.git
cd hydra-terminal
```

**2. Install Frontend Dependencies**
```bash
cd frontend
npm install
```

**3. Install Backend Dependencies**
```bash
cd ../backend
pip install -r requirements.txt
```

**4. Set up the Environment Variables**
Create a `.env` file in the `backend/` directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
PORT=8000
ENVIRONMENT=development
```

**5. Regenerate the Scaler Artifact**
```bash
python regenerate_scaler.py
```

**6. Start the Backend Server**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**7. Start the Frontend Server**
Open a new terminal window:
```bash
cd frontend
npm run dev
```

**8. Access the Dashboard**
Open your browser and navigate to: [http://localhost:3000](http://localhost:3000)

---

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI Studio key for Gemini qualitative analysis | Yes |
| `PORT` | FastAPI backend port (default `8000`) | No |
| `ENVIRONMENT` | Specifies `production` or `development` | No |

---

## 🌐 API Endpoints

The FastAPI backend exposes the following REST endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/universe` | Returns the full list of supported S&P 500 stocks. |
| `GET` | `/predict?ticker=AAPL` | Runs the ML inference pipeline and returns signals, confidence, prices, and NLP text. |
| `GET` | `/health` | Simple health check endpoint for system monitoring. |

### Example `/predict` Response:
```json
{
  "ticker": "AAPL",
  "current_price": 189.43,
  "signal": "SELL",
  "confidence_score": 72.5,
  "models": {
    "DL_LSTM_V4": {"signal": "SELL", "probability": 0.25},
    "XGB_AGENT": {"signal": "SELL", "probability": 0.30}
  },
  "projections": {
    "floor": 182.10,
    "ceiling": 195.00
  },
  "qualitative_alpha": "Market continues to show downward pressure amidst supply chain concerns. Bearish consensus achieved."
}
```

---

## 🎨 Theme System

Hydra Terminal features a completely custom, dual-theme architecture:

- **Dark Mode:** The default theme (`data-theme="dark"`). Untouched, highly-optimized CSS designed to reduce eye strain.
- **Light Mode:** A premium off-white (`#FDFAF5`) and orange (`#E8650A`) theme. 
- **Toggle Mechanism:** Activated via the sun/moon icon in the top-right navbar. 
- **Scoping:** All light-mode styles are strictly scoped under the `[data-theme="light"]` CSS selector, ensuring that dark mode base styles are never permanently overwritten.
- **Persistence:** User preference is saved to `localStorage` and applied instantly on load to prevent flickering.

---

## ⚠️ Known Limitations

> **CRITICAL:** Never run `clean_artifacts.py` in a production environment. This script deletes necessary ML artifacts. If run accidentally, you must regenerate the scaler via `python regenerate_scaler.py`.

- **Rate Limits:** The Gemini qualitative analysis relies on the free tier of Google AI Studio. You may see a "Qualitative analysis unavailable" message if you exceed 15 requests per minute.
- **Brokerage Integration:** Currently, the system supports paper trading and analysis only. There is no live execution integration with a brokerage.
- **Low Conviction Warnings:** Signal confidence scores below 60% should be treated as low conviction and executed with caution.

---

## 🗺️ Roadmap

- [ ] Live brokerage integration (Alpaca / Interactive Brokers)
- [ ] Email/SMS alerts for high-confidence signals
- [ ] Backtesting engine with historical signal replay
- [ ] Multi-timeframe analysis (1H, 4H, Daily)
- [ ] Portfolio optimization using Modern Portfolio Theory
- [ ] Mobile responsive layout

---

## 🤝 Contributing

We welcome contributions from the community! If you're a new developer joining the project:

1. **Branch Naming:** Follow conventional naming such as `feature/your-feature`, `fix/your-fix`, or `docs/your-doc-update`.
2. **Commits:** We strictly follow [Conventional Commits](https://www.conventionalcommits.org/). Example: `feat(chart): add new MA indicator`.
3. **Pull Requests:** All PRs must pass CI checks and be reviewed by a core maintainer before merging.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

<div align="center">
  <i>Built for the modern quant. Removing emotion, executing with precision.</i>
</div>