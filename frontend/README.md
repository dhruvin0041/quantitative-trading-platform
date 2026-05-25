# Hydra Terminal — Quantitative Trading Dashboard

**Status: 100/100 Institutional Grade**

Hydra Terminal is a high-performance, SOTA 2026 quantitative trading dashboard designed for professional hedge funds and institutional traders. It serves as the monitoring hub for the Hydra Agentic Mesh.

## Key Features
- **Institutional Visualization**: Custom `lightweight-charts` with Bollinger clouds, moving averages, and AI entry/exit markers.
- **Ensemble Intelligence**: Real-time signal consensus from LSTM, XGBoost, LightGBM, and DQN agents.
- **Explainable AI (XAI)**: Dynamic SHAP bars providing a per-feature decomposition of model decisions.
- **FX-Aware Accounting**: Native support for global markets with real-time base currency normalization (USD/INR/EUR/GBP).
- **Command Center (⌘K)**: Quick asset switching, theme toggling, and action execution via fuzzy search.
- **Live Risk Console**: Real-time tracking of Jensen's Alpha, Kelly Fraction, Beta, and Stampede Risk.
- **Institutional Layout**: Uses an intrinsic sizing model for high-density information display without hardcoded whitespace.

## Technology Stack
- **Framework**: Next.js 16.2 (App Router)
- **Language**: TypeScript (Strict)
- **Styling**: Tailwind CSS v4 (Oxide Engine)
- **Charts**: TradingView Lightweight Charts v5
- **Icons**: Lucide & Phosphor

## Getting Started
```bash
npm install
npm run dev
```

The terminal dashboard will be available at `http://localhost:3000`. Ensure the backend is running at `http://localhost:8000`.

## Performance
Optimized for high-refresh rate displays and multi-monitor setups. Pure TypeScript implementation with zero `any` types.
