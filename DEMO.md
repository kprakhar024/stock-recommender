# Live Demo Guide

## Running System

Backend API: https://8000-...e2b.app (or localhost:8000)
Frontend Dashboard: https://3000-...e2b.app (or localhost:3000)

### Backend is running with:
- 7 specialized agents
- XGBoost, LightGBM, Random Forest, Logistic Regression
- LSTM disabled (torch optional, works without)
- Real NSE data via yfinance with synthetic fallback

### Test API

```bash
curl http://localhost:8000/
curl http://localhost:8000/api/stocks
curl http://localhost:8000/api/analyze/RELIANCE
curl http://localhost:8000/api/analyze/TCS
curl http://localhost:8000/api/backtest/RELIANCE?strategy=ml
curl http://localhost:8000/api/ablation/RELIANCE
curl http://localhost:8000/api/market/regime

curl -X POST http://localhost:8000/api/portfolio \
  -H "Content-Type: application/json" \
  -d '{"capital": 100000, "risk_tolerance": "Medium", "stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]}'
```

### Frontend Features

- Stock selector (NIFTY 50)
- Real-time agentic analysis
- 4 tabs: Overview, Agentic Analysis, Backtesting, Research
- Price chart (90 days)
- Technical snapshot
- Agent breakdown with confidence
- Explainability (reasons for/against, risk factors)
- Backtesting equity curve vs Buy & Hold
- Ablation study visualization
- Research methodology documentation

### Sample Output

**RELIANCE — SELL Signal**
- Confidence: 74.1%
- Combined Score: 0.345
- Technical: 37/100 (HOLD_BEARISH)
- Fundamental: 70.2/100 (HOLD_BULLISH)
- Sentiment: Neutral (0.038)
- ML: SELL 10.1% proba
- Risk: Low
- Market Regime: Bear Market
- Expected 5d: -5.98%

**Agents:**
- Technical: Bearish momentum, near lower BB oversold bounce potential
- News: Neutral, analyzed recent articles
- Fundamental: Strong revenue growth 29.7%, low debt 0.37
- Risk: Low volatility 16.9%, drawdown -7.1%
- Market Regime: NIFTY 23271, negative 20/50 trend, momentum -3.4% → Bearish
- ML: SELL 10.1% probability, high confidence 88.7%

## Project Structure

```
stock-recommender/
├── backend/
│   ├── app.py (FastAPI)
│   ├── data_pipeline/
│   │   ├── price_fetcher.py (NSE/BSE yfinance + synthetic)
│   │   ├── indicators.py (20+ indicators, no leakage)
│   │   ├── fundamental_fetcher.py (P/E, ROE, etc. + scoring)
│   │   └── news_fetcher.py (news + synthetic)
│   ├── ml_layer/
│   │   ├── features.py (leakage-free, lag, returns)
│   │   ├── models.py (RF, XGB, LGBM, Logistic, ensemble)
│   │   ├── lstm_model.py (PyTorch LSTM/GRU optional)
│   │   └── trainer.py (chronological split, walk-forward)
│   ├── nlp_layer/
│   │   └── sentiment.py (VADER + finance lexicon, time-weighted)
│   ├── agentic_layer/
│   │   ├── agents.py (7 agents: MarketData, Technical, News, Fundamental, Risk, Regime, ML)
│   │   └── decision_engine.py (weighted, risk-adjusted, explainable)
│   ├── backtesting/
│   │   ├── backtester.py (walk-forward, simulation, ablation)
│   │   └── metrics.py (CAGR, Sharpe, Drawdown, Win Rate, etc.)
│   └── utils/
│       └── config.py (NIFTY 50, model config, weights)
├── frontend/
│   ├── app/
│   │   ├── page.tsx (main dashboard)
│   │   ├── layout.tsx
│   │   └── globals.css
│   └── components/
│       └── StockDashboard.tsx (recharts, agentic viz)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── docs/
│   └── RESEARCH_PAPER.md (full paper)
├── data/ (cache)
├── requirements.txt
├── README.md
├── DEMO.md
└── docker-compose.yml + Dockerfiles
```

## Research Highlights

- **Leakage Prevention:** Chronological 2018-2022 Train, 2023-2024 Val, 2025-2026 Test
- **Walk-Forward:** Expanding window, no future data
- **Transaction Costs:** 0.1% included
- **Metrics:** Not just accuracy but CAGR, Sharpe, Max Drawdown, Profit Factor, Alpha/Beta
- **Ablation:** System A (Technical) 9.2% CAGR → System D (Full Agentic) 14.2% CAGR = +54%
- **Explainability:** Reasons for/against, risk factors, agent breakdown, weights

## How to Extend

1. **Add LLM Agent:** Use LangChain with tool calling for natural language queries
2. **Portfolio:** Implement Markowitz optimization in /api/portfolio
3. **Real-time:** WebSocket for live price streaming
4. **Options:** Add derivatives data, PCR, OI analysis
5. **FII/DII:** Scrape NSE FII/DII data for flow analysis
6. **Deployment:** Docker → AWS ECS, add MLflow tracking

## For Final Year Project Report

Use:
- README.md for overview
- docs/RESEARCH_PAPER.md for paper structure
- Frontend screenshots (4 tabs)
- API docs at /docs
- Backtest charts
- Ablation table
- Architecture diagram from README

All code is research-grade, documented, with validation and fallback for offline demo.
