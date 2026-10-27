"""
Unified Streamlit Dashboard — Indian Stock Recommendation System (Revision 7)
══════════════════════════════════════════════════════════════════════════════
Replaces the separate Next.js frontend + FastAPI server.

Sections:
  ① Ticker Search          — Full NSE Universe (NIFTY 50 / Next 50 / Midcap /
                              Popular) with search-by-name + sector filter
  ② Macro Banner           — NIFTY price/change, India VIX level (color-coded)
  ③ Central Decision Badge — 🟢 BUY / 🟡 HOLD / 🔴 SELL with confidence %
  ④ Probability Bars       — P(BUY), P(HOLD), P(SELL) via st.progress
  ⑤ Risk Panel             — Stop-loss, Target, RRR, Max Allocation, Veto status
  ⑥ Agent Breakdown Tabs   — Technical, Fundamental, ML, News, Regime
  ⑦ FII / DII Panel        — Stock-level institutional holding + market flows
  ⑧ PDF Report Button      — 1-page PDF via ReportLab
  ⑨ Info Tooltips (ⓘ)      — Beginner-friendly explanations for every financial term

Run: streamlit run ui/app.py --server.port 8501
"""
import sys
import os
from pathlib import Path

# ── Path setup: ensure backend is importable ───────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Stock Advisor — AI Multi-Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import backend modules ─────────────────────────────────────────────────────
try:
    from backend.data_pipeline.price_fetcher import fetcher
    from backend.data_pipeline.indicators import TechnicalIndicators
    from backend.agentic_layer.agents import (
        MarketDataAgent, TechnicalAnalysisAgent, NewsSentimentAgent,
        FundamentalAgent, RiskAgent, MarketRegimeAgent, MLPredictionAgent,
    )
    from backend.agentic_layer.decision_engine import decision_engine
    from backend.utils.config import STOCK_UNIVERSE
    from backend.data_pipeline.institutional_fetcher import institutional_fetcher
    from backend.data_pipeline.market_pulse import market_pulse_scanner
    from backend.agentic_layer.daily_recommender import daily_recommender
    from backend.verification.scenario_verifier import scenario_verifier
except ImportError as e:
    st.error(f"Import error: {e}\n\nMake sure to run from the project root:\n`streamlit run ui/app.py`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  GLOSSARY — Beginner-friendly explanations for every financial term
#  Used by the info_icon() helper throughout the dashboard.
# ══════════════════════════════════════════════════════════════════════════════
GLOSSARY = {
    # ── Market terms ────────────────────────────────────────────────────────
    "NIFTY 50": "NIFTY 50 is the benchmark index of the National Stock Exchange (NSE). It tracks the performance of the top 50 largest companies listed on NSE. Think of it as a 'health meter' for the Indian stock market.",
    "India VIX": "The India Volatility Index (VIX) measures how much the market expects prices to move in the near future. A high VIX (above 22) means high fear/uncertainty; a low VIX means calm markets. It is often called the 'Fear Index'.",
    "Market Regime": "The overall direction and condition of the market — Bullish (prices rising), Bearish (prices falling), or Sideways (no clear direction). It helps you know the 'mood' of the market.",
    "Breadth": "Market breadth compares how many stocks are rising vs falling. If most stocks are going up, it shows broad strength; if most are falling, it signals widespread weakness.",
    "Live Price": "The real-time or most recent traded price of the stock on the National Stock Exchange (NSE).",
    "Previous Close": "The closing price of the stock at the end of the previous trading day.",

    # ── Technical Analysis terms ────────────────────────────────────────────
    "RSI": "The Relative Strength Index (RSI) measures whether a stock is overbought (too expensive, RSI > 70) or oversold (undervalued, RSI < 30). It ranges from 0 to 100. Think of it like a speedometer — if the stock is running too fast (> 70), it might slow down.",
    "MACD": "Moving Average Convergence Divergence (MACD) is a trend-following indicator. When the MACD line crosses above the signal line, it may signal a buy opportunity; when it crosses below, it may signal a sell.",
    "SMA": "Simple Moving Average (SMA) is the average closing price over a set number of days (e.g., SMA 20 = average of last 20 days). It smooths out price fluctuations to show the overall trend direction.",
    "EMA": "Exponential Moving Average (EMA) is similar to SMA but gives more weight to recent prices, making it quicker to react to price changes. EMA 200 is a key long-term trend indicator.",
    "Bollinger Bands": "Bollinger Bands are two lines drawn above and below the moving average. When the price touches the upper band, the stock may be overbought; when it touches the lower band, it may be oversold. The wider the bands, the more volatile the stock.",
    "Golden Cross": "A bullish signal that occurs when the short-term moving average (e.g., 50-day) crosses above the long-term moving average (e.g., 200-day). It suggests prices may continue rising.",
    "Volume": "Volume is the number of shares traded in a given period. Higher volume confirms the strength of a price move — rising price with high volume is more reliable than rising price with low volume.",
    "Volatility": "Volatility measures how much a stock's price swings up and down. High volatility = bigger price swings (riskier). Low volatility = steadier prices (less risky). It helps you judge the risk of holding a stock.",
    "Candlestick": "A candlestick chart shows 4 prices for each day: Open, High, Low, Close. Green candles mean the price went up that day; red candles mean it went down.",
    "Technical Score": "A composite score (0–100) that combines multiple technical signals like RSI, MACD, moving averages, and volume to give an overall picture of a stock's technical health.",
    "Overbought": "When RSI is above 70 or price is near the upper Bollinger Band — the stock has risen very quickly and might be due for a correction (price drop).",
    "Oversold": "When RSI is below 30 or price is near the lower Bollinger Band — the stock has fallen sharply and might be due for a bounce (price recovery).",

    # ── Fundamental Analysis terms ──────────────────────────────────────────
    "P/E Ratio": "Price-to-Earnings Ratio = Stock Price ÷ Earnings Per Share. It tells you how many years of current earnings you're paying for. Lower P/E may mean undervalued; higher P/E may mean overvalued. Example: P/E of 20 means you pay ₹20 for every ₹1 the company earns.",
    "P/B Ratio": "Price-to-Book Ratio = Stock Price ÷ Book Value Per Share. It compares what the market values the company at vs what the company's assets are actually worth. P/B < 1 may indicate an undervalued company.",
    "ROE": "Return on Equity measures how much profit a company generates with shareholders' money. Higher ROE (> 15%) means the company is efficient at making profits. Formula: Net Profit ÷ Shareholders' Equity × 100.",
    "ROCE": "Return on Capital Employed measures how well a company uses ALL its capital (equity + debt) to generate profit. Higher ROCE means better overall efficiency.",
    "D/E Ratio": "Debt-to-Equity Ratio shows how much debt a company has compared to its equity (owner's money). High D/E (> 1) means the company relies heavily on borrowed money, which is riskier.",
    "Profit Margin": "Net Profit Margin = Net Profit ÷ Revenue × 100. It tells you what percentage of each rupee of sales the company keeps as profit. Higher margin = more efficient business.",
    "Revenue Growth": "The percentage increase in a company's sales compared to the previous year. Positive growth means the company is selling more; negative means it's selling less.",
    "Earnings Growth": "The percentage increase in a company's profit compared to the previous year. Strong earnings growth often leads to stock price appreciation.",
    "EPS": "Earnings Per Share = Company's Net Profit ÷ Total Shares Outstanding. It tells you how much profit each share earned. Higher EPS generally means a more profitable company.",
    "Market Cap": "Market Capitalisation = Stock Price × Total Shares. It represents the total market value of a company. Large cap (> ₹20,000 Cr) are generally safer; small cap companies are riskier but can grow faster.",
    "Fundamental Score": "A composite score (0–100) based on financial health metrics like P/E, ROE, debt, growth, and profitability. Higher score = financially stronger company.",
    "Current Ratio": "Current Ratio = Current Assets ÷ Current Liabilities. It measures a company's ability to pay short-term debts. A ratio above 1.5 is generally healthy.",

    # ── Dividend terms ──────────────────────────────────────────────────────
    "Dividend Yield": "Dividend Yield = (Annual Dividend Per Share ÷ Current Stock Price) × 100. It shows what percentage of the stock price you earn back in cash dividends each year. Example: If a stock costs ₹100 and pays ₹4 dividend, the yield is 4%.",
    "Dividend Per Share": "Dividend Per Share (DPS) is the exact cash payout in ₹ given to the shareholder for every single share owned. Formula: Total Dividends Paid ÷ Total Shares.",
    "DPS": "Dividend Per Share (DPS) — the cash amount in ₹ distributed to you for each share from the company's profits.",
    "Stock Price": "The current market price at which 1 share of this company is traded on the National Stock Exchange (NSE).",
    "Dividend": "A portion of a company's profit distributed in cash to shareholders. Not all companies pay dividends — fast-growing companies often reinvest their profits instead.",

    # ── FII / DII terms ─────────────────────────────────────────────────────
    "FII": "Foreign Institutional Investors (FII) — large foreign institutions like global mutual funds, hedge funds, and foreign pension funds that invest in Indian stocks. When FIIs buy heavily, market liquidity increases and prices often rise.",
    "DII": "Domestic Institutional Investors (DII) — Indian institutions like local mutual funds (SBI MF, HDFC MF), insurance giants (like LIC), and Indian banks that invest in the Indian stock market. DIIs provide continuous stability.",
    "FII Holding": "The percentage of this specific company's total shares owned by Foreign Institutional Investors (FII/FPI). Higher foreign ownership indicates strong international institutional trust.",
    "DII Holding": "The percentage of this specific company's total shares owned by Domestic Institutional Investors (Mutual Funds, LIC, Banks). High DII holding provides strong domestic support and stability.",
    "Promoter Holding": "The percentage of shares held by the original founders, promoters, and company management. When promoters own a high stake (> 50%), it shows they have strong confidence in their company's future.",
    "Retail / Public Holding": "The percentage of shares owned by individual retail investors, traders, and the general non-institutional public.",
    "FII/DII Net Flow": "The difference between how much FIIs/DIIs bought vs sold across the market in a day (in ₹ Crores). Positive = net buying (bullish); Negative = net selling (bearish).",
    "Institutional Holding": "The total percentage of a company's shares owned by all institutional investors (FIIs + DIIs combined). Higher institutional holding (> 40%) means the company is heavily sponsored by professionals.",
    "Insider Holding": "Shares owned by company promoters, founders, and key management insiders.",
    "Institutional Bias": "An assessment of whether institutional investors are actively accumulating or reducing their exposure to this stock.",
    "Support Score": "A score (0–100%) indicating how strongly large institutional investors (both FIIs and DIIs) are financially backing and supporting this stock.",

    # ── Risk terms ──────────────────────────────────────────────────────────
    "Stop-Loss": "A safety price level at which you automatically sell a stock to limit your loss if the price moves against you. Example: If you buy at ₹100 and set stop-loss at ₹92, your loss is capped at 8%.",
    "Target Price": "The expected price a stock could reach based on analysis. It tells you when to book your profits. Example: Buy at ₹100, target ₹120 = expected 20% gain.",
    "Risk-to-Reward": "Risk-to-Reward Ratio (RRR) compares potential loss to potential profit. RRR of 1:3 means for every ₹1 you risk losing, you stand to gain ₹3. Higher is better.",
    "Max Allocation": "The maximum percentage of your total portfolio you should invest in this single stock to stay diversified and avoid putting all eggs in one basket.",
    "Veto": "The Risk Agent's emergency safety override — if market conditions are dangerous (high VIX, broken moving averages), it automatically vetoes BUY signals down to HOLD.",
    "Risk Level": "An overall risk classification — Low (stable), Medium (normal), or High (volatile).",
    "Expected 5d Return": "The AI model's prediction of how much the stock price could change over the next 5 trading days, expressed as a percentage.",

    # ── ML / AI terms ───────────────────────────────────────────────────────
    "ML Prediction": "Machine Learning prediction — the AI algorithm's forecast of whether the stock is a BUY, HOLD, or SELL based on mathematical patterns from historical data.",
    "P(BUY)": "The AI model's estimated probability (0–100%) that the stock will rise and produce a profitable return in the coming days.",
    "P(HOLD)": "The AI model's estimated probability (0–100%) that the stock will stay flat or move sideways.",
    "P(SELL)": "The AI model's estimated probability (0–100%) that the stock may drop in price.",
    "Confidence": "How certain the multi-agent system is about its final recommendation (0–100%). Higher confidence = stronger consensus among agents.",
    "Walk-Forward": "A rigorous time-tested evaluation method that trains models on past data and evaluates them on future unseen periods to prevent memorisation.",
    "XGBoost": "Extreme Gradient Boosting — an advanced machine learning algorithm used by hedge funds to predict stock direction.",
    "FinBERT": "A specialised AI language model trained on financial text to accurately understand whether market news is positive, negative, or neutral.",
    "OOF Metrics": "Out-of-Fold metrics — validation scores proving the model's accuracy on unseen historical periods.",

    # ── News & Verification terms ───────────────────────────────────────────
    "Sentiment": "The overall mood or tone of news headlines — Positive (good news), Negative (bad news), or Neutral.",
    "Net Score": "A combined score from all analyzed news articles. Positive means bullish news; negative means bearish news.",
    "52W High": "The highest price this stock has traded at in the last 52 weeks (1 year).",
    "52W Low": "The lowest price this stock has traded at in the last 52 weeks (1 year).",
    "Alpha vs NIFTY": "Alpha is the excess return generated by an investment above the benchmark NIFTY 50 index return. Positive alpha means outperforming the market.",
    "Win Rate": "The percentage of stock recommendations that achieved a positive return.",
    "Entry Price": "The recommended price at which the stock position was opened.",
    "Actual Return": "The real return achieved on the stock compared to entry price.",
}


def info_icon(term: str, extra_text: str = "") -> str:
    """Return an HTML info icon ⓘ with a tooltip for the given term."""
    explanation = GLOSSARY.get(term, f"A financial metric used in stock analysis.")
    # Escape quotes for HTML attribute
    safe_explanation = explanation.replace('"', '&quot;').replace("'", "&#39;")
    display = extra_text if extra_text else ""
    return f'{display} <span class="info-tooltip" title="{safe_explanation}">ⓘ</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Premium dark design + tooltip styles
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

/* ── Info tooltip icon (ⓘ) ──────────────────────────────────────── */
.info-tooltip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: rgba(88, 166, 255, 0.15);
    color: #58a6ff;
    font-size: 11px;
    font-weight: 700;
    cursor: help;
    margin-left: 4px;
    vertical-align: middle;
    position: relative;
    border: 1px solid rgba(88, 166, 255, 0.3);
    transition: all 0.2s ease;
}
.info-tooltip:hover {
    background: rgba(88, 166, 255, 0.30);
    border-color: #58a6ff;
    transform: scale(1.15);
}
.info-tooltip::after {
    content: attr(title);
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: linear-gradient(135deg, #1c2333 0%, #21262d 100%);
    border: 1px solid #58a6ff;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    font-weight: 400;
    color: #c9d1d9;
    white-space: normal;
    width: 280px;
    line-height: 1.5;
    z-index: 9999;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s ease;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
}
.info-tooltip:hover::after {
    opacity: 1;
    pointer-events: auto;
}

/* Macro banner card */
.macro-card {
    background: linear-gradient(135deg, #1c2333 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    transition: transform 0.2s;
}
.macro-card:hover { transform: translateY(-2px); }
.macro-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.macro-value { font-size: 26px; font-weight: 700; margin: 4px 0; }
.macro-change { font-size: 13px; font-weight: 500; }
.green  { color: #3fb950; }
.red    { color: #f85149; }
.yellow { color: #d29922; }
.blue   { color: #58a6ff; }

/* Decision badge */
.badge-buy  { background: linear-gradient(135deg, #0d4f2a, #1a7a40); border: 2px solid #3fb950; border-radius: 16px; padding: 24px 40px; text-align: center; }
.badge-hold { background: linear-gradient(135deg, #4a3a0d, #7a5f1a); border: 2px solid #d29922; border-radius: 16px; padding: 24px 40px; text-align: center; }
.badge-sell { background: linear-gradient(135deg, #4f0d0d, #7a1a1a); border: 2px solid #f85149; border-radius: 16px; padding: 24px 40px; text-align: center; }
.badge-label { font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 2px; }
.badge-signal { font-size: 52px; font-weight: 800; line-height: 1.1; margin: 8px 0; }
.badge-conf { font-size: 16px; font-weight: 500; opacity: 0.9; }

/* Info card */
.info-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
}
.info-card-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; }
.info-card-value { font-size: 20px; font-weight: 600; margin-top: 4px; }

/* Veto banner */
.veto-active {
    background: linear-gradient(135deg, #4f1919, #7a2020);
    border: 1px solid #f85149;
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 600;
    font-size: 14px;
}

/* Section header */
.section-header {
    font-size: 13px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 24px;
}

/* Probability bar container */
.prob-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
.prob-label { font-size: 13px; font-weight: 600; width: 60px; }
.prob-value { font-size: 13px; color: #8b949e; margin-left: auto; }

/* Tab styling */
.stTabs [data-baseweb="tab"] { background: transparent; border: none; color: #8b949e; font-size: 13px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #58a6ff; border-bottom: 2px solid #58a6ff; }

/* Spinner */
.stSpinner > div { border-color: #58a6ff; }

/* Market Pulse & Notification styles */
.pulse-container {
    background: linear-gradient(135deg, #131922 0%, #1a2230 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 18px;
}
.pulse-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.pulse-title {
    font-size: 13px;
    font-weight: 700;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.pulse-timestamp {
    font-size: 11px;
    color: #8b949e;
}
.pulse-badge-green {
    background: rgba(63, 185, 80, 0.15);
    border: 1px solid #3fb950;
    color: #3fb950;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin: 2px 4px 4px 0;
}
.pulse-badge-red {
    background: rgba(248, 81, 73, 0.15);
    border: 1px solid #f85149;
    color: #f85149;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin: 2px 4px 4px 0;
}
.pulse-badge-gold {
    background: rgba(210, 153, 34, 0.15);
    border: 1px solid #d29922;
    color: #e3b341;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin: 2px 4px 4px 0;
}
.pulse-badge-blue {
    background: rgba(88, 166, 255, 0.15);
    border: 1px solid #58a6ff;
    color: #58a6ff;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin: 2px 4px 4px 0;
}

/* Recommendation card styles */
.rec-card {
    background: #161b22;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
    border: 1px solid #30363d;
}
.rec-card-buy {
    border-left: 5px solid #3fb950;
    background: linear-gradient(90deg, rgba(63,185,80,0.09) 0%, #161b22 65%);
}
.rec-card-sell {
    border-left: 5px solid #f85149;
    background: linear-gradient(90deg, rgba(248,81,73,0.09) 0%, #161b22 65%);
}
.factor-pill {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    color: #c9d1d9;
    margin-right: 6px;
    margin-bottom: 6px;
    display: inline-block;
}

/* FII/DII institutional panel */
.inst-panel {
    background: linear-gradient(135deg, #131922 0%, #1a2230 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.inst-panel-title {
    font-size: 14px;
    font-weight: 700;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 14px;
}
.inst-metric {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    margin-bottom: 8px;
}
.inst-metric-label {
    font-size: 10px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.inst-metric-value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
}
.inst-flow-bar {
    height: 6px;
    border-radius: 3px;
    background: #21262d;
    margin-top: 6px;
    overflow: hidden;
}
.inst-flow-fill-green {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #3fb950, #2ea043);
}
.inst-flow-fill-red {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #f85149, #da3633);
}

/* Sector badge */
.sector-badge {
    background: rgba(88, 166, 255, 0.12);
    border: 1px solid rgba(88, 166, 255, 0.3);
    color: #58a6ff;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_macro_data():
    """Cache NIFTY + VIX for 5 minutes."""
    try:
        nifty = fetcher.fetch_market_data("^NSEI")
        vix   = fetcher.fetch_india_vix()
        n_last  = float(nifty["Close"].iloc[-1])
        n_prev  = float(nifty["Close"].iloc[-2])
        n_chg   = (n_last - n_prev) / n_prev * 100
        vix_val = float(vix["Close"].iloc[-1])
        return {"nifty": n_last, "nifty_chg": n_chg, "vix": vix_val}
    except Exception:
        return {"nifty": None, "nifty_chg": None, "vix": None}


def run_analysis(symbol: str) -> dict:
    """Run full agentic pipeline for a symbol."""
    agents = {
        "market_data":   MarketDataAgent(),
        "technical":     TechnicalAnalysisAgent(),
        "news":          NewsSentimentAgent(),
        "fundamental":   FundamentalAgent(),
        "risk":          RiskAgent(),
        "market_regime": MarketRegimeAgent(),
        "ml":            MLPredictionAgent(),
    }

    md_result = agents["market_data"].analyze(symbol, {})
    if md_result["signal"] == "ERROR":
        return {"error": md_result["details"].get("error", "Data fetch failed")}

    shared = md_result["details"]
    outputs = {
        "market_data":   md_result,
        "technical":     agents["technical"].analyze(symbol, shared),
        "news":          agents["news"].analyze(symbol, shared),
        "fundamental":   agents["fundamental"].analyze(symbol, shared),
        "risk":          agents["risk"].analyze(symbol, shared),
        "market_regime": agents["market_regime"].analyze(symbol, shared),
        "ml":            agents["ml"].analyze(symbol, shared),
    }

    decision = decision_engine.decide(outputs, symbol)

    # ── Fetch FII/DII data for this specific stock ──────────────────────
    try:
        inst_profile = institutional_fetcher.get_stock_institutional_profile(symbol)
    except Exception:
        inst_profile = {}

    return {
        "decision": decision,
        "agents": outputs,
        "shared": shared,
        "institutional_profile": inst_profile,
    }


def generate_pdf_report(symbol: str, decision: dict, agents: dict, inst_profile: dict = None) -> bytes:
    """Generate a 1-page PDF report using ReportLab with FII/DII institutional profile."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                topMargin=1.5*cm, bottomMargin=1.5*cm,
                                leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Header
        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=18, spaceAfter=4,
                                     textColor=colors.HexColor("#1a1a2e"))
        sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                     fontSize=10, textColor=colors.grey)

        story.append(Paragraph(f"Stock Analysis Report — {symbol.replace('.NS', '')}", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}", sub_style))
        story.append(Spacer(1, 0.4*cm))

        # Recommendation badge
        rec   = decision.get("recommendation", "HOLD")
        conf  = decision.get("confidence", 0)
        color_map = {"BUY": colors.HexColor("#2e7d32"), "SELL": colors.HexColor("#c62828"), "HOLD": colors.HexColor("#f57f17")}
        rec_data  = [[f"{rec}  |  Confidence: {conf:.1f}%"]]
        rec_table = Table(rec_data, colWidths=[16*cm])
        rec_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color_map.get(rec, colors.grey)),
            ("TEXTCOLOR",  (0, 0), (-1, -1), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 16),
            ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [color_map.get(rec, colors.grey)]),
            ("BORDERRADIUS", (0, 0), (-1, -1), 8),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 0.5*cm))

        # ML Probabilities
        ml_d = decision.get("ml_prediction", {})
        prob_data = [
            ["ML 3-Class Probabilities", ""],
            ["P(BUY)",  f"{ml_d.get('p_buy',  0):.1%}"],
            ["P(HOLD)", f"{ml_d.get('p_hold', 0):.1%}"],
            ["P(SELL)", f"{ml_d.get('p_sell', 0):.1%}"],
        ]
        pt = Table(prob_data, colWidths=[8*cm, 8*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(pt)
        story.append(Spacer(1, 0.4*cm))

        # Risk panel
        risk_d  = decision.get("risk", {})
        exp_ret = decision.get("expected_return_5d", 0)
        risk_data = [
            ["Risk Metrics", ""],
            ["Risk Level",        risk_d.get("level", "—")],
            ["Stop-Loss Price",   f"₹{risk_d.get('stop_loss_price', '—')}"],
            ["Target Price",      f"₹{risk_d.get('target_price', '—')}"],
            ["Risk-to-Reward",    str(risk_d.get("rrr", "—"))],
            ["Max Allocation",    f"{risk_d.get('max_allocation_pct', '—')}%"],
            ["Expected 5d Return",f"{exp_ret:+.2f}%"],
            ["Veto Active",       "⛔ YES" if risk_d.get("veto_active") else "✓ No"],
        ]
        rt = Table(risk_data, colWidths=[8*cm, 8*cm])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(rt)
        story.append(Spacer(1, 0.4*cm))

        # Institutional Profile (FII / DII)
        if inst_profile:
            fii_h = inst_profile.get("fii_holding_pct", 0.0)
            dii_h = inst_profile.get("dii_holding_pct", 0.0)
            prom_h = inst_profile.get("promoter_holding_pct", 0.0)
            bias_h = inst_profile.get("institutional_bias", "MODERATE").replace("_", " ").title()
            inst_data = [
                ["Institutional Ownership (FII / DII)", ""],
                ["FII Holding (Foreign)", f"{fii_h:.1f}%"],
                ["DII Holding (Domestic)", f"{dii_h:.1f}%"],
                ["Promoter / Insider", f"{prom_h:.1f}%"],
                ["Institutional Bias", bias_h],
            ]
            it = Table(inst_data, colWidths=[8*cm, 8*cm])
            it.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTSIZE",   (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(it)
            story.append(Spacer(1, 0.4*cm))

        # Reasons for / against
        expl = decision.get("explanation", {})
        story.append(Paragraph("<b>Why this recommendation:</b>", styles["Heading3"]))
        for r in expl.get("for", [])[:4]:
            story.append(Paragraph(f"• {r}", styles["Normal"]))
        if expl.get("against"):
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("<b>Risk considerations:</b>", styles["Heading3"]))
            for r in expl.get("against", [])[:3]:
                story.append(Paragraph(f"• {r}", styles["Normal"]))

        # Footer
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            "<i>For educational/research purposes only. Not financial advice. "
            "Always do your own due diligence before investing.</i>",
            ParagraphStyle("disclaimer", parent=styles["Normal"],
                           fontSize=8, textColor=colors.grey)
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return b""   # ReportLab not installed


def render_market_pulse_banner():
    """Renders top gainers, top losers, 52W alerts, and FII/DII flow in a sleek notification bar."""
    try:
        pulse = market_pulse_scanner.get_market_pulse()
        flows = institutional_fetcher.get_daily_flows()
    except Exception as e:
        st.warning(f"Market pulse scanner initializing: {e}")
        return

    st.markdown('<div class="pulse-container">', unsafe_allow_html=True)

    sentiment = pulse.get("breadth_sentiment", "MARKET PULSE")
    adv = pulse.get("advances", 0)
    dec = pulse.get("declines", 0)
    time_str = pulse.get("timestamp", "")
    fii_net = flows.get("fii_net_cr", 0.0)
    dii_net = flows.get("dii_net_cr", 0.0)

    st.markdown(f"""
    <div class="pulse-header">
        <div class="pulse-title">🔔 NIFTY 50 Market Pulse & Institutional Flow {info_icon("Breadth")}</div>
        <div class="pulse-timestamp">{time_str} | Breadth: <b>{adv} Adv / {dec} Dec</b> ({sentiment})</div>
    </div>
    """, unsafe_allow_html=True)

    c_g, c_l, c_alerts = st.columns([1.3, 1.3, 1.5])

    with c_g:
        st.markdown("<small style='color:#3fb950; font-weight:700;'>🟢 TOP 3 GAINERS (IN PROFIT TODAY)</small>", unsafe_allow_html=True)
        g_html = ""
        for g in pulse.get("top_gainers", [])[:3]:
            g_html += f"""<div class="pulse-badge-green">
                <b>{g['name']}</b> ₹{g['current_price']:,.1f} <span style="color:#2ea043; font-weight:700;">+{g['change_pct']:.2f}%</span>
            </div>"""
        st.markdown(g_html if g_html else "<small>No gainers</small>", unsafe_allow_html=True)

    with c_l:
        st.markdown("<small style='color:#f85149; font-weight:700;'>🔴 TOP 3 LOSERS (IN LOSS TODAY)</small>", unsafe_allow_html=True)
        l_html = ""
        for l in pulse.get("top_losers", [])[:3]:
            l_html += f"""<div class="pulse-badge-red">
                <b>{l['name']}</b> ₹{l['current_price']:,.1f} <span style="color:#da3633; font-weight:700;">{l['change_pct']:.2f}%</span>
            </div>"""
        st.markdown(l_html if l_html else "<small>No losers</small>", unsafe_allow_html=True)

    with c_alerts:
        st.markdown(f"<small style='color:#e3b341; font-weight:700;'>⚡ 52W HIGHS {info_icon('52W High')} / LOWS {info_icon('52W Low')} & FII-DII {info_icon('FII/DII Net Flow')}</small>", unsafe_allow_html=True)
        a_html = ""
        highs = pulse.get("fifty_two_week_high", [])
        lows = pulse.get("fifty_two_week_low", [])
        for h in highs[:2]:
            a_html += f"""<div class="pulse-badge-gold">⭐ <b>{h['name']}</b> {h['status']} (₹{h['price']:,.1f})</div>"""
        for lw in lows[:2]:
            a_html += f"""<div class="pulse-badge-red">🔻 <b>{lw['name']}</b> {lw['status']} (₹{lw['price']:,.1f})</div>"""

        flow_cls = "pulse-badge-blue" if (fii_net + dii_net) >= 0 else "pulse-badge-red"
        a_html += f"""<div class="{flow_cls}">🏦 <b>FII:</b> {fii_net:+,.0f}Cr | <b>DII:</b> {dii_net:+,.0f}Cr</div>"""
        st.markdown(a_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_daily_picks_and_verification_hub():
    """
    Renders section for:
      1. Daily Recommendations (BUY Today & SELL Today based on RSI, MACD, News, FII/DII, Financial Reports)
      2. Current Market Scenario Verification Report
    """
    with st.expander("🎯 Daily AI Recommendations (BUY/SELL Today) & Reality Verification Report", expanded=False):
        hub_tab1, hub_tab2 = st.tabs([
            "🔥 Stocks to BUY Today & SELL Today",
            "📋 Current Scenario Verification Report"
        ])

        with hub_tab1:
            try:
                recs = daily_recommender.get_daily_recommendations()
            except Exception as e:
                st.error(f"Error computing daily recommendations: {e}")
                return

            st.caption(f"Generated at: {recs.get('generated_at', '')} | Evaluated across: RSI (14), MACD, Financial News Sentiment, FII/DII Flows, Financial Reports")
            if recs.get("fii_dii_summary"):
                st.info(f"💡 **Institutional Market Context:** {recs['fii_dii_summary']}")

            col_buys, col_sells = st.columns(2)

            with col_buys:
                st.markdown("<h4 style='color:#3fb950; margin-bottom: 12px;'>🟢 Top Stocks to BUY Today</h4>", unsafe_allow_html=True)
                for b in recs.get("top_buys_today", []):
                    st.markdown(f"""
                    <div class="rec-card rec-card-buy">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:18px; font-weight:700; color:#e6edf3;">{b['name']}</span>
                            <span style="font-size:14px; font-weight:600; color:#3fb950;">Score: {b['composite_score']:.0f}/100</span>
                        </div>
                        <div style="font-size:13px; color:#8b949e; margin-bottom:8px;">
                            Price: <b>₹{b['price']:,.2f}</b> ({b['change_pct_1d']:+.2f}%) | Target: <b style="color:#3fb950">₹{b['target_buy']:,.2f}</b> {info_icon("Target Price")} | SL: <b style="color:#f85149">₹{b['stop_loss_buy']:,.2f}</b> {info_icon("Stop-Loss")} (RRR: {b['rrr_buy']}x) {info_icon("Risk-to-Reward")}
                        </div>
                        <div style="margin-top:6px;">
                            <span class="factor-pill">📈 <b>RSI:</b> {b['rsi_signal']} {info_icon("RSI")}</span>
                            <span class="factor-pill">📊 <b>MACD:</b> {b['macd_signal']} {info_icon("MACD")}</span>
                            <span class="factor-pill">📰 <b>News:</b> {b['news_sentiment']} {info_icon("Sentiment")}</span>
                            <span class="factor-pill">🏦 <b>FII/DII:</b> {b['institutional_signal']} {info_icon("FII/DII Net Flow")}</span>
                            <span class="factor-pill">📑 <b>Financials:</b> {b['financial_report']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_sells:
                st.markdown("<h4 style='color:#f85149; margin-bottom: 12px;'>🔴 Top Stocks to SELL / AVOID Today</h4>", unsafe_allow_html=True)
                for s in recs.get("top_sells_today", []):
                    st.markdown(f"""
                    <div class="rec-card rec-card-sell">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:18px; font-weight:700; color:#e6edf3;">{s['name']}</span>
                            <span style="font-size:14px; font-weight:600; color:#f85149;">Caution Score: {s['composite_score']:.0f}/100</span>
                        </div>
                        <div style="font-size:13px; color:#8b949e; margin-bottom:8px;">
                            Price: <b>₹{s['price']:,.2f}</b> ({s['change_pct_1d']:+.2f}%) | Target: <b style="color:#f85149">₹{s['target_sell']:,.2f}</b> {info_icon("Target Price")} | SL: ₹{s['stop_loss_sell']:,.2f} {info_icon("Stop-Loss")}
                        </div>
                        <div style="margin-top:6px;">
                            <span class="factor-pill">📈 <b>RSI:</b> {s['rsi_signal']} {info_icon("RSI")}</span>
                            <span class="factor-pill">📊 <b>MACD:</b> {s['macd_signal']} {info_icon("MACD")}</span>
                            <span class="factor-pill">📰 <b>News:</b> {s['news_sentiment']} {info_icon("Sentiment")}</span>
                            <span class="factor-pill">🏦 <b>FII/DII:</b> {s['institutional_signal']} {info_icon("FII/DII Net Flow")}</span>
                            <span class="factor-pill">📑 <b>Financials:</b> {s['financial_report']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with hub_tab2:
            try:
                rep = scenario_verifier.generate_verification_report()
            except Exception as e:
                st.error(f"Error generating verification report: {e}")
                return

            st.markdown(f"#### 📋 Real-World Scenario Verification: Recommendations vs Actual Returns")
            st.caption(f"Audit Window: {rep.get('audit_period', '')} | Verified at: {rep.get('verified_at', '')} | Benchmark: {rep.get('benchmark_index', '')} ({rep.get('benchmark_return_pct', 0):+.2f}%)")

            # Metric cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Directional Win Rate", f"{rep.get('win_rate_pct', 0):.1f}%", help="% of BUY recommendations that generated positive returns")
            with m2:
                st.metric("Avg BUY Return", f"{rep.get('avg_buy_return_pct', 0):+.2f}%", help="Average return achieved across BUY recommendations")
            with m3:
                st.metric("Alpha vs NIFTY 50", f"{rep.get('portfolio_alpha_pct', 0):+.2f}%", help="Excess return generated over NIFTY benchmark")
            with m4:
                st.metric("Capital Protected", f"{rep.get('capital_protected_pct', 0):.2f}%", help="Average loss avoided on SELL/HOLD calls")

            st.markdown("<br>", unsafe_allow_html=True)

            # Table of audit results
            audit_data = rep.get("audit_table", [])
            if audit_data:
                rows = []
                for a in audit_data:
                    rows.append({
                        "Stock": a["name"],
                        "Signal": a["signal"],
                        "Confidence": f"{a['confidence']:.1f}%",
                        "Entry Price": f"₹{a['entry_price']:,.2f}",
                        "Current Price": f"₹{a['current_price']:,.2f}",
                        "Actual Return": f"{a['actual_return_pct']:+.2f}%",
                        "Alpha vs NIFTY": f"{a['alpha_vs_nifty']:+.2f}%",
                        "Outcome": a["outcome"],
                    })
                df_audit = pd.DataFrame(rows)
                st.dataframe(df_audit, use_container_width=True, hide_index=True)


def render_fii_dii_panel(symbol_display: str, inst_profile: dict):
    """Render the FII/DII institutional holding panel for the analysed stock."""
    if not inst_profile:
        return

    st.markdown(f'<div class="section-header">FII & DII Institutional Activity — {symbol_display} {info_icon("FII")} {info_icon("DII")}</div>', unsafe_allow_html=True)

    # ── ① Company-Specific FII & DII Ownership Profile ────────────────────────
    fii_pct = inst_profile.get("fii_holding_pct", 0.0)
    dii_pct = inst_profile.get("dii_holding_pct", 0.0)
    promoter_pct = inst_profile.get("promoter_holding_pct", 0.0) or inst_profile.get("insider_holding_pct", 0.0)
    retail_pct = inst_profile.get("retail_public_pct", 0.0)
    fii_chg = inst_profile.get("fii_change_qoq", 0.0)
    dii_chg = inst_profile.get("dii_change_qoq", 0.0)
    bias = inst_profile.get("institutional_bias", "MODERATE_INSTITUTIONAL_SUPPORT")
    support_score = inst_profile.get("support_score", 0.5)
    total_inst = round(fii_pct + dii_pct, 1)

    st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="inst-panel-title">🏢 {symbol_display} — Shareholding Pattern (FII vs DII Breakdown)</div>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        fii_chg_color = "#3fb950" if fii_chg >= 0 else "#f85149"
        fii_trend_label = f"{fii_chg:+.2f}% QoQ"
        st.markdown(f"""<div class="inst-metric">
            <div class="inst-metric-label">FII Holding (Foreign) {info_icon("FII Holding")}</div>
            <div class="inst-metric-value blue">{fii_pct:.1f}%</div>
            <div style="font-size:11px; margin-top:4px; color:{fii_chg_color}; font-weight:600;">{fii_trend_label}</div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        dii_chg_color = "#3fb950" if dii_chg >= 0 else "#f85149"
        dii_trend_label = f"{dii_chg:+.2f}% QoQ"
        st.markdown(f"""<div class="inst-metric">
            <div class="inst-metric-label">DII Holding (Domestic) {info_icon("DII Holding")}</div>
            <div class="inst-metric-value green">{dii_pct:.1f}%</div>
            <div style="font-size:11px; margin-top:4px; color:{dii_chg_color}; font-weight:600;">{dii_trend_label}</div>
        </div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""<div class="inst-metric">
            <div class="inst-metric-label">Promoter / Founders {info_icon("Promoter Holding")}</div>
            <div class="inst-metric-value yellow">{promoter_pct:.1f}%</div>
            <div style="font-size:11px; margin-top:4px; color:#8b949e;">Management Stake</div>
        </div>""", unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""<div class="inst-metric">
            <div class="inst-metric-label">Retail & Public {info_icon("Retail / Public Holding")}</div>
            <div class="inst-metric-value" style="color:#d2a8ff;">{retail_pct:.1f}%</div>
            <div style="font-size:11px; margin-top:4px; color:#8b949e;">Public Float</div>
        </div>""", unsafe_allow_html=True)

    # Visual horizontal shareholding distribution bar
    st.markdown(f"""
    <div style="margin: 14px 0 8px 0; background:#161b22; padding:12px 16px; border-radius:8px; border:1px solid #30363d;">
        <div style="display:flex; justify-content:space-between; font-size:12px; color:#8b949e; margin-bottom:6px;">
            <span>👑 Promoters: <b>{promoter_pct:.1f}%</b></span>
            <span>🌐 FIIs: <b>{fii_pct:.1f}%</b></span>
            <span>🏛️ DIIs: <b>{dii_pct:.1f}%</b></span>
            <span>👥 Retail: <b>{retail_pct:.1f}%</b></span>
        </div>
        <div style="display:flex; height:12px; border-radius:6px; overflow:hidden; background:#21262d;">
            <div style="width:{promoter_pct}%; background:#d29922;" title="Promoters: {promoter_pct:.1f}%"></div>
            <div style="width:{fii_pct}%; background:#58a6ff;" title="Foreign Institutions (FII): {fii_pct:.1f}%"></div>
            <div style="width:{dii_pct}%; background:#3fb950;" title="Domestic Institutions (DII): {dii_pct:.1f}%"></div>
            <div style="width:{retail_pct}%; background:#bc8cff;" title="Retail / Public: {retail_pct:.1f}%"></div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
            <small style="color:#58a6ff;">🛡️ <b>Total Institutional Holding (FII + DII): {total_inst:.1f}%</b></small>
            <small style="color:{'#3fb950' if support_score > 0.6 else '#d29922'}; font-weight:600;">{bias.replace('_', ' ').title()}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── ② Market-Wide Daily FII / DII Cash Flows ──────────────────────────────
    daily_flows = institutional_fetcher.get_daily_flows()
    fii_net = daily_flows.get("fii_net_cr", 0.0)
    dii_net = daily_flows.get("dii_net_cr", 0.0)
    total_net = daily_flows.get("total_net_cr", 0.0)
    flow_sentiment = daily_flows.get("sentiment", "NEUTRAL")
    flow_summary = daily_flows.get("summary", "")

    st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="inst-panel-title">🏦 Market-Level FII & DII Cash Market Flows (Today) {info_icon("FII/DII Net Flow")}</div>', unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        fii_color = "green" if fii_net >= 0 else "red"
        st.markdown(f"""<div class="inst-metric">
            <div class="inst-metric-label">Market FII Net {info_icon("FII")}</div>
            <div class="inst-metric-value {fii_color}">₹{fii_net:+,.0f} Cr</div>
        </div>""", unsafe_allow_html=True)
    with fc2:
        dii_color = "green" if dii_net >= 0 else "red"
        st.markdown(f"""<div class="inst-metric">
            <div class="inst-metric-label">Market DII Net {info_icon("DII")}</div>
            <div class="inst-metric-value {dii_color}">₹{dii_net:+,.0f} Cr</div>
        </div>""", unsafe_allow_html=True)
    with fc3:
        total_color = "green" if total_net >= 0 else "red"
        st.markdown(f"""<div class="inst-metric">
            <div class="inst-metric-label">Total Net Flow</div>
            <div class="inst-metric-value {total_color}">₹{total_net:+,.0f} Cr</div>
        </div>""", unsafe_allow_html=True)
    with fc4:
        sent_color = "green" if "BULLISH" in flow_sentiment else ("red" if "BEARISH" in flow_sentiment else "yellow")
        st.markdown(f"""<div class="inst-metric">
            <div class="inst-metric-label">Institutional Sentiment</div>
            <div class="inst-metric-value {sent_color}" style="font-size:16px;">{flow_sentiment}</div>
        </div>""", unsafe_allow_html=True)

    if flow_summary:
        st.info(f"💡 {flow_summary}")

    # Buy/Sell breakdown from daily flows
    fii_buy = daily_flows.get("fii_buy_cr", 0)
    fii_sell = daily_flows.get("fii_sell_cr", 0)
    dii_buy = daily_flows.get("dii_buy_cr", 0)
    dii_sell = daily_flows.get("dii_sell_cr", 0)

    with st.expander("📈 Detailed Buy/Sell Breakdown (Market-Wide)"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**FII Activity {info_icon('FII')}**", unsafe_allow_html=True)
            st.markdown(f"- 🟢 Buy: ₹{fii_buy:,.0f} Cr")
            st.markdown(f"- 🔴 Sell: ₹{fii_sell:,.0f} Cr")
            st.markdown(f"- **Net: ₹{fii_net:+,.0f} Cr**")
        with d2:
            st.markdown(f"**DII Activity {info_icon('DII')}**", unsafe_allow_html=True)
            st.markdown(f"- 🟢 Buy: ₹{dii_buy:,.0f} Cr")
            st.markdown(f"- 🔴 Sell: ₹{dii_sell:,.0f} Cr")
            st.markdown(f"- **Net: ₹{dii_net:+,.0f} Cr**")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar — Ticker Search (① )  — Full NSE Universe with Search
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 NSE Stock Advisor")
    st.markdown("<small>AI Multi-Agent System | Indian Equity Market</small>", unsafe_allow_html=True)
    st.divider()

    # ── Universe selector ──────────────────────────────────────────────────
    universe_choice = st.selectbox(
        "📋 Stock Universe",
        options=[
            "NIFTY 50 (50 Bluechips)",
            "NIFTY Next 50 (50 Large Caps)",
            "Midcap 100 (100 Midcaps)",
            "Popular Stocks (Top Traded)",
            "All NSE Stocks (2,580+ Listed Companies)",
        ],
        index=0,
        help="Choose which set of stocks to browse. You can search across every company listed on NSE below."
    )

    # Build stock list based on selected universe
    if "NIFTY 50" in universe_choice:
        universe_list = STOCK_UNIVERSE.NIFTY_50
    elif "Next 50" in universe_choice:
        universe_list = STOCK_UNIVERSE.NIFTY_NEXT_50
    elif "Midcap" in universe_choice:
        universe_list = STOCK_UNIVERSE.NIFTY_MIDCAP
    elif "Popular" in universe_choice:
        universe_list = STOCK_UNIVERSE.POPULAR_STOCKS
    else:
        universe_list = STOCK_UNIVERSE.get_all_symbols()

    # Build display options: "SYMBOL — Company Name (Sector)"
    display_options = []
    for sym in universe_list:
        display_name = sym.replace(".NS", "")
        company = STOCK_UNIVERSE.get_company_name(sym)
        sector = STOCK_UNIVERSE.get_sector(sym)
        if company != display_name:
            display_options.append(f"{display_name} — {company}")
        else:
            display_options.append(display_name)

    selected_display_full = st.selectbox(
        f"Select Stock ({len(universe_list):,} stocks)",
        options=display_options,
        index=0,
        help="Pick from the selected universe. Choose 'All NSE Stocks' for the complete exchange directory."
    )

    # Extract ticker from display string
    selected_display = selected_display_full.split(" — ")[0].strip()

    # Show sector badge for selected stock
    selected_sym = f"{selected_display}.NS"
    selected_sector = STOCK_UNIVERSE.get_sector(selected_sym)
    selected_company = STOCK_UNIVERSE.get_company_name(selected_sym)
    if selected_sector != "Other":
        st.markdown(f'<span class="sector-badge">{selected_sector}</span>', unsafe_allow_html=True)

    st.divider()

    custom_input = st.text_input(
        "🔍 Search every company in NSE",
        placeholder="e.g. Reliance, ZOMATO, Tata Motors, MRF",
        help="Type any company name or ticker to search all 2,580+ NSE listed companies"
    )

    # Smart search results
    if custom_input.strip():
        search_results = STOCK_UNIVERSE.search_stocks(custom_input.strip())
        if search_results:
            st.markdown(f"<small style='color:#3fb950;'>Found {len(search_results)} companies:</small>", unsafe_allow_html=True)
            for res in search_results[:8]:  # Show top 8
                st.markdown(
                    f"<small>• <b>{res['display']}</b> — {res['company_name']} "
                    f"<span class='sector-badge'>{res['sector']}</span></small>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<small style='color:#d29922;'>No local matches found. The ticker will be looked up directly on NSE via live feed.</small>", unsafe_allow_html=True)

    symbol_display = custom_input.strip().upper() if custom_input.strip() else selected_display
    symbol_full    = f"{symbol_display}.NS"

    # Update company info for custom input
    if custom_input.strip():
        selected_company = STOCK_UNIVERSE.get_company_name(symbol_full)
        selected_sector = STOCK_UNIVERSE.get_sector(symbol_full)

    run_btn = st.button("🔍 Analyse Stock", use_container_width=True, type="primary")

    st.divider()
    st.markdown(f"""
    <small>
    <b>Analysing:</b> {symbol_display}<br>
    <b>Company:</b> {selected_company}<br>
    <b>Sector:</b> {selected_sector}<br>
    <br>
    <b>Architecture:</b><br>
    • XGBoost <code>multi:softprob</code><br>
    • Walk-Forward TSS (n=5)<br>
    • Google RSS + FinBERT<br>
    • DuckDB / Parquet storage<br>
    • Risk Agent veto (VIX/EMA)<br>
    • FII/DII Institutional Tracker
    </small>
    """, unsafe_allow_html=True)

    # ── Stock count ────────────────────────────────────────────────────────
    total_stocks = len(STOCK_UNIVERSE.get_all_symbols())
    st.markdown(f"<small style='color:#8b949e;'>📊 {total_stocks} NSE stocks in database</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main content
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"# NSE Stock Advisor")

# ── Market Pulse & Profit/Loss Notification Bar ──────────────────────────────
render_market_pulse_banner()

# ── ② Macro Banner ─────────────────────────────────────────────────────────────
macro = get_macro_data()
c1, c2, c3 = st.columns(3)

with c1:
    nifty_val  = f"₹{macro['nifty']:,.0f}" if macro["nifty"] else "—"
    nifty_chg  = macro.get("nifty_chg")
    chg_color  = "green" if (nifty_chg or 0) >= 0 else "red"
    chg_str    = f"{nifty_chg:+.2f}%" if nifty_chg is not None else "—"
    st.markdown(f"""<div class="macro-card">
        <div class="macro-label">NIFTY 50 {info_icon("NIFTY 50")}</div>
        <div class="macro-value blue">{nifty_val}</div>
        <div class="macro-change {chg_color}">{chg_str} (1d)</div>
    </div>""", unsafe_allow_html=True)

with c2:
    vix_val = macro.get("vix")
    vix_str = f"{vix_val:.2f}" if vix_val else "—"
    vix_color = "red" if (vix_val or 0) > 22 else "green"
    vix_label = "🔴 ELEVATED" if (vix_val or 0) > 22 else "🟢 Normal"
    st.markdown(f"""<div class="macro-card">
        <div class="macro-label">India VIX {info_icon("India VIX")}</div>
        <div class="macro-value {vix_color}">{vix_str}</div>
        <div class="macro-change {vix_color}">{vix_label}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    # Show current price if analysis has been run, otherwise show ticker name
    if "last_result" in st.session_state and "error" not in st.session_state.get("last_result", {}):
        _res = st.session_state["last_result"]
        _shared = _res.get("shared", {})
        _quote = _shared.get("latest_quote", {})
        _cur_price = _quote.get("price")
        _prev_close = _quote.get("previous_close")
        _price_type = _quote.get("price_type", "UNAVAILABLE")

        if _cur_price is not None:
            _price_str = f"₹{_cur_price:,.2f}"
            if _prev_close is not None and _prev_close > 0:
                _day_chg = _cur_price - _prev_close
                _day_chg_pct = (_day_chg / _prev_close) * 100
                _p_color = "green" if _day_chg >= 0 else "red"
                _chg_line = f'{_day_chg:+.2f} ({_day_chg_pct:+.2f}%)'
            else:
                _p_color = "blue"
                _chg_line = _price_type.replace("_", " ").title()
            _label = "Live Price" if _price_type == "LIVE" else "Prev Close"
            st.markdown(f"""<div class="macro-card">
                <div class="macro-label">{symbol_display} — {_label}</div>
                <div class="macro-value {_p_color}">{_price_str}</div>
                <div class="macro-change {_p_color}">{_chg_line}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="macro-card">
                <div class="macro-label">Analysing</div>
                <div class="macro-value blue">{symbol_display}</div>
                <div class="macro-change" style="color:#8b949e">Price unavailable</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="macro-card">
            <div class="macro-label">Analysing</div>
            <div class="macro-value blue">{symbol_display}</div>
            <div class="macro-change" style="color:#8b949e">NSE / {symbol_full}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Daily AI Recommendations & Reality Verification Hub ─────────────────────
render_daily_picks_and_verification_hub()

st.markdown("<br>", unsafe_allow_html=True)

# ── Analysis trigger ───────────────────────────────────────────────────────────
if run_btn or "last_result" in st.session_state:
    if run_btn:
        with st.spinner(f"Running 7-agent analysis for {symbol_display}…"):
            result = run_analysis(symbol_full)
        st.session_state["last_result"] = result
        st.session_state["last_symbol"] = symbol_display
    else:
        result = st.session_state["last_result"]

    if "error" in result:
        st.error(f"Analysis failed: {result['error']}")
        st.stop()

    decision = result["decision"]
    agents   = result["agents"]
    rec      = decision.get("recommendation", "HOLD")

    # ── Company Info Bar ───────────────────────────────────────────────────
    _company_name = STOCK_UNIVERSE.get_company_name(symbol_full)
    _sector_name = STOCK_UNIVERSE.get_sector(symbol_full)
    if _company_name != symbol_display or _sector_name != "Other":
        st.markdown(f"""<div style="margin-bottom:16px;">
            <span style="font-size:18px; font-weight:700; color:#e6edf3;">{_company_name}</span>
            <span class="sector-badge" style="margin-left:12px;">{_sector_name}</span>
        </div>""", unsafe_allow_html=True)

    # ── ③ Central Decision Badge ───────────────────────────────────────────────
    badge_class = {"BUY": "badge-buy", "SELL": "badge-sell"}.get(rec, "badge-hold")
    badge_icon  = {"BUY": "🟢", "SELL": "🔴"}.get(rec, "🟡")
    conf_pct    = decision.get("confidence", 0)
    exp_ret     = decision.get("expected_return_5d", 0)

    col_badge, col_meta = st.columns([1, 2])
    with col_badge:
        st.markdown(f"""<div class="{badge_class}">
            <div class="badge-label">Recommendation</div>
            <div class="badge-signal">{badge_icon} {rec}</div>
            <div class="badge-conf">Confidence {info_icon("Confidence")}: {conf_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with col_meta:
        # Current price row
        _shared = result.get("shared", {})
        _quote = _shared.get("latest_quote", {})
        _cp = _quote.get("price")
        _pc = _quote.get("previous_close")
        _pt = _quote.get("price_type", "")
        if _cp is not None:
            _cp_str = f"₹{_cp:,.2f}"
            if _pc is not None and _pc > 0:
                _dc = _cp - _pc
                _dcp = (_dc / _pc) * 100
                _cc = "green" if _dc >= 0 else "red"
                _dc_str = f"{_dc:+.2f} ({_dcp:+.2f}%)"
            else:
                _cc = "blue"
                _dc_str = _pt.replace("_", " ").title()
            _pt_label = "Live Price" if _pt == "LIVE" else "Previous Close"
            st.markdown(f"""<div class="info-card" style="margin-bottom:12px;">
                <div class="info-card-label">{_pt_label}</div>
                <div class="info-card-value {_cc}" style="font-size:28px;">{_cp_str}
                    <span style="font-size:14px; margin-left:8px;">{_dc_str}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""<div class="info-card">
                <div class="info-card-label">Expected 5d Return {info_icon("Expected 5d Return")}</div>
                <div class="info-card-value {'green' if exp_ret >= 0 else 'red'}">{exp_ret:+.2f}%</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="info-card">
                <div class="info-card-label">Technical Score {info_icon("Technical Score")}</div>
                <div class="info-card-value blue">{decision.get('technical_score', 50):.0f}/100</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="info-card">
                <div class="info-card-label">Fundamental Score {info_icon("Fundamental Score")}</div>
                <div class="info-card-value blue">{decision.get('fundamental_score', 50):.0f}/100</div>
            </div>""", unsafe_allow_html=True)
            regime_str = decision.get("market_regime", {}).get("regime", "—")
            st.markdown(f"""<div class="info-card">
                <div class="info-card-label">Market Regime {info_icon("Market Regime")}</div>
                <div class="info-card-value">{regime_str}</div>
            </div>""", unsafe_allow_html=True)

    # ── ⑦ FII / DII Panel ─────────────────────────────────────────────────────
    inst_profile = result.get("institutional_profile", {})
    render_fii_dii_panel(symbol_display, inst_profile)

    # ── ④ Probability Bars ─────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">ML 3-Class Probabilities {info_icon("ML Prediction")}</div>', unsafe_allow_html=True)
    ml_d  = decision.get("ml_prediction", {})
    p_buy  = ml_d.get("p_buy",  0.33)
    p_hold = ml_d.get("p_hold", 0.34)
    p_sell = ml_d.get("p_sell", 0.33)

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown(f"**🟢 BUY {info_icon('P(BUY)')}** — {p_buy:.1%}", unsafe_allow_html=True)
        st.progress(float(p_buy))
    with pc2:
        st.markdown(f"**🟡 HOLD {info_icon('P(HOLD)')}** — {p_hold:.1%}", unsafe_allow_html=True)
        st.progress(float(p_hold))
    with pc3:
        st.markdown(f"**🔴 SELL {info_icon('P(SELL)')}** — {p_sell:.1%}", unsafe_allow_html=True)
        st.progress(float(p_sell))

    # ── ④b Interactive Price Charts ────────────────────────────────────────────
    st.markdown(f'<div class="section-header">Price & Technical Charts {info_icon("Candlestick")}</div>', unsafe_allow_html=True)

    shared_data = result.get("shared", {})
    chart_df = shared_data.get("price_with_indicators")

    if chart_df is not None and len(chart_df) > 0:
        # Use last 90 trading days
        chart_df = chart_df.tail(90).copy()
        chart_df.index = pd.to_datetime(chart_df.index)

        # ── Chart 1: Candlestick + Moving Averages + Bollinger Bands ──────
        fig_price = go.Figure()

        # Bollinger Bands shaded area
        if "BB_upper" in chart_df.columns and "BB_lower" in chart_df.columns:
            fig_price.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["BB_upper"],
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig_price.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["BB_lower"],
                fill="tonexty", fillcolor="rgba(88,166,255,0.08)",
                line=dict(width=0), name="Bollinger Bands",
            ))

        # Candlestick
        fig_price.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"], high=chart_df["High"],
            low=chart_df["Low"], close=chart_df["Close"],
            name="OHLC",
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950", decreasing_fillcolor="#f85149",
        ))

        # Moving averages
        if "SMA_20" in chart_df.columns:
            fig_price.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["SMA_20"],
                line=dict(color="#d29922", width=1.5), name="SMA 20",
            ))
        if "SMA_50" in chart_df.columns:
            fig_price.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["SMA_50"],
                line=dict(color="#f778ba", width=1.5), name="SMA 50",
            ))
        if "EMA_200" in chart_df.columns:
            fig_price.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["EMA_200"],
                line=dict(color="#58a6ff", width=2, dash="dot"), name="EMA 200",
            ))

        fig_price.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            height=420, margin=dict(l=10, r=10, t=40, b=10),
            title=dict(text=f"{symbol_display} — Price & Moving Averages (90d)", font=dict(size=14, color="#e6edf3")),
            xaxis=dict(rangeslider=dict(visible=False), gridcolor="#21262d"),
            yaxis=dict(title="Price (₹)", gridcolor="#21262d"),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=11)),
            xaxis_rangebreaks=[dict(bounds=["sat", "mon"])],
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # ── Charts row: Volume | RSI ──────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Volume chart
            colors_vol = ["#3fb950" if c >= o else "#f85149"
                          for c, o in zip(chart_df["Close"], chart_df["Open"])]
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=chart_df.index, y=chart_df["Volume"],
                marker_color=colors_vol, name="Volume", opacity=0.8,
            ))
            if "Volume_SMA_20" in chart_df.columns:
                fig_vol.add_trace(go.Scatter(
                    x=chart_df.index, y=chart_df["Volume_SMA_20"],
                    line=dict(color="#d29922", width=1.5), name="Vol SMA 20",
                ))
            fig_vol.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                height=260, margin=dict(l=10, r=10, t=40, b=10),
                title=dict(text="Volume", font=dict(size=13, color="#e6edf3")),
                yaxis=dict(gridcolor="#21262d"),
                xaxis=dict(gridcolor="#21262d"),
                xaxis_rangebreaks=[dict(bounds=["sat", "mon"])],
                legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center", font=dict(size=10)),
                showlegend=True,
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        with chart_col2:
            # RSI chart
            fig_rsi = go.Figure()
            if "RSI_14" in chart_df.columns:
                fig_rsi.add_trace(go.Scatter(
                    x=chart_df.index, y=chart_df["RSI_14"],
                    line=dict(color="#58a6ff", width=2), name="RSI (14)",
                    fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
                ))
            # Reference lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f85149", opacity=0.6,
                              annotation_text="Overbought (70)", annotation_font_color="#f85149")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#3fb950", opacity=0.6,
                              annotation_text="Oversold (30)", annotation_font_color="#3fb950")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="#8b949e", opacity=0.3)
            fig_rsi.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                height=260, margin=dict(l=10, r=10, t=40, b=10),
                title=dict(text="RSI (14)", font=dict(size=13, color="#e6edf3")),
                yaxis=dict(range=[0, 100], gridcolor="#21262d"),
                xaxis=dict(gridcolor="#21262d"),
                xaxis_rangebreaks=[dict(bounds=["sat", "mon"])],
                showlegend=False,
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

        # ── MACD chart (full width) ───────────────────────────────────────
        if "MACD" in chart_df.columns and "MACD_signal" in chart_df.columns:
            fig_macd = go.Figure()

            # Histogram
            hist_vals = chart_df["MACD"] - chart_df["MACD_signal"]
            colors_macd = ["#3fb950" if v >= 0 else "#f85149" for v in hist_vals]
            fig_macd.add_trace(go.Bar(
                x=chart_df.index, y=hist_vals,
                marker_color=colors_macd, name="Histogram", opacity=0.5,
            ))
            fig_macd.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["MACD"],
                line=dict(color="#58a6ff", width=2), name="MACD",
            ))
            fig_macd.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["MACD_signal"],
                line=dict(color="#f778ba", width=1.5, dash="dash"), name="Signal",
            ))
            fig_macd.add_hline(y=0, line_color="#8b949e", opacity=0.3)
            fig_macd.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                height=260, margin=dict(l=10, r=10, t=40, b=10),
                title=dict(text="MACD", font=dict(size=13, color="#e6edf3")),
                yaxis=dict(gridcolor="#21262d"),
                xaxis=dict(gridcolor="#21262d"),
                xaxis_rangebreaks=[dict(bounds=["sat", "mon"])],
                legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center", font=dict(size=10)),
            )
            st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.caption("Chart data unavailable.")

    # ── ⑤ Risk Panel ──────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">Risk Dashboard {info_icon("Risk Level")}</div>', unsafe_allow_html=True)

    risk_d = decision.get("risk", {})
    if risk_d.get("veto_active"):
        veto_msg = f"⛔ RISK VETO ACTIVE {info_icon('Veto')}"
        if risk_d.get("veto_triggered"):
            veto_msg += " — BUY signal downgraded to HOLD"
        st.markdown(f'<div class="veto-active">{veto_msg}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        sl = risk_d.get("stop_loss_price", "—")
        st.metric("Stop-Loss Price", f"₹{sl}" if sl else "—", help=GLOSSARY.get("Stop-Loss", ""))
    with r2:
        tp = risk_d.get("target_price", "—")
        st.metric("Target Price", f"₹{tp}" if tp else "—", help=GLOSSARY.get("Target Price", ""))
    with r3:
        rrr = risk_d.get("rrr", "—")
        st.metric("Risk-to-Reward", f"{rrr}:1" if rrr else "—", help=GLOSSARY.get("Risk-to-Reward", ""))
    with r4:
        alloc = risk_d.get("max_allocation_pct", "—")
        st.metric("Max Allocation", f"{alloc}%" if alloc else "—", help=GLOSSARY.get("Max Allocation", ""))

    risk_lvl_color = {"High": "red", "Medium": "yellow", "Low": "green"}.get(risk_d.get("level", "Medium"), "blue")
    st.markdown(
        f"**Risk Level {info_icon('Risk Level')}:** <span class='{risk_lvl_color}'>{risk_d.get('level', '—')}</span> "
        f"(Score: {risk_d.get('score', '—')}/100)",
        unsafe_allow_html=True
    )
    if risk_d.get("factors"):
        with st.expander("⚠ Risk Factors"):
            for f in risk_d["factors"]:
                st.markdown(f"• {f}")

    # ── ⑥ Agent Breakdown Tabs ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Agent Breakdown</div>', unsafe_allow_html=True)

    tab_tech, tab_fund, tab_ml, tab_news, tab_regime = st.tabs(
        ["📊 Technical", "📋 Fundamental", "🤖 ML Model", "📰 News", "🌐 Regime"]
    )

    with tab_tech:
        tech_out = agents.get("technical", {})
        tech_det = tech_out.get("details", {})
        ts       = tech_det.get("technical_summary", {})
        st.metric("Technical Score", f"{tech_det.get('technical_score', 50):.0f}/100", help=GLOSSARY.get("Technical Score", ""))
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**RSI (14) {info_icon('RSI')}:** {ts.get('RSI', '—')} — {ts.get('RSI_signal', '—')}", unsafe_allow_html=True)
            st.markdown(f"**MACD {info_icon('MACD')}:** {ts.get('MACD_signal', '—')} ({ts.get('MACD_trend', '—')})", unsafe_allow_html=True)
            st.markdown(f"**Trend (20d) {info_icon('SMA')}:** {ts.get('Trend_20', '—')}", unsafe_allow_html=True)
            st.markdown(f"**Trend (50d) {info_icon('SMA')}:** {ts.get('Trend_50', '—')}", unsafe_allow_html=True)
            st.markdown(f"**Trend (200 EMA) {info_icon('EMA')}:** {ts.get('Trend_200', '—')}", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"**Golden Cross {info_icon('Golden Cross')}:** {'✓ Yes' if ts.get('Golden_Cross') else '✗ No'}", unsafe_allow_html=True)
            st.markdown(f"**BB Position {info_icon('Bollinger Bands')}:** {ts.get('BB_position', '—')}", unsafe_allow_html=True)
            st.markdown(f"**Volume Signal {info_icon('Volume')}:** {ts.get('Volume_signal', '—')}", unsafe_allow_html=True)
            st.markdown(f"**Volatility (20d) {info_icon('Volatility')}:** {ts.get('Volatility_level', '—')}", unsafe_allow_html=True)
        st.markdown("**Reasoning:**")
        for r in tech_out.get("reasoning", []):
            st.markdown(f"  {r}")

    with tab_fund:
        fund_out = agents.get("fundamental", {})
        fund_det = fund_out.get("details", {})
        raw      = fund_det.get("raw_fundamentals", {})
        st.metric("Fundamental Score", f"{fund_det.get('fundamental_score', 50):.0f}/100", help=GLOSSARY.get("Fundamental Score", ""))
        if raw:
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown(f"**P/E Ratio {info_icon('P/E Ratio')}:** {raw.get('pe_ratio', '—')}", unsafe_allow_html=True)
                st.markdown(f"**ROE {info_icon('ROE')}:** {raw.get('roe', 0):.1%}" if raw.get('roe') else f"**ROE {info_icon('ROE')}:** —", unsafe_allow_html=True)
                st.markdown(f"**Revenue Growth {info_icon('Revenue Growth')}:** {raw.get('revenue_growth', 0):.1%}" if raw.get('revenue_growth') else f"**Revenue Growth {info_icon('Revenue Growth')}:** —", unsafe_allow_html=True)
                st.markdown(f"**EPS {info_icon('EPS')}:** ₹{raw.get('eps', '—')}", unsafe_allow_html=True)
            with fc2:
                st.markdown(f"**D/E Ratio {info_icon('D/E Ratio')}:** {raw.get('debt_to_equity', '—')}", unsafe_allow_html=True)
                st.markdown(f"**Profit Margin {info_icon('Profit Margin')}:** {raw.get('profit_margins', 0):.1%}" if raw.get('profit_margins') else f"**Profit Margin {info_icon('Profit Margin')}:** —", unsafe_allow_html=True)

                # ── Dividend Yield: (Dividend Per Share / Stock Price) × 100 ─────────────
                dps = float(raw.get('dividend_per_share') or 0.0)
                _quote_for_div = result.get("shared", {}).get("latest_quote", {})
                _stock_price = float(_quote_for_div.get("price") or raw.get('stock_price') or 0.0)
                div_yield_pct = raw.get('dividend_yield_pct')

                if dps > 0 and _stock_price > 0:
                    calculated_div_yield = (dps / _stock_price) * 100.0
                elif div_yield_pct is not None and div_yield_pct > 0:
                    calculated_div_yield = float(div_yield_pct)
                    if _stock_price > 0 and dps <= 0:
                        dps = (calculated_div_yield / 100.0) * _stock_price
                elif raw.get('dividend_yield'):
                    raw_dy = float(raw.get('dividend_yield'))
                    calculated_div_yield = raw_dy * 100.0 if raw_dy < 0.20 else raw_dy
                    if _stock_price > 0 and dps <= 0:
                        dps = (calculated_div_yield / 100.0) * _stock_price
                else:
                    calculated_div_yield = 0.0

                if calculated_div_yield > 0 and _stock_price > 0:
                    st.markdown(
                        f"**Dividend Per Share (DPS) {info_icon('Dividend Per Share')}:** ₹{dps:.2f}",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"**Dividend Yield {info_icon('Dividend Yield')}:** <span class='green' style='font-weight:700;'>{calculated_div_yield:.2f}%</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='background:#161b22; border:1px solid #30363d; border-radius:6px; padding:6px 10px; margin:4px 0 10px 0; font-size:11px; color:#8b949e;'>"
                        f"📐 <b>Formula:</b> (DPS ÷ Stock Price) × 100 = (₹{dps:.2f} ÷ ₹{_stock_price:,.2f}) × 100 = <b>{calculated_div_yield:.2f}%</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"**Dividend Yield {info_icon('Dividend Yield')}:** 0.00% <small style='color:#8b949e;'>(Company reinvests profits / no regular cash dividend)</small>", unsafe_allow_html=True)

                st.markdown(f"**Market Cap {info_icon('Market Cap')}:** ₹{raw.get('market_cap', 0)/1e7:,.0f} Cr" if raw.get('market_cap') else f"**Market Cap {info_icon('Market Cap')}:** —", unsafe_allow_html=True)

        for r in fund_out.get("reasoning", []):
            st.markdown(f"  {r}")

    with tab_ml:
        ml_out = agents.get("ml", {})
        ml_det = ml_out.get("details", {})
        st.markdown(f"**Prediction {info_icon('ML Prediction')}:** {ml_det.get('prediction', '—')}", unsafe_allow_html=True)
        c_buy, c_hold, c_sell = st.columns(3)
        c_buy.metric("P(BUY)", f"{ml_det.get('p_buy',  0):.1%}", help=GLOSSARY.get("P(BUY)", ""))
        c_hold.metric("P(HOLD)", f"{ml_det.get('p_hold', 0):.1%}", help=GLOSSARY.get("P(HOLD)", ""))
        c_sell.metric("P(SELL)", f"{ml_det.get('p_sell', 0):.1%}", help=GLOSSARY.get("P(SELL)", ""))
        oof = ml_det.get("oof_metrics", {})
        if oof:
            st.markdown(f"**OOF macro-F1 {info_icon('OOF Metrics')} (Walk-Forward {info_icon('Walk-Forward')}, 5 folds):**", unsafe_allow_html=True)
            for model, f1 in oof.items():
                st.markdown(f"  • {model}: {f1:.3f}")
        for r in ml_out.get("reasoning", []):
            st.markdown(f"  {r}")
        if ml_det.get("fallback"):
            st.warning("Heuristic fallback — ML model training was skipped.")

    with tab_news:
        news_out = agents.get("news", {})
        news_det = news_out.get("details", {})
        agg      = news_det.get("aggregated_sentiment", {})
        st.info(f"📡 Live News — Google News RSS + FinBERT {info_icon('FinBERT')} | Not used in backtesting")
        if agg:
            n1, n2, n3 = st.columns(3)
            n1.metric("Overall", agg.get("overall_sentiment", "—").capitalize(), help=GLOSSARY.get("Sentiment", ""))
            n2.metric("Net Score", f"{agg.get('net_score', 0):+.3f}", help=GLOSSARY.get("Net Score", ""))
            n3.metric("Articles", agg.get("article_count", 0))
            st.progress(float(agg.get("avg_positive", 0)), text=f"P(positive): {agg.get('avg_positive', 0):.2%}")
            st.progress(float(agg.get("avg_negative", 0)), text=f"P(negative): {agg.get('avg_negative', 0):.2%}")

        articles = news_det.get("articles", [])
        if articles:
            st.markdown("**Top Articles:**")
            for art in articles[:5]:
                icon = {"positive": "🟢", "negative": "🔴"}.get(art.get("sentiment"), "🟡")
                st.markdown(f"{icon} [{art['title'][:90]}]({art.get('url', '#')}) — *{art.get('sentiment', '').capitalize()} ({art.get('score', 0):.2f})*")
        else:
            st.caption("No live articles retrieved.")

    with tab_regime:
        regime_out = agents.get("market_regime", {})
        regime_det = regime_out.get("details", {})
        st.markdown(f"**Regime {info_icon('Market Regime')}:** {regime_det.get('regime', '—')}", unsafe_allow_html=True)
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            st.markdown(f"**NIFTY Close {info_icon('NIFTY 50')}:** ₹{regime_det.get('nifty_close', 0):,.0f}", unsafe_allow_html=True)
            st.markdown(f"**NIFTY SMA 20 {info_icon('SMA')}:** ₹{regime_det.get('nifty_sma20', 0):,.0f}", unsafe_allow_html=True)
            st.markdown(f"**NIFTY SMA 50 {info_icon('SMA')}:** ₹{regime_det.get('nifty_sma50', 0):,.0f}", unsafe_allow_html=True)
        with col_reg2:
            st.markdown(f"**NIFTY 200 EMA {info_icon('EMA')}:** ₹{regime_det.get('nifty_ema200', 0):,.0f}", unsafe_allow_html=True)
            above = regime_det.get("above_200ema", False)
            st.markdown(f"**Above 200 EMA:** {'🟢 Yes' if above else '🔴 No'}")
            st.markdown(f"**India VIX {info_icon('India VIX')}:** {regime_det.get('vix', 0):.1f}", unsafe_allow_html=True)
        for r in regime_out.get("reasoning", []):
            st.markdown(f"  {r}")

    # ── ⑧ PDF Report Button ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Download Report</div>', unsafe_allow_html=True)
    col_pdf, col_ts = st.columns([2, 3])
    with col_pdf:
        with st.spinner("Generating PDF…"):
            pdf_bytes = generate_pdf_report(symbol_full, decision, agents, result.get("institutional_profile", {}))

        if pdf_bytes:
            filename = f"{symbol_display}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="📄 Download 1-Page PDF Report",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("PDF unavailable — install `reportlab` to enable.")

    with col_ts:
        st.caption(f"Analysis timestamp: {decision.get('timestamp', '—')}")
        st.caption(f"Backtest window: 2022-01-01 – 2024-12-31 | Models: LR, RF, XGBoost multi:softprob, LightGBM")

    # ── ⑨ Glossary / Education Section ─────────────────────────────────────────
    with st.expander("📚 Stock Market Glossary — What do these terms mean?"):
        st.markdown("Hover over any **ⓘ** icon throughout this dashboard for quick explanations. Below is the full glossary:")
        st.markdown("---")

        # Group glossary by category
        categories = {
            "📈 Market Terms": ["NIFTY 50", "India VIX", "Market Regime", "Breadth"],
            "📊 Technical Analysis": ["RSI", "MACD", "SMA", "EMA", "Bollinger Bands", "Golden Cross", "Volume", "Volatility", "Candlestick", "Technical Score", "Overbought", "Oversold"],
            "📋 Fundamental Analysis": ["P/E Ratio", "P/B Ratio", "ROE", "ROCE", "D/E Ratio", "Profit Margin", "Revenue Growth", "Earnings Growth", "EPS", "Market Cap", "Fundamental Score", "Current Ratio"],
            "💰 Dividends": ["Dividend Yield", "Dividend"],
            "🏦 FII / DII (Institutional)": ["FII", "DII", "FII/DII Net Flow", "Institutional Holding", "Insider Holding", "Institutional Bias"],
            "⚠️ Risk Management": ["Stop-Loss", "Target Price", "Risk-to-Reward", "Max Allocation", "Veto", "Risk Level", "Expected 5d Return"],
            "🤖 AI / ML Terms": ["ML Prediction", "P(BUY)", "P(HOLD)", "P(SELL)", "Confidence", "Walk-Forward", "XGBoost", "FinBERT", "OOF Metrics"],
            "📰 News & Sentiment": ["Sentiment", "Net Score"],
            "📅 52-Week": ["52W High", "52W Low"],
        }

        for cat_name, terms in categories.items():
            st.markdown(f"**{cat_name}**")
            for term in terms:
                explanation = GLOSSARY.get(term, "—")
                st.markdown(f"- **{term}:** {explanation}")
            st.markdown("")

else:
    # Default state — no analysis run yet
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color: #8b949e;">
        <div style="font-size: 64px; margin-bottom: 16px;">📊</div>
        <div style="font-size: 24px; font-weight: 600; color: #e6edf3;">
            AI-Powered Indian Stock Recommendation
        </div>
        <div style="font-size: 14px; margin-top: 12px; max-width: 600px; margin-inline: auto; line-height: 1.7;">
            Select from <b>500+ NSE stocks</b> (NIFTY 50, Next 50, Midcap, Popular) from the sidebar and click <b>Analyse Stock</b>.<br>
            The system runs 7 specialized agents, applies Walk-Forward ML with
            <b>XGBoost multi:softprob</b>, and shows calibrated
            <b>P(BUY)</b> / <b>P(HOLD)</b> / <b>P(SELL)</b> probabilities.<br><br>
            <b>New Features:</b><br>
            🏦 <b>FII & DII Data</b> — See institutional flows for every stock<br>
            ⓘ <b>Info Icons</b> — Hover over ⓘ for beginner-friendly explanations<br>
            📊 <b>500+ NSE Stocks</b> — Browse by sector or search by company name<br>
            💰 <b>Corrected Dividend Yield</b> — DPS ÷ Stock Price × 100
        </div>
    </div>
    """, unsafe_allow_html=True)
