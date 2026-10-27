"""
FastAPI Backend - Indian Stock Recommendation System
Research-grade API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import logging
from datetime import datetime

# Import our modules - support both direct and package imports
try:
    from data_pipeline.price_fetcher import fetcher
    from data_pipeline.indicators import TechnicalIndicators
    from data_pipeline.fundamental_fetcher import fundamental_analyzer
    from data_pipeline.news_fetcher import news_fetcher
    from nlp_layer.sentiment import sentiment_analyzer
    from agentic_layer.agents import (
        MarketDataAgent, TechnicalAnalysisAgent, NewsSentimentAgent,
        FundamentalAgent, RiskAgent, MarketRegimeAgent, MLPredictionAgent
    )
    from agentic_layer.decision_engine import decision_engine
    from backtesting.backtester import backtester
    from utils.config import STOCK_UNIVERSE
except ImportError:
    from backend.data_pipeline.price_fetcher import fetcher
    from backend.data_pipeline.indicators import TechnicalIndicators
    from backend.data_pipeline.fundamental_fetcher import fundamental_analyzer
    from backend.data_pipeline.news_fetcher import news_fetcher
    from backend.nlp_layer.sentiment import sentiment_analyzer
    from backend.agentic_layer.agents import (
        MarketDataAgent, TechnicalAnalysisAgent, NewsSentimentAgent,
        FundamentalAgent, RiskAgent, MarketRegimeAgent, MLPredictionAgent
    )
    from backend.agentic_layer.decision_engine import decision_engine
    from backend.backtesting.backtester import backtester
    from backend.utils.config import STOCK_UNIVERSE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Powered Indian Stock Recommendation System",
    description="Research-grade multi-agent system for NSE/BSE stock recommendations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents (singleton pattern for demo)
agents = {
    "market_data": MarketDataAgent(),
    "technical": TechnicalAnalysisAgent(),
    "news": NewsSentimentAgent(),
    "fundamental": FundamentalAgent(),
    "risk": RiskAgent(),
    "market_regime": MarketRegimeAgent(),
    "ml": MLPredictionAgent()
}

class StockRequest(BaseModel):
    symbol: str

class PortfolioRequest(BaseModel):
    capital: float = 100000
    risk_tolerance: str = "Medium"  # Low, Medium, High
    stocks: Optional[List[str]] = None

@app.get("/")
def root():
    return {
        "message": "AI-Powered Indian Stock Recommendation System",
        "version": "1.0.0",
        "market": "NSE/BSE",
        "agents": list(agents.keys()),
        "endpoints": ["/api/stocks", "/api/analyze/{symbol}", "/api/backtest/{symbol}", "/api/portfolio"]
    }

@app.get("/api/stocks")
def get_stocks():
    """Get list of supported Indian stocks (NIFTY 50)"""
    return {
        "count": len(STOCK_UNIVERSE.NIFTY_50),
        "stocks": [
            {
                "symbol": s,
                "display": s.replace('.NS', ''),
                "full_symbol": s
            } for s in STOCK_UNIVERSE.NIFTY_50
        ],
        "display_names": STOCK_UNIVERSE.DISPLAY_NAMES
    }

def _normalize_stock_symbol(symbol: str) -> str:
    clean_symbol = symbol.upper().strip()
    if not clean_symbol.endswith('.NS') and not clean_symbol.endswith('.BO'):
        return f"{clean_symbol}.NS"
    return clean_symbol

@app.get("/api/quote/{symbol}")
def get_stock_quote(symbol: str):
    """Return the latest live quote, or the most recent verified daily close."""
    full_symbol = _normalize_stock_symbol(symbol)
    quote = fetcher.fetch_latest_quote(full_symbol)
    return {"symbol": full_symbol, **quote}

@app.get("/api/history/{symbol}")
def get_stock_history(symbol: str):
    """Return recent OHLCV and technical data for the price chart."""
    full_symbol = _normalize_stock_symbol(symbol)
    price_data = fetcher.fetch_price_data(full_symbol, start="2018-01-01")
    price_with_indicators = TechnicalIndicators.add_all_indicators(price_data)
    recent = price_with_indicators.tail(90)

    def rounded_value(row, column, digits=2):
        value = row.get(column)
        return float(round(value, digits)) if pd.notna(value) else None

    history = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": rounded_value(row, "Open"),
            "high": rounded_value(row, "High"),
            "low": rounded_value(row, "Low"),
            "close": rounded_value(row, "Close"),
            "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else 0,
            "sma20": rounded_value(row, "SMA_20"),
            "sma50": rounded_value(row, "SMA_50"),
            "rsi14": rounded_value(row, "RSI_14"),
            "macd": rounded_value(row, "MACD", 4),
            "macd_signal": rounded_value(row, "MACD_signal", 4),
            "macd_histogram": rounded_value(row, "MACD_histogram", 4),
            "bb_upper": rounded_value(row, "BB_upper"),
            "bb_middle": rounded_value(row, "BB_middle"),
            "bb_lower": rounded_value(row, "BB_lower"),
        }
        for idx, row in recent.iterrows()
    ]
    return {"symbol": full_symbol, "price_history": history}

@app.get("/api/analyze/{symbol}")
def analyze_stock(symbol: str):
    """
    Full agentic analysis for a stock
    Symbol can be RELIANCE, TCS, etc. or RELIANCE.NS
    """
    try:
        # Normalize symbol
        clean_symbol = symbol.upper().strip()
        if not clean_symbol.endswith('.NS') and not clean_symbol.endswith('.BO'):
            # Check if it's in our universe
            full_symbol = f"{clean_symbol}.NS"
            if full_symbol not in STOCK_UNIVERSE.NIFTY_50:
                # Try as is
                full_symbol = clean_symbol if '.' in clean_symbol else f"{clean_symbol}.NS"
        else:
            full_symbol = clean_symbol
        
        logger.info(f"Analyzing {full_symbol}")
        
        # Step 1: Market Data Agent
        market_data_result = agents["market_data"].analyze(full_symbol, {})
        if market_data_result['signal'] == 'ERROR':
            raise HTTPException(status_code=500, detail=f"Data fetch failed: {market_data_result['details'].get('error')}")
        
        shared_data = market_data_result['details']
        
        # Step 2: Parallel agent analysis (sequential for demo, could be async)
        technical_result = agents["technical"].analyze(full_symbol, shared_data)
        news_result = agents["news"].analyze(full_symbol, shared_data)
        fundamental_result = agents["fundamental"].analyze(full_symbol, shared_data)
        risk_result = agents["risk"].analyze(full_symbol, shared_data)
        regime_result = agents["market_regime"].analyze(full_symbol, shared_data)
        ml_result = agents["ml"].analyze(full_symbol, shared_data)
        
        agent_outputs = {
            "technical": technical_result,
            "news": news_result,
            "fundamental": fundamental_result,
            "risk": risk_result,
            "market_regime": regime_result,
            "ml": ml_result,
            "market_data": market_data_result
        }
        
        # Step 3: Decision Engine
        final_decision = decision_engine.decide(agent_outputs, full_symbol)
        
        # Add price history for chart
        price_data = shared_data.get('price_data')
        price_history = []
        if price_data is not None and len(price_data) > 0:
            # Last 90 days for chart
            recent = price_data.tail(90)

            def rounded_value(row, column, digits=2):
                value = row.get(column)
                return float(round(value, digits)) if pd.notna(value) else None

            price_history = [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": rounded_value(row, "Open"),
                    "high": rounded_value(row, "High"),
                    "low": rounded_value(row, "Low"),
                    "close": rounded_value(row, "Close"),
                    "volume": int(row['Volume']),
                    "sma20": rounded_value(row, "SMA_20"),
                    "sma50": rounded_value(row, "SMA_50"),
                    "rsi14": rounded_value(row, "RSI_14"),
                    "macd": rounded_value(row, "MACD", 4),
                    "macd_signal": rounded_value(row, "MACD_signal", 4),
                    "macd_histogram": rounded_value(row, "MACD_histogram", 4),
                    "bb_upper": rounded_value(row, "BB_upper"),
                    "bb_middle": rounded_value(row, "BB_middle"),
                    "bb_lower": rounded_value(row, "BB_lower")
                }
                for idx, row in recent.iterrows()
            ]
        
        # Sanitize agent outputs to remove DataFrames and non-serializable
        def sanitize_agent_output(output):
            """Remove DataFrames and convert numpy types"""
            import numpy as np
            sanitized = {}
            for k, v in output.items():
                if k in ['price_data', 'price_with_indicators', 'market_data', 'vix_data']:
                    continue
                if isinstance(v, dict):
                    sanitized[k] = sanitize_agent_output(v)
                elif isinstance(v, (np.integer, np.floating)):
                    sanitized[k] = float(v)
                elif isinstance(v, np.ndarray):
                    sanitized[k] = v.tolist()
                elif isinstance(v, (list, tuple)):
                    # Sanitize list items
                    new_list = []
                    for item in v:
                        if isinstance(item, dict):
                            # Skip if contains DataFrame
                            if any(isinstance(x, pd.DataFrame) for x in item.values()):
                                continue
                            new_list.append(sanitize_agent_output(item) if isinstance(item, dict) else item)
                        elif not isinstance(item, pd.DataFrame):
                            if isinstance(item, (np.integer, np.floating)):
                                new_list.append(float(item))
                            else:
                                new_list.append(item)
                    sanitized[k] = new_list[:10]  # limit
                elif isinstance(v, pd.DataFrame):
                    continue
                else:
                    try:
                        # Try to check if JSON serializable
                        import json
                        json.dumps(v)
                        sanitized[k] = v
                    except:
                        # Convert to string
                        sanitized[k] = str(v)[:500]
            return sanitized
        
        sanitized_agents = {}
        for agent_name, agent_output in agent_outputs.items():
            sanitized_agents[agent_name] = sanitize_agent_output(agent_output)
        
        return {
            "symbol": full_symbol,
            "display_symbol": full_symbol.replace('.NS', ''),
            "recommendation": final_decision,
            "agents": sanitized_agents,
            "price_history": price_history,
            "current_price": shared_data.get('latest_price'),
            "previous_close": shared_data.get('latest_quote', {}).get('previous_close'),
            "price_type": shared_data.get('latest_quote', {}).get('price_type', 'UNAVAILABLE'),
            "price_as_of": shared_data.get('latest_quote', {}).get('as_of'),
            "price_source": shared_data.get('latest_quote', {}).get('source'),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/technical/{symbol}")
def get_technical_analysis(symbol: str):
    """Get only technical analysis"""
    try:
        clean_symbol = symbol.upper().strip()
        full_symbol = f"{clean_symbol}.NS" if not clean_symbol.endswith('.NS') else clean_symbol
        
        price_data = fetcher.fetch_price_data(full_symbol)
        price_with_ind = TechnicalIndicators.add_all_indicators(price_data)
        summary = TechnicalIndicators.get_technical_summary(price_with_ind)
        score = TechnicalIndicators.compute_technical_score(summary)
        
        return {
            "symbol": full_symbol,
            "technical_summary": summary,
            "technical_score": score,
            "price_history": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": round(row['Close'], 2),
                    "sma20": round(row.get('SMA_20', 0), 2),
                    "sma50": round(row.get('SMA_50', 0), 2),
                    "rsi": round(row.get('RSI_14', 50), 2)
                }
                for idx, row in price_with_ind.tail(90).iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest/{symbol}")
def backtest_stock(symbol: str, strategy: str = "technical"):
    """Backtest a stock"""
    try:
        clean_symbol = symbol.upper().strip()
        full_symbol = f"{clean_symbol}.NS" if not clean_symbol.endswith('.NS') else clean_symbol
        
        if strategy == "ml":
            result = backtester.backtest_with_ml(full_symbol)
        else:
            result = backtester.backtest_single_stock(full_symbol, strategy=strategy)
        
        if "error" in result and "equity_curve" not in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Convert equity curves to list for JSON
        equity_curve = result.get('equity_curve')
        equity_list = []
        if equity_curve is not None:
            equity_list = [
                {"date": idx.strftime("%Y-%m-%d"), "equity": round(val, 2)}
                for idx, val in equity_curve.tail(252).items()
            ]
        
        bh_equity = result.get('buy_hold_equity')
        bh_list = []
        if bh_equity is not None:
            bh_list = [
                {"date": idx.strftime("%Y-%m-%d"), "equity": round(val, 2)}
                for idx, val in bh_equity.tail(252).items()
            ]
        
        # Remove non-serializable
        result_copy = {k: v for k, v in result.items() if k not in ['equity_curve', 'buy_hold_equity', 'signals', 'test_equity', 'buy_hold_equity']}
        
        return {
            **result_copy,
            "equity_curve": equity_list,
            "benchmark_equity": bh_list,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ablation/{symbol}")
def ablation_study(symbol: str):
    """Research ablation study"""
    try:
        clean_symbol = symbol.upper().strip()
        full_symbol = f"{clean_symbol}.NS" if not clean_symbol.endswith('.NS') else clean_symbol
        
        result = backtester.ablation_study(full_symbol)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio")
def analyze_portfolio(request: PortfolioRequest):
    """Portfolio analysis - Phase 2"""
    try:
        stocks = request.stocks or STOCK_UNIVERSE.NIFTY_50[:10]
        capital = request.capital
        risk_tolerance = request.risk_tolerance
        
        results = []
        for symbol in stocks[:10]:  # limit for demo
            try:
                # Quick analysis
                price_data = fetcher.fetch_price_data(symbol)
                price_with_ind = TechnicalIndicators.add_all_indicators(price_data)
                summary = TechnicalIndicators.get_technical_summary(price_with_ind)
                score = TechnicalIndicators.compute_technical_score(summary)
                
                # Simple signal
                if score >= 65:
                    signal = "BUY"
                elif score <= 35:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                results.append({
                    "symbol": symbol,
                    "display": symbol.replace('.NS', ''),
                    "price": round(price_data['Close'].iloc[-1], 2),
                    "technical_score": score,
                    "signal": signal,
                    "rsi": summary.get('RSI', 50),
                    "trend": summary.get('Trend_50', 'Neutral')
                })
            except Exception as e:
                logger.warning(f"Portfolio analysis failed for {symbol}: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x['technical_score'], reverse=True)
        
        # Allocation based on risk tolerance
        if risk_tolerance == "Low":
            top_n = 5
            allocation = "Equal weight among top 5, 20% each, with 30% cash"
        elif risk_tolerance == "High":
            top_n = 3
            allocation = "Concentrated: 40%, 35%, 25% in top 3"
        else:
            top_n = 5
            allocation = "Diversified: 25%, 25%, 20%, 15%, 15% in top 5"
        
        buy_signals = [r for r in results if r['signal'] == 'BUY']
        
        return {
            "capital": capital,
            "risk_tolerance": risk_tolerance,
            "allocation_strategy": allocation,
            "stocks_analyzed": len(results),
            "buy_signals": len(buy_signals),
            "recommendations": results,
            "top_picks": results[:top_n],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/regime")
def get_market_regime():
    """Get current market regime"""
    try:
        market_data = fetcher.fetch_market_data("^NSEI")
        vix_data = fetcher.fetch_india_vix()
        
        agent = MarketRegimeAgent()
        result = agent.analyze("NIFTY", {"market_data": market_data, "vix_data": vix_data})
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
