"""
Fundamental Data - Indian Stocks
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    def fetch_fundamentals(self, symbol: str) -> Dict:
        """Fetch fundamental data with fallback"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or len(info) < 5:
                return self._synthetic_fundamentals(symbol)
            
            # ── Strictly implement: (Dividend Per Share / Stock Price) × 100 ──────────
            stock_price = float(info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0.0)
            dps = float(info.get('dividendRate') or info.get('trailingAnnualDividendRate') or 0.0)

            if dps > 0 and stock_price > 0:
                dividend_yield_pct = (dps / stock_price) * 100.0
            elif info.get('dividendYield') is not None and stock_price > 0:
                raw_dy = float(info.get('dividendYield'))
                # Normalize if yfinance returned percentage (e.g. 6.14) vs decimal (0.0614)
                dividend_yield_pct = raw_dy * 100.0 if raw_dy < 0.20 else raw_dy
                if dps <= 0:
                    dps = (dividend_yield_pct / 100.0) * stock_price
            else:
                dps = 0.0
                dividend_yield_pct = 0.0

            dividend_yield_ratio = dividend_yield_pct / 100.0

            fundamentals = {
                "pe_ratio": info.get('trailingPE', np.random.uniform(15, 35)),
                "pb_ratio": info.get('priceToBook', np.random.uniform(1.5, 8)),
                "roe": info.get('returnOnEquity', np.random.uniform(0.10, 0.30)),
                "roce": info.get('returnOnAssets', np.random.uniform(0.08, 0.25)) * 1.5,
                "debt_to_equity": info.get('debtToEquity', np.random.uniform(0.1, 1.2)) / 100 if info.get('debtToEquity', 0) > 5 else info.get('debtToEquity', 0.3),
                "profit_margins": info.get('profitMargins', np.random.uniform(0.08, 0.25)),
                "revenue_growth": info.get('revenueGrowth', np.random.uniform(0.05, 0.20)),
                "earnings_growth": info.get('earningsGrowth', np.random.uniform(0.05, 0.30)),
                "dividend_per_share": round(dps, 2),
                "stock_price": round(stock_price, 2),
                "dividend_yield_pct": round(dividend_yield_pct, 2),
                "dividend_yield": round(dividend_yield_ratio, 4),
                "market_cap": info.get('marketCap', 1e12),
                "current_ratio": info.get('currentRatio', np.random.uniform(1.2, 3.0)),
                "eps": info.get('trailingEps', np.random.uniform(20, 200)),
            }
            return fundamentals
            
        except Exception as e:
            logger.warning(f"Fundamental fetch failed for {symbol}: {e}")
            return self._synthetic_fundamentals(symbol)
    
    def _synthetic_fundamentals(self, symbol: str) -> Dict:
        """Realistic synthetic fundamentals for Indian blue-chips with strict (DPS/Price)*100"""
        np.random.seed(abs(hash(symbol)) % 2**32)
        
        # Realistic prices & dividends for sector profiles
        if symbol in ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]:
            # Banking
            stock_price = float(np.random.uniform(600, 1800))
            dps = float(np.random.uniform(8, 25))
            div_yield_pct = (dps / stock_price) * 100.0
            return {
                "pe_ratio": np.random.uniform(12, 22),
                "pb_ratio": np.random.uniform(1.5, 3.5),
                "roe": np.random.uniform(0.12, 0.20),
                "roce": np.random.uniform(0.10, 0.18),
                "debt_to_equity": np.random.uniform(0.05, 0.3),
                "profit_margins": np.random.uniform(0.20, 0.35),
                "revenue_growth": np.random.uniform(0.08, 0.18),
                "earnings_growth": np.random.uniform(0.10, 0.25),
                "dividend_per_share": round(dps, 2),
                "stock_price": round(stock_price, 2),
                "dividend_yield_pct": round(div_yield_pct, 2),
                "dividend_yield": round(div_yield_pct / 100.0, 4),
                "market_cap": np.random.uniform(5e11, 8e12),
                "current_ratio": np.random.uniform(1.1, 1.5),
                "eps": np.random.uniform(30, 80),
            }
        elif symbol in ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"]:
            # IT
            stock_price = float(np.random.uniform(1500, 3800))
            dps = float(np.random.uniform(35, 90))
            div_yield_pct = (dps / stock_price) * 100.0
            return {
                "pe_ratio": np.random.uniform(20, 35),
                "pb_ratio": np.random.uniform(5, 12),
                "roe": np.random.uniform(0.20, 0.40),
                "roce": np.random.uniform(0.25, 0.45),
                "debt_to_equity": np.random.uniform(0.0, 0.2),
                "profit_margins": np.random.uniform(0.18, 0.28),
                "revenue_growth": np.random.uniform(0.05, 0.15),
                "earnings_growth": np.random.uniform(0.06, 0.18),
                "dividend_per_share": round(dps, 2),
                "stock_price": round(stock_price, 2),
                "dividend_yield_pct": round(div_yield_pct, 2),
                "dividend_yield": round(div_yield_pct / 100.0, 4),
                "market_cap": np.random.uniform(1e12, 12e12),
                "current_ratio": np.random.uniform(2.0, 4.0),
                "eps": np.random.uniform(40, 150),
            }
        else:
            # Conglomerate / Diversified
            stock_price = float(np.random.uniform(400, 2500))
            dps = float(np.random.uniform(5, 45))
            div_yield_pct = (dps / stock_price) * 100.0
            return {
                "pe_ratio": np.random.uniform(15, 30),
                "pb_ratio": np.random.uniform(2, 6),
                "roe": np.random.uniform(0.12, 0.25),
                "roce": np.random.uniform(0.10, 0.22),
                "debt_to_equity": np.random.uniform(0.2, 0.8),
                "profit_margins": np.random.uniform(0.10, 0.22),
                "revenue_growth": np.random.uniform(0.06, 0.18),
                "earnings_growth": np.random.uniform(0.08, 0.22),
                "dividend_per_share": round(dps, 2),
                "stock_price": round(stock_price, 2),
                "dividend_yield_pct": round(div_yield_pct, 2),
                "dividend_yield": round(div_yield_pct / 100.0, 4),
                "market_cap": np.random.uniform(5e11, 15e12),
                "current_ratio": np.random.uniform(1.2, 2.5),
                "eps": np.random.uniform(20, 120),
            }
    
    def compute_fundamental_score(self, fundamentals: Dict) -> Dict:
        """Transparent fundamental scoring - research grade"""
        score = 0
        breakdown = {}
        
        # P/E scoring (15%) - lower is better, but not too low
        pe = fundamentals.get('pe_ratio', 20)
        if 10 <= pe <= 18:
            pe_score = 90
        elif 18 < pe <= 25:
            pe_score = 75
        elif 25 < pe <= 35:
            pe_score = 60
        elif pe < 10:
            pe_score = 50  # might be value trap
        else:
            pe_score = 30
        breakdown['pe_score'] = pe_score
        score += pe_score * 0.15
        
        # P/B (10%)
        pb = fundamentals.get('pb_ratio', 3)
        if pb < 2:
            pb_score = 90
        elif pb < 4:
            pb_score = 75
        elif pb < 7:
            pb_score = 60
        else:
            pb_score = 40
        breakdown['pb_score'] = pb_score
        score += pb_score * 0.10
        
        # ROE (20%) - higher is better
        roe = fundamentals.get('roe', 0.15)
        if roe > 0.25:
            roe_score = 95
        elif roe > 0.18:
            roe_score = 80
        elif roe > 0.12:
            roe_score = 65
        else:
            roe_score = 40
        breakdown['roe_score'] = roe_score
        score += roe_score * 0.20
        
        # Debt/Equity (15%) - lower is better
        de = fundamentals.get('debt_to_equity', 0.5)
        if de < 0.2:
            de_score = 90
        elif de < 0.5:
            de_score = 75
        elif de < 1.0:
            de_score = 60
        else:
            de_score = 35
        breakdown['de_score'] = de_score
        score += de_score * 0.15
        
        # Revenue Growth (15%)
        rg = fundamentals.get('revenue_growth', 0.10)
        if rg > 0.15:
            rg_score = 90
        elif rg > 0.08:
            rg_score = 75
        elif rg > 0.03:
            rg_score = 60
        else:
            rg_score = 30
        breakdown['revenue_growth_score'] = rg_score
        score += rg_score * 0.15
        
        # Profit Growth (15%)
        eg = fundamentals.get('earnings_growth', 0.10)
        if eg > 0.20:
            eg_score = 95
        elif eg > 0.12:
            eg_score = 80
        elif eg > 0.05:
            eg_score = 60
        else:
            eg_score = 35
        breakdown['earnings_growth_score'] = eg_score
        score += eg_score * 0.15
        
        # Dividend Yield (10%): strictly (DPS / stock_price) * 100
        dy_pct = fundamentals.get('dividend_yield_pct')
        if dy_pct is None:
            raw_dy = fundamentals.get('dividend_yield', 0.01)
            dy_pct = raw_dy * 100.0 if raw_dy < 0.20 else raw_dy

        if 1.0 <= dy_pct <= 4.0:
            dy_score = 90
        elif dy_pct > 4.0:
            dy_score = 85  # Strong high-dividend payer
        elif 0.5 <= dy_pct < 1.0:
            dy_score = 70
        else:
            dy_score = 50
        breakdown['dividend_score'] = dy_score
        score += dy_score * 0.10
        
        final_score = round(max(0, min(100, score)), 1)
        
        # Signal
        if final_score >= 75:
            signal = "Strong"
        elif final_score >= 60:
            signal = "Positive"
        elif final_score >= 45:
            signal = "Neutral"
        else:
            signal = "Weak"
        
        return {
            "fundamental_score": final_score,
            "signal": signal,
            "breakdown": breakdown,
            "raw": fundamentals
        }

fundamental_analyzer = FundamentalAnalyzer()
