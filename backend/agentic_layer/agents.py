"""
Agentic AI Layer — Specialized Financial Agents (Revised v2.0)
═══════════════════════════════════════════════════════════════

Changes in v2.0 (6 Mandatory Revisions):
  - NewsSentimentAgent  → live-only (Google RSS + FinBERT), not used in backtest
  - RiskAgent           → enhanced with stop-loss, RRR, max allocation, VIX/EMA veto
  - MLPredictionAgent   → consumes 3-class probabilities (P_BUY, P_HOLD, P_SELL)
  - LSTM removed from MLPredictionAgent imports
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from datetime import datetime

try:
    from ..data_pipeline.price_fetcher import fetcher
    from ..data_pipeline.indicators import TechnicalIndicators
    from ..data_pipeline.fundamental_fetcher import fundamental_analyzer
    from ..data_pipeline.news_fetcher import news_fetcher
    from ..nlp_layer.sentiment import sentiment_analyzer
    from ..utils.config import MODEL_CONFIG, RISK_CONFIG
    from ..ml_layer.trainer import UnifiedTrainer
except ImportError:
    from backend.data_pipeline.price_fetcher import fetcher
    from backend.data_pipeline.indicators import TechnicalIndicators
    from backend.data_pipeline.fundamental_fetcher import fundamental_analyzer
    from backend.data_pipeline.news_fetcher import news_fetcher
    from backend.nlp_layer.sentiment import sentiment_analyzer
    from backend.utils.config import MODEL_CONFIG, RISK_CONFIG
    from backend.ml_layer.trainer import UnifiedTrainer

logger = logging.getLogger(__name__)

# ─── VIX / Regime veto thresholds (Revision 5) ───────────────────────────────
VIX_VETO_THRESHOLD = 22.0   # India VIX above this → veto BUY signals


# ════════════════════════════════════════════════════════════════════════════════
class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name      = name
        self.last_run  = None
        self.confidence = 0.5

    @abstractmethod
    def analyze(self, symbol: str, data: Dict) -> Dict:
        pass

    def _create_response(
        self,
        signal:     str,
        confidence: float,
        details:    Dict,
        reasoning:  List[str],
    ) -> Dict:
        return {
            "agent":        self.name,
            "signal":       signal,
            "confidence":   round(confidence, 3),
            "timestamp":    datetime.now().isoformat(),
            "details":      details,
            "reasoning":    reasoning,
            "risk_factors": details.get("risk_factors", []),
        }


# ════════════════════════════════════════════════════════════════════════════════
class MarketDataAgent(BaseAgent):
    """Responsible for collecting, validating, and preparing market data."""

    def __init__(self):
        super().__init__("MarketDataAgent")

    def analyze(self, symbol: str, data: Dict) -> Dict:
        try:
            price_data  = fetcher.fetch_price_data(symbol, start="2018-01-01")
            market_data = fetcher.fetch_market_data("^NSEI",    start="2018-01-01")
            vix_data    = fetcher.fetch_india_vix(start="2018-01-01")

            missing_pct = price_data.isnull().sum() / len(price_data) * 100
            is_valid    = missing_pct.max() < 10 and len(price_data) > 100

            price_with_indicators = TechnicalIndicators.add_all_indicators(price_data)
            latest_quote = fetcher.fetch_latest_quote(symbol)

            reasoning = [
                f"Fetched {len(price_data)} trading days for {symbol}",
                f"Data validation: {'Passed' if is_valid else 'Issues detected'}",
                f"Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}",
                f"Latest close: ₹{price_data['Close'].iloc[-1]:.2f}",
                f"Storage: Parquet cache active",
            ]

            return self._create_response(
                signal="VALID" if is_valid else "INVALID",
                confidence=0.95 if is_valid else 0.3,
                details={
                    "price_data":             price_data,
                    "price_with_indicators":  price_with_indicators,
                    "market_data":            market_data,
                    "vix_data":               vix_data,
                    "validation": {
                        "missing_pct": missing_pct.to_dict(),
                        "is_valid":    is_valid,
                        "data_points": len(price_data),
                    },
                    "latest_price": latest_quote["price"],
                    "latest_quote": latest_quote,
                },
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"MarketDataAgent failed: {e}")
            return self._create_response(
                signal="ERROR", confidence=0.1,
                details={"error": str(e)},
                reasoning=[f"Failed to fetch data: {e}"],
            )


# ════════════════════════════════════════════════════════════════════════════════
class TechnicalAnalysisAgent(BaseAgent):
    """Analyzes technical indicators and patterns."""

    def __init__(self):
        super().__init__("TechnicalAnalysisAgent")

    def analyze(self, symbol: str, data: Dict) -> Dict:
        try:
            price_with_ind = data.get("price_with_indicators")
            if price_with_ind is None:
                price_data     = fetcher.fetch_price_data(symbol)
                price_with_ind = TechnicalIndicators.add_all_indicators(price_data)

            summary = TechnicalIndicators.get_technical_summary(price_with_ind)
            score   = TechnicalIndicators.compute_technical_score(summary)

            if score >= 70:
                signal, confidence = "BUY",          0.7 + (score - 70) / 100
            elif score <= 30:
                signal, confidence = "SELL",         0.7 + (30 - score) / 100
            elif score >= 55:
                signal, confidence = "HOLD_BULLISH", 0.6
            elif score <= 45:
                signal, confidence = "HOLD_BEARISH", 0.6
            else:
                signal, confidence = "HOLD",         0.5

            if summary.get("Volume_signal") == "High" and signal in ["BUY", "SELL"]:
                confidence = min(0.95, confidence + 0.1)
            if summary.get("Volatility_level") == "High":
                confidence = max(0.3, confidence - 0.15)

            reasoning, risk_factors = [], []

            if summary.get("Trend_20") == "Positive" and summary.get("Trend_50") == "Positive":
                reasoning.append("✓ Positive 20/50-day trend indicates bullish momentum")
            if summary.get("Trend_200") == "Positive":
                reasoning.append("✓ Price above 200-day EMA — long-term bull trend")
            if summary.get("Golden_Cross"):
                reasoning.append("✓ Golden Cross (20-day SMA above 50-day SMA) — bullish signal")
            if summary.get("RSI_signal") in ["Bullish momentum", "Neutral"]:
                reasoning.append(f"✓ RSI at {summary.get('RSI')} indicates {summary.get('RSI_signal')}")
            if summary.get("MACD_signal") == "Bullish":
                reasoning.append(f"✓ MACD bullish with {summary.get('MACD_trend')} trend")
            if summary.get("Volume_signal") == "High":
                reasoning.append(f"✓ High volume ({summary.get('Volume_ratio')}x avg) confirms move")

            if summary.get("Volatility_level") == "High":
                risk_factors.append(f"High volatility ({summary.get('Volatility_20'):.2%})")
            if summary.get("RSI", 50) > 70:
                risk_factors.append(f"RSI overbought at {summary.get('RSI')}")
            if summary.get("BB_position", 0.5) > 0.85:
                risk_factors.append("Price near upper Bollinger Band — pullback risk")

            if not reasoning:
                reasoning.append(f"Technical indicators mixed, score {score}/100")

            return self._create_response(
                signal=signal, confidence=confidence,
                details={
                    "technical_summary": summary,
                    "technical_score":   score,
                    "risk_factors":      risk_factors,
                    "indicators": {
                        "RSI":    summary.get("RSI"),
                        "MACD":   summary.get("MACD"),
                        "SMA_20": summary.get("SMA_20"),
                        "SMA_50": summary.get("SMA_50"),
                        "EMA_200": summary.get("EMA_200"),
                    },
                },
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"TechnicalAnalysisAgent failed: {e}")
            return self._create_response(
                signal="HOLD", confidence=0.3,
                details={"error": str(e)},
                reasoning=[f"Technical analysis failed: {e}"],
            )


# ════════════════════════════════════════════════════════════════════════════════
class NewsSentimentAgent(BaseAgent):
    """
    LIVE-ONLY News & Sentiment Agent (Revision 3).

    Sources: Google News RSS → FinBERT scoring.
    NOT used in historical backtesting.
    Backtesting evaluates on: Technicals + Market Regime + Macro + Fundamentals.
    """

    def __init__(self):
        super().__init__("NewsSentimentAgent")

    def analyze(self, symbol: str, data: Dict) -> Dict:
        """Fetch live RSS news and score with FinBERT."""
        try:
            # Live-only: fetch from Google News RSS
            articles   = news_fetcher.fetch_live_rss(symbol, max_articles=15)
            aggregated = news_fetcher.aggregate_sentiment(articles)

            overall    = aggregated["overall_sentiment"]   # "positive" | "negative" | "neutral"
            net_score  = aggregated["net_score"]           # avg_positive - avg_negative
            n_articles = aggregated["article_count"]

            confidence = min(0.9, 0.3 + n_articles * 0.04)   # more articles → higher confidence

            if overall == "positive" and net_score > 0.2:
                signal = "BUY"
            elif overall == "positive":
                signal = "HOLD_BULLISH"
            elif overall == "negative" and net_score < -0.2:
                signal = "SELL"
            elif overall == "negative":
                signal = "HOLD_BEARISH"
            else:
                signal = "HOLD"

            reasoning, risk_factors = [], []

            if n_articles == 0:
                reasoning.append("No live news found — returning neutral sentiment")
                signal     = "HOLD"
                confidence = 0.3
            else:
                reasoning.append(f"Scored {n_articles} live articles via FinBERT (Google News RSS)")
                reasoning.append(f"Aggregate sentiment: {overall.capitalize()} (net score: {net_score:+.3f})")
                for title in aggregated.get("top_positive", [])[:2]:
                    reasoning.append(f"✓ Positive: {title[:80]}")
                for title in aggregated.get("top_negative", [])[:2]:
                    reasoning.append(f"⚠ Negative: {title[:80]}")

            if aggregated["avg_negative"] > 0.4:
                risk_factors.append(f"High average negative score ({aggregated['avg_negative']:.2f})")
            if n_articles < 3:
                risk_factors.append("Low news volume — sentiment signal less reliable")

            return self._create_response(
                signal=signal, confidence=confidence,
                details={
                    "aggregated_sentiment": aggregated,
                    "articles":             articles[:5],       # top 5 for display
                    "risk_factors":         risk_factors,
                    "live_only":            True,               # flag for decision engine
                    "article_count":        n_articles,
                },
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"NewsSentimentAgent failed: {e}")
            return self._create_response(
                signal="HOLD", confidence=0.3,
                details={"error": str(e), "live_only": True},
                reasoning=[f"Live sentiment fetch failed: {e}"],
            )


# ════════════════════════════════════════════════════════════════════════════════
class FundamentalAgent(BaseAgent):
    """Analyzes company fundamentals."""

    def __init__(self):
        super().__init__("FundamentalAgent")

    def analyze(self, symbol: str, data: Dict) -> Dict:
        try:
            fundamentals = fundamental_analyzer.fetch_fundamentals(symbol)
            analysis     = fundamental_analyzer.compute_fundamental_score(fundamentals)

            score = analysis["fundamental_score"]
            if score >= 75:
                signal, confidence = "BUY",          0.75 + (score - 75) / 100
            elif score >= 60:
                signal, confidence = "HOLD_BULLISH", 0.65
            elif score >= 45:
                signal, confidence = "HOLD",         0.55
            else:
                signal, confidence = "SELL",         0.65

            raw       = analysis["raw"]
            reasoning, risk_factors = [], []

            if raw.get("roe", 0) > 0.18:
                reasoning.append(f"✓ Strong ROE at {raw['roe']:.1%} — efficient capital use")
            if raw.get("revenue_growth", 0) > 0.12:
                reasoning.append(f"✓ Robust revenue growth {raw['revenue_growth']:.1%} YoY")
            if raw.get("debt_to_equity", 1) < 0.4:
                reasoning.append(f"✓ Low D/E {raw['debt_to_equity']:.2f} — financially stable")
            if raw.get("profit_margins", 0) > 0.15:
                reasoning.append(f"✓ Healthy profit margins {raw['profit_margins']:.1%}")
            if raw.get("pe_ratio", 20) > 35:
                reasoning.append(f"⚠ High P/E {raw['pe_ratio']:.1f} — valuation elevated")
            
            dy_pct = raw.get("dividend_yield_pct", 0)
            dps = raw.get("dividend_per_share", 0)
            stock_p = raw.get("stock_price", 0)
            if dy_pct >= 2.0 and dps > 0 and stock_p > 0:
                reasoning.append(f"✓ Attractive dividend yield {dy_pct:.2f}% (DPS: ₹{dps:.2f} / Price: ₹{stock_p:.2f} × 100)")
            elif dy_pct >= 1.0:
                reasoning.append(f"✓ Dividend yield at {dy_pct:.2f}%")

            if not reasoning:
                reasoning.append(f"Fundamental score {score}/100 — {analysis['signal']}")

            if raw.get("debt_to_equity", 0) > 0.8:
                risk_factors.append(f"High D/E {raw['debt_to_equity']:.2f}")
            if raw.get("pe_ratio", 0) > 40:
                risk_factors.append(f"Very high P/E {raw['pe_ratio']:.1f} — valuation risk")
            if raw.get("revenue_growth", 0) < 0.03:
                risk_factors.append(f"Low revenue growth {raw.get('revenue_growth', 0):.1%}")

            return self._create_response(
                signal=signal, confidence=confidence,
                details={
                    "fundamental_score": score,
                    "analysis":          analysis,
                    "risk_factors":      risk_factors,
                    "raw_fundamentals":  raw,
                },
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"FundamentalAgent failed: {e}")
            return self._create_response(
                signal="HOLD", confidence=0.4,
                details={"error": str(e)},
                reasoning=[f"Fundamental analysis failed: {e}"],
            )


# ════════════════════════════════════════════════════════════════════════════════
class RiskAgent(BaseAgent):
    """
    Risk assessment with enhanced outputs (Revision 5):
      - Stop-Loss Price   = Entry Price − 1.5 × ATR_14
      - Target Price      = Entry Price + 2.0 × ATR_14
      - Risk-to-Reward    = (Target − Entry) / (Entry − Stop)
      - Max Allocation %  = 5% (HIGH), 10% (MEDIUM), 15% (LOW)
      - Veto Power        = if VIX > 22 OR NIFTY < 200 EMA → downgrade BUY → HOLD
    """

    def __init__(self):
        super().__init__("RiskAgent")

    def analyze(self, symbol: str, data: Dict) -> Dict:
        try:
            price_with_ind = data.get("price_with_indicators")
            if price_with_ind is None:
                price_data     = fetcher.fetch_price_data(symbol)
                price_with_ind = TechnicalIndicators.add_all_indicators(price_data)

            vix_data    = data.get("vix_data")
            if vix_data is None:
                vix_data = fetcher.fetch_india_vix()

            market_data = data.get("market_data")
            if market_data is None:
                market_data = fetcher.fetch_market_data("^NSEI")

            latest = price_with_ind.iloc[-1]
            recent = price_with_ind.tail(20)

            # ── Core risk metrics ─────────────────────────────────────────────
            entry_price  = float(latest["Close"])
            atr_14       = float(latest.get("ATR_14", entry_price * 0.02))
            volatility   = float(latest.get("Volatility_20", 0.25))
            atr_pct      = atr_14 / entry_price if entry_price > 0 else 0.02
            vol_ratio    = float(latest.get("Volume_ratio", 1.0))
            returns_20   = float(recent["Returns"].std() * np.sqrt(252))

            # Stop-Loss, Target, RRR (Revision 5)
            stop_loss_price = round(entry_price - 1.5 * atr_14, 2)
            target_price    = round(entry_price + 2.0 * atr_14, 2)
            risk_pts        = entry_price - stop_loss_price
            reward_pts      = target_price - entry_price
            rrr             = round(reward_pts / risk_pts, 2) if risk_pts > 0 else 1.33

            # Drawdown
            rolling_max      = price_with_ind["Close"].rolling(60).max()
            drawdown         = (price_with_ind["Close"] - rolling_max) / rolling_max
            current_drawdown = float(drawdown.iloc[-1])
            max_drawdown_60  = float(drawdown.tail(60).min())

            # Volume anomaly
            volume_anomaly = vol_ratio > 2.5 or vol_ratio < 0.3

            # VIX
            current_vix  = float(vix_data["Close"].iloc[-1]) if len(vix_data) > 0 else 15.0
            vix_elevated = current_vix > VIX_VETO_THRESHOLD

            # NIFTY vs 200 EMA (veto condition)
            nifty_below_200ema = False
            try:
                if market_data is not None and not market_data.empty:
                    nifty_with_ind      = TechnicalIndicators.add_all_indicators(market_data)
                    nifty_latest        = nifty_with_ind.iloc[-1]
                    nifty_ema_200       = float(nifty_latest.get("EMA_200", nifty_latest["Close"]))
                    nifty_close         = float(nifty_latest["Close"])
                    nifty_below_200ema  = nifty_close < nifty_ema_200
            except Exception:
                pass

            # ── Veto decision (Revision 5) ────────────────────────────────────
            veto_active = vix_elevated or nifty_below_200ema

            # ── Risk scoring ──────────────────────────────────────────────────
            risk_score, risk_factors = 0, []

            vix_threshold = getattr(RISK_CONFIG, "ELEVATED_VIX_THRESHOLD", 20)
            high_vol_thresh = getattr(RISK_CONFIG, "HIGH_VOLATILITY_THRESHOLD", 0.35)

            if volatility > high_vol_thresh:
                risk_score += 30
                risk_factors.append(f"High volatility {volatility:.1%}")
            elif volatility > 0.25:
                risk_score += 15
                risk_factors.append(f"Elevated volatility {volatility:.1%}")

            if current_drawdown < -0.15:
                risk_score += 25
                risk_factors.append(f"Deep drawdown {current_drawdown:.1%} from 60-day high")
            elif current_drawdown < -0.08:
                risk_score += 10
                risk_factors.append(f"Moderate drawdown {current_drawdown:.1%}")

            if vix_elevated:
                risk_score += 20
                risk_factors.append(f"India VIX elevated at {current_vix:.1f} > {VIX_VETO_THRESHOLD}")

            if nifty_below_200ema:
                risk_score += 15
                risk_factors.append("NIFTY 50 below 200-day EMA — bearish market regime")

            if volume_anomaly:
                risk_score += 10
                risk_factors.append(f"Unusual volume {vol_ratio:.1f}× average")

            if returns_20 > 0.4:
                risk_score += 15
                risk_factors.append("Recent price instability high")

            if risk_score >= 50:
                risk_level, signal, confidence = "High",   "HIGH_RISK",   0.80
            elif risk_score >= 25:
                risk_level, signal, confidence = "Medium", "MEDIUM_RISK", 0.70
            else:
                risk_level, signal, confidence = "Low",    "LOW_RISK",    0.70

            # Max allocation recommendation
            max_allocation_pct = {"High": 5.0, "Medium": 10.0, "Low": 15.0}[risk_level]

            reasoning = [
                f"Volatility (20d annualised): {volatility:.1%} — {risk_level} risk level",
                f"Current drawdown: {current_drawdown:.1%} | Max 60d drawdown: {max_drawdown_60:.1%}",
                f"India VIX: {current_vix:.1f} ({'🔴 ELEVATED — VETO ACTIVE' if vix_elevated else '🟢 Normal'})",
                f"NIFTY 50 vs 200-EMA: {'🔴 BELOW — VETO ACTIVE' if nifty_below_200ema else '🟢 Above'}",
                f"ATR (14d): ₹{atr_14:.2f} = {atr_pct:.1%} of price",
                f"Stop-Loss: ₹{stop_loss_price:.2f} | Target: ₹{target_price:.2f} | RRR: {rrr:.2f}",
                f"Max position allocation: {max_allocation_pct:.0f}% of portfolio",
            ]
            if veto_active:
                reasoning.append("⛔ VETO: BUY signals will be downgraded to HOLD by decision engine")
            if risk_factors:
                reasoning.extend([f"⚠ {rf}" for rf in risk_factors])
            else:
                reasoning.append("✓ No major risk factors detected")

            return self._create_response(
                signal=signal, confidence=confidence,
                details={
                    "risk_level":        risk_level,
                    "risk_score":        risk_score,
                    "risk_factors":      risk_factors,
                    "veto_active":       veto_active,
                    "vix_elevated":      vix_elevated,
                    "nifty_below_200ema": nifty_below_200ema,
                    # Enhanced risk outputs (Revision 5)
                    "stop_loss_price":   stop_loss_price,
                    "target_price":      target_price,
                    "rrr":               rrr,
                    "max_allocation_pct": max_allocation_pct,
                    "entry_price":       entry_price,
                    "metrics": {
                        "volatility_20":     volatility,
                        "atr_14":            atr_14,
                        "atr_pct":           float(atr_pct),
                        "current_drawdown":  current_drawdown,
                        "max_drawdown_60":   max_drawdown_60,
                        "vix":               current_vix,
                        "volume_ratio":      float(vol_ratio),
                        "returns_vol_20":    returns_20,
                    },
                },
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"RiskAgent failed: {e}")
            return self._create_response(
                signal="MEDIUM_RISK", confidence=0.4,
                details={"error": str(e), "veto_active": False},
                reasoning=[f"Risk assessment failed: {e}"],
            )


# ════════════════════════════════════════════════════════════════════════════════
class MarketRegimeAgent(BaseAgent):
    """Determines market regime: Bull / Bear / Sideways / High Volatility."""

    def __init__(self):
        super().__init__("MarketRegimeAgent")

    def analyze(self, symbol: str, data: Dict) -> Dict:
        try:
            market_data = data.get("market_data")
            if market_data is None or market_data.empty:
                market_data = fetcher.fetch_market_data("^NSEI")
            vix_data = data.get("vix_data")
            if vix_data is None or vix_data.empty:
                vix_data = fetcher.fetch_india_vix()

            market_with_ind = TechnicalIndicators.add_all_indicators(market_data)
            latest = market_with_ind.iloc[-1]

            close        = float(latest["Close"])
            sma20        = float(latest["SMA_20"])
            sma50        = float(latest["SMA_50"])
            sma200       = float(latest.get("SMA_200", sma50))
            ema200       = float(latest.get("EMA_200", sma200))
            momentum     = float(latest.get("Momentum_20", 0))
            nifty_vol    = float(latest.get("Volatility_20", 0.2))
            current_vix  = float(vix_data["Close"].iloc[-1]) if len(vix_data) > 0 else 15.0

            above_ema200 = close > ema200

            if close > sma20 and close > sma50 and close > sma200 and momentum > 0.02 and current_vix < 18:
                regime, signal, confidence = "Bull Market",     "BULLISH",  0.80
            elif close < sma20 and close < sma50 and momentum < -0.02:
                regime, signal, confidence = "Bear Market",     "BEARISH",  0.75
            elif nifty_vol > 0.30 or current_vix > VIX_VETO_THRESHOLD:
                regime, signal, confidence = "High Volatility", "VOLATILE", 0.70
            else:
                regime, signal, confidence = "Sideways",        "NEUTRAL",  0.60

            reasoning = [
                f"NIFTY 50: ₹{close:.0f} | 20d SMA: ₹{sma20:.0f} | 50d SMA: ₹{sma50:.0f}",
                f"200d EMA: ₹{ema200:.0f} | {'🟢 Above' if above_ema200 else '🔴 Below'} 200-EMA",
                f"Momentum (20d): {momentum:.1%} | Volatility: {nifty_vol:.1%}",
                f"India VIX: {current_vix:.1f} | Regime: {regime}",
            ]
            regime_implications = {
                "Bull Market":     "✓ Bull market — favorable for longs, higher risk tolerance",
                "Bear Market":     "⚠ Bear market — defensive, reduce position sizes",
                "High Volatility": "⚠ High-vol regime — reduce leverage, wider stops",
                "Sideways":        "→ Sideways — stock-specific factors dominate",
            }
            reasoning.append(regime_implications.get(regime, ""))

            return self._create_response(
                signal=signal, confidence=confidence,
                details={
                    "regime":            regime,
                    "nifty_close":       close,
                    "nifty_sma20":       sma20,
                    "nifty_sma50":       sma50,
                    "nifty_ema200":      ema200,
                    "above_200ema":      above_ema200,
                    "nifty_momentum":    momentum,
                    "nifty_volatility":  nifty_vol,
                    "vix":               current_vix,
                },
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"MarketRegimeAgent failed: {e}")
            return self._create_response(
                signal="NEUTRAL", confidence=0.4,
                details={"error": str(e)},
                reasoning=[f"Market regime detection failed: {e}"],
            )


# ════════════════════════════════════════════════════════════════════════════════
class MLPredictionAgent(BaseAgent):
    """
    ML Prediction Agent — 3-class probabilities (Revision 1 & 2).
    Uses Walk-Forward trained models (LogReg, RF, XGBoost multi:softprob, LightGBM).
    LSTM removed.
    """

    def __init__(self):
        super().__init__("MLPredictionAgent")
        self._trainer: Optional[UnifiedTrainer] = None

    def analyze(self, symbol: str, data: Dict) -> Dict:
        try:
            price_with_ind = data.get("price_with_indicators")
            if price_with_ind is None:
                price_data     = fetcher.fetch_price_data(symbol)
                price_with_ind = TechnicalIndicators.add_all_indicators(price_data)

            # Lazy-train on demand
            if self._trainer is None:
                self._trainer = UnifiedTrainer()

            # Use cached models or train fresh
            benchmarks = fetcher.fetch_benchmarks()
            result = self._trainer.train(
                symbol,
                nifty_df=benchmarks.get("nifty"),
                vix_df=benchmarks.get("vix"),
                restrict_to_backtest=True,
            )

            if "error" in result:
                return self._fallback_heuristic(price_with_ind, reason=result["error"])

            models = result["models"]
            fe     = self._trainer.feature_engineer

            # Build latest feature row
            df_feat = fe.create_features(price_with_ind)
            feature_cols = models.feature_cols or fe.get_feature_columns(df_feat)
            # Keep only available columns
            feature_cols = [c for c in feature_cols if c in df_feat.columns]
            X_latest     = df_feat[feature_cols].dropna().tail(1)

            if X_latest.empty:
                return self._fallback_heuristic(price_with_ind, reason="No valid feature row")

            ensemble = models.predict_ensemble(X_latest)

            p_buy  = ensemble["p_buy"]
            p_hold = ensemble["p_hold"]
            p_sell = ensemble["p_sell"]
            pred   = ensemble["prediction"]   # "BUY" | "HOLD" | "SELL"
            conf   = ensemble["confidence"]

            signal_map = {"BUY": "BUY", "HOLD": "HOLD", "SELL": "SELL"}
            signal = signal_map.get(pred, "HOLD")

            oof = result.get("oof_metrics", {})
            reasoning = [
                f"Walk-Forward ensemble: {pred} | P(BUY)={p_buy:.2%} P(HOLD)={p_hold:.2%} P(SELL)={p_sell:.2%}",
                f"Model confidence (1−entropy): {conf:.2%}",
                f"Models: 4 (LogReg, RF, XGBoost multi:softprob, LightGBM)",
                f"OOF macro-F1 → XGB: {oof.get('xgboost', 0):.3f} | LGB: {oof.get('lightgbm', 0):.3f}",
                f"Training samples: {result.get('n_samples', '?')} | Backtest: 2022-01-01 – 2024-12-31",
            ]

            # Top XGBoost features
            xgb_model = models.models.get("xgboost")
            if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
                top3 = sorted(
                    zip(feature_cols, xgb_model.feature_importances_),
                    key=lambda x: x[1], reverse=True
                )[:3]
                reasoning.append(f"Top XGB features: {', '.join(f[0] for f in top3)}")

            return self._create_response(
                signal=signal, confidence=conf,
                details={
                    "p_buy":          p_buy,
                    "p_hold":         p_hold,
                    "p_sell":         p_sell,
                    "prediction":     pred,
                    "oof_metrics":    oof,
                    "label_dist":     result.get("label_dist", {}),
                    "risk_factors":   [] if conf > 0.6 else ["Model uncertainty elevated"],
                    "all_probas":     ensemble.get("all_probas", {}),
                },
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"MLPredictionAgent failed: {e}", exc_info=True)
            return self._fallback_heuristic(
                data.get("price_with_indicators"), reason=str(e)
            )

    def _fallback_heuristic(self, price_with_ind, reason: str = "") -> Dict:
        """3-signal heuristic when ML training is unavailable."""
        try:
            latest = price_with_ind.iloc[-1]
            score  = 0
            if latest["Close"] > latest["SMA_20"]:       score += 1
            if latest["RSI_14"] > 50:                    score += 1
            if latest["MACD_histogram"] > 0:             score += 1

            p_buy  = 0.25 + score * 0.15
            p_sell = max(0.1, 0.6 - score * 0.15)
            p_hold = max(0.1, 1.0 - p_buy - p_sell)
            signal = "BUY" if score >= 2 else "SELL" if score == 0 else "HOLD"

            return self._create_response(
                signal=signal, confidence=0.45,
                details={
                    "p_buy": round(p_buy, 3), "p_hold": round(p_hold, 3), "p_sell": round(p_sell, 3),
                    "prediction": signal, "fallback": True, "heuristic_score": score,
                    "risk_factors": ["ML model unavailable — heuristic fallback"],
                },
                reasoning=[
                    f"Heuristic fallback (reason: {reason})",
                    f"3-signal score: {score}/3 → {signal}",
                    f"P(BUY)={p_buy:.2%} P(HOLD)={p_hold:.2%} P(SELL)={p_sell:.2%}",
                ],
            )
        except Exception:
            return self._create_response(
                signal="HOLD", confidence=0.3,
                details={"p_buy": 0.33, "p_hold": 0.34, "p_sell": 0.33, "fallback": True},
                reasoning=["ML and heuristic both failed — defaulting to HOLD"],
            )


# ── Agent Registry ─────────────────────────────────────────────────────────────
AGENT_REGISTRY = {
    "market_data":   MarketDataAgent,
    "technical":     TechnicalAnalysisAgent,
    "news":          NewsSentimentAgent,
    "fundamental":   FundamentalAgent,
    "risk":          RiskAgent,
    "market_regime": MarketRegimeAgent,
    "ml":            MLPredictionAgent,
}
