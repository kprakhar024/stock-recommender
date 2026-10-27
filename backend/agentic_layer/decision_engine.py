"""
Decision Engine — Revised v2.0
════════════════════════════════
Changes (6 Mandatory Revisions):
  - Consumes 3-class probabilities: P(BUY), P(HOLD), P(SELL) from MLPredictionAgent
  - Risk Agent veto: if veto_active=True (VIX>22 OR NIFTY<200EMA) → BUY → HOLD
  - Expected return: (P_buy - P_sell) × 15% instead of binary formula
  - Enhanced output: includes stop_loss, target_price, RRR, max_allocation_pct
"""
from typing import Dict, List
import numpy as np
from datetime import datetime
import logging

try:
    from ..utils.config import AGENT_WEIGHTS
except ImportError:
    from backend.utils.config import AGENT_WEIGHTS

logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(self):
        self.weights = AGENT_WEIGHTS

    # ── Signal → Scalar score ─────────────────────────────────────────────────
    def _signal_to_score(self, signal: str) -> float:
        """Convert agent signals to numeric scores [0, 1]."""
        return {
            "BUY":          1.0,
            "STRONG_BUY":   1.0,
            "HOLD_BULLISH": 0.65,
            "HOLD":         0.5,
            "HOLD_BEARISH": 0.38,
            "SELL":         0.0,
            "STRONG_SELL":  0.0,
            "VALID":        0.5,
            "INVALID":      0.0,
            "LOW_RISK":     1.0,
            "MEDIUM_RISK":  0.5,
            "HIGH_RISK":    0.0,
            "BULLISH":      1.0,
            "BEARISH":      0.0,
            "NEUTRAL":      0.5,
            "VOLATILE":     0.35,
            "ERROR":        0.5,
        }.get(signal, 0.5)

    def _risk_adjustment(self, risk_signal: str, base_score: float) -> float:
        """Adjust combined score based on risk level."""
        if risk_signal == "HIGH_RISK" and base_score > 0.5:
            return base_score * 0.7
        if risk_signal == "LOW_RISK" and base_score > 0.5:
            return min(1.0, base_score * 1.1)
        return base_score

    # ── Main Decide ───────────────────────────────────────────────────────────
    def decide(self, agent_outputs: Dict[str, Dict], symbol: str) -> Dict:
        """
        Combine all agent outputs into a final BUY / HOLD / SELL recommendation.

        Key v2.0 changes:
          1. ML score derived from P(BUY) instead of binary probability.
          2. Expected return = (P_buy - P_sell) × 15%.
          3. Risk Agent veto: if veto_active → downgrade any BUY to HOLD.
          4. Enhanced risk section includes stop_loss, target_price, RRR, max_allocation_pct.
        """
        try:
            technical    = agent_outputs.get("technical",    {})
            ml           = agent_outputs.get("ml",           {})
            news         = agent_outputs.get("news",         {})
            fundamental  = agent_outputs.get("fundamental",  {})
            market_regime = agent_outputs.get("market_regime", {})
            risk         = agent_outputs.get("risk",         {})

            # ── ML 3-class probabilities ─────────────────────────────────────
            ml_details  = ml.get("details", {})
            p_buy       = float(ml_details.get("p_buy",  0.33))
            p_hold      = float(ml_details.get("p_hold", 0.34))
            p_sell      = float(ml_details.get("p_sell", 0.33))

            # ML signal score: use P(BUY) as the ML contribution to combined score
            # This replaces the old binary probability approach
            ml_score_direct = p_buy  # ranges [0, 1]

            # ── Per-agent signal scores ───────────────────────────────────────
            technical_score   = self._signal_to_score(technical.get("signal",     "HOLD"))
            news_score        = self._signal_to_score(news.get("signal",           "HOLD"))
            fundamental_score = self._signal_to_score(fundamental.get("signal",    "HOLD"))
            regime_score      = self._signal_to_score(market_regime.get("signal",  "NEUTRAL"))
            risk_score_raw    = self._signal_to_score(risk.get("signal",           "MEDIUM_RISK"))

            # ── Weighted combination ──────────────────────────────────────────
            combined_score = (
                technical_score    * self.weights.TECHNICAL +
                ml_score_direct    * self.weights.ML_PREDICTION +
                news_score         * self.weights.SENTIMENT +
                fundamental_score  * self.weights.FUNDAMENTAL +
                regime_score       * self.weights.MARKET_REGIME +
                risk_score_raw     * self.weights.RISK
            )
            combined_score = self._risk_adjustment(risk.get("signal", "MEDIUM_RISK"), combined_score)

            # ── Confidence ────────────────────────────────────────────────────
            confidences = [
                technical.get("confidence",    0.5),
                ml.get("confidence",           0.5),
                news.get("confidence",         0.5),
                fundamental.get("confidence",  0.5),
                market_regime.get("confidence", 0.5),
                risk.get("confidence",         0.5),
            ]
            avg_confidence = float(np.mean(confidences))
            scores_list = [technical_score, ml_score_direct, news_score, fundamental_score, regime_score]
            agreement      = float(1 - np.std(scores_list))
            final_confidence = 0.6 * avg_confidence + 0.4 * agreement
            final_confidence = max(0.1, min(0.95, final_confidence))

            # ── Final signal ──────────────────────────────────────────────────
            if combined_score >= 0.65:
                final_signal = "BUY"
            elif combined_score <= 0.35:
                final_signal = "SELL"
            else:
                final_signal = "HOLD"

            # ── Risk Agent Veto (Revision 5) ──────────────────────────────────
            risk_details   = risk.get("details", {})
            veto_active    = bool(risk_details.get("veto_active", False))
            veto_triggered = False
            if veto_active and final_signal == "BUY":
                final_signal   = "HOLD"
                veto_triggered = True
                logger.info(
                    f"[DecisionEngine] ⛔ VETO applied for {symbol}: BUY → HOLD "
                    f"(VIX={risk_details.get('vix_elevated')} | NIFTY_below_200EMA={risk_details.get('nifty_below_200ema')})"
                )

            # ── Expected return (Revision 1) ─────────────────────────────────
            # (P_buy - P_sell) × 15% — accounts for both upside and downside probability
            expected_return_5d = (p_buy - p_sell) * 0.15

            # ── Display scores ────────────────────────────────────────────────
            tech_score_display = technical.get("details", {}).get("technical_score", 50)
            fund_score_display = fundamental.get("details", {}).get("fundamental_score", 50)

            # ── Enhanced risk outputs (Revision 5) ───────────────────────────
            stop_loss_price   = risk_details.get("stop_loss_price")
            target_price      = risk_details.get("target_price")
            rrr               = risk_details.get("rrr")
            max_allocation_pct = risk_details.get("max_allocation_pct")

            # ── Explainability ────────────────────────────────────────────────
            reasons_for, reasons_against, all_risk_factors = [], [], []

            for agent_name, output in agent_outputs.items():
                reasoning = output.get("reasoning", [])
                sig       = output.get("signal", "")

                if sig in ("BUY", "HOLD_BULLISH", "BULLISH", "LOW_RISK"):
                    reasons_for.extend(reasoning[:2])
                elif sig in ("SELL", "HOLD_BEARISH", "BEARISH", "HIGH_RISK"):
                    reasons_against.extend(reasoning[:2])

                all_risk_factors.extend(output.get("details", {}).get("risk_factors", []))

            if veto_triggered:
                reasons_against.insert(0, "⛔ Risk veto applied: BUY downgraded to HOLD (VIX > 22 or NIFTY below 200-EMA)")

            reasons_for      = list(dict.fromkeys(reasons_for))[:6]
            reasons_against  = list(dict.fromkeys(reasons_against))[:4]
            all_risk_factors = list(dict.fromkeys(all_risk_factors))[:5]

            if not reasons_for and final_signal in ("BUY", "HOLD"):
                reasons_for = [
                    f"Combined score {combined_score:.2f} in {'BUY' if combined_score >= 0.65 else 'HOLD'} zone",
                    f"Technical score {tech_score_display}/100",
                    f"ML → P(BUY)={p_buy:.0%} P(HOLD)={p_hold:.0%} P(SELL)={p_sell:.0%}",
                ]

            regime = market_region = market_regime.get("details", {}).get("regime", "Unknown")

            return {
                "symbol":              symbol,
                "recommendation":      final_signal,
                "confidence":          round(final_confidence * 100, 1),
                "combined_score":      round(combined_score, 3),
                "expected_return_5d":  round(expected_return_5d * 100, 2),
                "technical_score":     round(float(tech_score_display), 1) if isinstance(tech_score_display, (int, float)) else 50,
                "fundamental_score":   round(float(fund_score_display), 1) if isinstance(fund_score_display, (int, float)) else 50,
                # ── ML 3-class probabilities ─────────────────────────────────
                "ml_prediction": {
                    "signal":     ml.get("signal", "HOLD"),
                    "p_buy":      round(p_buy,  3),
                    "p_hold":     round(p_hold, 3),
                    "p_sell":     round(p_sell, 3),
                    "confidence": ml.get("confidence", 0.5),
                    "oof_metrics": ml_details.get("oof_metrics", {}),
                },
                # ── Sentiment ────────────────────────────────────────────────
                "sentiment": {
                    "label":      news.get("details", {}).get("aggregated_sentiment", {}).get("overall_sentiment", "neutral"),
                    "net_score":  news.get("details", {}).get("aggregated_sentiment", {}).get("net_score", 0),
                    "confidence": news.get("confidence", 0.3),
                    "articles":   news.get("details", {}).get("article_count", 0),
                },
                # ── Enhanced risk (Revision 5) ────────────────────────────────
                "risk": {
                    "level":             risk_details.get("risk_level", "Medium"),
                    "score":             risk_details.get("risk_score", 25),
                    "veto_active":       veto_active,
                    "veto_triggered":    veto_triggered,
                    "stop_loss_price":   stop_loss_price,
                    "target_price":      target_price,
                    "rrr":               rrr,
                    "max_allocation_pct": max_allocation_pct,
                    "factors":           all_risk_factors,
                },
                # ── Market regime ─────────────────────────────────────────────
                "market_regime": {
                    "regime": regime,
                    "signal": market_regime.get("signal", "NEUTRAL"),
                },
                # ── Explainability ────────────────────────────────────────────
                "explanation": {
                    "for":          reasons_for,
                    "against":      reasons_against,
                    "risk_factors": all_risk_factors,
                },
                # ── Per-agent breakdown ───────────────────────────────────────
                "agent_breakdown": {
                    "technical":     {"signal": technical.get("signal"),    "confidence": technical.get("confidence"),    "score": technical_score},
                    "ml":            {"signal": ml.get("signal"),           "confidence": ml.get("confidence"),           "p_buy": p_buy, "p_hold": p_hold, "p_sell": p_sell},
                    "sentiment":     {"signal": news.get("signal"),         "confidence": news.get("confidence"),         "score": news_score},
                    "fundamental":   {"signal": fundamental.get("signal"),  "confidence": fundamental.get("confidence"),  "score": fundamental_score},
                    "market_regime": {"signal": market_regime.get("signal"), "confidence": market_regime.get("confidence"), "score": regime_score},
                    "risk":          {"signal": risk.get("signal"),         "confidence": risk.get("confidence"),         "veto_active": veto_active},
                },
                "weights_used": {
                    "technical":     self.weights.TECHNICAL,
                    "ml":            self.weights.ML_PREDICTION,
                    "sentiment":     self.weights.SENTIMENT,
                    "fundamental":   self.weights.FUNDAMENTAL,
                    "market_regime": self.weights.MARKET_REGIME,
                    "risk":          self.weights.RISK,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"DecisionEngine failed: {e}", exc_info=True)
            return {
                "symbol":             symbol,
                "recommendation":     "HOLD",
                "confidence":         30.0,
                "combined_score":     0.5,
                "expected_return_5d": 0.0,
                "technical_score":    50,
                "fundamental_score":  50,
                "ml_prediction":      {"signal": "HOLD", "p_buy": 0.33, "p_hold": 0.34, "p_sell": 0.33, "confidence": 0.3},
                "sentiment":          {"label": "neutral", "net_score": 0, "confidence": 0.3, "articles": 0},
                "risk": {
                    "level": "Medium", "score": 30, "veto_active": False, "veto_triggered": False,
                    "stop_loss_price": None, "target_price": None, "rrr": None, "max_allocation_pct": 10,
                    "factors": [f"Decision error: {e}"],
                },
                "market_regime":   {"regime": "Unknown", "signal": "NEUTRAL"},
                "explanation":     {"for": ["Fallback HOLD"], "against": [str(e)], "risk_factors": []},
                "agent_breakdown": {},
                "weights_used":    {},
                "timestamp":       datetime.now().isoformat(),
                "error":           str(e),
            }


decision_engine = DecisionEngine()
