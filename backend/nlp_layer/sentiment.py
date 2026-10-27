"""
Financial Sentiment Analysis - Indian Market

Two analyzers:
  1. FinancialSentimentAnalyzer (VADER + Indian finance lexicon)
     — used as a fast, offline-capable fallback.

  2. FinBERTAnalyzer (ProsusAI/finbert via HuggingFace Transformers)
     — used by the News/Sentiment Agent for live inference.
     — lazy-loaded; falls back to VADER if unavailable.

NOTE: Sentiment is used ONLY for live inference.
      Historical backtests do NOT include news sentiment.
"""
from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import re
import logging

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except:
    VADER_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

class FinancialSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Finance-specific lexicon adjustments for Indian market
        self.bullish_keywords = {
            'beat', 'beats', 'upgrade', 'buy', 'growth', 'profit', 'surge', 'rally',
            'gain', 'jump', 'soar', 'strong', 'robust', 'expansion', 'order', 'bag',
            'partnership', 'dividend', 'bonus', 'record', 'high', 'bullish', 'positive',
            'outperform', 'overweight', 'accumulate', 'target raised', 'fii buying',
            'stake increase', 'investment', 'renewable', 'ai', 'tech'
        }
        
        self.bearish_keywords = {
            'miss', 'downgrade', 'sell', 'loss', 'fall', 'drop', 'decline', 'weak',
            'slump', 'crash', 'bearish', 'negative', 'underperform', 'concern',
            'regulatory', 'scrutiny', 'pledge', 'debt', 'slowdown', 'competition',
            'dispute', 'penalty', 'fine', 'probe', 'underweight'
        }
        
        # Try to load FinBERT if available (optional, heavy)
        self.finbert = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # Don't load by default to keep demo light - use VADER + lexicon
                # self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                pass
            except:
                pass
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze single text"""
        if not text:
            return {"positive": 0.33, "neutral": 0.34, "negative": 0.33, "compound": 0, "label": "Neutral"}
        
        text_lower = text.lower()
        
        # VADER baseline
        if self.vader:
            vader_scores = self.vader.polarity_scores(text)
            compound = vader_scores['compound']
        else:
            compound = 0
        
        # Finance lexicon overlay
        bullish_count = sum(1 for word in self.bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_keywords if word in text_lower)
        
        # Lexicon adjustment
        lexicon_score = (bullish_count - bearish_count) / max(1, bullish_count + bearish_count)
        # Blend VADER and lexicon
        blended_compound = 0.6 * compound + 0.4 * lexicon_score
        
        # Convert to probabilities
        if blended_compound >= 0.3:
            positive = 0.6 + min(0.35, blended_compound * 0.5)
            neutral = 0.3 - blended_compound * 0.2
            negative = 0.1
            label = "Positive"
        elif blended_compound <= -0.3:
            positive = 0.1
            neutral = 0.3 + blended_compound * 0.2  # blended negative, so this reduces
            negative = 0.6 + min(0.35, abs(blended_compound) * 0.5)
            label = "Negative"
        else:
            # Neutral
            positive = 0.25 + max(0, blended_compound) * 0.5
            negative = 0.25 + max(0, -blended_compound) * 0.5
            neutral = 1 - positive - negative
            label = "Neutral"
        
        # Normalize
        total = positive + neutral + negative
        positive /= total
        neutral /= total
        negative /= total
        
        return {
            "positive": round(positive, 3),
            "neutral": round(neutral, 3),
            "negative": round(negative, 3),
            "compound": round(blended_compound, 3),
            "label": label,
            "bullish_keywords": bullish_count,
            "bearish_keywords": bearish_count
        }
    
    def analyze_news_list(self, news_list: List[Dict]) -> Dict:
        """Analyze list of news with time-weighting"""
        if not news_list:
            return {
                "overall_sentiment": "Neutral",
                "sentiment_score": 0,
                "positive": 0.33,
                "neutral": 0.34,
                "negative": 0.33,
                "confidence": 0.5,
                "article_count": 0,
                "time_weighted_score": 0,
                "details": []
            }
        
        now = datetime.now(timezone.utc)
        analyzed = []
        
        for article in news_list:
            title = article.get('title', '')
            sentiment = self.analyze_text(title)
            
            # Time weighting - exponential decay, recent matters more
            try:
                pub_time_str = article.get('publish_time', '')
                if pub_time_str:
                    # Parse ISO format
                    if pub_time_str.endswith('Z'):
                        pub_time = datetime.fromisoformat(pub_time_str.replace('Z', '+00:00'))
                    else:
                        pub_time = datetime.fromisoformat(pub_time_str)
                        if pub_time.tzinfo is None:
                            pub_time = pub_time.replace(tzinfo=timezone.utc)
                else:
                    pub_time = now - pd.Timedelta(hours=12)
            except:
                pub_time = now - pd.Timedelta(hours=12)
            
            hours_ago = (now - pub_time).total_seconds() / 3600
            # Weight: 1 for now, 0.5 for 24h ago, 0.25 for 48h, etc. Half-life ~24h
            time_weight = np.exp(-hours_ago / 24.0 * np.log(2))
            
            analyzed.append({
                "title": title,
                "sentiment": sentiment,
                "publish_time": article.get('publish_time'),
                "hours_ago": round(hours_ago, 1),
                "time_weight": round(time_weight, 3),
                "publisher": article.get('publisher', 'Unknown')
            })
        
        # Aggregate with time weighting
        total_weight = sum(a['time_weight'] for a in analyzed)
        if total_weight == 0:
            total_weight = 1
        
        weighted_compound = sum(a['sentiment']['compound'] * a['time_weight'] for a in analyzed) / total_weight
        weighted_positive = sum(a['sentiment']['positive'] * a['time_weight'] for a in analyzed) / total_weight
        weighted_negative = sum(a['sentiment']['negative'] * a['time_weight'] for a in analyzed) / total_weight
        weighted_neutral = sum(a['sentiment']['neutral'] * a['time_weight'] for a in analyzed) / total_weight
        
        # Overall label
        if weighted_compound >= 0.2:
            overall = "Positive"
        elif weighted_compound <= -0.2:
            overall = "Negative"
        else:
            overall = "Neutral"
        
        # Confidence based on agreement and volume
        compounds = [a['sentiment']['compound'] for a in analyzed]
        agreement = 1 - np.std(compounds) if len(compounds) > 1 else 0.5
        # More articles = higher confidence up to a point
        volume_factor = min(1.0, len(analyzed) / 8.0)
        confidence = 0.5 * agreement + 0.5 * volume_factor
        confidence = max(0.3, min(0.95, confidence))
        
        return {
            "overall_sentiment": overall,
            "sentiment_score": round(weighted_compound, 3),
            "positive": round(weighted_positive, 3),
            "neutral": round(weighted_neutral, 3),
            "negative": round(weighted_negative, 3),
            "confidence": round(confidence, 3),
            "article_count": len(analyzed),
            "time_weighted_score": round(weighted_compound, 3),
            "details": analyzed[:10],  # top 10
            "raw_compounds": compounds
        }
    
    def get_sentiment_features(self, sentiment_result: Dict) -> Dict:
        """Convert sentiment to ML features"""
        return {
            "sentiment_score": sentiment_result.get('sentiment_score', 0),
            "sentiment_positive": sentiment_result.get('positive', 0.33),
            "sentiment_negative": sentiment_result.get('negative', 0.33),
            "sentiment_confidence": sentiment_result.get('confidence', 0.5),
            "news_count": sentiment_result.get('article_count', 0),
            "sentiment_label_encoded": 1 if sentiment_result.get('overall_sentiment') == 'Positive' else -1 if sentiment_result.get('overall_sentiment') == 'Negative' else 0
        }

sentiment_analyzer = FinancialSentimentAnalyzer()


# ─── FinBERT Analyzer (Revision 5) ───────────────────────────────────────────
class FinBERTAnalyzer:
    """
    Wraps ProsusAI/finbert for domain-accurate financial sentiment.
    Lazy-loaded to avoid adding startup overhead.

    Usage:
        result = finbert_analyzer.score("Reliance beats Q2 estimates")
        # → {"positive": 0.87, "negative": 0.05, "neutral": 0.08,
        #    "label": "positive", "score": 0.87}

        batch = finbert_analyzer.score_batch([text1, text2, ...])
    """

    def __init__(self):
        self._pipeline = None
        self._loaded = False
        self._vader_fallback = FinancialSentimentAnalyzer()

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("[FinBERT] Loading ProsusAI/finbert pipeline…")
            self._pipeline = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("[FinBERT] Loaded ✓")
        except Exception as e:
            logger.warning(f"[FinBERT] Could not load ({e}). Using VADER fallback.")
            self._pipeline = None
        self._loaded = True

    def score(self, text: str) -> Dict:
        """
        Score a single text string.
        Returns {positive, negative, neutral, label, score}.
        """
        self._load()
        if not text or not text.strip():
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "label": "neutral", "score": 0.34}

        if self._pipeline is not None:
            try:
                raw = self._pipeline(text)[0]  # list of {label, score} dicts
                scores = {r["label"].lower(): r["score"] for r in raw}
                label = max(scores, key=scores.get)
                return {**scores, "label": label, "score": round(scores[label], 4)}
            except Exception as e:
                logger.warning(f"[FinBERT] Inference error: {e}")

        # VADER fallback
        vader_result = self._vader_fallback.analyze_text(text)
        label = vader_result["label"].lower()
        return {
            "positive": vader_result["positive"],
            "negative": vader_result["negative"],
            "neutral":  vader_result["neutral"],
            "label":    label,
            "score":    vader_result.get("positive" if label == "positive" else
                                         "negative" if label == "negative" else "neutral", 0.34),
        }

    def score_batch(self, texts: List[str]) -> List[Dict]:
        """
        Score a batch of texts. More efficient than calling score() in a loop.
        """
        self._load()
        if not texts:
            return []

        if self._pipeline is not None:
            try:
                raw_batch = self._pipeline(texts)
                results = []
                for raw in raw_batch:
                    scores = {r["label"].lower(): r["score"] for r in raw}
                    label = max(scores, key=scores.get)
                    results.append({**scores, "label": label, "score": round(scores[label], 4)})
                return results
            except Exception as e:
                logger.warning(f"[FinBERT] Batch inference error: {e}")

        # VADER fallback (per item)
        return [self.score(t) for t in texts]


# Singletons
finbert_analyzer = FinBERTAnalyzer()
