"""
News Fetcher — Live RSS + FinBERT (Revision 3)
═══════════════════════════════════════════════
Design Decision:
  News/Sentiment is used EXCLUSIVELY for live inference, NOT for historical backtesting.
  Backtesting evaluates models on: Technicals + Market Regime + Macro + Fundamentals.

Live flow:
  1. Fetch headlines from Google News RSS for the given NSE ticker.
  2. Score each headline with FinBERT (ProsusAI/finbert).
  3. Return structured list of {title, url, published, sentiment, score}.

Fallback:
  If network unavailable or FinBERT not loaded, returns an empty list so the
  NewsSentimentAgent can report "No live news available" cleanly.
"""
import feedparser
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# ─── Google News RSS template ─────────────────────────────────────────────────
# hl=en-IN  → English, India locale; gl=IN → India geo; ceid=IN:en → correct feed
_RSS_TEMPLATE = (
    "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
)
_MAX_ARTICLES = 15     # cap to avoid FinBERT latency spike
_FETCH_TIMEOUT = 10    # seconds


class LiveNewsFetcher:
    """
    Fetches and scores news articles in real-time using Google News RSS + FinBERT.
    Does NOT maintain any historical news cache — strictly live.
    """

    def __init__(self):
        self._finbert = None          # lazy-loaded
        self._finbert_loaded = False

    # ── FinBERT lazy loader ───────────────────────────────────────────────────
    def _load_finbert(self):
        """Lazy-load FinBERT pipeline to avoid startup overhead."""
        if self._finbert_loaded:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("[NewsFetcher] Loading FinBERT (ProsusAI/finbert)…")
            self._finbert = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,           # return all 3 class scores
                truncation=True,
                max_length=512,
            )
            self._finbert_loaded = True
            logger.info("[NewsFetcher] FinBERT loaded ✓")
        except Exception as e:
            logger.warning(f"[NewsFetcher] FinBERT unavailable ({e}). Will use neutral fallback.")
            self._finbert = None
            self._finbert_loaded = True   # mark as "attempted" so we don't retry every call

    def _score_with_finbert(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Score a list of texts with FinBERT.
        Returns list of dicts: {positive, negative, neutral} probabilities.
        """
        self._load_finbert()
        if self._finbert is None:
            # Neutral fallback
            return [{"positive": 0.33, "negative": 0.33, "neutral": 0.34} for _ in texts]

        results = []
        try:
            raw = self._finbert(texts)
            for item in raw:
                # item is a list of {"label": ..., "score": ...}
                scores = {r["label"].lower(): r["score"] for r in item}
                results.append({
                    "positive": scores.get("positive", 0.0),
                    "negative": scores.get("negative", 0.0),
                    "neutral":  scores.get("neutral", 0.0),
                })
        except Exception as e:
            logger.warning(f"[NewsFetcher] FinBERT scoring error: {e}")
            results = [{"positive": 0.33, "negative": 0.33, "neutral": 0.34} for _ in texts]
        return results

    # ── RSS Fetching ──────────────────────────────────────────────────────────
    def _build_query(self, symbol: str) -> str:
        """Build a search query from a symbol like RELIANCE.NS → 'RELIANCE NSE India stock'."""
        base = symbol.replace(".NS", "").replace(".BO", "").strip()
        return quote_plus(f"{base} NSE India stock")

    def _parse_published(self, entry) -> str:
        """Safely parse a feedparser entry's published date to ISO string."""
        try:
            t = entry.get("published_parsed") or entry.get("updated_parsed")
            if t:
                dt = datetime(*t[:6], tzinfo=timezone.utc)
                return dt.isoformat()
        except Exception:
            pass
        return datetime.now(timezone.utc).isoformat()

    def fetch_live_rss(self, symbol: str, max_articles: int = _MAX_ARTICLES) -> List[Dict]:
        """
        Fetch live news articles from Google News RSS and score them with FinBERT.

        Returns:
            List of dicts, each containing:
              - title      : str
              - url        : str
              - published  : ISO datetime string
              - sentiment  : "positive" | "negative" | "neutral"
              - score      : float (dominant class score)
              - scores     : {positive, negative, neutral} probabilities
        """
        query = self._build_query(symbol)
        rss_url = _RSS_TEMPLATE.format(query=query)

        try:
            logger.info(f"[NewsFetcher] Fetching RSS for {symbol}: {rss_url}")
            feed = feedparser.parse(rss_url, request_headers={"User-Agent": "Mozilla/5.0"})

            entries = feed.get("entries", [])[:max_articles]
            if not entries:
                logger.warning(f"[NewsFetcher] No RSS entries for {symbol}")
                return []

            titles = [e.get("title", "") for e in entries]

            # Score all titles in one batch
            sentiments = self._score_with_finbert(titles)

            articles = []
            for entry, sent in zip(entries, sentiments):
                dominant = max(sent, key=sent.get)
                articles.append({
                    "title":     entry.get("title", ""),
                    "url":       entry.get("link", ""),
                    "published": self._parse_published(entry),
                    "sentiment": dominant,
                    "score":     round(sent[dominant], 4),
                    "scores":    {k: round(v, 4) for k, v in sent.items()},
                })

            logger.info(f"[NewsFetcher] {len(articles)} articles scored for {symbol}")
            return articles

        except Exception as e:
            logger.error(f"[NewsFetcher] RSS fetch failed for {symbol}: {e}")
            return []

    def aggregate_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Aggregate article-level FinBERT scores into a single sentiment summary.

        Returns:
            {
              overall_sentiment: "positive" | "negative" | "neutral",
              avg_positive: float,
              avg_negative: float,
              avg_neutral: float,
              net_score: float,   # avg_positive - avg_negative ∈ [-1, 1]
              article_count: int,
              top_positive: [titles],
              top_negative: [titles],
            }
        """
        if not articles:
            return {
                "overall_sentiment": "neutral",
                "avg_positive": 0.33,
                "avg_negative": 0.33,
                "avg_neutral":  0.34,
                "net_score":    0.0,
                "article_count": 0,
                "top_positive": [],
                "top_negative": [],
            }

        avg_pos = sum(a["scores"]["positive"] for a in articles) / len(articles)
        avg_neg = sum(a["scores"]["negative"] for a in articles) / len(articles)
        avg_neu = sum(a["scores"]["neutral"]  for a in articles) / len(articles)
        net     = round(avg_pos - avg_neg, 4)

        if avg_pos > avg_neg and avg_pos > avg_neu:
            overall = "positive"
        elif avg_neg > avg_pos and avg_neg > avg_neu:
            overall = "negative"
        else:
            overall = "neutral"

        top_pos = [a["title"] for a in articles if a["sentiment"] == "positive"][:3]
        top_neg = [a["title"] for a in articles if a["sentiment"] == "negative"][:3]

        return {
            "overall_sentiment": overall,
            "avg_positive":  round(avg_pos, 4),
            "avg_negative":  round(avg_neg, 4),
            "avg_neutral":   round(avg_neu, 4),
            "net_score":     net,
            "article_count": len(articles),
            "top_positive":  top_pos,
            "top_negative":  top_neg,
        }


# Singleton
news_fetcher = LiveNewsFetcher()
