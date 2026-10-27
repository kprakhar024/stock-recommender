"""
Indian Stock Price Data Pipeline - NSE/BSE
Research-grade with validation, leakage prevention, and Parquet caching.

Default fetch window:
  - start="2018-01-01" for indicator warmup (200-day SMA needs ~10 months)
  - Backtest evaluation labels are restricted to 2022-01-01 – 2024-12-31
    upstream in ml_layer/features.py
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from datetime import datetime
import logging

try:
    from .storage import parquet_store
except ImportError:
    from backend.data_pipeline.storage import parquet_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evaluation window for backtesting (labels restricted here)
BACKTEST_START = "2022-01-01"
BACKTEST_END   = "2024-12-31"

class IndianStockDataFetcher:
    def __init__(self):
        # In-process memory cache for the current session
        self._mem_cache: dict = {}

    def fetch_price_data(
        self,
        symbol: str,
        start: str = "2018-01-01",
        end: str = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for Indian stocks (RELIANCE.NS, TCS.NS …).

        Resolution order:
          1. In-memory session cache  (instant)
          2. Local Parquet cache      (fast, persisted across runs)
          3. yfinance live download   (network, writes back to Parquet)
          4. Synthetic fallback       (offline / testing)
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        cache_key = f"{symbol}_{start}_{end}"

        # 1. Memory cache
        if not force_refresh and cache_key in self._mem_cache:
            return self._mem_cache[cache_key]

        # 2. Parquet cache (fresh = updated within last 1 calendar day)
        if not force_refresh and parquet_store.ohlcv_is_fresh(symbol, max_age_days=1):
            df_parquet = parquet_store.load_ohlcv(symbol)
            if df_parquet is not None and not df_parquet.empty:
                # Slice to requested window
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                index_tz = getattr(df_parquet.index, "tz", None)
                if index_tz is not None:
                    start_ts = start_ts.tz_localize(index_tz) if start_ts.tzinfo is None else start_ts.tz_convert(index_tz)
                    end_ts = end_ts.tz_localize(index_tz) if end_ts.tzinfo is None else end_ts.tz_convert(index_tz)
                else:
                    start_ts = start_ts.tz_localize(None) if start_ts.tzinfo is not None else start_ts
                    end_ts = end_ts.tz_localize(None) if end_ts.tzinfo is not None else end_ts
                df_parquet = df_parquet.loc[
                    (df_parquet.index >= start_ts) &
                    (df_parquet.index <= end_ts)
                ]
                if not df_parquet.empty:
                    self._mem_cache[cache_key] = df_parquet
                    return df_parquet

        # 3. Live yfinance download
        try:
            logger.info(f"[PriceFetcher] Downloading {symbol} from {start} to {end}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=False)

            if df.empty:
                logger.warning(f"No data from yfinance for {symbol}, using synthetic")
                df = self._generate_synthetic_data(symbol, start, end)
            else:
                df = self._validate_and_clean(df, symbol)
                # Persist full download (from start) so future calls hit Parquet
                parquet_store.save_ohlcv(df, symbol)

            self._mem_cache[cache_key] = df
            return df

        except Exception as e:
            logger.error(f"[PriceFetcher] Error fetching {symbol}: {e}, using synthetic")
            df = self._generate_synthetic_data(symbol, start, end)
            self._mem_cache[cache_key] = df
            return df

    def fetch_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch the latest non-adjusted quote separately from the historical cache.

        Yahoo's ``fast_info.lastPrice`` is the freshest value available through
        yfinance.  If it is unavailable, use the latest dated daily close and
        label it as a previous close rather than calling it live.
        """
        ticker = yf.Ticker(symbol)

        try:
            intraday = ticker.history(period="1d", interval="1m", auto_adjust=False)
            intraday = intraday.dropna(subset=["Close"])
            if not intraday.empty:
                live_price = intraday["Close"].iloc[-1]
                previous_close = None
                daily = ticker.history(period="5d", interval="1d", auto_adjust=False)
                if not daily.empty:
                    daily = daily.dropna(subset=["Close"])
                    if len(daily) >= 2:
                        previous_close = float(daily["Close"].iloc[-2])
                return {
                    "price": float(live_price),
                    "previous_close": previous_close,
                    "price_type": "LIVE",
                    "as_of": intraday.index[-1].isoformat(),
                    "source": "Yahoo Finance intraday",
                    "is_synthetic": False,
                }
        except Exception as exc:
            logger.warning("[PriceFetcher] Live quote unavailable for %s: %s", symbol, exc)

        try:
            history = ticker.history(period="5d", interval="1d", auto_adjust=False)
            history = history.dropna(subset=["Close"])
            if not history.empty:
                latest = history.iloc[-1]
                previous = history.iloc[-2]["Close"] if len(history) > 1 else None
                latest_timestamp = history.index[-1]
                return {
                    "price": float(latest["Close"]),
                    "previous_close": float(previous) if previous is not None else None,
                    "price_type": "PREVIOUS_CLOSE",
                    "as_of": str(latest_timestamp.isoformat()),
                    "source": "Yahoo Finance daily close",
                    "is_synthetic": False,
                }
        except Exception as exc:
            logger.warning("[PriceFetcher] Daily close unavailable for %s: %s", symbol, exc)

        try:
            downloaded = yf.download(
                symbol,
                period="5d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if not downloaded.empty:
                close = downloaded["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna()
                if not close.empty:
                    previous = close.iloc[-2] if len(close) > 1 else None
                    return {
                        "price": float(close.iloc[-1]),
                        "previous_close": float(previous) if previous is not None else None,
                        "price_type": "PREVIOUS_CLOSE",
                        "as_of": close.index[-1].isoformat(),
                        "source": "Yahoo Finance daily close",
                        "is_synthetic": False,
                    }
        except Exception as exc:
            logger.warning("[PriceFetcher] Download fallback unavailable for %s: %s", symbol, exc)

        return {
            "price": None,
            "previous_close": None,
            "price_type": "UNAVAILABLE",
            "as_of": None,
            "source": None,
            "is_synthetic": True,
        }

    def _validate_and_clean(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Data validation - critical for research"""
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Check for missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        if missing_pct.max() > 5:
            logger.warning(f"{symbol} has high missing: {missing_pct}")
        
        # Forward fill small gaps, but log
        df = df.ffill().bfill()
        
        # Validate OHLC logic
        invalid_ohlc = (df['High'] < df['Low']) | (df['High'] < df['Open']) | (df['High'] < df['Close']) | (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
        if invalid_ohlc.any():
            logger.warning(f"{symbol} has {invalid_ohlc.sum()} invalid OHLC rows")
            df = df[~invalid_ohlc]
        
        # Add returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        
        return df
    
    def _generate_synthetic_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """High-quality synthetic data for demo/research when offline"""
        np.random.seed(hash(symbol) % 2**32)
        dates = pd.date_range(start=start, end=end, freq='B')
        n = len(dates)
        
        # Different base prices for different stocks
        base_prices = {
            "RELIANCE.NS": 2500, "TCS.NS": 3500, "HDFCBANK.NS": 1600,
            "INFY.NS": 1500, "ICICIBANK.NS": 1000, "SBIN.NS": 500
        }
        base = base_prices.get(symbol, 1000 + hash(symbol) % 2000)
        
        # Geometric Brownian Motion with Indian market characteristics
        mu = 0.12 / 252  # 12% annual drift
        sigma = 0.25 / np.sqrt(252)  # 25% annual vol
        
        returns = np.random.normal(mu, sigma, n)
        # Add occasional jumps (Indian market feature)
        jumps = np.random.choice([0, 1], n, p=[0.98, 0.02]) * np.random.normal(-0.03, 0.05, n)
        returns += jumps
        
        prices = base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.005, n))
        df['Open'].iloc[0] = base
        df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, 0.01, n)))
        df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, 0.01, n)))
        df['Volume'] = np.random.lognormal(14, 0.5, n).astype(int)
        df['Adj Close'] = df['Close']
        
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        
        return df
    
    def fetch_market_data(
        self,
        symbol: str = "^NSEI",
        start: str = "2018-01-01",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch NIFTY 50 index for market regime detection. Cached in Parquet."""
        if not force_refresh and parquet_store.benchmark_is_fresh(symbol):
            cached = parquet_store.load_benchmark(symbol)
            if cached is not None and not cached.empty:
                return cached
        try:
            df = yf.Ticker(symbol).history(start=start, auto_adjust=False)
            if df.empty:
                return self._generate_synthetic_data(symbol, start, datetime.now().strftime("%Y-%m-%d"))
            df = self._validate_and_clean(df, symbol)
            parquet_store.save_benchmark(df, symbol)
            return df
        except Exception:
            return self._generate_synthetic_data(symbol, start, datetime.now().strftime("%Y-%m-%d"))

    def fetch_india_vix(
        self,
        start: str = "2018-01-01",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch India VIX for risk assessment. Cached in Parquet."""
        vix_name = "^INDIAVIX"
        if not force_refresh and parquet_store.benchmark_is_fresh(vix_name):
            cached = parquet_store.load_benchmark(vix_name)
            if cached is not None and not cached.empty:
                return cached
        try:
            df = yf.Ticker(vix_name).history(start=start)
            if df.empty:
                raise ValueError("Empty VIX response")
            parquet_store.save_benchmark(df, vix_name)
            return df
        except Exception:
            # Synthetic fallback
            dates = pd.date_range(start=start, end=datetime.now(), freq="B")
            vix = 15 + 5 * np.sin(np.arange(len(dates)) / 50) + np.random.normal(0, 2, len(dates))
            return pd.DataFrame({"Close": np.clip(vix, 10, 45)}, index=dates)

    def fetch_benchmarks(self, start: str = "2018-01-01", force_refresh: bool = False) -> dict:
        """
        Convenience: fetch and cache both NIFTY 50 and India VIX together.
        Returns {"nifty": DataFrame, "vix": DataFrame}.
        """
        return {
            "nifty": self.fetch_market_data("^NSEI", start=start, force_refresh=force_refresh),
            "vix":   self.fetch_india_vix(start=start, force_refresh=force_refresh),
        }


# Singleton
fetcher = IndianStockDataFetcher()
