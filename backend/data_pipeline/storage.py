"""
DuckDB + Parquet Storage Layer
Replaces PostgreSQL with local file-based storage.
- Raw OHLCV per ticker: data/raw/ohlcv/<SYMBOL>.parquet
- Benchmarks:           data/raw/benchmarks/nifty.parquet, vix.parquet
- Feature matrices:     data/features/<SYMBOL>_features.parquet
"""
import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import duckdb

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[3]  # stock-recommender/
DATA_DIR        = _BASE_DIR / "data"
RAW_OHLCV_DIR   = DATA_DIR / "raw" / "ohlcv"
BENCHMARK_DIR   = DATA_DIR / "raw" / "benchmarks"
FEATURES_DIR    = DATA_DIR / "features"


def _ensure_dirs() -> None:
    for d in [RAW_OHLCV_DIR, BENCHMARK_DIR, FEATURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


class ParquetStore:
    """
    Thin persistence layer over local Parquet files.
    DuckDB is used for ad-hoc SQL queries when needed.
    """

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _ohlcv_path(symbol: str) -> Path:
        safe = symbol.replace(".", "_").replace("^", "")
        return RAW_OHLCV_DIR / f"{safe}.parquet"

    def save_ohlcv(self, df: pd.DataFrame, symbol: str) -> None:
        """Persist OHLCV dataframe to parquet. Index must be DatetimeIndex."""
        path = self._ohlcv_path(symbol)
        df.reset_index().to_parquet(path, index=False, compression="snappy")
        logger.info(f"[ParquetStore] Saved OHLCV for {symbol} → {path}")

    def load_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load OHLCV from parquet; returns None if file not found."""
        path = self._ohlcv_path(symbol)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        # Restore DatetimeIndex
        date_col = next((c for c in ["Date", "Datetime", "index"] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        df.index.name = None
        logger.info(f"[ParquetStore] Loaded OHLCV for {symbol} ({len(df)} rows)")
        return df

    def ohlcv_is_fresh(self, symbol: str, max_age_days: int = 1) -> bool:
        """Return True if cached parquet file was modified within max_age_days."""
        path = self._ohlcv_path(symbol)
        if not path.exists():
            return False
        import time
        age_days = (time.time() - path.stat().st_mtime) / 86400
        return age_days < max_age_days

    # ── Benchmarks ────────────────────────────────────────────────────────────
    @staticmethod
    def _benchmark_path(name: str) -> Path:
        safe = name.replace("^", "").lower()
        return BENCHMARK_DIR / f"{safe}.parquet"

    def save_benchmark(self, df: pd.DataFrame, name: str) -> None:
        path = self._benchmark_path(name)
        df.reset_index().to_parquet(path, index=False, compression="snappy")
        logger.info(f"[ParquetStore] Saved benchmark {name} → {path}")

    def load_benchmark(self, name: str) -> Optional[pd.DataFrame]:
        path = self._benchmark_path(name)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        date_col = next((c for c in ["Date", "Datetime", "index"] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        df.index.name = None
        return df

    def benchmark_is_fresh(self, name: str, max_age_days: int = 1) -> bool:
        path = self._benchmark_path(name)
        if not path.exists():
            return False
        import time
        age_days = (time.time() - path.stat().st_mtime) / 86400
        return age_days < max_age_days

    # ── Feature Matrices ──────────────────────────────────────────────────────
    @staticmethod
    def _features_path(symbol: str) -> Path:
        safe = symbol.replace(".", "_").replace("^", "")
        return FEATURES_DIR / f"{safe}_features.parquet"

    def save_features(self, df: pd.DataFrame, symbol: str) -> None:
        path = self._features_path(symbol)
        df.reset_index().to_parquet(path, index=False, compression="snappy")
        logger.info(f"[ParquetStore] Saved features for {symbol} → {path}")

    def load_features(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self._features_path(symbol)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        date_col = next((c for c in ["Date", "Datetime", "index"] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        df.index.name = None
        return df

    # ── DuckDB Ad-Hoc Queries ─────────────────────────────────────────────────
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute an arbitrary SQL query over parquet files.
        Reference files directly in SQL using their absolute paths.

        Example:
            store.query(
                f"SELECT * FROM '{RAW_OHLCV_DIR}/RELIANCE_NS.parquet' WHERE Close > 2500"
            )
        """
        con = duckdb.connect(database=":memory:")
        result = con.execute(sql).df()
        con.close()
        return result

    def query_ohlcv(self, symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
        """Convenience: load OHLCV via DuckDB with optional date filter."""
        path = self._ohlcv_path(symbol)
        if not path.exists():
            return pd.DataFrame()
        where_clauses = []
        if start:
            where_clauses.append(f"Date >= '{start}'")
        if end:
            where_clauses.append(f"Date <= '{end}'")
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"SELECT * FROM '{path}' {where_sql} ORDER BY Date"
        df = self.query(sql)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df.index.name = None
        return df

    def list_cached_symbols(self) -> list:
        """Return list of symbols with cached OHLCV parquet files."""
        return [
            p.stem.replace("_NS", ".NS").replace("_BO", ".BO")
            for p in RAW_OHLCV_DIR.glob("*.parquet")
        ]


# Singleton
parquet_store = ParquetStore()
