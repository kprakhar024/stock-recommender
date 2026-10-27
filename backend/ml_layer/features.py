"""
Feature Engineering — 3-Class Dynamic Volatility Labeling (Revision 1)
═══════════════════════════════════════════════════════════════════════

Target Variable: 3-class classification
  +1  BUY   — 5-day forward return > Threshold_t
   0  HOLD  — |5-day forward return| <= Threshold_t
  -1  SELL  — 5-day forward return < -Threshold_t

Dynamic Threshold (per row):
  rolling_std_20  = std of daily returns over 20 trading days
  Threshold_t     = max(0.015, 1.0 * rolling_std_20 * sqrt(5))
  (Note: rolling_std_20 is already a daily std, so we scale to 5-day horizon.)

Backtest Window:
  Labels are restricted to 2022-01-01 – 2024-12-31 when calling
  prepare_dataset(restrict_to_backtest=True). Live inference uses all dates.

Macro Features (added for richer signal):
  - NIFTY_return_5d   : 5-day return of NIFTY 50
  - VIX_level         : India VIX close
  - NIFTY_above_200EMA: binary flag (1 / 0)
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Backtest evaluation window (Revision 3)
BACKTEST_START = "2022-01-01"
BACKTEST_END   = "2024-12-31"


class FeatureEngineer:
    def __init__(
        self,
        lookback: int = 60,
        horizon:  int = 5,
        min_threshold: float = 0.015,
        threshold_multiplier: float = 1.0,
    ):
        self.lookback              = lookback
        self.horizon               = horizon
        self.min_threshold         = min_threshold
        self.threshold_multiplier  = threshold_multiplier

    # ── Core Feature Set ──────────────────────────────────────────────────────
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML features from a DataFrame that already has technical indicators
        (output of TechnicalIndicators.add_all_indicators).
        All features are strictly past/present values — no look-ahead.
        """
        df = df.copy()

        # ── Existing indicator features ──────────────────────────────────────
        base_cols = [
            "RSI_14", "RSI_7",
            "MACD", "MACD_signal", "MACD_histogram",
            "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "BB_width", "BB_position", "ATR_14",
            "Volume_ratio", "MFI", "Momentum_10", "Momentum_20",
            "Volatility_10", "Volatility_20", "Volatility_30",
            "Price_position_20", "Return_1d", "Return_5d", "Return_10d", "Return_20d",
            "Returns", "Log_Returns",
            # New from Revision 3 (indicators.py)
            "Rolling_Std_20",       # raw daily return std — used in dynamic threshold
            "Trend_200",            # long-term regime binary
        ]
        for col in base_cols:
            if col not in df.columns:
                df[col] = 0

        # ── Engineered features ───────────────────────────────────────────────
        df["RSI_MA_diff"]       = df["RSI_14"] - df["RSI_14"].rolling(10).mean()
        df["Price_SMA20_diff"]  = (df["Close"] - df["SMA_20"]) / df["SMA_20"].replace(0, np.nan)
        df["Price_SMA50_diff"]  = (df["Close"] - df["SMA_50"]) / df["SMA_50"].replace(0, np.nan)
        df["SMA20_SMA50_diff"]  = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"].replace(0, np.nan)
        df["Volume_trend"]      = df["Volume"].rolling(10).mean() / df["Volume"].rolling(30).mean() - 1
        df["High_Low_range"]    = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
        df["Open_Close_diff"]   = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)

        # Lag features (past returns only — strictly no future leakage)
        for lag in [1, 2, 3, 5]:
            df[f"Return_lag_{lag}"]       = df["Returns"].shift(lag)
            df[f"RSI_lag_{lag}"]          = df["RSI_14"].shift(lag)
            df[f"Volume_ratio_lag_{lag}"] = df["Volume_ratio"].shift(lag)

        return df

    # ── Dynamic 3-Class Label ─────────────────────────────────────────────────
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 3-class labels using dynamic volatility thresholding.

        Label_3class ∈ {-1 (SELL), 0 (HOLD), +1 (BUY)}

        Threshold_t = max(min_threshold, multiplier * rolling_std_20 * sqrt(horizon))

        where rolling_std_20 is the std of *daily* returns over 20 days,
        scaled to the 5-day forward horizon.
        """
        df = df.copy()

        # 5-day forward return (leakage-safe: label uses future, features use past)
        df["Future_Return"] = df["Close"].shift(-self.horizon) / df["Close"] - 1

        # Dynamic per-row threshold
        # rolling_std_20 is already computed in indicators.py; use it if present
        if "Rolling_Std_20" in df.columns and df["Rolling_Std_20"].notna().sum() > 0:
            rolling_std = df["Rolling_Std_20"]
        else:
            rolling_std = df["Returns"].rolling(20).std()

        threshold_t = np.maximum(
            self.min_threshold,
            self.threshold_multiplier * rolling_std * np.sqrt(self.horizon),
        )
        df["Threshold_t"] = threshold_t

        # 3-class label
        df["Label_3class"] = np.where(
            df["Future_Return"] > threshold_t,   1,   # BUY
            np.where(
                df["Future_Return"] < -threshold_t, -1,  # SELL
                0,                                        # HOLD
            ),
        )

        # Keep regression label for diagnostics
        df["Label_return"] = df["Future_Return"]

        return df

    # ── Macro Features (optional enrichment) ─────────────────────────────────
    def merge_macro_features(
        self,
        df: pd.DataFrame,
        nifty_df: Optional[pd.DataFrame] = None,
        vix_df:   Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Merge NIFTY and VIX data as additional macro features.
        Aligns on DatetimeIndex; forward-fills to handle market holidays.
        """
        df = df.copy()

        if nifty_df is not None and not nifty_df.empty:
            nifty_5d_return = nifty_df["Close"].pct_change(5).rename("NIFTY_return_5d")
            nifty_ema200    = nifty_df["Close"].ewm(span=200, adjust=False).mean()
            nifty_above_200 = (nifty_df["Close"] > nifty_ema200).astype(int).rename("NIFTY_above_200EMA")

            df = df.join(nifty_5d_return, how="left")
            df = df.join(nifty_above_200, how="left")
            df["NIFTY_return_5d"]    = df["NIFTY_return_5d"].ffill()
            df["NIFTY_above_200EMA"] = df["NIFTY_above_200EMA"].ffill()

        if vix_df is not None and not vix_df.empty:
            vix_close = vix_df["Close"].rename("VIX_level")
            df = df.join(vix_close, how="left")
            df["VIX_level"] = df["VIX_level"].ffill()

        return df

    # ── Feature Column Selection ──────────────────────────────────────────────
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Return the list of valid feature columns, excluding OHLCV raw values
        and any look-ahead (Future_*, Label_*) columns.
        """
        exclude = {
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Future_Return", "Threshold_t",
            "Label_3class", "Label_return",
        }
        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and not c.startswith("Future")
            and not c.startswith("Label")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        return feature_cols

    # ── Full Dataset Preparation ──────────────────────────────────────────────
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        nifty_df:  Optional[pd.DataFrame] = None,
        vix_df:    Optional[pd.DataFrame] = None,
        restrict_to_backtest: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepare X, y_3class for training.

        Args:
            df:                     Price + indicator DataFrame (post add_all_indicators).
            nifty_df:               Optional NIFTY OHLCV DataFrame for macro features.
            vix_df:                 Optional VIX DataFrame for macro features.
            restrict_to_backtest:   If True, restrict label window to 2022-01-01 – 2024-12-31.

        Returns:
            (X, y, feature_cols)
        """
        df = self.create_features(df)
        df = self.create_labels(df)

        if nifty_df is not None or vix_df is not None:
            df = self.merge_macro_features(df, nifty_df=nifty_df, vix_df=vix_df)

        if restrict_to_backtest:
            df = df.loc[BACKTEST_START:BACKTEST_END]
            logger.info(f"[FeatureEngineer] Restricted to backtest window: {BACKTEST_START} – {BACKTEST_END}")

        feature_cols = self.get_feature_columns(df)

        # Drop rows with any NaN in features or label
        required = feature_cols + ["Label_3class"]
        df_clean = df.dropna(subset=required)

        n_total = len(df_clean)
        if n_total < 100:
            logger.warning(f"[FeatureEngineer] Only {n_total} clean samples — results may be unstable.")

        # Log class distribution
        dist = df_clean["Label_3class"].value_counts().to_dict()
        logger.info(
            f"[FeatureEngineer] Label distribution → "
            f"BUY(+1): {dist.get(1, 0)} | HOLD(0): {dist.get(0, 0)} | SELL(-1): {dist.get(-1, 0)}"
        )

        X = df_clean[feature_cols]
        y = df_clean["Label_3class"]

        return X, y, feature_cols

    # ── Walk-Forward Compatible Split ────────────────────────────────────────
    def chronological_split(
        self,
        df: pd.DataFrame,
        train_end: str,
        val_end:   str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Hard chronological split.
        Walk-Forward TimeSeriesSplit (n_splits=5) is handled in trainer.py.
        This method is kept for single-pass quick diagnostics.
        """
        train_boundary = pd.Timestamp(train_end)
        val_boundary = pd.Timestamp(val_end)
        index_tz = getattr(df.index, "tz", None)
        if index_tz is not None:
            train_boundary = train_boundary.tz_localize(index_tz) if train_boundary.tzinfo is None else train_boundary.tz_convert(index_tz)
            val_boundary = val_boundary.tz_localize(index_tz) if val_boundary.tzinfo is None else val_boundary.tz_convert(index_tz)
        else:
            train_boundary = train_boundary.tz_localize(None) if train_boundary.tzinfo is not None else train_boundary
            val_boundary = val_boundary.tz_localize(None) if val_boundary.tzinfo is not None else val_boundary

        train_df = df[df.index <= train_boundary]
        val_df   = df[(df.index > train_boundary) & (df.index <= val_boundary)]
        test_df  = df[df.index > val_boundary]
        return train_df, val_df, test_df


# Singleton
feature_engineer = FeatureEngineer()
