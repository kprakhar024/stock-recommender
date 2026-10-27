"""
Walk-Forward Training Pipeline (Revision 2)
════════════════════════════════════════════
Replaces simple chronological_split with proper Walk-Forward TimeSeriesSplit
(n_splits=5) to prevent data leakage and produce reliable OOF metrics.

Flow:
  1. Fetch price data + indicators (from 2018 for warmup)
  2. Feature engineering + 3-class dynamic labeling
  3. Merge macro features (NIFTY, VIX)
  4. Walk-Forward TSS: train each of the 4 models across 5 folds
  5. Report OOF macro-F1 per model
  6. Final model: retrain on full training window, evaluate on held-out test set
  7. Return trained models ready for live inference

Backtest window: features.BACKTEST_START – features.BACKTEST_END (2022–2024)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit

try:
    from .features import FeatureEngineer, BACKTEST_START, BACKTEST_END
    from .models import StockMLModels, _encode_labels, IDX_TO_LABEL, CLASS_NAMES
    from ..data_pipeline.price_fetcher import fetcher
    from ..data_pipeline.indicators import TechnicalIndicators
except ImportError:
    from backend.ml_layer.features import FeatureEngineer, BACKTEST_START, BACKTEST_END
    from backend.ml_layer.models import StockMLModels, _encode_labels, IDX_TO_LABEL, CLASS_NAMES
    from backend.data_pipeline.price_fetcher import fetcher
    from backend.data_pipeline.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

N_SPLITS  = 5   # Walk-Forward folds
MIN_TRAIN = 200  # minimum trading days required in train fold


class UnifiedTrainer:
    """
    Orchestrates Walk-Forward training for a single stock symbol.

    Usage:
        trainer = UnifiedTrainer()
        result  = trainer.train(symbol="RELIANCE.NS")
        # result["models"]     → trained StockMLModels instance
        # result["oof_metrics"] → per-fold OOF macro-F1
        # result["final_metrics"] → metrics on held-out test set
    """

    def __init__(self, n_splits: int = N_SPLITS):
        self.n_splits        = n_splits
        self.feature_engineer = FeatureEngineer()
        self._trained_models: Dict[str, StockMLModels] = {}

    # ── Public API ────────────────────────────────────────────────────────────
    def train(
        self,
        symbol:              str,
        nifty_df:            Optional[pd.DataFrame] = None,
        vix_df:              Optional[pd.DataFrame] = None,
        restrict_to_backtest: bool = True,
    ) -> Dict:
        """
        Full walk-forward training pipeline for a symbol.

        Returns dict with:
          - models        : StockMLModels (trained on full backtest window)
          - oof_metrics   : OOF macro-F1 per model across all folds
          - final_metrics : held-out metrics (last 20% of data)
          - feature_cols  : list of feature column names
          - label_dist    : class distribution in training set
        """
        logger.info(f"[Trainer] Starting Walk-Forward training for {symbol}")

        # ── Data preparation ─────────────────────────────────────────────────
        price_df = fetcher.fetch_price_data(symbol, start="2018-01-01")
        price_df = TechnicalIndicators.add_all_indicators(price_df)

        X, y, feature_cols = self.feature_engineer.prepare_dataset(
            price_df,
            nifty_df=nifty_df,
            vix_df=vix_df,
            restrict_to_backtest=restrict_to_backtest,
        )

        if len(X) < MIN_TRAIN:
            logger.warning(
                f"[Trainer] {symbol}: only {len(X)} samples after cleaning "
                f"(min required: {MIN_TRAIN}). Skipping."
            )
            return {"error": f"Insufficient data ({len(X)} rows)"}

        label_dist = y.value_counts().to_dict()

        # ── Walk-Forward TimeSeriesSplit ─────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof_metrics = {name: [] for name in ["logistic", "random_forest", "xgboost", "lightgbm"]}

        X_arr = X.values
        y_arr = _encode_labels(y)

        logger.info(f"[Trainer] Running {self.n_splits}-fold Walk-Forward TSS on {len(X)} samples")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
            if len(train_idx) < MIN_TRAIN:
                logger.debug(f"[Trainer] Fold {fold}: train size {len(train_idx)} < {MIN_TRAIN}, skipping")
                continue

            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            fold_models = StockMLModels()
            fold_results = fold_models.train_all(X_tr, y_tr, X_va, y_va)

            for model_name, res in fold_results.items():
                oof_metrics[model_name].append(res["metrics"]["macro_f1"])

            logger.info(
                f"[Trainer] Fold {fold+1}/{self.n_splits} | "
                f"train={len(train_idx)} val={len(val_idx)} | "
                f"XGB macro-F1={fold_results['xgboost']['metrics']['macro_f1']:.3f}"
            )

        # Average OOF metrics
        avg_oof = {
            name: round(float(np.mean(scores)), 4) if scores else 0.0
            for name, scores in oof_metrics.items()
        }
        logger.info(f"[Trainer] OOF macro-F1 averages: {avg_oof}")

        # ── Final model: train on full window, eval on last 20% ──────────────
        split_idx   = int(len(X) * 0.80)
        X_train_all = X.iloc[:split_idx]
        y_train_all = y.iloc[:split_idx]
        X_test      = X.iloc[split_idx:]
        y_test      = y.iloc[split_idx:]

        final_models = StockMLModels()
        if len(X_test) > 10:
            final_results = final_models.train_all(X_train_all, y_train_all, X_test, y_test)
            final_metrics = {k: v["metrics"] for k, v in final_results.items()}
        else:
            # Not enough test data — train on all, report OOF as proxy
            final_models.train_all(X_train_all, y_train_all, X_train_all.tail(50), y_train_all.tail(50))
            final_metrics = {"note": "test set too small; OOF metrics used as proxy"}

        final_models.feature_cols = feature_cols
        self._trained_models[symbol] = final_models

        return {
            "symbol":        symbol,
            "models":        final_models,
            "oof_metrics":   avg_oof,
            "final_metrics": final_metrics,
            "feature_cols":  feature_cols,
            "label_dist":    label_dist,
            "n_samples":     len(X),
        }

    def get_models(self, symbol: str) -> Optional[StockMLModels]:
        """Return cached trained models for a symbol, or None if not trained."""
        return self._trained_models.get(symbol)

    def predict_live(self, symbol: str, X_latest: pd.DataFrame) -> Dict:
        """
        Run ensemble inference on the latest feature row.
        Trains models on demand if not already cached.
        """
        if symbol not in self._trained_models:
            logger.info(f"[Trainer] No cached model for {symbol}, training now…")
            result = self.train(symbol)
            if "error" in result:
                return {
                    "prediction": "HOLD", "p_buy": 0.33, "p_hold": 0.34,
                    "p_sell": 0.33, "confidence": 0.0, "error": result["error"]
                }

        models = self._trained_models[symbol]
        return models.predict_ensemble(X_latest)


# Singleton
unified_trainer = UnifiedTrainer()
