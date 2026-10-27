"""
ML Models — 4-Model Stack with 3-Class Calibrated Probabilities (Revision 2)
═════════════════════════════════════════════════════════════════════════════

Models (LSTM and Transformers removed):
  1. LogisticRegression   — interpretable baseline (multi-class, multinomial)
  2. RandomForestClassifier — non-linear baseline
  3. XGBClassifier        — primary production model (multi:softprob)
  4. LGBMClassifier       — benchmark challenger (multiclass)

Each model outputs P(SELL), P(HOLD), P(BUY) — 3 calibrated probabilities.

Label encoding for tree-based models:
  XGBoost and LightGBM require labels in {0, 1, 2}.
  Internal mapping: -1 → 0 (SELL), 0 → 1 (HOLD), +1 → 2 (BUY)
  Probabilities returned in order [SELL, HOLD, BUY].

Ensemble:
  Average of softmax probabilities across all 4 models.
  Final decision: argmax(avg_probas) → {BUY, HOLD, SELL}.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, List, Optional
import logging
import os

logger = logging.getLogger(__name__)

# ─── Label mapping ────────────────────────────────────────────────────────────
# 3-class labels from features.py: -1 (SELL), 0 (HOLD), +1 (BUY)
# XGB/LGB require contiguous 0-based integers
LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}      # SELL→0, HOLD→1, BUY→2
IDX_TO_LABEL = {0: -1, 1: 0, 2: 1}       # reverse
CLASS_NAMES  = ["SELL", "HOLD", "BUY"]   # index order


def _encode_labels(y: pd.Series) -> np.ndarray:
    """Map {-1, 0, 1} → {0, 1, 2} for XGB/LGB."""
    return np.array([LABEL_TO_IDX[v] for v in y])


def _decode_probas(probas: np.ndarray) -> dict:
    """
    Convert [P(SELL), P(HOLD), P(BUY)] array (shape N×3) into named dict
    for the last sample (used for live inference).
    """
    return {
        "p_sell": float(probas[-1, 0]),
        "p_hold": float(probas[-1, 1]),
        "p_buy":  float(probas[-1, 2]),
    }


def _evaluate(y_true_raw, y_pred_idx, probas: np.ndarray) -> dict:
    """Compute evaluation metrics for a 3-class problem."""
    # Convert predicted indices back to {-1, 0, 1} for reporting
    y_pred_label = np.array([IDX_TO_LABEL[i] for i in y_pred_idx])
    y_true_label = np.array(list(y_true_raw))

    # Convert true labels to indices for sklearn
    y_true_idx = _encode_labels(pd.Series(y_true_label))

    acc        = accuracy_score(y_true_idx, y_pred_idx)
    macro_f1   = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)

    return {
        "accuracy":    round(acc, 4),
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "class_report": classification_report(
            y_true_idx, y_pred_idx,
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }


class StockMLModels:
    """
    Container for all 4 trained models.
    Provides unified train / predict / ensemble interface.
    """

    def __init__(self):
        self.models:      Dict = {}
        self.scalers:     Dict = {}
        self.feature_cols: List[str] = []
        self.metrics:     Dict = {}

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    def train_logistic_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val:   pd.DataFrame, y_val:   pd.Series,
    ) -> Dict:
        """Multinomial Logistic Regression — interpretable 3-class baseline."""
        logger.info("[Models] Training Logistic Regression (multinomial)")

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_va_sc = scaler.transform(X_val)

        y_tr_idx = _encode_labels(y_train)
        y_va_idx = _encode_labels(y_val)

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=0.1,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_tr_sc, y_tr_idx)

        y_pred      = model.predict(X_va_sc)
        probas      = model.predict_proba(X_va_sc)   # shape (N, 3) → [SELL, HOLD, BUY]
        metrics     = _evaluate(y_val, y_pred, probas)

        self.models["logistic"]  = model
        self.scalers["logistic"] = scaler
        self.metrics["logistic"] = metrics

        logger.info(f"[Models] LR → macro-F1={metrics['macro_f1']:.3f}, acc={metrics['accuracy']:.3f}")
        return {"model": model, "scaler": scaler, "metrics": metrics, "probas": probas}

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    def train_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val:   pd.DataFrame, y_val:   pd.Series,
    ) -> Dict:
        """Non-linear 3-class baseline."""
        logger.info("[Models] Training Random Forest (3-class)")

        y_tr_idx = _encode_labels(y_train)
        y_va_idx = _encode_labels(y_val)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X_train.values, y_tr_idx)

        y_pred  = model.predict(X_val.values)
        probas  = model.predict_proba(X_val.values)
        metrics = _evaluate(y_val, y_pred, probas)

        self.models["random_forest"]  = model
        self.metrics["random_forest"] = metrics

        importance = dict(zip(X_train.columns, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info(f"[Models] RF  → macro-F1={metrics['macro_f1']:.3f}, acc={metrics['accuracy']:.3f}")
        return {"model": model, "metrics": metrics, "probas": probas, "feature_importance": top_features}

    # ── 3. XGBoost (multi:softprob) ───────────────────────────────────────────
    def train_xgboost(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val:   pd.DataFrame, y_val:   pd.Series,
    ) -> Dict:
        """Primary production model with calibrated 3-class probabilities."""
        logger.info("[Models] Training XGBoost (multi:softprob, num_class=3)")

        y_tr_idx = _encode_labels(y_train)
        y_va_idx = _encode_labels(y_val)

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
            use_label_encoder=False,
        )
        model.fit(
            X_train.values, y_tr_idx,
            eval_set=[(X_val.values, y_va_idx)],
            verbose=False,
        )

        y_pred  = model.predict(X_val.values)
        probas  = model.predict_proba(X_val.values)   # shape (N, 3)
        metrics = _evaluate(y_val, y_pred, probas)

        self.models["xgboost"]  = model
        self.metrics["xgboost"] = metrics

        importance = dict(zip(X_train.columns, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info(f"[Models] XGB → macro-F1={metrics['macro_f1']:.3f}, acc={metrics['accuracy']:.3f}")
        return {"model": model, "metrics": metrics, "probas": probas, "feature_importance": top_features}

    # ── 4. LightGBM ──────────────────────────────────────────────────────────
    def train_lightgbm(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val:   pd.DataFrame, y_val:   pd.Series,
    ) -> Dict:
        """Benchmark challenger — efficient 3-class multiclass."""
        logger.info("[Models] Training LightGBM (multiclass, num_class=3)")

        y_tr_idx = _encode_labels(y_train)
        y_va_idx = _encode_labels(y_val)

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(
            X_train.values, y_tr_idx,
            eval_set=[(X_val.values, y_va_idx)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=-1)],
        )

        y_pred  = model.predict(X_val.values)
        probas  = model.predict_proba(X_val.values)
        metrics = _evaluate(y_val, y_pred, probas)

        self.models["lightgbm"]  = model
        self.metrics["lightgbm"] = metrics

        logger.info(f"[Models] LGB → macro-F1={metrics['macro_f1']:.3f}, acc={metrics['accuracy']:.3f}")
        return {"model": model, "metrics": metrics, "probas": probas}

    # ── Train All 4 ───────────────────────────────────────────────────────────
    def train_all(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val:   pd.DataFrame, y_val:   pd.Series,
    ) -> Dict:
        """Train all 4 models and return combined results dict."""
        results = {}
        results["logistic"]     = self.train_logistic_regression(X_train, y_train, X_val, y_val)
        results["random_forest"] = self.train_random_forest(X_train, y_train, X_val, y_val)
        results["xgboost"]      = self.train_xgboost(X_train, y_train, X_val, y_val)
        results["lightgbm"]     = self.train_lightgbm(X_train, y_train, X_val, y_val)

        best = max(results.items(), key=lambda kv: kv[1]["metrics"]["macro_f1"])
        logger.info(f"[Models] Best model: {best[0]} (macro-F1={best[1]['metrics']['macro_f1']:.3f})")
        self.feature_cols = list(X_train.columns)
        return results

    # ── Inference ─────────────────────────────────────────────────────────────
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align inference DataFrame to the stored training feature columns.
        - Missing columns are filled with 0.
        - Extra columns are dropped.
        - Column order matches training exactly.
        This prevents "Feature shape mismatch" errors in XGBoost / LightGBM.
        """
        if not self.feature_cols:
            return X

        # Add missing columns with 0
        for col in self.feature_cols:
            if col not in X.columns:
                X = X.copy()
                X[col] = 0.0

        # Select only training columns in the correct order
        return X[self.feature_cols]

    def predict_proba_3class(
        self, X: pd.DataFrame, model_name: str = "xgboost"
    ) -> np.ndarray:
        """
        Return 3-class probability array (shape N×3) for a given model.
        Columns: [P(SELL), P(HOLD), P(BUY)]
        """
        if model_name not in self.models:
            model_name = next(iter(self.models), None)
        if model_name is None:
            return np.array([[0.33, 0.34, 0.33]] * len(X))

        model  = self.models[model_name]
        X_aligned = self._align_features(X)
        X_input = X_aligned.values

        if model_name == "logistic" and "logistic" in self.scalers:
            X_input = self.scalers["logistic"].transform(X_input)

        return model.predict_proba(X_input)

    def predict_ensemble(self, X: pd.DataFrame) -> Dict:
        """
        Ensemble all available models by averaging 3-class softmax probabilities.

        Returns:
            {
              p_buy:      float,
              p_hold:     float,
              p_sell:     float,
              prediction: str  ("BUY" | "HOLD" | "SELL"),
              confidence: float,   # 1 - entropy of avg proba
              model_count: int,
              all_probas: dict per model
            }
        """
        all_probas_list = []
        per_model = {}

        for name in self.models:
            try:
                p = self.predict_proba_3class(X, model_name=name)
                all_probas_list.append(p)
                per_model[name] = _decode_probas(p)
            except Exception as e:
                logger.warning(f"[Models] Ensemble skip {name}: {e}")

        if not all_probas_list:
            return {
                "p_buy": 0.33, "p_hold": 0.34, "p_sell": 0.33,
                "prediction": "HOLD", "confidence": 0.5, "model_count": 0,
                "all_probas": {},
            }

        avg_p = np.mean(all_probas_list, axis=0)   # shape (N, 3)
        last  = avg_p[-1]                           # single sample

        p_sell, p_hold, p_buy = float(last[0]), float(last[1]), float(last[2])
        best_idx   = int(np.argmax(last))
        prediction = CLASS_NAMES[best_idx]          # "SELL" | "HOLD" | "BUY"

        # Confidence: 1 - normalised entropy
        eps     = 1e-9
        entropy = -np.sum(last * np.log(last + eps))
        max_ent = np.log(3)
        confidence = float(1.0 - entropy / max_ent)

        return {
            "p_buy":       round(p_buy,  4),
            "p_hold":      round(p_hold, 4),
            "p_sell":      round(p_sell, 4),
            "prediction":  prediction,
            "confidence":  round(confidence, 4),
            "model_count": len(all_probas_list),
            "all_probas":  per_model,
        }
