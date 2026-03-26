#!/usr/bin/env python3
"""
Three-level cascade hierarchical classifier for sepsis outcomes.

Pipeline
--------
  Level 1 – sepsis (binary)
    Trained on all patients with a valid sepsis label.

  Level 2 – resultado_hemo_grouped (multiclass)
    Trained on original features + Level-1 OOF sepsis probability.
    Very rare blood-culture classes are dropped automatically.

  Level 3 – resistente_cefalosporina (binary)
    Trained on original features + Level-1 OOF sepsis probability
    + Level-2 OOF per-class probabilities from resultado_hemo_grouped.
    Row scope: only patients with a positive blood culture
               (resultado_hemo_grouped != "NEGATIVE").

During training, out-of-fold (OOF) predictions from each level are used as
features for the next level to prevent label leakage.

At inference time the three trained models are applied sequentially.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import shap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
JOB_ID = os.environ.get("SLURM_JOB_ID", 1)
TODAY = datetime.today().strftime("%Y%m%d%H%M%S")

FOCUS_MAP = {
    1: "pulmonar",
    2: "intraabdominal",
    3: "biliar",
    4: "urinario",
    5: "cardiovascular",
    6: "piel",
    7: "sistema nervioso central",
    8: "cateter venoso",
    9: "vías altas respiratorias",
    10: "osteoarticular",
    11: "genital",
    12: "desconocido",
}

# All potential target columns – none of these should appear as features
TARGET_REMOVE = [
    "sepsis",
    "resultado_hemo",
    "resultado_hemo_grouped",
    "all_cult_org",
    "infected_yes_no",
    "bmr_etiologia",
    "fenotipo_resistencia",
    "fenotipo_resistencia_grouped",
    "resistente_cefalosporina",
]

DELETE_COLUMNS = [
    "qsofa",
    "vasopresores",
    "hipotension",
    "freq_bacteria",
    "freq_bac_foco",
    "Unnamed: 0",
    "person_id",
    "fecha_ingreso_urgencias",
    "fecha_ingreso_urgencias_x",
    "shock_septico",
    "sintoma_nan",
    "fecha_nacimiento",
    "codigo_postal",
    "center",
    "dag",
    "ultima_fecha",
    "mujer_gestante",
]

FOCUS_TO_EXCLUDE = {
    "piel",
    "osteoarticular",
    "biliar",
    "genital",
    "sistema nervioso central",
    "cateter venoso",
    "vías altas respiratorias",
    "cardiovascular",
}

# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def safe_drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        try:
            df = df.drop(columns=col)
        except KeyError:
            print(f"Warning: column '{col}' not found – skipping drop.")
    return df


def load_processed_dataframe(csv_path: Path, cols_to_delete: List[str]) -> pd.DataFrame:
    """Load dataset and apply focus filter. No target-based row filtering here."""
    df = pd.read_csv(csv_path)
    if "foco" in df.columns:
        df["foco"] = df["foco"].map(FOCUS_MAP).fillna(df["foco"])
        df = df[~df["foco"].isin(FOCUS_TO_EXCLUDE)]
    df = safe_drop_columns(df, cols_to_delete)
    return df


def impute_missing_values(loaded_df: pd.DataFrame, exclude_cols: set) -> pd.DataFrame:
    """Impute missing values; target/weight columns are kept as-is.

    Strategy
    --------
    * Binary (0/1) numeric columns → mode (SimpleImputer).
    * All other numeric columns (continuous AND low-cardinality ordinal scores
      such as SOFA sub-scores, bilirrubina 0-4, snc_glasgow 0-4, respiracion
      0-3) → KNNImputer(k=10, distance-weighted).  Using KNN for ordinal
      scores is strictly better than mode because it exploits the patient's
      other measurements; mode always collapses to the most common value
      regardless of clinical context.
    * String/category columns → mode (SimpleImputer).
    * Missing-indicator flags: for every numeric column whose missing rate
      exceeds 10 %, a companion ``<col>_missing`` binary column is added
      *before* imputation.  Tree models can use these flags directly as a
      signal that the original value was absent (informative missingness,
      e.g. SOFA not recorded often means the patient was less severe).
    """
    df_copy = loaded_df.drop(columns=list(exclude_cols), errors="ignore").copy()
    numeric_cols = df_copy.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()

    binary_cols = [c for c in numeric_cols if set(df_copy[c].dropna().unique()) <= {0, 1}]
    # Ordinal clinical scores (low-cardinality) and continuous values both
    # benefit from multivariate KNN — route all non-binary numeric to KNN.
    knn_cols = [c for c in numeric_cols if c not in binary_cols]

    # Add missing-indicator flags for numerics with >10 % missingness.
    # These flags remain even after imputation so the model can learn from them.
    high_missing = [
        c for c in numeric_cols
        if df_copy[c].isna().mean() > 0.10
    ]
    for col in high_missing:
        df_copy[f"{col}_missing"] = df_copy[col].isna().astype(int)

    if binary_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df_copy[binary_cols] = imp.fit_transform(df_copy[binary_cols]).astype(int)
    if knn_cols:
        # k=10 gives more stable estimates than k=5 for datasets of ~3 000+ rows.
        imp = KNNImputer(n_neighbors=10, weights="distance")
        df_copy[knn_cols] = imp.fit_transform(df_copy[knn_cols])
        # Re-cast originally-integer ordinal columns back to int after KNN.
        for col in knn_cols:
            if loaded_df[col].dropna().astype(float).apply(float.is_integer).all():
                df_copy[col] = df_copy[col].round().astype(int)
    if categorical_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df_copy[categorical_cols] = imp.fit_transform(df_copy[categorical_cols]).astype(str)
    print("Imputed values and included marker columns for high_missings")
    for col in exclude_cols:
        if col in loaded_df.columns:
            df_copy[col] = loaded_df[col]
    return df_copy


def compute_balanced_sample_weight(
    labels: pd.Series, base_sample_weight: Optional[pd.Series] = None
) -> pd.Series:
    classes = np.unique(labels)
    cw = compute_class_weight("balanced", classes=classes, y=labels)
    balanced = labels.map(dict(zip(classes, cw)))
    if base_sample_weight is not None:
        balanced = balanced * pd.Series(base_sample_weight, index=labels.index)
    return balanced


def _build_ranking_model(model_type: str, y: pd.Series, random_state: int) -> object:
    """Return a fast, fixed-param model of *model_type* used only for importance ranking.

    Parameters are intentionally conservative (shallow, few estimators) so the
    ranking step finishes quickly.  The result is never used for prediction.
    """
    n_classes = int(y.nunique())
    is_multi = n_classes > 2

    if model_type == "lgbm":
        params: Dict = {
            "n_estimators": 200, "num_leaves": 31, "max_depth": 6,
            "class_weight": "balanced", "n_jobs": 1,
            "random_state": random_state, "verbosity": -1,
        }
        if is_multi:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
        else:
            params["objective"] = "binary"
        return LGBMClassifier(**params)

    elif model_type == "xgb":
        if is_multi:
            return XGBClassifier(
                objective="multi:softprob", num_class=n_classes,
                n_estimators=200, max_depth=6,
                random_state=random_state, n_jobs=1, verbosity=0,
            )
        n_pos = float((y == 1).sum())
        spw = float(len(y) - n_pos) / max(n_pos, 1.0)
        return XGBClassifier(
            objective="binary:logistic", n_estimators=200, max_depth=6,
            scale_pos_weight=spw,
            random_state=random_state, n_jobs=1, verbosity=0,
        )

    elif model_type == "catb":
        return CatBoostClassifier(
            iterations=200, depth=6,
            auto_class_weights="Balanced", verbose=False, random_state=42,
        )

    else:  # "rf" or anything else
        return RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced_subsample",
            random_state=random_state, n_jobs=1,
        )


def _shap_importance(model, X: pd.DataFrame) -> np.ndarray:
    """Return mean absolute SHAP values (shape [n_features,]) using TreeExplainer.

    Handles binary (2-D array), multiclass list-of-arrays (XGB/RF style), and
    multiclass 3-D array (LGBM style) transparently.

    Raises ImportError if shap is not installed (caller handles the fallback).
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    if isinstance(sv, list):
        # XGB / RF multiclass: list of (n_samples, n_features), one per class
        return np.mean([np.abs(a).mean(axis=0) for a in sv], axis=0)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        # LGBM multiclass: (n_samples, n_features, n_classes)
        return np.abs(sv).mean(axis=(0, 2))
    # Binary: (n_samples, n_features)
    return np.abs(sv).mean(axis=0)


def cap_features(
    features: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    max_features: Optional[int],
    random_state: int,
    model_type: Optional[str] = "rf",
    rank_model: Optional[BaseEstimator] = None,
) -> List[str]:
    """Trim *features* to the top *max_features* by mean |SHAP| value.

    A lightweight version of *model_type* (200 estimators, fixed depth) is
    fitted on *X[features]* solely to rank candidates – no hyperparameter
    tuning is performed.  Mean absolute SHAP values are used for ranking when
    the ``shap`` package is available; the model's own ``feature_importances_``
    are used as a fallback otherwise.

    Using the same model family as the one being trained avoids the MDI bias
    of Random Forest importance (bias toward high-cardinality / continuous
    features) and makes the selected subset consistent with what the final
    model will actually rely on.

    If *max_features* is None, or the feature list is already within budget,
    the original list is returned unchanged.
    """
    if max_features is None or len(features) <= max_features:
        return features
    if rank_model is None:
        model = _build_ranking_model(model_type, y, random_state)
    else:
        model = copy.deepcopy(rank_model)
    model.fit(X[features], y)

    try:
        raw = _shap_importance(model, X[features])
        method = "mean |SHAP|"
    except ImportError:
        raw = model.feature_importances_
        method = "feature_importances_ (install shap for SHAP-based ranking)"
    except Exception as exc:
        # TreeExplainer can fail for some model configurations; degrade gracefully
        print(f"  SHAP ranking failed ({exc}); falling back to feature_importances_.")
        raw = model.feature_importances_
        method = "feature_importances_"

    importance = pd.Series(raw, index=features)
    top_features = importance.nlargest(max_features).index.tolist()
    print(
        f"  Feature cap applied: {len(features)} → {max_features} features"
        f" (ranked by {method})."
    )
    return top_features


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_binary_model(model_type: str, params: Dict) -> object:
    cls_map = {
        "xgb": XGBClassifier,
        "lgbm": LGBMClassifier,
        "rf": RandomForestClassifier,
        "catb": CatBoostClassifier,
    }
    if model_type not in cls_map:
        raise ValueError(f"Unknown model_type '{model_type}'.")
    return cls_map[model_type](**params)


def build_multiclass_model(model_type: str, params: Dict, num_classes: int) -> object:
    if model_type == "xgb":
        params["objective"] = "multi:softprob"
        params["num_class"] = num_classes
        return XGBClassifier(**params)
    elif model_type == "lgbm":
        params["objective"] = "multiclass"
        params["num_class"] = num_classes
        return LGBMClassifier(**params)
    elif model_type == "rf":
        params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type for multiclass: '{model_type}'.")


def _fit_model(model, model_type: str, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None) -> None:
    """Fit a model; CatBoost uses early-stopping with the validation fold."""
    sw = sample_weight
    if model_type == "catb" and X_va is not None:
        n_iter = getattr(model, "iterations", 500)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=max(20, int(0.05 * n_iter)),
            verbose=False,
            sample_weight=sw,
        )
    else:
        if sw is not None:
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)


# ---------------------------------------------------------------------------
# Optuna optimisation
# ---------------------------------------------------------------------------


def _find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0) -> float:
    """Find the threshold that maximises F-beta score (change #3).

    β=2 weights recall twice as much as precision — appropriate for rare clinical
    events where missing a true positive is costlier than a false alarm.
    A dense grid of 181 thresholds in [0.05, 0.95] is searched exhaustively.
    """
    best_thresh, best_score = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        score = fbeta_score(y_true, (y_proba >= t).astype(int), beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = float(t)
    return best_thresh


def _binary_params_for_trial(trial: optuna.Trial, model_type: str, scale_pos_weight: float, random_state: int) -> Dict:
    if model_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": random_state,
        }
    elif model_type == "xgb":
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
            "random_state": random_state,
            "n_jobs": 1,
        }
    elif model_type == "lgbm":
        return {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 300, 4000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": -1,
        }
    elif model_type == "catb":
        return {
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "auto_class_weights": "Balanced",
            "verbose": False,
            "random_state": 42,
        }
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")


def optimise_binary_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    n_trials: int,
    random_state: int,
    sample_weight: Optional[pd.Series],
    model_type: str,
) -> Tuple[Dict, float, optuna.Study]:
    """Tune a binary classifier with Optuna maximising macro F1.

    Change #2: objective is now macro F1 (directly matches the evaluation metric)
    instead of ROC-AUC.  The decision threshold is included in the Optuna search
    space so it is jointly optimised with the hyperparameters, eliminating the
    need for a separate post-hoc OOF threshold search.
    """
    weight_series = (
        sample_weight.reindex(X.index)
        if isinstance(sample_weight, pd.Series)
        else (pd.Series(sample_weight, index=X.index) if sample_weight is not None else None)
    )
    if weight_series is not None:
        pos_mask = y == 1
        spw = float(weight_series[~pos_mask].sum()) / float(weight_series[pos_mask].sum() or 1)
    else:
        n_pos = float((y == 1).sum())
        spw = float(len(y) - n_pos) / (n_pos or 1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _binary_params_for_trial(trial, model_type, spw, random_state)
        # Threshold is co-optimised with hyperparameters (change #2)
        threshold = trial.suggest_float("threshold", 0.10, 0.90)
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            w_tr = weight_series.iloc[tr_idx].to_numpy() if weight_series is not None else None
            model = build_binary_model(model_type, params.copy())
            _fit_model(model, model_type, X_tr, y_tr, X_va, y_va, sample_weight=w_tr)
            proba = model.predict_proba(X_va)[:, 1]
            scores.append(f1_score(y_va, (proba >= threshold).astype(int),
                                   average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=N_CPUS, gc_after_trial=True)
    best = study.best_trial.params.copy()
    # Threshold was optimised as part of Optuna; extract before passing params to model builder
    best_threshold = float(best.pop("threshold", 0.5))
    return best, best_threshold, study


def _multiclass_params_for_trial(
    trial: optuna.Trial, model_type: str, num_classes: int, random_state: int
) -> Dict:
    if model_type == "xgb":
        return {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "num_class": num_classes,
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": 1,
        }
    elif model_type == "lgbm":
        return {
            "objective": "multiclass",
            "boosting_type": "gbdt",
            "num_class": num_classes,
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 150),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": -1,
        }
    elif model_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": random_state,
        }
    else:
        raise ValueError(f"Unsupported multiclass model_type: '{model_type}'")


def optimise_multiclass_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    n_trials: int,
    random_state: int,
    sample_weight: Optional[pd.Series],
    model_type: str,
    num_classes: int,
) -> Tuple[Dict, optuna.Study]:
    """Tune a multiclass classifier with Optuna maximising macro F1.

    sample_weight is applied during each fold's fit so that balanced class
    weights actually influence the search (previously they were silently ignored).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    weight_series = (
        sample_weight.reindex(X.index)
        if isinstance(sample_weight, pd.Series)
        else (pd.Series(sample_weight, index=X.index) if sample_weight is not None else None)
    )

    def objective(trial: optuna.Trial) -> float:
        params = _multiclass_params_for_trial(trial, model_type, num_classes, random_state)
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            w_tr = weight_series.iloc[tr_idx].to_numpy() if weight_series is not None else None
            model = build_multiclass_model(model_type, params.copy(), num_classes)
            if w_tr is not None:
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            scores.append(f1_score(y_va, y_pred, average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=N_CPUS, gc_after_trial=True)
    return study.best_trial.params.copy(), study


# ---------------------------------------------------------------------------
# Out-of-fold prediction generators
# ---------------------------------------------------------------------------


def generate_oof_probas_binary(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict,
    model_type: str,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Return calibrated OOF positive-class probabilities (shape [n,]) for a binary classifier.

    Within each outer fold the base model is wrapped with CalibratedClassifierCV
    (isotonic, cv=3) so that the probabilities fed to the next cascade level are
    well-calibrated and consistent with what the final calibrated model will produce.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        base = build_binary_model(model_type, best_params.copy())
        # cv=3 calibration within the training fold (never touches the OOF validation slice)
        cal_cv = min(3, int(y_tr.value_counts().min()))
        cal_cv = max(cal_cv, 2)
        model = CalibratedClassifierCV(base, cv=cal_cv, method="isotonic")
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
    return oof


def generate_oof_probas_multiclass(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict,
    model_type: str,
    num_classes: int,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Return calibrated OOF per-class probabilities (shape [n, num_classes]).

    Same calibration strategy as generate_oof_probas_binary.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros((len(X), num_classes))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        base = build_multiclass_model(model_type, best_params.copy(), num_classes)
        cal_cv = min(3, int(y_tr.value_counts().min()))
        cal_cv = max(cal_cv, 2)
        model = CalibratedClassifierCV(base, cv=cal_cv, method="isotonic")
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict_proba(X_va)
    return oof


def shap_rfecv(
        model: BaseEstimator,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv: int = 5,
        min_features: int = 5,
        max_features: int = 50,
        scoring: str = "roc_auc",
        random_state: int = 42,
    ) -> Tuple[List[str], Dict[str, Tuple[float, List[str]]]]:
    """
    SHAP-based Recursive Feature Elimination with Cross Validation.

    At each step the CV score is recorded for the *current* feature set,
    then the globally least-important feature (by mean |SHAP| on the full
    training data) is removed.  After all iterations the feature set whose
    CV score was highest is returned together with the full score history.
    """
    remaining_features = list(X.columns)
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    is_multiclass = n_classes > 2

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    history: Dict[str, Tuple[float, List[str]]] = {}

    def _cv_score(features: List[str]) -> float:
        """Return mean CV score for *features* using the current skf splits."""
        if scoring != "roc_auc":
            raise ValueError("Only roc_auc is supported for shap_rfecv")
        scores = []
        for train_idx, val_idx in skf.split(X[features], y):
            X_tr = X.iloc[train_idx][features]
            X_va = X.iloc[val_idx][features]
            y_tr, y_va = y[train_idx], y[val_idx]
            est = copy.deepcopy(model)
            est.fit(X_tr, y_tr)
            try:
                if is_multiclass:
                    proba = est.predict_proba(X_va)
                    score = roc_auc_score(y_va, proba, multi_class="ovr", average="macro")
                else:
                    proba = est.predict_proba(X_va)[:, 1]
                    score = roc_auc_score(y_va, proba)
            except Exception:
                score = 0.0
            scores.append(score)
        return float(np.mean(scores))

    def _global_worst_feature(features: List[str]) -> str:
        """Fit on all training data; return the least-important feature by mean |SHAP|."""
        est = copy.deepcopy(model)
        est.fit(X[features], y)
        try:
            explainer = shap.TreeExplainer(est)
        except Exception:
            explainer = shap.Explainer(est, X[features])
        sv = explainer.shap_values(X[features])
        if isinstance(sv, list):
            # XGB / RF multiclass: list of (n_samples, n_features), one per class
            shap_matrix = np.mean([np.abs(a) for a in sv], axis=0)
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            # LGBM multiclass: (n_samples, n_features, n_classes)
            shap_matrix = np.abs(sv).mean(axis=2)
        else:
            shap_matrix = np.abs(sv)
        importance = np.mean(shap_matrix, axis=0)
        return features[int(np.argmin(importance))]

    while len(remaining_features) > min_features:
        # Score the CURRENT feature set — label and score are in sync
        mean_score = _cv_score(remaining_features)
        history[str(len(remaining_features))] = (mean_score, remaining_features.copy())
        print(f"Features: {len(remaining_features)} | CV score: {mean_score:.4f}")

        # Remove the globally least important feature
        removed = _global_worst_feature(remaining_features)
        remaining_features.remove(removed)
        print(f"Removed feature: {removed}")

    # Score and record the final minimal feature set
    final_score = _cv_score(remaining_features)
    history[str(len(remaining_features))] = (final_score, remaining_features.copy())
    print(f"Features: {len(remaining_features)} | CV score: {final_score:.4f}")

    # Return the feature set whose CV score was highest (true RFECV selection),
    # constrained to at most max_features.  Keys are strings; cast before comparing.
    best_n = max(history, key=lambda k: history[k][0] if int(k) <= max_features else 0.0)
    best_score, best_features = history[best_n]
    print(
        f"Best CV score: {best_score:.4f} at {best_n} features → "
        f"Selected {len(best_features)} features."
    )
    return best_features, history

# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------


def remove_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.90,
) -> List[str]:
    """Return the subset of *X.columns* that survives a Spearman correlation filter.

    All feature columns are numeric after ``pd.get_dummies``, so Spearman rank
    correlation is a single valid measure for every column type:

    * **Binary 0/1** (one-hot dummies, presence/absence flags):
      Spearman = phi coefficient for binary pairs, which equals Pearson.
    * **Ordinal** (recoded vital signs 0-3, clinical scores 0-4):
      Spearman is designed for ordered discrete data.
    * **Continuous** (raw vital signs, labs, counts):
      Spearman is valid and more robust to outliers than Pearson.

    No Cramér's V is needed because string/categorical columns have already
    been one-hot encoded before this function is called.

    Algorithm
    ---------
    1. Sort columns by **descending variance** so the more informative
       representation wins when a pair must be pruned.  For example,
       ``temperatura`` (continuous, higher variance) beats
       ``temperatura_recoded`` (0-3 quartile bin, lower variance).
    2. Walk the sorted list; keep each column unless it is already
       scheduled for removal because it is too similar to an earlier-kept
       column.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix built from **training rows only**.  Test rows must
        not be included to avoid test-set leakage into the correlation
        estimates.
    threshold : float
        Absolute Spearman correlation above which a pair is considered
        redundant.  Default 0.90.

    Returns
    -------
    List[str]
        Column names to keep, preserved in their original order.
    """
    if threshold >= 1.0:
        return X.columns.tolist()

    # Pairwise Spearman on training data.  Constant columns yield NaN
    # correlations; fill with 0 so they are treated as uncorrelated and
    # kept (SHAP can remove them later if they carry no information).
    corr = X.corr(method="spearman").abs().fillna(0.0)

    # Higher-variance columns are preferred when breaking ties (a continuous
    # vital sign carries more information than its 0-3 quartile-binned twin).
    col_order = X.var().sort_values(ascending=False).index.tolist()

    dropped: set = set()
    kept_ordered: List[str] = []
    for col in col_order:
        if col in dropped:
            continue
        kept_ordered.append(col)
        # Schedule every column that is too similar to *col* for removal.
        redundant = corr.index[corr[col] > threshold].tolist()
        for partner in redundant:
            if partner != col:
                dropped.add(partner)

    # Return names in the original DataFrame column order.
    kept_set = set(kept_ordered)
    return [c for c in X.columns if c in kept_set]


def run_training(args: argparse.Namespace) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("SELECTED ARGS:", args)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    all_targets = {args.sepsis_target, args.hemo_target, args.cef_target, args.weight_column}
    cols_to_delete = list(DELETE_COLUMNS)
    cols_to_delete.extend([x for x in TARGET_REMOVE if x not in all_targets])

    df = load_processed_dataframe(args.database_file, cols_to_delete)

    missing = [c for c in all_targets if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataframe: {missing}")

    working_df = df.copy()
    working_df[args.weight_column] = pd.to_numeric(
        working_df[args.weight_column], errors="coerce"
    )
    # Drop rows where sepsis label or weight is missing (needed for split stratification)
    working_df = working_df.dropna(subset=[args.sepsis_target, args.weight_column])
    working_df = working_df[working_df[args.weight_column] > 0]

    exclude_cols = all_targets.copy()
    feature_cols = [c for c in working_df.columns if c not in exclude_cols]

    # Drop high-NA feature columns
    dropped_na = [
        col for col in feature_cols if working_df[col].isna().mean() > args.na_perc_limit
    ]
    if dropped_na:
        print(f"Dropping {len(dropped_na)} high-NA columns.")
        working_df.drop(columns=dropped_na, inplace=True)
        feature_cols = [c for c in feature_cols if c not in dropped_na]

    if not feature_cols:
        raise ValueError("No feature columns remain after NA filtering.")

    if args.impute_missing:
        working_df = impute_missing_values(working_df, exclude_cols)
    else:
        working_df = working_df.dropna(subset=feature_cols)

    feature_df = working_df[feature_cols]
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
        feature_df.columns = feature_df.columns.str.replace(
            "[^0-9a-zA-Z_]+", "_", regex=True
        )

    # ------------------------------------------------------------------
    # 2. Train / test split – stratified on sepsis
    # ------------------------------------------------------------------
    y_sepsis = working_df.loc[feature_df.index, args.sepsis_target]
    y_hemo = working_df.loc[feature_df.index, args.hemo_target]
    y_cef = working_df.loc[feature_df.index, args.cef_target]
    w = working_df.loc[feature_df.index, args.weight_column]

    (
        X_train, X_test,
        y_sep_train, y_sep_test,
        y_hemo_train, y_hemo_test,
        y_cef_train, y_cef_test,
        w_train, w_test,
    ) = train_test_split(
        feature_df, y_sepsis, y_hemo, y_cef, w,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_sepsis,
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # ------------------------------------------------------------------
    # 2b. Remove highly correlated features (training data only → no leakage)
    # ------------------------------------------------------------------
    if args.max_corr < 1.0:
        print(f"\nRemoving features with |Spearman corr| > {args.max_corr} …")
        kept_cols = remove_correlated_features(X_train, threshold=args.max_corr)
        n_removed = len(X_train.columns) - len(kept_cols)
        if n_removed:
            print(f"  Dropped {n_removed} redundant features → {len(kept_cols)} remain.")
        X_train = X_train[kept_cols]
        X_test = X_test[kept_cols]


    # ------------------------------------------------------------------
    # 3. Output directory
    # ------------------------------------------------------------------
    output_dir = Path(str(args.output_dir) + "_" + TODAY + "_" + JOB_ID)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict = {"args": {str(k): str(v) for k,v in args.__dict__.items()}}

    # ==================================================================
    # LEVEL 1 – sepsis
    # ==================================================================
    print("\n" + "=" * 60)
    print("LEVEL 1: sepsis")
    print("=" * 60)
    l1_dir = output_dir / "level1_sepsis"
    l1_dir.mkdir(exist_ok=True)

    le_sep = LabelEncoder()
    y_sep_train_enc = pd.Series(
        le_sep.fit_transform(y_sep_train.astype(str)),
        index=y_sep_train.index,
        name="sepsis_enc",
    )
    y_sep_test_enc = pd.Series(
        le_sep.transform(y_sep_test.astype(str)),
        index=y_sep_test.index,
        name="sepsis_enc",
    )

    sw_l1 = compute_balanced_sample_weight(y_sep_train_enc, w_train)

    print("  Selecting Level 1 features by SHAP importance …")
    rank_model_l1 = _build_ranking_model(args.model_type, y_sep_train_enc, args.random_state)
    selected_features, l1_rfecv_history = shap_rfecv(
        rank_model_l1, X_train, y_sep_train_enc, min_features=2, max_features=args.max_features
    )
    l1_features = cap_features(
        selected_features, X_train, y_sep_train_enc,
        args.max_features, args.random_state, args.model_type, rank_model=rank_model_l1
    )
    print(f"  SHAP selection kept {len(l1_features)} features.")

    scaler_l1 = MinMaxScaler()
    X_l1_train = pd.DataFrame(
        scaler_l1.fit_transform(X_train[l1_features]),
        columns=l1_features, index=X_train.index,
    )
    X_l1_test = pd.DataFrame(
        scaler_l1.transform(X_test[l1_features]),
        columns=l1_features, index=X_test.index,
    )

    print("  Optimising Level 1 model …")
    l1_params, l1_threshold, l1_study = optimise_binary_model(
        X_l1_train, y_sep_train_enc,
        n_splits=args.cv_splits,
        n_trials=args.binary_trials,
        random_state=args.random_state,
        sample_weight=sw_l1,
        model_type=args.model_type,
    )
    print(f"  Best threshold: {l1_threshold:.3f}")

    # Final Level 1 model trained on all training data (no calibration — change #1)
    # Change #4: L1 OOF probabilities are no longer computed; cascade features removed.
    l1_model = build_binary_model(args.model_type, l1_params.copy())
    l1_model.fit(X_l1_train, y_sep_train_enc)

    l1_test_proba = l1_model.predict_proba(X_l1_test)[:, 1]
    l1_test_pred = (l1_test_proba >= l1_threshold).astype(int)

    l1_report = classification_report(
        y_sep_test_enc, l1_test_pred,
        target_names=[str(c) for c in le_sep.classes_],
        zero_division=0,
    )
    l1_auc = roc_auc_score(y_sep_test_enc, l1_test_proba)
    l1_f1 = f1_score(y_sep_test_enc, l1_test_pred, average="macro", zero_division=0)
    print(f"  Level 1 – Macro F1: {l1_f1:.3f}  |  ROC-AUC: {l1_auc:.3f}")

    (l1_dir / "report.txt").write_text(l1_report)
    (l1_dir / "summary.json").write_text(json.dumps({
        "params": l1_params, "threshold": l1_threshold,
        "macro_f1": l1_f1, "roc_auc": l1_auc,
        "classes": le_sep.classes_.tolist(),
        "shaprfecv_features": selected_features,
        "features": l1_features,
    }, indent=2))
    l1_study.trials_dataframe().to_csv(l1_dir / "optuna_trials.csv", index=False)
    pd.DataFrame({
        "true": y_sep_test_enc.values,
        "pred": l1_test_pred,
        "proba": l1_test_proba,
    }).to_csv(l1_dir / "predictions.csv", index=False)

    confusion_matrix(y_sep_test_enc, l1_test_pred, labels=[0, 1])
    all_summaries["level1_sepsis"] = {"macro_f1": l1_f1, "roc_auc": l1_auc}

    # ==================================================================
    # LEVEL 2 – resultado_hemo_grouped
    # ==================================================================
    print("\n" + "=" * 60)
    print("LEVEL 2: resultado_hemo_grouped")
    print("=" * 60)
    l2_dir = output_dir / "level2_hemo"
    l2_dir.mkdir(exist_ok=True)

    # Drop rare hemo classes (< 2 % of hemo-valid training samples)
    hemo_valid_train = y_hemo_train.notna()
    hemo_counts = y_hemo_train[hemo_valid_train].value_counts()
    rare_thresh = max(5, hemo_valid_train.sum() / 50)
    rare_classes = hemo_counts[hemo_counts < rare_thresh].index.tolist()
    if rare_classes:
        print(f"  Dropping rare hemo classes: {rare_classes}")

    l2_train_mask = hemo_valid_train & ~y_hemo_train.isin(rare_classes)
    l2_test_mask = y_hemo_test.notna() & ~y_hemo_test.isin(rare_classes)

    # Encode hemo labels on train first, then filter test to known classes
    le_hemo = LabelEncoder()
    y2_train_raw = y_hemo_train.loc[l2_train_mask]
    le_hemo.fit(y2_train_raw)
    l2_test_mask = l2_test_mask & y_hemo_test.isin(le_hemo.classes_)

    y2_test_raw = y_hemo_test.loc[l2_test_mask]
    y2_train_enc = pd.Series(
        le_hemo.transform(y2_train_raw), index=y2_train_raw.index, name="hemo_enc"
    )
    y2_test_enc = pd.Series(
        le_hemo.transform(y2_test_raw), index=y2_test_raw.index, name="hemo_enc"
    )
    num_hemo_classes = len(le_hemo.classes_)
    print(f"  Hemo classes ({num_hemo_classes}): {le_hemo.classes_.tolist()}")

    # CatBoost does not support multiclass in this setup, so fall back to lgbm
    _l2_model_type = args.model_type if args.model_type != "catb" else "lgbm"

    # Base feature matrices for this level
    X2_train_base = X_train.loc[l2_train_mask].copy()
    X2_test_base = X_test.loc[l2_test_mask].copy()

    print("  Selecting Level 2 features by SHAP importance …")
    rank_model_l2 = _build_ranking_model(args.model_type, y2_train_enc, args.random_state)
    l2_shap_feats, l2_rfecv_history = shap_rfecv(
        rank_model_l2, X2_train_base, y2_train_enc, min_features=2, max_features=args.max_features
    )
    l2_features = cap_features(
        l2_shap_feats, X2_train_base, y2_train_enc,
        args.max_features, args.random_state, _l2_model_type, rank_model=rank_model_l2
    )
    print(f"  SHAP selection kept {len(l2_features)} features.")

    scaler_l2 = MinMaxScaler()
    X2_train_scaled = pd.DataFrame(
        scaler_l2.fit_transform(X2_train_base[l2_features]),
        columns=l2_features, index=X2_train_base.index,
    )
    X2_test_scaled = pd.DataFrame(
        scaler_l2.transform(X2_test_base[l2_features]),
        columns=l2_features, index=X2_test_base.index,
    )

    # Change #4: cascade sepsis_proba feature removed — L2 trained on base features only.
    sw_l2 = compute_balanced_sample_weight(y2_train_enc, w_train.reindex(y2_train_enc.index))

    print("  Optimising Level 2 multiclass model …")
    l2_params, l2_study = optimise_multiclass_model(
        X2_train_scaled, y2_train_enc,
        n_splits=args.cv_splits,
        n_trials=args.binary_trials,
        random_state=args.random_state,
        sample_weight=sw_l2,
        model_type=_l2_model_type,
        num_classes=num_hemo_classes,
    )

    # Change #4: L2 OOF probabilities not generated — cascade to L3 removed.

    # Final Level 2 model (no calibration — change #1)
    l2_model = build_multiclass_model(_l2_model_type, l2_params.copy(), num_hemo_classes)
    l2_model.fit(X2_train_scaled, y2_train_enc)
    l2_test_proba = l2_model.predict_proba(X2_test_scaled)  # [n_l2_test, num_hemo_classes]
    l2_test_pred = l2_model.predict(X2_test_scaled)

    l2_report = classification_report(
        y2_test_enc, l2_test_pred,
        target_names=[str(c) for c in le_hemo.classes_],
        zero_division=0,
    )
    l2_f1 = f1_score(y2_test_enc, l2_test_pred, average="macro", zero_division=0)
    l2_auc = roc_auc_score(
        y2_test_enc,
        l2_test_proba,
        multi_class="ovr",
        average="macro"
    )
    print(f"  Level 2 – Macro F1: {l2_f1:.3f}")

    (l2_dir / "report.txt").write_text(l2_report)
    (l2_dir / "summary.json").write_text(json.dumps({
        "params": l2_params, "macro_f1": l2_f1,
        "hemo_classes": le_hemo.classes_.tolist(),
        "features": l2_features, "roc_auc": l2_auc
    }, indent=2))
    l2_study.trials_dataframe().to_csv(l2_dir / "optuna_trials.csv", index=False)
    pd.DataFrame(l2_test_proba, columns=[f"proba_{c}" for c in le_hemo.classes_]).assign(
        true=y2_test_enc.values, pred=l2_test_pred
    ).to_csv(l2_dir / "predictions.csv", index=False)

    all_summaries["level2_hemo"] = {
        "macro_f1": l2_f1,
        "roc_auc": l2_auc,
        "hemo_classes": le_hemo.classes_.tolist(),
    }

    # ==================================================================
    # LEVEL 3 – resistente_cefalosporina
    # ==================================================================
    print("\n" + "=" * 60)
    print("LEVEL 3: resistente_cefalosporina")
    print("=" * 60)
    l3_dir = output_dir / "level3_cefalosporina"
    l3_dir.mkdir(exist_ok=True)

    # Filter: positive blood culture AND valid cef label AND within L2 scope
    # (positive blood culture = resultado_hemo_grouped != "NEGATIVE")
    cef_train_mask = (
        y_cef_train.notna()
        & y_hemo_train.notna()
        & (y_hemo_train != "NEGATIVE")
        & l2_train_mask  # ensures we have a valid L2 OOF proba for every row
    )
    cef_test_mask = (
        y_cef_test.notna()
        & y_hemo_test.notna()
        & (y_hemo_test != "NEGATIVE")
        & l2_test_mask
    )

    if not cef_train_mask.any():
        raise ValueError(
            "No training rows with positive blood culture and valid cef label. "
            "Check that 'resultado_hemo_grouped' and 'resistente_cefalosporina' "
            "are present and properly coded."
        )

    X3_train_base = X_train.loc[cef_train_mask].copy()
    X3_test_base = X_test.loc[cef_test_mask].copy()
    y3_train_raw = y_cef_train.loc[cef_train_mask]
    y3_test_raw = y_cef_test.loc[cef_test_mask]

    le_cef = LabelEncoder()
    y3_train_enc = pd.Series(
        le_cef.fit_transform(y3_train_raw), index=y3_train_raw.index, name="cef_enc"
    )
    # Filter test to known cef classes only
    cef_test_mask = cef_test_mask & y_cef_test.isin(le_cef.classes_)
    X3_test_base = X_test.loc[cef_test_mask].copy()
    y3_test_raw = y_cef_test.loc[cef_test_mask]
    y3_test_enc = pd.Series(
        le_cef.transform(y3_test_raw), index=y3_test_raw.index, name="cef_enc"
    )

    print(f"  Cef classes: {le_cef.classes_.tolist()}")
    print(f"  Level 3 train rows: {len(X3_train_base)}  |  test rows: {len(X3_test_base)}")

    # ------------------------------------------------------------------
    # L3 size guard
    # ------------------------------------------------------------------
    l3_minority_count = int(y3_train_enc.value_counts().min())
    l3_small_dataset = l3_minority_count < args.l3_min_positive

    # Cap CV splits for L3 (used by Optuna, OOF generation, and calibration)
    l3_cv_splits = max(min(args.cv_splits, l3_minority_count), 2)
    if l3_cv_splits != args.cv_splits:
        print(f"  Reducing cv_splits from {args.cv_splits} to {l3_cv_splits} for Level 3.")

    if l3_small_dataset:
        print(
            f"\n  {'!' * 60}"
            f"\n  WARNING: Level 3 minority class has only {l3_minority_count} training"
            f" samples (threshold: {args.l3_min_positive})."
            f"\n  Results should be interpreted with caution."
            f"\n  {'!' * 60}\n"
        )
    print("  Selecting Level 3 features by SHAP importance …")
    rank_model_l3 = _build_ranking_model(args.model_type, y3_train_enc, args.random_state)
    l3_shap_feats, l3_rfecv_history = shap_rfecv(
        rank_model_l3, X3_train_base, y3_train_enc, min_features=2, max_features=args.max_features
    )
    l3_features = cap_features(
        l3_shap_feats, X3_train_base, y3_train_enc,
        args.max_features, args.random_state, args.model_type, rank_model=rank_model_l3
    )
    print(f"  SHAP selection kept {len(l3_features)} features.")

    scaler_l3 = MinMaxScaler()
    X3_train_scaled = pd.DataFrame(
        scaler_l3.fit_transform(X3_train_base[l3_features]),
        columns=l3_features, index=X3_train_base.index,
    )
    X3_test_scaled = pd.DataFrame(
        scaler_l3.transform(X3_test_base[l3_features]),
        columns=l3_features, index=X3_test_base.index,
    )

    # Change #4: cascade features (sepsis_proba, hemo_proba_*) removed —
    # L3 trained on base features only.
    sw_l3 = compute_balanced_sample_weight(y3_train_enc, w_train.reindex(y3_train_enc.index))

    print("  Optimising Level 3 model …")
    l3_params, l3_threshold, l3_study = optimise_binary_model(
        X3_train_scaled, y3_train_enc,
        n_splits=l3_cv_splits,
        n_trials=args.binary_trials,
        random_state=args.random_state,
        sample_weight=sw_l3,
        model_type=args.model_type,
    )
    print(f"  Best threshold: {l3_threshold:.3f}")

    # Final Level 3 model (no calibration — change #1)
    l3_model = build_binary_model(args.model_type, l3_params.copy())
    l3_model.fit(X3_train_scaled, y3_train_enc)

    l3_test_proba = l3_model.predict_proba(X3_test_scaled)[:, 1]
    l3_test_pred = (l3_test_proba >= l3_threshold).astype(int)

    l3_report = classification_report(
        y3_test_enc, l3_test_pred,
        target_names=[str(c) for c in le_cef.classes_],
        zero_division=0,
    )
    l3_f1 = f1_score(y3_test_enc, l3_test_pred, average="macro", zero_division=0)
    l3_auc: Optional[float] = None
    if y3_test_enc.nunique() > 1:
        l3_auc = float(roc_auc_score(y3_test_enc, l3_test_proba))

    print(
        f"  Level 3 – Macro F1: {l3_f1:.3f}"
        + (f"  |  ROC-AUC: {l3_auc:.3f}" if l3_auc is not None else "")
    )

    (l3_dir / "report.txt").write_text(l3_report)
    (l3_dir / "summary.json").write_text(json.dumps({
        "params": l3_params, "threshold": l3_threshold,
        "macro_f1": l3_f1, "roc_auc": l3_auc,
        "classes": le_cef.classes_.tolist(),
        "features": l3_features,
        "cascade_features": [],  # change #4: no cascade features used
    }, indent=2))
    l3_study.trials_dataframe().to_csv(l3_dir / "optuna_trials.csv", index=False)
    pd.DataFrame({
        "true": y3_test_enc.values,
        "pred": l3_test_pred,
        "proba": l3_test_proba,
    }).to_csv(l3_dir / "predictions.csv", index=False)

    all_summaries["level3_cefalosporina"] = {"macro_f1": l3_f1, "roc_auc": l3_auc}
    all_summaries["l1_rfecv_scores"] = l1_rfecv_history
    all_summaries["l2_rfecv_scores"] = l2_rfecv_history
    all_summaries["l3_rfecv_scores"] = l3_rfecv_history

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )
    print("\n" + "=" * 60)
    print("COMPLETED.  Results saved in:", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    home = Path.cwd()
    default_db = os.path.join(
        home, "mepram_data", "df_merged_full_multilabel_grouped.csv"
    )
    default_out = os.path.join(home, "mepram_data", "outputs", "cascade_three_level")

    parser = argparse.ArgumentParser(
        description=(
            "Three-level cascade: "
            "sepsis → resultado_hemo_grouped → resistente_cefalosporina"
        )
    )
    parser.add_argument("--database-file", "-db", type=Path, default=default_db)
    parser.add_argument("--output-dir", "-o", type=Path, default=default_out)
    parser.add_argument(
        "--sepsis-target", type=str, default="sepsis",
        help="Column name for the Level-1 binary target (default: sepsis).",
    )
    parser.add_argument(
        "--hemo-target", type=str, default="resultado_hemo_grouped",
        help="Column name for the Level-2 multiclass target (default: resultado_hemo_grouped).",
    )
    parser.add_argument(
        "--cef-target", type=str, default="resistente_cefalosporina",
        help="Column name for the Level-3 binary target (default: resistente_cefalosporina).",
    )
    parser.add_argument(
        "--weight-column", type=str, default="sample_weight",
        help="Column containing per-sample weights (default: sample_weight).",
    )
    parser.add_argument(
        "--model-type", type=str, choices=["xgb", "lgbm", "rf", "catb"],
        default="lgbm",
        help=(
            "Estimator used for all three levels. "
            "Note: Level 2 always uses lgbm/xgb/rf (not catb) for multiclass. "
            "Default: lgbm."
        ),
    )
    parser.add_argument(
        "--binary-trials", "-btrials", type=int, default=500,
        help="Optuna trials per level (default: 500).",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--test-size", "-tsize", type=float, default=0.35,
        help="Hold-out fraction (default: 0.35).",
    )
    parser.add_argument("--random-state", type=int, default=99)
    parser.add_argument(
        "--na-perc-limit", "-na", type=float, default=0.20,
        help="Drop columns with more than this fraction of missing values.",
    )
    parser.add_argument(
        "--max-features", type=int, default=None,
        help=(
            "Maximum number of features to keep per level, ranked by mean |SHAP|. "
            "Cascade features (sepsis_proba, hemo_proba_*) are appended afterwards "
            "and do NOT count against this budget. Default: None (keep all features)."
        ),
    )
    parser.add_argument(
        "--l3-min-positive", type=int, default=30,
        help=(
            "Minimum number of minority-class training samples for Level 3 "
            "before a warning is printed about unreliable results. "
            "CV folds are also capped to this count if it is lower than --cv-splits. "
            "Default: 30."
        ),
    )
    parser.add_argument(
        "--max-corr", type=float, default=0.90,
        help=(
            "Remove features whose absolute Spearman correlation with any "
            "previously-kept feature exceeds this threshold. "
            "Set to 1.0 to disable. Default: 0.90 (removes near-duplicate "
            "continuous/recoded vital-sign pairs and perfectly correlated "
            "one-hot complements)."
        ),
    )
    parser.add_argument(
        "--no-impute", dest="impute_missing", action="store_false",
        help="Disable missing-value imputation (default: enabled).",
    )
    parser.set_defaults(impute_missing=True)
    return parser


def main() -> None:
    start = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()
    print("Parsed args:", args)

    output_folder = Path(str(args.output_dir) + "_" + TODAY + "_" + JOB_ID)
    try:
        run_training(args)
    except Exception:
        if output_folder.exists():
            shutil.rmtree(output_folder)
        raise
    print(f"\nElapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
