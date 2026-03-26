#!/usr/bin/env python3
"""
Shared-feature analysis: all three levels train on the same feature set.

Runs a **joint** SHAP-RFECV that finds a single feature set optimal for all
three clinical targets simultaneously, then performs per-level Optuna
hyperparameter optimisation using that shared set.

Joint feature selection
-----------------------
At each RFECV elimination step the retained feature set is scored against *all
three* levels at once via a weighted-average CV AUC.  The globally
least-important feature (by weighted-mean |SHAP| across levels) is removed.
Level weights are configurable via ``--level-weights L1 L2 L3``
(default: 1 1 2, giving double weight to L3 which is the hardest target).

Targets / data subsets are identical to ``independent_target_analysis.py``:
  Level 1 – sepsis              (binary,      all rows)
  Level 2 – resultado_hemo_grouped (multiclass, hemo-valid rows)
  Level 3 – resistente_cefalosporina_multi (binary/multiclass, positive-hemo rows)

Outputs  (<output_dir>_TIMESTAMP_JOBID/)
-----------------------------------------
  shared_rfecv_history.json    joint elimination history
  level1_sepsis/
      summary.json, predictions.csv, optuna_trials.csv, report.txt
  level2_hemo/
      (same)
  level3_cef/
      (same)
  aggregate_summary.json
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
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import shap
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
JOB_ID  = os.environ.get("SLURM_JOB_ID", 1)
TODAY   = datetime.today().strftime("%Y%m%d%H%M%S")

_PER_MODEL_THREADS = min(N_CPUS, 8)
MODEL_JOBS  = _PER_MODEL_THREADS
OPTUNA_JOBS = max(1, N_CPUS // _PER_MODEL_THREADS)

FOCUS_MAP = {
    1: "pulmonar", 2: "intraabdominal", 3: "biliar", 4: "urinario",
    5: "cardiovascular", 6: "piel", 7: "sistema nervioso central",
    8: "cateter venoso", 9: "vías altas respiratorias",
    10: "osteoarticular", 11: "genital", 12: "desconocido",
}

TARGET_REMOVE = [
    "sepsis", "resultado_hemo", "resultado_hemo_grouped", "all_cult_org",
    "infected_yes_no", "bmr_etiologia", "fenotipo_resistencia",
    "fenotipo_resistencia_grouped", "resistente_cefalosporina",
    "resistente_cefalosporina_multi",
]

DELETE_COLUMNS = [
    "qsofa", "vasopresores", "hipotension", "freq_bacteria", "freq_bac_foco",
    "Unnamed: 0", "person_id", "fecha_ingreso_urgencias",
    "fecha_ingreso_urgencias_x", "shock_septico", "sintoma_nan",
    "fecha_nacimiento", "codigo_postal", "center", "dag",
    "ultima_fecha", "mujer_gestante",
]

FOCUS_TO_EXCLUDE = {
    "piel", "osteoarticular", "biliar", "genital",
    "sistema nervioso central", "cateter venoso",
    "vías altas respiratorias", "cardiovascular",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def safe_drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        try:
            df = df.drop(columns=col)
        except KeyError:
            pass
    return df


def load_processed_dataframe(csv_path: Path, cols_to_delete: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "foco" in df.columns:
        df["foco"] = df["foco"].map(FOCUS_MAP).fillna(df["foco"])
        df = df[~df["foco"].isin(FOCUS_TO_EXCLUDE)]
    return safe_drop_columns(df, cols_to_delete)


def impute_missing_values(loaded_df: pd.DataFrame, exclude_cols: set) -> pd.DataFrame:
    df_copy = loaded_df.drop(columns=list(exclude_cols), errors="ignore").copy()
    numeric_cols     = df_copy.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()
    binary_cols = [c for c in numeric_cols if set(df_copy[c].dropna().unique()) <= {0, 1}]
    knn_cols    = [c for c in numeric_cols if c not in binary_cols]
    high_missing = [c for c in numeric_cols if df_copy[c].isna().mean() > 0.10]
    for col in high_missing:
        df_copy[f"{col}_missing"] = df_copy[col].isna().astype(int)
    if binary_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df_copy[binary_cols] = imp.fit_transform(df_copy[binary_cols]).astype(int)
    if knn_cols:
        imp = KNNImputer(n_neighbors=10, weights="distance")
        df_copy[knn_cols] = imp.fit_transform(df_copy[knn_cols])
        for col in knn_cols:
            if loaded_df[col].dropna().astype(float).apply(float.is_integer).all():
                df_copy[col] = df_copy[col].round().astype(int)
    if categorical_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df_copy[categorical_cols] = imp.fit_transform(df_copy[categorical_cols]).astype(str)
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


def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.90) -> List[str]:
    if threshold >= 1.0:
        return X.columns.tolist()
    corr = X.corr(method="spearman").abs().fillna(0.0)
    col_order = X.var().sort_values(ascending=False).index.tolist()
    dropped: set = set()
    kept: List[str] = []
    for col in col_order:
        if col in dropped:
            continue
        kept.append(col)
        for partner in corr.index[corr[col] > threshold].tolist():
            if partner != col:
                dropped.add(partner)
    kept_set = set(kept)
    return [c for c in X.columns if c in kept_set]

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_binary_model(model_type: str, params: Dict) -> object:
    cls_map = {
        "xgb": XGBClassifier, "lgbm": LGBMClassifier,
        "rf": RandomForestClassifier, "catb": CatBoostClassifier,
    }
    if model_type not in cls_map:
        raise ValueError(f"Unknown model_type '{model_type}'.")
    return cls_map[model_type](**params)


def build_multiclass_model(model_type: str, params: Dict, num_classes: int) -> object:
    if model_type == "xgb":
        params["objective"] = "multi:softprob"
        params["num_class"]  = num_classes
        return XGBClassifier(**params)
    elif model_type == "lgbm":
        params["objective"] = "multiclass"
        params["num_class"]  = num_classes
        return LGBMClassifier(**params)
    elif model_type == "rf":
        params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type for multiclass: '{model_type}'.")


def _fit_model(model, model_type: str, X_tr, y_tr, X_va=None, y_va=None,
               sample_weight=None) -> None:
    sw = sample_weight
    if model_type == "catb" and X_va is not None:
        n_iter = getattr(model, "iterations", 500)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                  early_stopping_rounds=max(20, int(0.05 * n_iter)),
                  verbose=False, sample_weight=sw)
    else:
        if sw is not None:
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)

# ---------------------------------------------------------------------------
# SHAP importance helper
# ---------------------------------------------------------------------------

def _shap_importance(model, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.mean([np.abs(a).mean(axis=0) for a in sv], axis=0)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        return np.abs(sv).mean(axis=(0, 2))
    return np.abs(sv).mean(axis=0)


def _build_ranking_model(model_type: str, y: pd.Series, random_state: int) -> object:
    n_classes = int(y.nunique())
    is_multi  = n_classes > 2
    if model_type == "lgbm":
        params: Dict = {
            "n_estimators": 200, "num_leaves": 31, "max_depth": 6,
            "class_weight": "balanced", "n_jobs": N_CPUS,
            "random_state": random_state, "verbosity": -1,
        }
        params["objective"] = "multiclass" if is_multi else "binary"
        if is_multi:
            params["num_class"] = n_classes
        return LGBMClassifier(**params)
    elif model_type == "xgb":
        if is_multi:
            return XGBClassifier(objective="multi:softprob", num_class=n_classes,
                                 n_estimators=200, max_depth=6,
                                 random_state=random_state, n_jobs=MODEL_JOBS, verbosity=0)
        spw = float(len(y) - (y == 1).sum()) / max(float((y == 1).sum()), 1.0)
        return XGBClassifier(objective="binary:logistic", n_estimators=200, max_depth=6,
                             scale_pos_weight=spw, random_state=random_state,
                             n_jobs=MODEL_JOBS, verbosity=0)
    elif model_type == "catb":
        return CatBoostClassifier(iterations=200, depth=6, auto_class_weights="Balanced",
                                  verbose=False, random_state=42, thread_count=MODEL_JOBS)
    else:
        return RandomForestClassifier(n_estimators=200, max_depth=10,
                                      class_weight="balanced_subsample",
                                      random_state=random_state, n_jobs=MODEL_JOBS)

# ---------------------------------------------------------------------------
# Shared SHAP-RFECV  (joint across all levels)
# ---------------------------------------------------------------------------

def shap_rfecv_shared(
    levels: List[Dict],
    cv: int = 5,
    min_features: int = 5,
    max_features: int = 50,
    level_weights: Optional[List[float]] = None,
    random_state: int = 42,
) -> Tuple[List[str], Dict]:
    """
    Joint SHAP-RFECV: find a single feature set that maximises a weighted-
    average CV AUC across all levels.

    Parameters
    ----------
    levels : list of dicts, one per level, each containing:
        - 'model'        : unfitted ranking estimator (cloned at each step)
        - 'X'            : pd.DataFrame  (training rows for this level, all columns)
        - 'y'            : np.ndarray    (encoded integer labels)
        - 'label'        : str           (name used in log messages)
        - 'is_multiclass': bool
    level_weights : relative weights for the composite score (normalised internally).
                   Defaults to equal weights.
    """
    if not levels:
        raise ValueError("At least one level is required.")

    n_levels = len(levels)
    if level_weights is None:
        level_weights = [1.0] * n_levels
    w = np.array(level_weights, dtype=float)
    w = w / w.sum()                          # normalise

    # All levels must share the same feature columns
    all_cols = levels[0]["X"].columns.tolist()
    for lvl in levels[1:]:
        if set(lvl["X"].columns) != set(all_cols):
            raise ValueError("All levels must have the same feature columns.")

    remaining = list(all_cols)
    history: Dict = {}

    # ── per-level CV scorer ──────────────────────────────────────────────────
    def _level_cv_score(lvl: Dict, features: List[str]) -> float:
        X_lvl = lvl["X"]
        y_lvl = lvl["y"]
        is_mc = lvl["is_multiclass"]
        skf   = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        def _fold(tr_idx, va_idx):
            est = copy.deepcopy(lvl["model"])
            est.fit(X_lvl.iloc[tr_idx][features], y_lvl[tr_idx])
            try:
                proba = est.predict_proba(X_lvl.iloc[va_idx][features])
                if is_mc:
                    return roc_auc_score(y_lvl[va_idx], proba,
                                        multi_class="ovr", average="macro")
                return roc_auc_score(y_lvl[va_idx], proba[:, 1])
            except Exception:
                return 0.0

        scores = Parallel(n_jobs=OPTUNA_JOBS, prefer="threads")(
            delayed(_fold)(tr, va) for tr, va in skf.split(X_lvl[features], y_lvl)
        )
        return float(np.mean(scores))

    def _combined_cv_score(features: List[str]) -> Tuple[float, List[float]]:
        per_level = [_level_cv_score(lvl, features) for lvl in levels]
        combined  = float(np.dot(w, per_level))
        return combined, per_level

    # ── per-level SHAP importance (fit on full training split) ───────────────
    def _combined_importance(features: List[str]) -> np.ndarray:
        """Returns shape (n_features,) combined SHAP importance vector."""
        combined_imp = np.zeros(len(features))
        for wi, lvl in zip(w, levels):
            est = copy.deepcopy(lvl["model"])
            est.fit(lvl["X"][features], lvl["y"])
            try:
                explainer = shap.TreeExplainer(est)
            except Exception:
                explainer = shap.Explainer(est, lvl["X"][features])
            sv = explainer.shap_values(lvl["X"][features])
            if isinstance(sv, list):
                shap_mat = np.mean([np.abs(a) for a in sv], axis=0)
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                shap_mat = np.abs(sv).mean(axis=2)
            else:
                shap_mat = np.abs(sv)
            imp = np.mean(shap_mat, axis=0)          # shape (n_features,)
            # normalise per-level so scale differences don't dominate
            norm = imp.sum()
            if norm > 0:
                imp = imp / norm
            combined_imp += wi * imp
        return combined_imp

    # ── elimination loop ─────────────────────────────────────────────────────
    while len(remaining) > min_features:
        combined_score, per_level_scores = _combined_cv_score(remaining)
        history[str(len(remaining))] = {
            "combined_score": combined_score,
            "per_level_scores": per_level_scores,
            "features": remaining.copy(),
        }
        level_info = "  ".join(
            f"{lvl['label']} {s:.4f}" for lvl, s in zip(levels, per_level_scores)
        )
        print(f"  Features: {len(remaining):4d}  |  Combined: {combined_score:.4f}"
              f"  |  [{level_info}]")

        imp = _combined_importance(remaining)
        worst_idx  = int(np.argmin(imp))
        worst_feat = remaining[worst_idx]
        remaining.remove(worst_feat)
        print(f"  Removed: {worst_feat}  (combined importance {imp[worst_idx]:.6f})")

    # Score the minimum set
    combined_score, per_level_scores = _combined_cv_score(remaining)
    history[str(len(remaining))] = {
        "combined_score": combined_score,
        "per_level_scores": per_level_scores,
        "features": remaining.copy(),
    }
    level_info = "  ".join(
        f"{lvl['label']} {s:.4f}" for lvl, s in zip(levels, per_level_scores)
    )
    print(f"  Features: {len(remaining):4d}  |  Combined: {combined_score:.4f}"
          f"  |  [{level_info}]")

    # Select the feature set with the best combined score, subject to max_features
    best_n = max(
        history,
        key=lambda k: history[k]["combined_score"] if int(k) <= max_features else 0.0,
    )
    best_entry = history[best_n]
    best_features = best_entry["features"]
    print(
        f"\n  → Best combined {best_entry['combined_score']:.4f} "
        f"at {best_n} features → kept {len(best_features)}"
    )
    return best_features, history

# ---------------------------------------------------------------------------
# Optuna parameter spaces  (unchanged from original)
# ---------------------------------------------------------------------------

def _find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def _binary_params_for_trial(trial: optuna.Trial, model_type: str,
                              scale_pos_weight: float, random_state: int) -> Dict:
    if model_type == "rf":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "criterion":         trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth":         trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": "balanced", "n_jobs": N_CPUS, "random_state": random_state,
        }
    elif model_type == "xgb":
        return {
            "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "n_estimators":       trial.suggest_int("n_estimators", 300, 3000),
            "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":          trial.suggest_int("max_depth", 3, 12),
            "min_child_weight":   trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 1.0, 20.0),
            "random_state": random_state, "n_jobs": N_CPUS,
        }
    elif model_type == "lgbm":
        return {
            "objective": "binary", "boosting_type": "gbdt",
            "n_estimators":    trial.suggest_int("n_estimators", 300, 4000),
            "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 16, 512),
            "max_depth":       trial.suggest_int("max_depth", 3, 16),
            "min_data_in_leaf":trial.suggest_int("min_data_in_leaf", 5, 200),
            "lambda_l1":       trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2":       trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction":trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction":trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq":    trial.suggest_int("bagging_freq", 1, 10),
            "class_weight": "balanced", "n_jobs": N_CPUS,
            "random_state": random_state, "verbosity": -1,
        }
    elif model_type == "catb":
        return {
            "iterations":          trial.suggest_int("iterations", 300, 2000),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth":               trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count":        trial.suggest_int("border_count", 32, 255),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 2.0),
            "auto_class_weights": "Balanced", "verbose": False,
            "thread_count": N_CPUS, "random_state": 42,
        }
    raise ValueError(f"Unknown model_type: '{model_type}'")


def _multiclass_params_for_trial(trial: optuna.Trial, model_type: str,
                                  num_classes: int, random_state: int) -> Dict:
    if model_type == "xgb":
        return {
            "objective": "multi:softprob", "eval_metric": "mlogloss", "tree_method": "hist",
            "num_class":        num_classes,
            "n_estimators":     trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state, "n_jobs": N_CPUS,
        }
    elif model_type == "lgbm":
        return {
            "objective": "multiclass", "boosting_type": "gbdt", "num_class": num_classes,
            "n_estimators":    trial.suggest_int("n_estimators", 300, 3000),
            "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 16, 256),
            "max_depth":       trial.suggest_int("max_depth", 3, 14),
            "min_data_in_leaf":trial.suggest_int("min_data_in_leaf", 5, 150),
            "lambda_l1":       trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2":       trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction":trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction":trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq":    trial.suggest_int("bagging_freq", 1, 10),
            "class_weight": "balanced", "n_jobs": N_CPUS,
            "random_state": random_state, "verbosity": -1,
        }
    elif model_type == "rf":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced", "n_jobs": N_CPUS, "random_state": random_state,
        }
    raise ValueError(f"Unsupported multiclass model_type: '{model_type}'")

# ---------------------------------------------------------------------------
# Optuna studies  (unchanged from original)
# ---------------------------------------------------------------------------

def optimise_binary(
    X: pd.DataFrame, y: pd.Series, *,
    n_splits: int, n_trials: int, random_state: int,
    sample_weight: Optional[pd.Series], model_type: str,
) -> Tuple[Dict, float, optuna.Study]:
    w = (sample_weight.reindex(X.index)
         if isinstance(sample_weight, pd.Series) else None)
    n_pos = float((y == 1).sum())
    spw   = float(len(y) - n_pos) / (n_pos or 1)
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = _binary_params_for_trial(trial, model_type, spw, random_state)
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            w_tr = w.iloc[tr_idx].to_numpy() if w is not None else None
            m = build_binary_model(model_type, params.copy())
            _fit_model(m, model_type, X_tr, y_tr, X_va, y_va, sample_weight=w_tr)
            scores.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=OPTUNA_JOBS, gc_after_trial=True)
    best = study.best_trial.params.copy()

    oof_true, oof_proba = [], []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        w_tr = w.iloc[tr_idx].to_numpy() if w is not None else None
        m = build_binary_model(model_type, best.copy())
        _fit_model(m, model_type, X_tr, y_tr, sample_weight=w_tr)
        oof_proba.append(m.predict_proba(X_va)[:, 1])
        oof_true.append(y_va.to_numpy())
    threshold = _find_best_threshold(np.concatenate(oof_true), np.concatenate(oof_proba))
    return best, threshold, study


def optimise_multiclass(
    X: pd.DataFrame, y: pd.Series, *,
    n_splits: int, n_trials: int, random_state: int,
    sample_weight: Optional[pd.Series], model_type: str, num_classes: int,
) -> Tuple[Dict, optuna.Study]:
    w = (sample_weight.reindex(X.index)
         if isinstance(sample_weight, pd.Series) else None)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = _multiclass_params_for_trial(trial, model_type, num_classes, random_state)
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            w_tr = w.iloc[tr_idx].to_numpy() if w is not None else None
            m = build_multiclass_model(model_type, params.copy(), num_classes)
            if w_tr is not None:
                m.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                m.fit(X_tr, y_tr)
            scores.append(f1_score(y_va, m.predict(X_va), average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=OPTUNA_JOBS, gc_after_trial=True)
    return study.best_trial.params.copy(), study

# ---------------------------------------------------------------------------
# Per-level runner  (uses pre-selected shared features — no RFECV here)
# ---------------------------------------------------------------------------

def _run_level_shared_features(
    *,
    label: str,
    level_dir: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    w_train: pd.Series,
    shared_features: List[str],
    is_binary: bool,
    model_type: str,
    n_splits: int,
    n_trials: int,
    random_state: int,
) -> Dict:
    """Optuna → eval → save using the pre-determined shared feature set."""
    level_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    print(f"  Shared features used: {len(shared_features)}")

    # Encode labels
    le = LabelEncoder()
    y_train_enc = pd.Series(le.fit_transform(y_train.astype(str)),
                             index=y_train.index, name="label_enc")
    test_known = y_test.astype(str).isin(le.classes_)
    if not test_known.all():
        print(f"  Dropping {(~test_known).sum()} test rows with unseen classes.")
    X_test  = X_test.loc[test_known]
    y_test  = y_test.loc[test_known]
    y_test_enc = pd.Series(le.transform(y_test.astype(str)),
                            index=y_test.index, name="label_enc")

    num_classes = len(le.classes_)
    is_binary   = is_binary or (num_classes == 2)
    print(f"  Classes ({num_classes}): {le.classes_.tolist()}")
    print(f"  Mode: {'binary' if is_binary else 'multiclass'}")

    eff_model_type = model_type if (is_binary or model_type != "catb") else "lgbm"

    # Only keep shared features that are actually present in this level's data
    available = [f for f in shared_features if f in X_train.columns]
    if len(available) < len(shared_features):
        missing_feats = set(shared_features) - set(available)
        print(f"  WARNING: {len(missing_feats)} shared features not in this level's "
              f"columns — proceeding with {len(available)} features.")
    selected_features = available

    sw = compute_balanced_sample_weight(y_train_enc, w_train.reindex(y_train_enc.index))

    # Scale
    scaler = MinMaxScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train[selected_features]),
                            columns=selected_features, index=X_train.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_test[selected_features]),
                            columns=selected_features, index=X_test.index)

    # Optuna
    print(f"\n  Optimising model ({n_trials} trials) …")
    if is_binary:
        best_params, threshold, study = optimise_binary(
            X_tr_sc, y_train_enc,
            n_splits=n_splits, n_trials=n_trials, random_state=random_state,
            sample_weight=sw, model_type=model_type,
        )
    else:
        best_params, study = optimise_multiclass(
            X_tr_sc, y_train_enc,
            n_splits=n_splits, n_trials=n_trials, random_state=random_state,
            sample_weight=sw, model_type=eff_model_type, num_classes=num_classes,
        )
        threshold = None

    # Final calibrated model
    if is_binary:
        base_model = build_binary_model(model_type, best_params.copy())
    else:
        base_model = build_multiclass_model(eff_model_type, best_params.copy(), num_classes)
    cal_cv = min(5, int(y_train_enc.value_counts().min()))
    cal_cv = max(cal_cv, 2)
    final_model = CalibratedClassifierCV(base_model, cv=cal_cv, method="isotonic")
    final_model.fit(X_tr_sc, y_train_enc)

    # Predict
    test_proba = final_model.predict_proba(X_te_sc)
    if is_binary:
        test_proba_pos = test_proba[:, 1]
        test_pred = (test_proba_pos >= threshold).astype(int)
        auc_score = float(roc_auc_score(y_test_enc, test_proba_pos))
    else:
        test_pred = np.argmax(test_proba, axis=1)
        try:
            auc_score = float(roc_auc_score(
                y_test_enc, test_proba, multi_class="ovr", average="macro"
            )) if y_test_enc.nunique() > 1 else None
        except ValueError:
            auc_score = None

    macro_f1 = float(f1_score(y_test_enc, test_pred, average="macro", zero_division=0))
    report   = classification_report(y_test_enc, test_pred,
                                     target_names=[str(c) for c in le.classes_],
                                     zero_division=0)
    print(f"\n  Macro F1: {macro_f1:.4f}" +
          (f"  |  ROC-AUC: {auc_score:.4f}" if auc_score is not None else ""))
    print(report)

    # Save outputs
    summary = {
        "label": label,
        "model_type": model_type,
        "is_binary": is_binary,
        "classes": le.classes_.tolist(),
        "features": selected_features,
        "n_features": len(selected_features),
        "params": best_params,
        "threshold": threshold,
        "macro_f1": macro_f1,
        "roc_auc": auc_score,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    (level_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (level_dir / "report.txt").write_text(report)
    study.trials_dataframe().to_csv(level_dir / "optuna_trials.csv", index=False)

    if is_binary:
        pred_df = pd.DataFrame({
            "true": y_test_enc.values, "pred": test_pred, "proba": test_proba_pos,
        })
    else:
        pred_df = pd.DataFrame(
            test_proba, columns=[f"proba_{c}" for c in le.classes_]
        ).assign(true=y_test_enc.values, pred=test_pred,
                 proba=test_proba[np.arange(len(test_pred)), test_pred])
    pred_df.to_csv(level_dir / "predictions.csv", index=False)

    return {"macro_f1": macro_f1, "roc_auc": auc_score,
            "classes": le.classes_.tolist(), "n_features": len(selected_features)}

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_analysis(args: argparse.Namespace) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("ARGS:", args)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    all_targets = {args.sepsis_target, args.hemo_target, args.cef_target,
                   args.weight_column}
    cols_to_delete = list(DELETE_COLUMNS)
    cols_to_delete.extend([x for x in TARGET_REMOVE if x not in all_targets])

    df = load_processed_dataframe(args.database_file, cols_to_delete)
    missing = [c for c in all_targets if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    df[args.weight_column] = pd.to_numeric(df[args.weight_column], errors="coerce")
    df = df.dropna(subset=[args.sepsis_target, args.weight_column])
    df = df[df[args.weight_column] > 0]

    exclude_cols  = all_targets.copy()
    feature_cols  = [c for c in df.columns if c not in exclude_cols]

    high_na = [c for c in feature_cols if df[c].isna().mean() > args.na_perc_limit]
    if high_na:
        print(f"Dropping {len(high_na)} high-NA columns.")
        df.drop(columns=high_na, inplace=True)
        feature_cols = [c for c in feature_cols if c not in high_na]

    if args.impute_missing:
        df = impute_missing_values(df, exclude_cols)
    else:
        df = df.dropna(subset=feature_cols)

    feature_df  = df[feature_cols].copy()
    cat_cols    = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
        feature_df.columns = feature_df.columns.str.replace("[^0-9a-zA-Z_]+", "_", regex=True)

    # ------------------------------------------------------------------
    # 2. Global train/test split stratified on sepsis
    # ------------------------------------------------------------------
    y_sepsis = df.loc[feature_df.index, args.sepsis_target]
    y_hemo   = df.loc[feature_df.index, args.hemo_target]
    y_cef    = df.loc[feature_df.index, args.cef_target]
    w        = df.loc[feature_df.index, args.weight_column]

    X_train, X_test, y_sep_tr, y_sep_te, y_hemo_tr, y_hemo_te, \
    y_cef_tr, y_cef_te, w_tr, w_te = train_test_split(
        feature_df, y_sepsis, y_hemo, y_cef, w,
        test_size=args.test_size, random_state=args.random_state, stratify=y_sepsis,
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    if args.max_corr < 1.0:
        print(f"\nRemoving features with |Spearman| > {args.max_corr} …")
        kept = remove_correlated_features(X_train, threshold=args.max_corr)
        print(f"  Dropped {len(X_train.columns) - len(kept)} redundant features "
              f"→ {len(kept)} remain.")
        X_train = X_train[kept]
        X_test  = X_test[kept]

    # ------------------------------------------------------------------
    # 3. Output directory
    # ------------------------------------------------------------------
    output_dir = Path(str(args.output_dir) + "_" + TODAY + "_" + str(JOB_ID))
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict = {"args": {str(k): str(v) for k, v in args.__dict__.items()}}

    # ------------------------------------------------------------------
    # 4. Build per-level data descriptors for joint RFECV
    # ------------------------------------------------------------------

    # — Level 1 masks —
    l1_mask_tr = y_sep_tr.notna()
    l1_mask_te = y_sep_te.notna()

    # — Level 2 masks (rare-class filtering on train) —
    hemo_valid_tr = y_hemo_tr.notna()
    hemo_counts   = y_hemo_tr[hemo_valid_tr].value_counts()
    rare_thresh   = max(5, hemo_valid_tr.sum() / 50)
    rare_classes  = hemo_counts[hemo_counts < rare_thresh].index.tolist()
    if rare_classes:
        print(f"\n  Dropping rare hemo classes: {rare_classes}")
    l2_mask_tr = hemo_valid_tr & ~y_hemo_tr.isin(rare_classes)
    l2_mask_te = y_hemo_te.notna() & ~y_hemo_te.isin(rare_classes)

    # — Level 3 masks —
    l3_mask_tr = (
        y_cef_tr.notna()
        & y_hemo_tr.notna()
        & (y_hemo_tr != "NEGATIVE")
        & l2_mask_tr
    )
    l3_mask_te = (
        y_cef_te.notna()
        & y_hemo_te.notna()
        & (y_hemo_te != "NEGATIVE")
        & l2_mask_te
    )

    # Encode labels for RFECV (identical to _run_level logic)
    def _encode(y: pd.Series) -> np.ndarray:
        le = LabelEncoder()
        return le.fit_transform(y.astype(str))

    y_l1_enc = _encode(y_sep_tr.loc[l1_mask_tr])
    y_l2_enc = _encode(y_hemo_tr.loc[l2_mask_tr])

    _l2_model_type = args.model_type if args.model_type != "catb" else "lgbm"
    _l3_model_type = args.model_type if args.model_type != "catb" else "lgbm"

    has_l3 = l3_mask_tr.any()
    if has_l3:
        y_l3_enc = _encode(y_cef_tr.loc[l3_mask_tr])
        n_cef_classes = len(np.unique(y_l3_enc))
        l3_is_mc = n_cef_classes > 2
        l3_cv = max(min(args.cv_splits, int(y_cef_tr.loc[l3_mask_tr].value_counts().min())), 2)
    else:
        print("\nWARNING: No L3 training rows — L3 will be excluded from joint RFECV.")

    rfecv_cv = min(args.cv_splits, int(y_sep_tr.loc[l1_mask_tr].value_counts().min()),
                   int(y_hemo_tr.loc[l2_mask_tr].value_counts().min()))
    rfecv_cv = max(rfecv_cv, 2)

    levels_for_rfecv = [
        {
            "model":         _build_ranking_model(args.model_type, pd.Series(y_l1_enc), args.random_state),
            "X":             X_train.loc[l1_mask_tr],
            "y":             y_l1_enc,
            "label":         "L1-sepsis",
            "is_multiclass": False,
        },
        {
            "model":         _build_ranking_model(_l2_model_type, pd.Series(y_l2_enc), args.random_state),
            "X":             X_train.loc[l2_mask_tr],
            "y":             y_l2_enc,
            "label":         "L2-hemo",
            "is_multiclass": True,
        },
    ]
    if has_l3:
        levels_for_rfecv.append({
            "model":         _build_ranking_model(_l3_model_type, pd.Series(y_l3_enc), args.random_state),
            "X":             X_train.loc[l3_mask_tr],
            "y":             y_l3_enc,
            "label":         "L3-cef",
            "is_multiclass": l3_is_mc,
        })

    # ------------------------------------------------------------------
    # 5. Joint SHAP-RFECV to find shared feature set
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  JOINT SHAP-RFECV  (shared feature selection)")
    print(f"{'='*60}")
    print(f"  Levels: {[lvl['label'] for lvl in levels_for_rfecv]}")
    print(f"  Level weights: {args.level_weights}")
    print(f"  CV folds: {rfecv_cv}  |  max_features: {args.max_features}  "
          f"|  min_features: {args.min_features}")

    shared_features, rfecv_history = shap_rfecv_shared(
        levels=levels_for_rfecv,
        cv=rfecv_cv,
        min_features=args.min_features,
        max_features=args.max_features,
        level_weights=args.level_weights,
        random_state=args.random_state,
    )
    print(f"\n  Shared feature set ({len(shared_features)} features): {shared_features}")

    # Serialise RFECV history  (convert per_level_scores lists to plain lists)
    rfecv_history_serial = {}
    for k, v in rfecv_history.items():
        rfecv_history_serial[k] = {
            "combined_score": v["combined_score"],
            "per_level_scores": v["per_level_scores"],
            "features": v["features"],
        }
    (output_dir / "shared_rfecv_history.json").write_text(
        json.dumps(rfecv_history_serial, indent=2)
    )
    summaries["shared_features"] = shared_features
    summaries["shared_n_features"] = len(shared_features)

    # ------------------------------------------------------------------
    # 6. Per-level Optuna + evaluation on shared features
    # ------------------------------------------------------------------

    # LEVEL 1
    summaries["level1_sepsis"] = _run_level_shared_features(
        label="Level 1 – sepsis (binary)",
        level_dir=output_dir / "level1_sepsis",
        X_train=X_train.loc[l1_mask_tr],
        X_test=X_test.loc[l1_mask_te],
        y_train=y_sep_tr.loc[l1_mask_tr],
        y_test=y_sep_te.loc[l1_mask_te],
        w_train=w_tr,
        shared_features=shared_features,
        is_binary=True,
        model_type=args.model_type,
        n_splits=args.cv_splits,
        n_trials=args.n_trials,
        random_state=args.random_state,
    )

    # LEVEL 2
    summaries["level2_hemo"] = _run_level_shared_features(
        label="Level 2 – resultado_hemo_grouped (multiclass)",
        level_dir=output_dir / "level2_hemo",
        X_train=X_train.loc[l2_mask_tr],
        X_test=X_test.loc[l2_mask_te],
        y_train=y_hemo_tr.loc[l2_mask_tr],
        y_test=y_hemo_te.loc[l2_mask_te],
        w_train=w_tr,
        shared_features=shared_features,
        is_binary=False,
        model_type=_l2_model_type,
        n_splits=args.cv_splits,
        n_trials=args.n_trials,
        random_state=args.random_state,
    )

    # LEVEL 3
    if not has_l3:
        print("\nWARNING: No L3 training rows — skipping level 3.")
        summaries["level3_cef"] = None
    else:
        summaries["level3_cef"] = _run_level_shared_features(
            label=f"Level 3 – resistente_cefalosporina "
                  f"({'binary' if not l3_is_mc else 'multiclass'})",
            level_dir=output_dir / "level3_cef",
            X_train=X_train.loc[l3_mask_tr],
            X_test=X_test.loc[l3_mask_te],
            y_train=y_cef_tr.loc[l3_mask_tr],
            y_test=y_cef_te.loc[l3_mask_te],
            w_train=w_tr,
            shared_features=shared_features,
            is_binary=(not l3_is_mc),
            model_type=_l3_model_type,
            n_splits=l3_cv,
            n_trials=args.n_trials,
            random_state=args.random_state,
        )

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    print(f"\n{'='*60}")
    print("COMPLETED.  Results in:", output_dir)
    print(f"{'='*60}")
    print(f"  Shared features ({len(shared_features)}): {shared_features}")
    for lvl, m in summaries.items():
        if not isinstance(m, dict) or "macro_f1" not in m:
            continue
        auc_str = f"  ROC-AUC {m['roc_auc']:.4f}" if m.get("roc_auc") else ""
        print(f"  {lvl:30s}  Macro-F1 {m['macro_f1']:.4f}{auc_str}"
              f"  ({m['n_features']} features)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    home = Path.cwd()
    default_db  = str(home / "mepram_data" / "df_merged_full_multilabel_grouped.csv")
    default_out = str(home / "mepram_data" / "outputs" / "shared_features_analysis")

    p = argparse.ArgumentParser(
        description="Joint SHAP-RFECV shared-feature analysis for all target levels."
    )
    p.add_argument("--database-file", "-db", type=Path, default=default_db)
    p.add_argument("--output-dir",    "-o",  type=Path, default=default_out)
    p.add_argument("--sepsis-target", type=str, default="sepsis")
    p.add_argument("--hemo-target",   type=str, default="resultado_hemo_grouped")
    p.add_argument("--cef-target",    type=str, default="resistente_cefalosporina_multi")
    p.add_argument("--weight-column", type=str, default="sample_weight")
    p.add_argument(
        "--model-type", type=str, choices=["xgb", "lgbm", "rf", "catb"],
        default="lgbm",
        help="Estimator used for all levels.  catb is automatically replaced "
             "by lgbm for multiclass levels. Default: lgbm.",
    )
    p.add_argument(
        "--n-trials", "-t", type=int, default=300,
        help="Optuna trials per level (default: 300).",
    )
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument(
        "--test-size", "-ts", type=float, default=0.30,
        help="Hold-out fraction (default: 0.30).",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--max-features", type=int, default=50,
        help="Maximum features kept after joint SHAP-RFECV (default: 50).",
    )
    p.add_argument(
        "--min-features", type=int, default=5,
        help="Minimum features that joint SHAP-RFECV will not go below (default: 5).",
    )
    p.add_argument(
        "--na-perc-limit", "-na", type=float, default=0.20,
        help="Drop columns with more than this fraction of NaN (default: 0.20).",
    )
    p.add_argument(
        "--max-corr", type=float, default=0.90,
        help="Spearman correlation threshold for redundancy removal (default: 0.90). "
             "Set to 1.0 to disable.",
    )
    p.add_argument(
        "--no-impute", dest="impute_missing", action="store_false",
        help="Disable missing-value imputation (default: enabled).",
    )
    p.add_argument(
        "--level-weights", type=float, nargs="+", default=[1.0, 1.0, 2.0],
        metavar="W",
        help="Relative weights for L1 / L2 / L3 in the joint RFECV composite score. "
             "Weights are normalised internally.  Default: 1 1 2 (double weight to L3).",
    )
    p.set_defaults(impute_missing=True)
    return p


def main() -> None:
    start  = time.time()
    parser = build_arg_parser()
    args   = parser.parse_args()

    # Validate level-weights length
    if len(args.level_weights) not in (2, 3):
        parser.error("--level-weights must have 2 values (if L3 always absent) or 3 values.")

    print("Parsed args:", args)

    output_folder = Path(str(args.output_dir) + "_" + TODAY + "_" + str(JOB_ID))
    try:
        run_analysis(args)
    except Exception:
        if output_folder.exists():
            shutil.rmtree(output_folder)
        raise
    print(f"\nElapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
