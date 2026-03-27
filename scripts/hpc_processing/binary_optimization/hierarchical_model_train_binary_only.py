#!/usr/bin/env python3
"""
Binary-only Optuna training script (no multiclass head).

This reuses the preprocessing and RFECV flow from the hierarchical trainer but
optimises only a single binary model.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import shutil

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

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

TARGET_REMOVE = ["sepsis", "resultado_hemo", "resultado_hemo_grouped", "all_cult_org", "infected_yes_no", "bmr_etiologia", "fenotipo_resistencia",  "fenotipo_resistencia_grouped", "resistente_cefalosporina"]

DELETE_COLUMNS = ["qsofa", "vasopresores", "hipotension", "freq_bacteria", "freq_bac_foco", "Unnamed: 0", "person_id", "fecha_ingreso_urgencias", "fecha_ingreso_urgencias_x", "shock_septico", "sintoma_nan", "fecha_nacimiento", "codigo_postal", "center", "dag", "ultima_fecha", "mujer_gestante"]

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
# Data preparation
# ---------------------------------------------------------------------------


def safe_drop_columns(df, columns):
    """Drop a list of columns from a DataFrame, silently skipping any that are absent.

    Args:
        df: Input DataFrame.
        columns: Column names to attempt to drop.

    Returns:
        DataFrame with the specified columns removed (where present).
    """
    for col in columns:
        try:
            df = df.drop(columns=col)
        except KeyError:
            print(f"Warning: Column {col} not found in DataFrame. Skipping drop.")
    return df


def load_processed_dataframe(csv_path: Path, cols_to_delete: list, target: str) -> pd.DataFrame:
    """Load the merged dataset from CSV and apply domain-specific filtering.

    Steps performed:
    - Reads the CSV at csv_path.
    - Maps the numeric foco column to human-readable Spanish labels using FOCUS_MAP.
    - Removes rows whose multiclass_target belongs to MINOR_CLASSES_TO_DROP (only for resultado_hemo / all_cult_org targets).
    - For fenotipo_resistencia targets, drops phenotype classes that represent fewer than 1/50th of the total sample count (very rare classes).
    - Drops administrative / leakage columns listed in cols_to_delete.

    Args:
        csv_path: Path to the merged input CSV.
        cols_to_delete: Column names to drop before returning the DataFrame.
        multiclass_target: Name of the multiclass label column; governs which class-filtering rules are applied.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = pd.read_csv(csv_path)
    if "foco" in df.columns:
        df = df.copy()
        df["foco"] = df["foco"].map(FOCUS_MAP).fillna(df["foco"])
    if "resistente_cefalosporina" == target:
        df = df[df["resultado_hemo"] != "NEGATIVE"] 
    elif "resultado_hemo_grouped" == target:
        df = df[df["resultado_hemo_grouped"].isin(["Bacilo gram-", "Coco gram+"])]
    df = safe_drop_columns(df=df, columns=cols_to_delete)
    return df


def impute_missing_values(loaded_df, exclude_cols):
    """Impute missing values using column-type-aware strategies.

    Columns are split into four groups and imputed separately:
    - Binary columns (values in {0, 1}): mode imputation via SimpleImputer.
    - Continuous numeric columns (>=15 unique values): KNN imputation (k=5, distance-weighted) to preserve local data structure.
    - Low-cardinality numeric columns (<15 unique values, treated as categorical-numeric): mode imputation, result cast to int.
    - String / categorical columns: mode imputation, result cast to str.

    Target and weight columns listed in exclude_cols are excluded from imputation and re-attached to the result unchanged.

    Args:
        loaded_df: DataFrame that may contain missing values.
        exclude_cols: Column names to skip during imputation (e.g. target labels, sample weights).

    Returns:
        DataFrame with the same shape as loaded_df but with NaNs filled in all non-excluded columns.
    """
    exclude_cols = set(exclude_cols)
    df_copy = loaded_df.drop(columns=list(exclude_cols), errors="ignore").copy()
    numeric_cols = df_copy.select_dtypes(include=["int", "float"]).columns
    categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns
    binary_cols = [col for col in numeric_cols if set(df_copy[col].dropna().unique()) <= {0, 1}]

    continuous_cols = [col for col in numeric_cols if col not in binary_cols]
    categorical_numeric_cols = [col for col in continuous_cols if df_copy[col].nunique() < 15]
    continuous_cols = [col for col in continuous_cols if col not in categorical_numeric_cols]

    if binary_cols:
        binary_imputer = SimpleImputer(strategy="most_frequent")
        df_copy[binary_cols] = binary_imputer.fit_transform(df_copy[binary_cols]).astype(int)

    if continuous_cols:
        numeric_imputer = KNNImputer(n_neighbors=5, weights="distance")
        df_copy[continuous_cols] = numeric_imputer.fit_transform(df_copy[continuous_cols])

    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        df_copy[categorical_cols] = categorical_imputer.fit_transform(df_copy[categorical_cols])
        df_copy[categorical_cols] = df_copy[categorical_cols].astype(str)

    if categorical_numeric_cols:
        categorical_numeric_imputer = SimpleImputer(strategy="most_frequent")
        df_copy[categorical_numeric_cols] = categorical_numeric_imputer.fit_transform(df_copy[categorical_numeric_cols])
        df_copy[categorical_numeric_cols] = df_copy[categorical_numeric_cols].astype(int)

    for col in exclude_cols:
        if col in loaded_df.columns:
            df_copy[col] = loaded_df[col]
    return df_copy


def compute_balanced_sample_weight(labels: pd.Series, base_sample_weight: pd.Series | None = None) -> pd.Series:
    """Compute per-sample weights that correct for class imbalance.

    Uses sklearn.utils.class_weight.compute_class_weight with class_weight='balanced' to derive a weight for each class, then maps those weights onto every sample. If base_sample_weight is provided (e.g. clinical cohort weights), the balanced weights are multiplied element-wise so both sources of weighting are combined.

    Args:
        labels: Series of class labels for the training split.
        base_sample_weight: Optional pre-existing per-sample weights. When supplied the result is balanced_weight * base_weight.

    Returns:
        Series of per-sample weights with the same index as labels.
    """
    classes = np.unique(labels)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    weight_map = {cls: weight for cls, weight in zip(classes, class_weights)}
    balanced = labels.map(weight_map)
    if base_sample_weight is not None:
        base_series = pd.Series(base_sample_weight, index=labels.index, name="base_weight")
        balanced = balanced * base_series
    return balanced


def perform_rfecv_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    cv_splits: int,
    step: int | float,
    min_features_to_select: int,
    scoring: str,
) -> Tuple[List[str], RFECV]:
    """Select an optimal feature subset via Recursive Feature Elimination with CV.

    Wraps sklearn's RFECV around a RandomForestClassifier (400 trees, balanced subsample weights) and a StratifiedKFold cross-validator. Features are ranked by importance and eliminated step at a time until the cross-validated score stops improving, subject to the min_features_to_select floor.

    Args:
        X: Full feature matrix (training split).
        y: Target label series aligned with X.
        random_state: Seed for the Random Forest and the CV splitter.
        cv_splits: Number of stratified folds used during cross-validation.
        step: Number (int) or fraction (float in (0, 1]) of features to remove at each iteration.
        min_features_to_select: Hard lower bound on the number of features retained. Must be between 1 and X.shape[1].
        scoring: Sklearn scoring string passed to RFECV (e.g. 'f1_macro').

    Returns:
        Tuple of (selected_columns, fitted_selector) where selected_columns is the list of column names chosen by RFECV and fitted_selector is the fitted RFECV object (useful for inspecting support_, ranking_, and cv_results_).

    Raises:
        ValueError: If X is empty, step is out of range, or min_features_to_select is outside [1, n_features].
    """
    if X.empty:
        raise ValueError("Cannot run RFECV on an empty feature matrix.")

    available_features = X.shape[1]
    if min_features_to_select <= 0 or min_features_to_select > available_features:
        raise ValueError(
            f"min_features_to_select must be between 1 and {available_features}; received {min_features_to_select}."
        )

    if isinstance(step, int):
        if step <= 0:
            raise ValueError("RFECV step must be a positive integer.")
    else:
        if not (0 < step <= 1):
            raise ValueError("RFECV fractional step must be in the (0, 1] range.")

    estimator = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=1,
    )
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=2,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=1,
    )
    selector.fit(X, y)
    selected_columns = X.columns[selector.support_].tolist()
    return selected_columns, selector


# ---------------------------------------------------------------------------
# Optuna optimisation
# ---------------------------------------------------------------------------

def optimise_binary_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    n_trials: int,
    random_state: int,
    sample_weight: pd.Series | np.ndarray | None,
    use_stacking: bool,
    model_type: str,
    meta_model,
) -> Tuple[Dict[str, object], float, optuna.study.Study]:
    """Tune the binary model and choose the best probability threshold."""
    weight_series = None
    if sample_weight is not None:
        if isinstance(sample_weight, pd.Series):
            weight_series = sample_weight.reindex(X.index)
        else:
            weight_series = pd.Series(sample_weight, index=X.index, name="sample_weight")

    if weight_series is not None:
        pos_mask = y == 1
        pos_weight_sum = float(weight_series.loc[pos_mask].sum())
        neg_weight_sum = float(weight_series.loc[~pos_mask].sum())
        scale_pos_weight = neg_weight_sum / pos_weight_sum if pos_weight_sum else 1.0
    else:
        pos_count = float((y == 1).sum())
        neg_count = float(len(y) - pos_count)
        scale_pos_weight = neg_count / pos_count if pos_count else 1.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    def stacking_objective(trial: optuna.Trial) -> float:
        threshold = trial.suggest_float("threshold", 0.2, 0.8)
        params_lgr = {
            "C": trial.suggest_float("lr_C", 0.05, 2.0, log=True),
            "penalty": "l2",
            "class_weight": "balanced",
            "max_iter": 3000,
            "n_jobs": 1,
        }
        params_lgbm = {
            "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("lgbm_n_estimators", 300, 1500, step=100),
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 16, 64, step=4),
            "max_depth": trial.suggest_int("lgbm_max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 30, 120, step=10),
            "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("lgbm_reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain": trial.suggest_float("lgbm_min_split_gain", 0.0, 0.5),
        }
        class_weight = {0: 1.0, 1: scale_pos_weight}

        params_xgb = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 2000, step=100),
            "max_depth": trial.suggest_int("xgb_max_depth", 4, 10),
            "min_child_weight": trial.suggest_float("xgb_min_child_weight", 3.0, 15.0),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
            "gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
        }

        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            w_tr = sample_weight.iloc[tr_idx].to_numpy() if sample_weight is not None else None
            w_va = sample_weight.iloc[va_idx].to_numpy() if sample_weight is not None else None

            if weight_series is not None:
                w_tr = weight_series.loc[X_tr.index].to_numpy()
                w_va = weight_series.loc[X_va.index].to_numpy()
            else:
                w_tr = w_va = None

            base_models = [
                ("lr", LogisticRegression(**params_lgr)),
                ("lgbm", LGBMClassifier(**params_lgbm, random_state=random_state, n_jobs=1)),
                #("xgb", XGBClassifier(**params_xgb, random_state=random_state, n_jobs=-1)),
            ]
            # Stacking
            model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=3,
                n_jobs=1,
                passthrough=False,
            )
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            probas = model.predict_proba(X_va)[:, 1]
            preds = (probas >= threshold).astype(int)
            score = fbeta_score(y_va, preds, beta=2, sample_weight=w_va) #roc_auc_score(y_va, probas, sample_weight=w_va) 
            scores.append(score)
        return float(np.mean(scores))

    def objective(trial: optuna.Trial) -> float:
        if model_type == "rf":
            rf_params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "class_weight": "balanced",
                "n_jobs": 1,
                "random_state": random_state
            }
            models_dict = {"rf": (RandomForestClassifier, rf_params)}
        elif model_type == "xgb":
            xgb_params = {
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
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
                "random_state": random_state,
                "n_jobs": 1
            }
            models_dict = {"xgb": (XGBClassifier, xgb_params)}
        elif model_type == "lgbm":
            lgbm_params = {
                "objective": "binary",
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 300, 4000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 512),
                "max_depth": trial.suggest_int("max_depth", 3, 16),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
                "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "class_weight": "balanced",
                "max_bin": trial.suggest_int("max_bin", 128, 512),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "n_jobs": 1,
                "random_state": random_state,
                "verbosity": -1
            }
            models_dict = {"lgbm": (LGBMClassifier, lgbm_params)}
        elif model_type == "catb":
            catb_params = {
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
            raise ValueError(f"Invalid model_type: {model_type}")

        threshold = trial.suggest_float("threshold", 0.1, 0.9)
        scores = []

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            w_tr = sample_weight.iloc[tr_idx].to_numpy() if sample_weight is not None else None
            w_va = sample_weight.iloc[va_idx].to_numpy() if sample_weight is not None else None

            if weight_series is not None:
                w_tr = weight_series.loc[X_tr.index].to_numpy()
                w_va = weight_series.loc[X_va.index].to_numpy()
            else:
                w_tr = w_va = None
            if model_type == "catb":
                model = CatBoostClassifier(**catb_params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_va, y_va),
                    early_stopping_rounds=max(20, int(0.05 * catb_params["iterations"])),
                    verbose=False
                )
            else:
                model = models_dict[model_type][0](**models_dict[model_type][1])
                model.fit(X_tr, y_tr)

            y_proba = model.predict_proba(X_va)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            # --- 5. compute macro-F1 ---
            score = f1_score(y_va, y_pred, average="macro")
            scores.append(score)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    if use_stacking:
        study.optimize(stacking_objective, n_trials=n_trials, n_jobs=N_CPUS, gc_after_trial=True)
    else:
        study.optimize(objective, n_trials=n_trials, n_jobs=N_CPUS, gc_after_trial=True)

    best_params = study.best_trial.params.copy()
    best_threshold = best_params.pop("threshold")
    return best_params, best_threshold, study

def build_final_stacking_model(best_params, random_state, meta_model):
    params_lgr = {k.replace("lr_", ""): v for k, v in best_params.items() if k.startswith("lr_")}
    params_lgbm = {k.replace("lgbm_", ""): v for k, v in best_params.items() if k.startswith("lgbm_")}
    params_xgb = {k.replace("xgb_", ""): v for k, v in best_params.items() if k.startswith("xgb_")}
    params_rf  = {k.replace("rf_", ""): v for k, v in best_params.items() if k.startswith("rf_")}
    final_base_models = [
        (
            "lr", 
            LogisticRegression(
                **params_lgr,
                class_weight="balanced",
                max_iter=5000,
                n_jobs=1,
            )
        ),
        (
            "lgbm",
            LGBMClassifier(
                **params_lgbm,
                objective="binary",
                random_state=random_state,
                n_jobs=1,
                verbose=-1,
            )
        )
    ]
    binary_model = StackingClassifier(
        estimators=final_base_models,
        final_estimator=meta_model,
        passthrough=False,
    )
    """(
            "xgb",
            XGBClassifier(
                **params_xgb,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=1,
            )
        ),
        (
            "rf",
            RandomForestClassifier(
                **params_rf,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            )
        ),"""
    return binary_model

def find_best_threshold(y_true, probas, sample_weight=None, beta=2.0):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr, best_score = 0.5, -np.inf

    for t in thresholds:
        preds = (probas >= t).astype(int)
        score = fbeta_score(
            y_true, preds, beta=beta, sample_weight=sample_weight
        )
        if score > best_score:
            best_score = score
            best_thr = t

    return best_thr, best_score

def build_binary_model(model_type, best_params):
    model_typedict = {
        "xgb": XGBClassifier,
        "lgbm": LGBMClassifier,
        "rf": RandomForestClassifier,
        "catb": CatBoostClassifier
    }
    binary_model = model_typedict[model_type](**best_params)

    return binary_model

# ---------------------------------------------------------------------------
# Training with one feature subset
# ---------------------------------------------------------------------------


def train_with_feature_subset(
    *,
    subset_name: str,
    binary_features: List[str],
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_train_binary: pd.Series,
    y_test_binary: pd.Series,
    sample_weight_train: pd.Series | None,
    sample_weight_test: pd.Series | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    """Train, evaluate, and persist the full two-stage hierarchical classifier.

    This is the core training function. Given pre-selected feature lists for the binary gate and the phenotype head, it:
    1. Creates a timestamped output subdirectory and saves selected_features.json.
    2. Scales features with MinMaxScaler (fit on train, applied to test).
    3. Converts binary labels to 0/1 and computes balanced gate weights.
    4. Runs optimise_binary_gate and trains the final gate model on the full training set.
    5. Filters to positive-only training samples, encodes phenotype labels, and runs optimise_binary_gate again for the two-class phenotype head.
    6. Applies the hierarchical prediction pipeline on the test set: gate predicts POSITIVE/NEGATIVE; for POSITIVE samples, the phenotype head predicts the resistance class.
    7. Computes and saves: confusion matrices (binary + multiclass), classification reports, macro F1, binary ROC-AUC, multiclass ROC-AUC, Optuna trial CSVs, and summary.json.

    Args:
        subset_name: Human-readable label for this feature subset (used as the subdirectory name and in log output).
        binary_features: Feature columns to use for the binary gate.
        multiclass_features: Feature columns to use for the phenotype head.
        X_train_full / X_test_full: Full (unscaled) feature matrices.
        y_train_binary / y_test_binary: Binary gate label series.
        y_train_multiclass / y_test_multiclass: Phenotype label series.
        sample_weight_train / sample_weight_test: Optional per-sample weights; pass None to omit weighting.
        args: Parsed CLI arguments (model types, Optuna trial counts, CV splits, random state, negative label, etc.).
        output_dir: Root directory under which the subset subdirectory is created.

    Returns:
        Summary dict containing model parameters, thresholds, evaluation metrics, text reports, and the output directory path. This dict is also written to <subset_output_dir>/summary.json.
    """
    subset_output_dir = output_dir / subset_name
    subset_output_dir.mkdir(parents=True, exist_ok=True)
    if not subset_output_dir.exists():
        raise OSError(f"Could not create subset output directory: {subset_output_dir}")

    (subset_output_dir / "selected_features.json").write_text(
        json.dumps(
            {
                "subset": subset_name,
                "binary": {"feature_count": len(binary_features), "features": binary_features},
            },
            indent=2,
        )
    )

    X_train_binary = X_train_full[binary_features].copy()
    X_test_binary = X_test_full[binary_features].copy()
    binary_scaler = MinMaxScaler()
    X_train_binary_scaled = pd.DataFrame(
        binary_scaler.fit_transform(X_train_binary),
        columns=binary_features,
        index=X_train_binary.index,
    )
    X_test_binary_scaled = pd.DataFrame(
        binary_scaler.transform(X_test_binary),
        columns=binary_features,
        index=X_test_binary.index,
    )

    train_weights = None
    if sample_weight_train is not None:
        train_weights = pd.Series(sample_weight_train, name="sample_weight").loc[X_train_binary.index]
    test_weights = None
    if sample_weight_test is not None:
        test_weights = pd.Series(sample_weight_test, name="sample_weight").loc[X_test_binary.index]

    label_encoder = LabelEncoder()
    y_train_binary_enc = pd.Series(
        label_encoder.fit_transform(y_train_binary),
        index=y_train_binary.index,
        name="encoded_target",
    )
    y_test_binary_enc = pd.Series(
        label_encoder.transform(y_test_binary),
        index=y_test_binary.index,
        name="encoded_target",
    )
    gate_sample_weight = compute_balanced_sample_weight(y_train_binary_enc, base_sample_weight=train_weights)

    meta_model = LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=15,
        max_depth=3,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=args.random_state,
        n_jobs=1,
    )

    # Binary optimisation + training
    binary_params, best_threshold, binary_study = optimise_binary_model(
        X_train_binary_scaled,
        y_train_binary_enc,
        n_splits=args.cv_splits,
        n_trials=args.binary_trials,
        random_state=args.random_state,
        sample_weight=gate_sample_weight,
        use_stacking=args.use_stacking,
        model_type=args.binary_model,
        meta_model=meta_model
    )

    pos_weight_sum = float(gate_sample_weight.loc[y_train_binary_enc == 1].sum())
    neg_weight_sum = float(gate_sample_weight.loc[y_train_binary_enc == 0].sum())
    gate_scale_pos_weight = neg_weight_sum / pos_weight_sum if pos_weight_sum else 1.0
    if args.use_stacking:
        binary_model = build_final_stacking_model(
            best_params=binary_params,
            random_state=args.random_state,
            meta_model=meta_model
        )
    else:
        binary_model = build_binary_model(
            model_type = args.binary_model,
            best_params = binary_params,
        )
    binary_model.fit(X_train_binary_scaled, y_train_binary_enc)
    # Evaluation
    """train_proba = binary_model.predict_proba(X_train_binary_scaled)[:, 1]
    best_threshold, _ = find_best_threshold(
        y_train_binary_enc,
        train_proba,
        sample_weight=gate_sample_weight,
    )"""
    binary_proba_test = binary_model.predict_proba(X_test_binary_scaled)[:, 1]
    binary_pred_test = (binary_proba_test >= best_threshold).astype(int)

    binary_conf_matrix = confusion_matrix(
        y_test_binary_enc,
        binary_pred_test,
        labels=[0, 1],
        sample_weight=test_weights,
    )
    class_names = [str(lbl) for lbl in label_encoder.classes_]
    binary_conf_df = pd.DataFrame(
        binary_conf_matrix,
        index=[f"true_{lbl}" for lbl in class_names],
        columns=[f"pred_{lbl}" for lbl in class_names],
    )

    binary_report = classification_report(
        y_test_binary_enc,
        binary_pred_test,
        target_names=class_names,
        zero_division=0,
        sample_weight=test_weights,
    )
    macro_f1 = f1_score(
        y_test_binary_enc,
        binary_pred_test,
        labels=[0, 1],
        average="macro",
        sample_weight=test_weights,
    )
    macro_auc = roc_auc_score(
        y_test_binary_enc,
        binary_proba_test,
        sample_weight=test_weights,
    )

    pred_labels = label_encoder.inverse_transform(binary_pred_test)
    positive_label = label_encoder.classes_[1] if len(label_encoder.classes_) > 1 else label_encoder.classes_[0]
    results_payload = {
        "true_binary_label": y_test_binary.reset_index(drop=True),
        "binary_positive_prob": binary_proba_test,
        "binary_pred": pred_labels,
    }
    if test_weights is not None:
        results_payload["sample_weight"] = test_weights.reset_index(drop=True)

    binary_results = pd.DataFrame(results_payload)
    binary_results["is_positive_true"] = binary_results["true_binary_label"] == positive_label

    macro_f1 = float(macro_f1)
    macro_auc = float(macro_auc)

    # Persist artefacts
    binary_results.to_csv(subset_output_dir / "binary_predictions.csv", index=False)
    binary_conf_df.to_csv(subset_output_dir / "binary_confusion_matrix.csv")

    summary = {
        "subset": subset_name,
        "binary_model": args.binary_model,
        "binary_feature_count": len(binary_features),
        "binary_params": binary_params,
        "binary_threshold": best_threshold,
        "macro_f1": macro_f1,
        "binary_roc_auc": macro_auc,
        "binary_report": binary_report,
        "class_names": class_names,
        "binary_trials": args.binary_trials,
        "binary_study_best_value": binary_study.best_value if binary_study else None,
        "output_dir": str(subset_output_dir),
        "n_cpus": N_CPUS
    }
    (subset_output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (subset_output_dir / "binary_report.txt").write_text(binary_report)

    trials_dir = subset_output_dir / "optuna_trials"
    trials_dir.mkdir(exist_ok=True)
    binary_study.trials_dataframe().to_csv(trials_dir / "binary_trials.csv", index=False)

    # Final console summary for SLURM logs
    print(f"[{subset_name}] Binary best params:", binary_params)
    print(f"[{subset_name}] Binary threshold:", best_threshold)
    print(f"[{subset_name}] Hold-out macro F1:", f"{macro_f1:.3f}")
    print(f"[{subset_name}] Hold-out binary ROC-AUC:", f"{macro_auc:.3f}")

    return summary


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    """Orchestrate the end-to-end training pipeline from CLI arguments.

    Executes the following sequence:
    1. Builds the column-deletion list by merging DELETE_COLUMNS with target columns that are not the active binary/multiclass targets.
    2. Loads and cleans the dataset via load_processed_dataframe.
    3. Drops columns exceeding the --na-perc-limit missing-value threshold.
    4. Optionally imputes remaining missing values (--no-impute disables this).
    5. One-hot encodes categorical feature columns; sanitises column names.
    6. Performs a stratified train/test split (stratified on the binary target).
    7. Runs RFECV independently for the binary gate and the positive-only multiclass subset, producing separate optimal feature sets.
    8. Calls train_with_feature_subset once with the RFECV-selected features.
    9. Writes rfecv_selected_features.json and aggregate_summary.json to the timestamped output directory.

    Args:
        args: argparse.Namespace produced by build_arg_parser. All training configuration (file paths, model choices, trial counts, CV splits, random seed, imputation flag, etc.) is read from here.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("SELECTED ARGS: ", args)
    cols_to_delete = list(DELETE_COLUMNS)
    keep_targets = {args.binary_target, args.weight_column}
    cols_to_delete.extend([x for x in TARGET_REMOVE if x not in keep_targets])
    df = load_processed_dataframe(args.database_file, cols_to_delete=cols_to_delete, target=args.binary_target)

    missing_cols = [col for col in [args.binary_target, args.weight_column] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from dataframe: {missing_cols}")

    working_df = df.copy()
    working_df[args.weight_column] = pd.to_numeric(working_df[args.weight_column], errors="coerce")
    working_df = working_df.dropna(subset=[args.binary_target, args.weight_column])
    working_df = working_df[working_df[args.weight_column] > 0]

    exclude_cols = {args.binary_target, args.weight_column}
    feature_cols = [col for col in working_df.columns if col not in exclude_cols]
    dropped_for_na = []
    for col in feature_cols:
        na_per = working_df[col].isna().mean()
        if na_per > args.na_perc_limit:
            print(f"Column {col} --> %NaN = {na_per}. deleted")
            dropped_for_na.append(col)
    if dropped_for_na:
        working_df = working_df.drop(columns=dropped_for_na)
        feature_cols = [col for col in feature_cols if col not in dropped_for_na]
    if not feature_cols:
        raise ValueError("No usable feature columns remain after NA filtering.")
    if args.impute_missing:
        working_df = impute_missing_values(working_df, exclude_cols)
    else:
        working_df = working_df.dropna(subset=feature_cols)
    feature_df = working_df[feature_cols]
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        print(categorical_cols)
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=False)
        feature_df.columns = feature_df.columns.str.replace("[^0-9a-zA-Z_]+", "_", regex=True)
    print(feature_df.columns)

    binary_target_series = working_df.loc[feature_df.index, args.binary_target]
    weight_series = working_df.loc[feature_df.index, args.weight_column]

    X_train_full, X_test_full, y_train_binary, y_test_binary, sample_weight_train, sample_weight_test = train_test_split(
        feature_df,
        binary_target_series,
        weight_series,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=binary_target_series,
    )
    print("Finished traintest split")
    available_features = X_train_full.shape[1]
    if available_features == 0:
        raise ValueError("No features available after preprocessing; aborting.")

    min_features = min(args.rfecv_min_features, available_features)

    if args.rfecv_step is None:
        step_value: int | float = max(1, int(0.1 * available_features))
    else:
        step_value = args.rfecv_step
        if step_value < 1:
            step_value = float(step_value)
        else:
            step_value = int(step_value)

    print("Starting binary RFECV...")
    binary_selected_columns, binary_rfecv_selector = perform_rfecv_feature_selection(
        X_train_full,
        y_train_binary,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
        step=step_value,
        min_features_to_select=min_features,
        scoring=args.rfecv_scoring,
    )
    print(f"Finished binary RFECV. Selected {len(binary_selected_columns)} features.")
    binary_selected_columns = X_train_full.columns.tolist()

    output_folder = Path(str(args.output_dir) + "_" + str(TODAY))
    output_folder.mkdir(parents=True, exist_ok=True)
    if not output_folder.exists():
        raise OSError(f"Could not create output directory: {output_folder}")
    subset_name = f"rfecv_bin{len(binary_selected_columns)}"
    summary = train_with_feature_subset(
        subset_name=subset_name,
        binary_features=binary_selected_columns,
        X_train_full=X_train_full,
        X_test_full=X_test_full,
        y_train_binary=y_train_binary,
        y_test_binary=y_test_binary,
        sample_weight_train=sample_weight_train,
        sample_weight_test=sample_weight_test,
        args=args,
        output_dir=output_folder,
    )

    selection_metadata = {
        "binary": {
            "selected_feature_count": len(binary_selected_columns),
            "selected_features": binary_selected_columns,
            "rfecv_support_mask": binary_rfecv_selector.support_.tolist(),
            "rfecv_ranking": binary_rfecv_selector.ranking_.tolist(),
        },
        "rfecv_step": step_value,
        "rfecv_min_features_requested": args.rfecv_min_features,
        "rfecv_min_features_used": min_features,
        "rfecv_scoring": args.rfecv_scoring,
    }
    (output_folder / "rfecv_selected_features.json").write_text(json.dumps(selection_metadata, indent=2))

    binary_auc_str = f"{summary['binary_roc_auc']:.3f}" if summary.get("binary_roc_auc") is not None else "N/A"
    print(f"[{subset_name}] Macro F1: {summary['macro_f1']:.3f} | Binary ROC-AUC: {binary_auc_str}")

    def _extract_scores(selector: RFECV) -> List[float] | None:
        """Extract mean CV scores from an RFECV selector, handling API differences.

        Tries cv_results_['mean_test_score'] (sklearn >=1.0) then falls back to the legacy grid_scores_ attribute. Returns None if neither attribute is present.
        """
        scores = None
        if hasattr(selector, "cv_results_"):
            vals = selector.cv_results_.get("mean_test_score")
            if vals is not None:
                scores = [float(val) for val in vals]
        elif hasattr(selector, "grid_scores_"):
            vals = getattr(selector, "grid_scores_", None)
            if vals is not None:
                scores = [float(val) for val in np.atleast_1d(vals)]
        return scores

    aggregate_summary = {
        "subset_summary": summary,
        "subset_output_dir": summary.get("output_dir"),
        "rfecv_step": step_value,
        "rfecv_min_features_requested": args.rfecv_min_features,
        "rfecv_min_features_used": min_features,
        "rfecv_scoring": args.rfecv_scoring,
        "binary": {
            "selected_feature_count": len(binary_selected_columns),
            "selected_features": binary_selected_columns,
            "support_mask": binary_rfecv_selector.support_.tolist(),
            "ranking": binary_rfecv_selector.ranking_.tolist(),
            "mean_test_scores": _extract_scores(binary_rfecv_selector),
        },
    }
    (output_folder / "aggregate_summary.json").write_text(json.dumps(aggregate_summary, indent=2))
    print("Completed binary training with RFECV-selected features.")
    print(f"Results saved in {output_folder}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for the training script.

    Defines all command-line flags consumed by run_training, including:
    - --database-file / -db: path to the merged input CSV.
    - --output-dir / -o: root output directory (a timestamp suffix is appended at runtime).
    - --binary-target / --multiclass-target: target column names.
    - --binary-model / --multiclass-model: estimator family ('xgb', 'lgbm', or 'rf' where applicable).
    - --weight-column: column holding per-sample cohort weights.
    - --negative-label / -n: string label for the negative class.
    - --na-perc-limit / -na: maximum allowed missing-value fraction per feature column before it is dropped.
    - --no-impute: flag to disable KNN / mode imputation.
    - --test-size / -tsize: hold-out fraction for final evaluation.
    - --binary-trials / -btrials: Optuna trial budget for the gate.
    - --multiclass-trials / -mtrials: Optuna trial budget for the head.
    - --cv-splits: number of stratified folds.
    - --random-state: global random seed.
    - --rfecv-step: feature-elimination step size for RFECV.
    - --rfecv-min-features: minimum features to retain after RFECV.
    - --rfecv-scoring: scoring metric for RFECV cross-validation.

    Returns:
        Configured ArgumentParser instance ready for parse_args().
    """
    home = Path.cwd()
    default_db = os.path.join(home, "mepram_data", "df_merged_full.csv")
    default_out = os.path.join(home, "mepram_data", "outputs", "binary_optuna")

    parser = argparse.ArgumentParser(description="Run Optuna + RFECV search for a single binary model.")
    parser.add_argument(
        "--database-file",
        "-db",
        type=Path,
        default=default_db,
        help=f"Path to the merged dataframe (default: {default_db})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=default_out,
        help=f"Directory to store outputs (default: {default_out})",
    )
    parser.add_argument(
        "--binary-target",
        type=str,
        default="bmr_etiologia",
        help="Column used as the binary target (default: bmr_etiologia).",
    )
    parser.add_argument(
        "--binary-model",
        type=str,
        choices=["xgb", "lgbm", "rf", "catb"],
        default="xgb",
        help="Binary estimator to optimise.",
    )
    parser.add_argument(
        "--weight-column",
        type=str,
        default="sample_weight",
        help="Column containing per-sample weights (default: sample_weight).",
    )
    parser.add_argument(
        "--na-perc-limit",
        "-na",
        type=float,
        default=0.052,
        help="Minimum NA percentage for columns in data.",
    )
    parser.add_argument(
        "--no-impute",
        dest="impute_missing",
        action="store_false",
        help="Disable missing-value imputation (default: imputation enabled).",
    )
    parser.add_argument(
        "--test-size",
        "-tsize",
        type=float,
        default=0.35,
        help="Hold-out fraction used for final evaluation.",
    )
    parser.add_argument(
        "--binary-trials",
        "-btrials",
        type=int,
        default=500,
        help="Optuna trials for the binary model.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="StratifiedKFold splits for the study.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=99,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--rfecv-step",
        type=float,
        default=None,
        help="RFECV step – if <1 treated as fraction, otherwise rounded to an integer count (default: 10% of features).",
    )
    parser.add_argument(
        "--rfecv-min-features",
        type=int,
        default=5,
        help="Minimum number of features to retain during RFECV.",
    )
    parser.add_argument(
        "--rfecv-scoring",
        type=str,
        default="f1_macro",
        help="Scoring metric used by RFECV (default: f1_macro).",
    )
    parser.add_argument(
        "--use-stacking",
        action="store_true",
        help="Wether to use stacking or not.",
    )
    parser.set_defaults(impute_missing=True)
    return parser


def main() -> None:
    """Entry point: parse CLI arguments, run training, and report elapsed time.

    Measures wall-clock time from start to finish and prints the total elapsed minutes to stdout for SLURM job logs.
    """
    start = time.time()
    print("Checking parsed args...")
    parser = build_arg_parser()
    args = parser.parse_args()
    print("Parsed args: ", args)
    print("Starting script...")
    try:
        run_training(args)
    except Exception as e:
        output_folder = Path(str(args.output_dir) + "_" + str(TODAY))
        if output_folder.exists():
            shutil.rmtree(output_folder)
        raise
    end = time.time()
    print("ELAPSED TIME (minutes): ", (end - start) / 60)


if __name__ == "__main__":
    main()
