#!/usr/bin/env python3
"""
Multi-label Optuna training script for fenotipo_resistencia (no binary gate) using CatBoost.

Key differences vs. the LGBM/XGB variant:
  - Keeps categorical columns native for CatBoost; no one-hot encoding.
  - Uses CatBoostClassifier in One-vs-Rest with per-label thresholds tuned by Optuna.
  - Optional RFECV for feature selection remains, but uses a RandomForest baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import ast
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_safe(obj):
    """Convert numpy/Path/sets to JSON-serialisable structures."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

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

DELETE_COLUMNS = {
    "qsofa",
    "vasopresores",
    "hipotension",
    "freq_bacteria",
    "freq_bac_foco",
    "Unnamed: 0",
    "person_id",
    "fecha_ingreso_urgencias",
    "fecha_ingreso_urgencias_x",
    "ultima_fecha",
    "shock_septico",
    "sintoma_nan",
    "fecha_nacimiento",
    "codigo_postal",
    "center",
    "dag",
    "mujer_gestante",
}

TARGET_REMOVE = ["sepsis", "resultado_hemo", "infected_yes_no", "bmr_etiologia", "fenotipo_resistencia", "resistente_cefalosporina"]

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
# Helpers
# ---------------------------------------------------------------------------


def parse_multilabel_cell(value, *, delimiter: str) -> list[str]:
    """Parse a cell that may contain a tuple/list literal, delimiter-separated string, or already-parsed iterable."""
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        if pd.isna(value):
            return []
        text = str(value).strip()
        if text in {"", "()", "[]", "{}"}:
            return []
        if text[0] in "([{":
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    raw_items = list(parsed)
                else:
                    raw_items = [parsed]
            except Exception:
                raw_items = text.split(delimiter)
        else:
            raw_items = text.split(delimiter)

    labels = []
    for item in raw_items:
        if pd.isna(item):
            continue
        txt = str(item).strip()
        if not txt or txt in {"()", "[]", "{}"}:
            continue
        labels.append(txt)
    return labels


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


def load_processed_dataframe(csv_path: Path, cols_to_delete: list, args) -> Tuple[pd.DataFrame, set]:
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
    df[args.multiclass_target] = df[args.multiclass_target].apply(
        lambda v: parse_multilabel_cell(v, delimiter=args.multilabel_delimiter)
    )
    if "foco" in df.columns:
        df = df.copy()
        df["foco"] = df["foco"].map(FOCUS_MAP).fillna(df["foco"])

    all_labels = df[args.multiclass_target].explode()
    counts = all_labels.value_counts()
    min_count = 5
    keep_labels = set(counts[counts >= min_count].index)
    dropped_labels = set(counts[counts < min_count].index)

    df = df.copy()
    df[args.multiclass_target] = df[args.multiclass_target].apply(lambda lbls: [l for l in lbls if l in keep_labels])
    df = safe_drop_columns(df=df, columns=cols_to_delete)
    return df, dropped_labels


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
            f"min_features_to_select must be between 1 and {available_features}; "
            f"received {min_features_to_select}."
        )

    if isinstance(step, int):
        if step <= 0:
            raise ValueError("RFECV step must be a positive integer.")
    else:
        if not (0 < step <= 1):
            raise ValueError("RFECV fractional step must be in the (0, 1] range.")

    estimator = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=skf,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    selector.fit(X, y)
    selected_columns = X.columns[selector.support_].tolist()
    return selected_columns, selector


def resample_multilabel_combinations(
    X: pd.DataFrame,
    labels_list: pd.Series,
    *,
    sample_weight: pd.Series | None,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Balance multilabel samples by oversampling combinations using ROS/SMOTE."""
    combos = pd.Series(
        ["|".join(sorted(lbls)) if len(lbls) else "__NONE__" for lbls in labels_list],
        index=X.index,
        name="combo",
    )
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(combos), index=X.index, name="combo_enc")
    class_counts = y_enc.value_counts()
    min_class = class_counts.min()
    if min_class <= 1:
        sampler = RandomOverSampler(random_state=random_state)
    else:
        k = max(1, min(5, min_class - 1))
        sampler = SMOTE(random_state=random_state, k_neighbors=k)

    X_res, y_res = sampler.fit_resample(X, y_enc)
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name="combo_enc")

    weight_res: pd.Series | None = None
    if sample_weight is not None:
        sample_weight = sample_weight.reindex(X.index)
        if hasattr(sampler, "sample_indices_"):
            idx = sampler.sample_indices_
            weight_res = sample_weight.iloc[idx].reset_index(drop=True)
        else:
            base = sample_weight.to_numpy()
            extra = len(X_res) - len(base)
            synthetic = np.full(extra, float(np.mean(base))) if extra > 0 else np.array([])
            weight_res = pd.Series(np.concatenate([base, synthetic]), name=sample_weight.name)

    combos_res = pd.Series(le.inverse_transform(y_res), name="combo")
    labels_res = combos_res.apply(lambda s: [] if s == "__NONE__" else s.split("|"))
    return X_res, labels_res, weight_res


# ---------------------------------------------------------------------------
# CatBoost helpers
# ---------------------------------------------------------------------------


def build_catboost_binary(params: dict, random_state: int, cat_features: List[int]):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
        cat_features=cat_features,
        **params,
    )


def optimise_multilabel_head_catboost(
    X: pd.DataFrame,
    Y: np.ndarray,
    *,
    n_splits: int,
    n_trials: int,
    random_state: int,
    cat_feature_indices: List[int],
    sample_weight: pd.Series | None,
) -> Tuple[Dict[str, object], optuna.study.Study]:
    """Tune CatBoost multilabel head using OVR + micro-F1 optimisation."""
    kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_labels = Y.shape[1]
    weight_series = sample_weight

    def objective(trial: optuna.Trial) -> float:
        thresholds = np.array([trial.suggest_float(f"thr_{j}", 0.01, 0.8) for j in range(n_labels)])
        params = {
            "auto_class_weights": "Balanced",
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 50.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "bootstrap_type": "Bayesian",
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0)
        }

        scores = []
        for train_idx, valid_idx in kf.split(X, Y):
            X_tr = X.iloc[train_idx]
            X_va = X.iloc[valid_idx]
            Y_tr = Y[train_idx]
            Y_va = Y[valid_idx]
            w_va = weight_series.iloc[valid_idx] if weight_series is not None else None

            label_models = []
            for j in range(n_labels):
                y_j = Y_tr[:, j]
                model = build_catboost_binary(params, random_state, cat_feature_indices)
                model.fit(X_tr, y_j, sample_weight=weight_series.iloc[train_idx] if weight_series is not None else None)
                label_models.append(model)

            y_proba = np.column_stack([m.predict_proba(X_va)[:, 1] for m in label_models])
            y_pred = (y_proba >= thresholds[np.newaxis, :]).astype(int)
            score = f1_score(
                Y_va,
                y_pred,
                average="macro",
                sample_weight=w_va,
                zero_division=0,
            )
            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return study.best_trial.params.copy(), study


# ---------------------------------------------------------------------------
# Training with one feature subset
# ---------------------------------------------------------------------------


def train_with_feature_subset(
    *,
    subset_name: str,
    selected_columns: List[str],
    categorical_cols: List[str],
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_train_multilabel: pd.Series,
    y_test_multilabel: pd.Series,
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

    (subset_output_dir / "selected_features.json").write_text(
        json.dumps(
            {
                "subset": subset_name,
                "feature_count": len(selected_columns),
                "features": selected_columns.to_list(),
            },
            indent=2,
        )
    )

    X_train = X_train_full[selected_columns].copy()
    X_test = X_test_full[selected_columns].copy()

    mlb = MultiLabelBinarizer()
    Y_train_bin = mlb.fit_transform(y_train_multilabel)
    Y_test_bin = mlb.transform(y_test_multilabel)

    train_weights = None
    if sample_weight_train is not None:
        train_weights = pd.Series(sample_weight_train, name="sample_weight").loc[X_train.index]
    test_weights = None
    if sample_weight_test is not None:
        test_weights = pd.Series(sample_weight_test, name="sample_weight").loc[X_test.index]

    # Balance multilabel combinations via ROS/SMOTE on categorical + numeric together
    X_train_bal, labels_bal, train_weights_bal = resample_multilabel_combinations(
        X_train,
        y_train_multilabel,
        sample_weight=train_weights,
        random_state=args.random_state,
    )
    Y_train_bin_bal = mlb.transform(labels_bal)
    if train_weights_bal is None:
        train_weights_bal = train_weights.loc[X_train_bal.index] if train_weights is not None else None

    cat_indices = [X_train.columns.get_loc(c) for c in categorical_cols if c in X_train.columns]

    # Optimisation + training
    best_params, study = optimise_multilabel_head_catboost(
        X_train_bal,
        Y_train_bin_bal,
        n_splits=args.cv_splits,
        n_trials=args.multiclass_trials,
        random_state=args.random_state,
        cat_feature_indices=cat_indices,
        sample_weight=train_weights_bal,
    )

    thresholds_dict = {}
    n_labels = len(mlb.classes_)
    for i, cls in enumerate(list(mlb.classes_)):
        key = f"thr_{i}"
        thr_i = float(best_params.pop(key, args.multilabel_threshold))
        thresholds_dict[cls] = thr_i
    thresholds_array = np.array([thresholds_dict[cls] for cls in mlb.classes_], dtype=float)

    # Fit final OVR CatBoost models
    label_models = []
    for j in range(n_labels):
        y_j = Y_train_bin_bal[:, j]
        model = build_catboost_binary(best_params, args.random_state, cat_indices)
        model.fit(X_train_bal, y_j, sample_weight=train_weights_bal)
        label_models.append(model)

    # Evaluation
    proba_test = np.column_stack([m.predict_proba(X_test)[:, 1] for m in label_models])
    y_pred_bin = (proba_test >= thresholds_array[np.newaxis, :]).astype(int)

    micro_f1 = f1_score(
        Y_test_bin,
        y_pred_bin,
        average="micro",
        sample_weight=test_weights,
        zero_division=0,
    )
    macro_f1 = f1_score(
        Y_test_bin,
        y_pred_bin,
        average="macro",
        sample_weight=test_weights,
        zero_division=0,
    )

    def _weighted_mean(arr):
        if test_weights is None:
            return float(np.mean(arr))
        return float(np.average(arr, weights=test_weights))

    try:
        micro_auc = roc_auc_score(
            Y_test_bin,
            proba_test,
            average="micro",
            sample_weight=test_weights,
        )
        micro_auc = float(micro_auc)
    except ValueError:
        micro_auc = None

    exact_match = np.all(Y_test_bin == y_pred_bin, axis=1)
    coverage_mask = y_pred_bin.sum(axis=1) > 0
    inclusion_mask = (y_pred_bin & Y_test_bin).sum(axis=1) > 0

    multilabel_metrics = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_roc_auc": micro_auc,
        "exact_match_ratio": _weighted_mean(exact_match),
        "coverage": _weighted_mean(coverage_mask),
        "inclusion_any_true": _weighted_mean(inclusion_mask),
        "avg_labels_predicted": float(
            np.average(
                y_pred_bin.sum(axis=1),
                weights=test_weights if test_weights is not None else None,
            )
        ),
        "per_label_thresholds": {cls: float(thr) for cls, thr in thresholds_dict.items()},
    }

    # Build prediction payload
    inv_true = mlb.inverse_transform(Y_test_bin)
    inv_pred = mlb.inverse_transform(y_pred_bin)
    probability_payload = []
    for idx in range(len(X_test)):
        proba_map = {cls: float(prob) for cls, prob in zip(mlb.classes_, proba_test[idx])}
        probability_payload.append(proba_map)

    results_df = pd.DataFrame(
        {
            "true_labels": [", ".join(labels) if labels else "" for labels in inv_true],
            "pred_labels": [", ".join(labels) if labels else "" for labels in inv_pred],
        }
    )
    results_df["proba"] = probability_payload

    # Persist artefacts
    results_df.to_csv(subset_output_dir / "multilabel_predictions.csv", index=False)
    (subset_output_dir / "multilabel_metrics.json").write_text(
        json.dumps(multilabel_metrics, indent=2, default=_json_safe)
    )

    summary = {
        "subset": subset_name,
        "feature_count": len(selected_columns),
        "selected_features": selected_columns,
        "multilabel_params": best_params,
        "multilabel_thresholds": thresholds_dict,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_roc_auc": micro_auc,
        "multilabel_metrics": multilabel_metrics,
        "class_names": mlb.classes_.tolist(),
        "multiclass_trials": args.multiclass_trials,
        "output_dir": str(subset_output_dir),
    }
    (subset_output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_safe))

    trials_dir = subset_output_dir / "optuna_trials"
    trials_dir.mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(trials_dir / "multilabel_trials.csv", index=False)

    print(f"[{subset_name}] Multilabel best params:", best_params)
    print(f"[{subset_name}] Micro F1:", f"{micro_f1:.3f}")
    if micro_auc is not None:
        print(f"[{subset_name}] Micro ROC-AUC:", f"{micro_auc:.3f}")
    else:
        print(f"[{subset_name}] Micro ROC-AUC: N/A")

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
    keep_targets = {args.multiclass_target, args.weight_column}
    cols_to_delete.extend([x for x in TARGET_REMOVE if x not in keep_targets])
    df, dropped_labels = load_processed_dataframe(args.database_file, cols_to_delete=cols_to_delete, args=args)
    if dropped_labels:
        print(f"Dropped {len(dropped_labels)} low freq labels: {dropped_labels}")

    missing_cols = [col for col in [args.multiclass_target, args.weight_column] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from dataframe: {missing_cols}")

    working_df = df.copy()
    working_df[args.weight_column] = pd.to_numeric(working_df[args.weight_column], errors="coerce")
    working_df = working_df.dropna(subset=[args.multiclass_target, args.weight_column])
    working_df = working_df[working_df[args.weight_column] > 0]

    # remove rows without labels
    working_df = working_df[working_df[args.multiclass_target].apply(lambda x: len(x) > 0)]

    exclude_cols = {args.multiclass_target, args.weight_column}
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
    # Keep categoricals for CatBoost; ensure string dtype
    if categorical_cols:
        feature_df[categorical_cols] = feature_df[categorical_cols].apply(lambda s: s.astype(str))
    print(feature_df.columns)

    target_series = working_df.loc[feature_df.index, args.multiclass_target]
    weight_series = working_df.loc[feature_df.index, args.weight_column]

    (
        X_train_full,
        X_test_full,
        y_train_multilabel,
        y_test_multilabel,
        sample_weight_train,
        sample_weight_test,
    ) = train_test_split(
        feature_df,
        target_series,
        weight_series,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=None,
    )
    print("Finished traintest split")
    available_features = X_train_full.shape[1]
    if available_features == 0:
        raise ValueError("No features available after preprocessing; aborting.")

    combo_labels = pd.Series(["|".join(sorted(lbls)) for lbls in y_train_multilabel], index=y_train_multilabel.index)
    min_features = min(args.rfecv_min_features, available_features)
    if args.rfecv_step is None:
        step_value: int | float = max(1, int(0.1 * available_features))
    else:
        step_value = args.rfecv_step
        if step_value < 1:
            step_value = float(step_value)
        else:
            step_value = int(step_value)

    print("NO RFECV PERFORMED HERE")
    """selected_columns, rfecv_selector = perform_rfecv_feature_selection(
        X_train_full,
        combo_labels,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
        step=step_value,
        min_features_to_select=min_features,
        scoring=args.rfecv_scoring,
    )
    print(f"Finished RFECV. Selected {len(selected_columns)} features.")
    """
    selected_columns = X_train_full.columns
    output_folder = Path(str(args.output_dir) + "_" + str(TODAY))
    output_folder.mkdir(parents=True, exist_ok=True)
    subset_name = f"rfecv_{len(selected_columns)}"
    summary = train_with_feature_subset(
        subset_name=subset_name,
        selected_columns=selected_columns,
        categorical_cols=[c for c in categorical_cols if c in selected_columns],
        X_train_full=X_train_full,
        X_test_full=X_test_full,
        y_train_multilabel=y_train_multilabel,
        y_test_multilabel=y_test_multilabel,
        sample_weight_train=sample_weight_train,
        sample_weight_test=sample_weight_test,
        args=args,
        output_dir=output_folder,
    )

    selection_metadata = {
        "selected_feature_count": len(selected_columns),
        "selected_features": selected_columns,
        #"rfecv_support_mask": rfecv_selector.support_.tolist(),
        #"rfecv_ranking": rfecv_selector.ranking_.tolist(),
        "rfecv_step": step_value,
        "rfecv_min_features_requested": args.rfecv_min_features,
        "rfecv_min_features_used": min_features,
        "rfecv_scoring": args.rfecv_scoring,
    }
    (output_folder / "rfecv_selected_features.json").write_text(
        json.dumps(selection_metadata, indent=2, default=_json_safe)
    )

    micro_auc_str = f"{summary['micro_roc_auc']:.3f}" if summary.get("micro_roc_auc") is not None else "N/A"
    print(
        f"[{subset_name}] Micro F1: {summary['micro_f1']:.3f} | "
        f"Macro F1: {summary['macro_f1']:.3f} | "
        f"Micro ROC-AUC: {micro_auc_str}"
    )
    print("Completed multilabel CatBoost training.")
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
    default_out = os.path.join(home, "mepram_data", "outputs", "fenotipo_multilabel_catboost")

    parser = argparse.ArgumentParser(
        description="Run Optuna search for fenotipo_resistencia multilabel with CatBoost (no binary gate)."
    )
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
        "--multiclass-target",
        type=str,
        default="fenotipo_resistencia",
        help="Column used as the multilabel target (default: fenotipo_resistencia).",
    )
    parser.add_argument(
        "--weight-column",
        type=str,
        default="sample_weight",
        help="Column containing per-sample weights (default: sample_weight).",
    )
    parser.add_argument(
        "--multiclass-trials",
        "-mtrials",
        type=int,
        default=200,
        help="Optuna trials for the multilabel head.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="KFold splits for the study.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=99,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--test-size",
        "-tsize",
        type=float,
        default=0.35,
        help="Hold-out fraction used for final evaluation.",
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
        "--multilabel-threshold",
        type=float,
        default=0.5,
        help="Default per-label threshold if Optuna does not override (default: 0.5).",
    )
    parser.add_argument(
        "--multilabel-delimiter",
        type=str,
        default=",",
        help="Delimiter to split multilabel string cells.",
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
    run_training(args)
    end = time.time()
    print("ELAPSED TIME (minutes): ", (end - start) / 60)


if __name__ == "__main__":
    main()
