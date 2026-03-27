#!/usr/bin/env python3
"""
Hierarchical Optuna training script for resultado_hemo.

This script reproduces the notebook workflow:
  1. Load and preprocess the merged dataset.
  2. Use RFECV to select an optimal feature subset automatically.
  3. Optimise a binary NEGATIVE vs POSITIVE gate with Optuna.
  4. Optimise a multiclass head (only positive species) with Optuna + SMOTE.
  5. Train the final hierarchy, evaluate on a hold-out set, and persist artefacts.

It is intended to be launched from SLURM or any non-interactive environment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

TARGET_REMOVE = ["resultado_hemo", "all_cult_org", "infected_yes_no", "bmr_etiologia", "fenotipo_resistencia"]

DELETE_COLUMNS = ["qsofa", "vasopresores", "hipotension", "freq_bacteria", "freq_bac_foco", "Unnamed: 0", "person_id", "fecha_ingreso_urgencias", "fecha_ingreso_urgencias_x", 'ultima_fecha', "shock_septico", "sintoma_nan", "fecha_nacimiento", "codigo_postal", "center", "dag", 'mujer_gestante']

MINOR_CLASSES_TO_DROP = {
    "Enterococcus",
    "_Fungi",
    "_Other bacteria",
}

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

def load_processed_dataframe(csv_path: Path, cols_to_delete: list) -> pd.DataFrame:
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
    df = safe_drop_columns(df=df, columns=cols_to_delete)
    return df


def resample_positive_classes(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance class distribution with SMOTE or random over-sampling.

    Selects the resampling strategy based on the smallest class size:
    - Fewer than 6 samples: RandomOverSampler (SMOTE requires at least k+1 neighbours; synthetic generation is unreliable at this scale).
    - 6-50 samples: SMOTE with k_neighbors capped at min_class - 1.
    - More than 50 samples: SMOTE with k_neighbors=5 (standard).

    When sample_weight is provided it is propagated to the resampled dataset: original weights are kept for real samples; synthetic samples receive the mean weight of the original set.

    Args:
        X: Feature matrix for positive-class samples.
        y: Label series aligned with X.
        sample_weight: Optional per-sample weights aligned with X. Pass None to skip weight propagation.
        random_state: Seed for the resampler's RNG.

    Returns:
        Tuple of (X_resampled, y_resampled, weight_resampled). The weight element is None when sample_weight was not supplied.
    """
    class_counts = y.value_counts()
    min_class = class_counts.min()
    if min_class <= 1:
        sampler = RandomOverSampler(random_state=random_state)
    else:
        k = max(1, min(5, min_class - 1))
        sampler = SMOTE(random_state=random_state, k_neighbors=k)

    X_res, y_res = sampler.fit_resample(X, y)
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=y.name)
    return X_res, y_res


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
        n_estimators=400,
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

# ---------------------------------------------------------------------------

# Optuna optimisation

def optimise_binary_gate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    n_trials: int,
    random_state: int,
) -> Tuple[Dict[str, object], float, optuna.study.Study]:
    """Jointly optimise hyperparameters and classification threshold for the binary gate.

    Runs an Optuna study that maximises the mean cross-validated Fbeta score (beta=2, favouring recall) over n_trials trials. Each trial samples both a probability threshold (0.3-0.8) and model hyperparameters for the chosen model_type ('lgbm', 'rf', or 'xgb').

    Class imbalance is handled via scale_pos_weight (XGBoost / LightGBM) or class_weight='balanced_subsample' (RandomForest), derived from the weighted positive/negative ratio when sample_weight is provided.

    Args:
        X: Scaled feature matrix for the training split.
        y: Binary label series (0 = negative, 1 = positive) aligned with X.
        n_splits: Stratified K-Fold splits used inside each trial.
        n_trials: Number of Optuna trials to run.
        random_state: Seed for model and CV splitter RNGs.
        sample_weight: Optional per-sample weights used for both training and Fbeta score computation.
        model_type: Estimator family to tune - one of 'lgbm', 'rf', or 'xgb'.

    Returns:
        Tuple of (best_params, best_threshold, study) where best_params is the dict of hyperparameters (threshold excluded), best_threshold is the optimal decision cutoff, and study is the completed Optuna study object.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 6, 8, 10, 12, 16, 20]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.6, 0.8]
            ),
        }
        threshold = trial.suggest_float("threshold", 0.35, 0.7)

        scores = []
        for train_idx, valid_idx in skf.split(X, y):
            X_tr = X.iloc[train_idx]
            X_va = X.iloc[valid_idx]
            y_tr = y.iloc[train_idx]
            y_va = y.iloc[valid_idx]

            model = RandomForestClassifier(
                **params,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            probas = model.predict_proba(X_va)[:, 1]
            preds = (probas >= threshold).astype(int)
            scores.append(fbeta_score(y_va, preds, beta=2))

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    best_params = study.best_trial.params.copy()
    best_threshold = best_params.pop("threshold")
    return best_params, best_threshold, study


def optimise_multiclass_head(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    n_trials: int,
    random_state: int,
) -> Tuple[Dict[str, object], optuna.study.Study]:
    """Optimise the positive-class (phenotype) head using stratified CV and SMOTE.

    Runs an Optuna study on the subset of samples that passed the binary gate (i.e. true positives). Class-balanced weights are computed once and fused with any supplied sample_weight. Inside each cross-validation fold the training split is over-sampled with resample_positive_classes before fitting.

    Supported estimators (model_type):
    - 'xgb': XGBoost with binary:logistic objective and a multiclass class_weight map.
    - 'lgbm': LightGBM with binary objective and the same class map.

    The objective function maximises mean binary F1 on the held-out fold.

    Args:
        X: Feature matrix for the positive training subset.
        y: Multiclass label series aligned with X.
        n_splits: Stratified K-Fold splits used inside each trial.
        n_trials: Number of Optuna trials to run.
        random_state: Seed for model, CV splitter, and SMOTE RNGs.
        sample_weight: Optional per-sample weights; if None, uniform weights of 1.0 are used.
        model_type: Estimator family to tune - one of 'xgb' or 'lgbm'.

    Returns:
        Tuple of (best_params, study) where best_params is the dict of optimised hyperparameters and study is the completed Optuna study.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    num_classes = len(np.unique(y))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64, step=4),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60, step=5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        scores = []
        for train_idx, valid_idx in skf.split(X, y):
            X_tr = X.iloc[train_idx]
            X_va = X.iloc[valid_idx]
            y_tr = y.iloc[train_idx]
            y_va = y.iloc[valid_idx]

            X_tr_bal, y_tr_bal = resample_positive_classes(
                X_tr, y_tr, random_state=random_state
            )

            model = LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
                **params,
            )
            model.fit(X_tr_bal, y_tr_bal)

            y_pred = model.predict(X_va)
            scores.append(f1_score(y_va, y_pred, average="macro"))

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    best_params = study.best_trial.params.copy()
    return best_params, study


def train_with_feature_subset(
    *,
    subset_name: str,
    selected_columns: List[str],
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_train_full: pd.Series,
    y_test_full: pd.Series,
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
                "features": selected_columns,
            },
            indent=2,
        )
    )

    X_train = X_train_full[selected_columns].copy()
    X_test = X_test_full[selected_columns].copy()

    y_train_binary = (y_train_full != args.negative_label).astype(int)
    y_test_binary = (y_test_full != args.negative_label).astype(int)

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=selected_columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=selected_columns,
        index=X_test.index,
    )

    # ------------------------------------------------------------------
    # Binary gate optimisation + training
    # ------------------------------------------------------------------
    binary_params, best_threshold, binary_study = optimise_binary_gate(
        X_train_scaled,
        y_train_binary,
        n_splits=args.cv_splits,
        n_trials=args.binary_trials,
        random_state=args.random_state,
    )

    binary_model = RandomForestClassifier(
        **binary_params,
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=-1,
    )
    binary_model.fit(X_train_scaled, y_train_binary)

    # ------------------------------------------------------------------
    # Multiclass head optimisation + training
    # ------------------------------------------------------------------
    pos_mask = y_train_full != args.negative_label
    if not pos_mask.any():
        raise ValueError("No positive samples available to train the multiclass head.")
    X_train_pos = X_train_scaled.loc[pos_mask]
    y_train_pos = y_train_full.loc[pos_mask]

    label_encoder = LabelEncoder()
    y_train_pos_enc = pd.Series(
        label_encoder.fit_transform(y_train_pos),
        index=y_train_pos.index,
        name="encoded_target",
    )

    multiclass_params, multiclass_study = optimise_multiclass_head(
        X_train_pos,
        y_train_pos_enc,
        n_splits=args.cv_splits,
        n_trials=args.multiclass_trials,
        random_state=args.random_state,
    )

    X_train_pos_bal, y_train_pos_bal = resample_positive_classes(
        X_train_pos, y_train_pos_enc, random_state=args.random_state
    )
    multiclass_model = LGBMClassifier(
        objective="multiclass",
        num_class=len(label_encoder.classes_),
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
        verbose=-1,
        **multiclass_params,
    )
    multiclass_model.fit(X_train_pos_bal, y_train_pos_bal)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    binary_proba_test = binary_model.predict_proba(X_test_scaled)[:, 1]
    binary_pred_test = (binary_proba_test >= best_threshold).astype(int)

    hierarchical_pred = np.full(len(X_test_scaled), fill_value=args.negative_label, dtype=object)
    hierarchical_conf = np.zeros(len(X_test_scaled))
    multiclass_proba_per_sample = [None] * len(X_test_scaled)

    pos_candidates = np.where(binary_pred_test == 1)[0]
    if len(pos_candidates):
        X_test_pos = X_test_scaled.iloc[pos_candidates]
        pos_proba = multiclass_model.predict_proba(X_test_pos)
        pos_pred_idx = np.argmax(pos_proba, axis=1)
        pos_pred_labels = label_encoder.inverse_transform(pos_pred_idx)
        hierarchical_pred[pos_candidates] = pos_pred_labels
        hierarchical_conf[pos_candidates] = pos_proba.max(axis=1)
        for local_idx, sample_idx in enumerate(pos_candidates):
            multiclass_proba_per_sample[sample_idx] = {
                cls: float(prob) for cls, prob in zip(label_encoder.classes_, pos_proba[local_idx])
            }

    all_labels = [args.negative_label] + label_encoder.classes_.tolist()

    binary_conf_matrix = confusion_matrix(
        y_test_binary,
        binary_pred_test,
        labels=[0, 1],
    )
    binary_conf_df = pd.DataFrame(
        binary_conf_matrix,
        index=[f"true_{args.negative_label}", "true_POSITIVE"],
        columns=[f"pred_{args.negative_label}", "pred_POSITIVE"],
    )

    binary_report = classification_report(
        y_test_binary,
        binary_pred_test,
        target_names=[args.negative_label, "POSITIVE"],
        zero_division=0,
    )
    hierarchical_report = classification_report(
        y_test_full,
        hierarchical_pred,
        labels=all_labels,
        target_names=all_labels,
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_test_full,
        hierarchical_pred,
        labels=all_labels,
        average="macro",
    )
    macro_auc = roc_auc_score(
        (y_test_full != args.negative_label).astype(int),
        binary_proba_test,
    )

    pos_mask_test = y_test_full != args.negative_label
    positive_report = None
    multiclass_conf_df = None
    multiclass_macro_auc = None
    if pos_mask_test.any():
        X_test_pos_true = X_test_scaled.loc[pos_mask_test]
        y_test_pos_true = y_test_full.loc[pos_mask_test]
        y_test_pos_true_enc = label_encoder.transform(y_test_pos_true)
        pos_true_proba = multiclass_model.predict_proba(X_test_pos_true)
        pos_true_pred_idx = np.argmax(pos_true_proba, axis=1)
        pos_true_pred_labels = label_encoder.inverse_transform(pos_true_pred_idx)

        positive_report = classification_report(
            y_test_pos_true,
            pos_true_pred_labels,
            labels=label_encoder.classes_,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
        multiclass_conf_matrix = confusion_matrix(
            y_test_pos_true,
            pos_true_pred_labels,
            labels=label_encoder.classes_,
        )
        multiclass_conf_df = pd.DataFrame(
            multiclass_conf_matrix,
            index=[f"true_{lbl}" for lbl in label_encoder.classes_],
            columns=[f"pred_{lbl}" for lbl in label_encoder.classes_],
        )
        try:
            multiclass_macro_auc = roc_auc_score(
                y_test_pos_true_enc,
                pos_true_proba,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            multiclass_macro_auc = None

    probability_payload = []
    for idx in range(len(X_test_scaled)):
        prob_entry = {
            "binary": {
                args.negative_label: float(1 - binary_proba_test[idx]),
                "POSITIVE": float(binary_proba_test[idx]),
            },
            "multiclass": multiclass_proba_per_sample[idx],
        }
        probability_payload.append(prob_entry)

    hierarchical_results = pd.DataFrame(
        {
            "true_label": y_test_full.reset_index(drop=True),
            "binary_positive_prob": binary_proba_test,
            "binary_pred": np.where(binary_pred_test == 1, "POSITIVE", args.negative_label),
            "hierarchical_pred": hierarchical_pred,
            "hierarchical_confidence": hierarchical_conf,
        }
    )
    hierarchical_results["is_positive_true"] = hierarchical_results["true_label"] != args.negative_label
    hierarchical_results["hierarchical_proba"] = probability_payload

    macro_f1 = float(macro_f1)
    macro_auc = float(macro_auc)
    if multiclass_macro_auc is not None:
        multiclass_macro_auc = float(multiclass_macro_auc)

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    hierarchical_results.to_csv(
        subset_output_dir / "hierarchical_predictions.csv", index=False
    )
    binary_conf_df.to_csv(subset_output_dir / "binary_confusion_matrix.csv")
    if multiclass_conf_df is not None:
        multiclass_conf_df.to_csv(subset_output_dir / "multiclass_confusion_matrix.csv")

    reports = {
        "binary_report.txt": binary_report,
        "hierarchical_report.txt": hierarchical_report,
    }
    if positive_report is not None:
        reports["positive_only_report.txt"] = positive_report
    for name, content in reports.items():
        (subset_output_dir / name).write_text(content)

    summary = {
        "subset_name": subset_name,
        "feature_count": len(selected_columns),
        "selected_features": selected_columns,
        "binary_best_params": binary_params,
        "binary_best_threshold": best_threshold,
        "multiclass_best_params": multiclass_params,
        "macro_f1": macro_f1,
        "binary_roc_auc": macro_auc,
        "multiclass_macro_auc": multiclass_macro_auc,
        "label_encoder_classes": label_encoder.classes_.tolist(),
        "binary_trials": args.binary_trials,
        "multiclass_trials": args.multiclass_trials,
        "output_dir": str(subset_output_dir),
    }
    (subset_output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    trials_dir = subset_output_dir / "optuna_trials"
    trials_dir.mkdir(exist_ok=True)
    binary_study.trials_dataframe().to_csv(trials_dir / "binary_trials.csv", index=False)
    multiclass_study.trials_dataframe().to_csv(
        trials_dir / "multiclass_trials.csv",
        index=False,
    )

    # Final console summary for SLURM logs
    print(f"[{subset_name}] Binary gate best params:", binary_params)
    print(f"[{subset_name}] Binary gate threshold:", best_threshold)
    print(f"[{subset_name}] Multiclass best params:", multiclass_params)
    print(f"[{subset_name}] Hold-out macro F1:", f"{macro_f1:.3f}")
    print(f"[{subset_name}] Hold-out binary ROC-AUC:", f"{macro_auc:.3f}")
    if multiclass_macro_auc is not None:
        print(f"[{subset_name}] Hold-out multiclass macro ROC-AUC:", f"{multiclass_macro_auc:.3f}")
    else:
        print(f"[{subset_name}] Hold-out multiclass macro ROC-AUC: N/A")

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
    cols_to_delete.extend([x for x in TARGET_REMOVE if x != args.target])
    df = load_processed_dataframe(args.database_file, cols_to_delete=cols_to_delete)

    feature_df = (
        df.dropna(subset=args.target)
        .copy()
        .loc[lambda d: ~d[args.target].isin(MINOR_CLASSES_TO_DROP)]
    )
    for col in feature_df.columns:
        na_per = 1 - len(feature_df[col].dropna()) / feature_df.shape[0]
        if na_per > args.na_perc_limit:
            print(f"Column {col} --> %NaN = {na_per}. deleted")
            feature_df = feature_df.drop(columns=col)
    feature_df = feature_df.dropna()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns
    categorical_cols = [x for x in categorical_cols if x != args.target]
    if categorical_cols:
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=False)

    target_series = feature_df[args.target]
 
    feature_df = feature_df.drop(columns=args.target)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        feature_df,
        target_series,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target_series,
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

    print("Starting RFECV...")
    selected_columns, rfecv_selector = perform_rfecv_feature_selection(
        X_train_full,
        y_train_full,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
        step=step_value,
        min_features_to_select=min_features,
        scoring=args.rfecv_scoring,
    )
    print(f"Finished RFECV. Selected {len(selected_columns)} features.")

    output_folder = Path(str(args.output_dir) + "_" + str(TODAY))
    output_folder.mkdir(parents=True, exist_ok=True)
    subset_name = f"rfecv_{len(selected_columns)}"
    summary = train_with_feature_subset(
        subset_name=subset_name,
        selected_columns=selected_columns,
        X_train_full=X_train_full,
        X_test_full=X_test_full,
        y_train_full=y_train_full,
        y_test_full=y_test_full,
        args=args,
        output_dir=output_folder,
    )

    selection_metadata = {
        "selected_feature_count": len(selected_columns),
        "selected_features": selected_columns,
        "rfecv_step": step_value,
        "rfecv_min_features_requested": args.rfecv_min_features,
        "rfecv_min_features_used": min_features,
        "rfecv_scoring": args.rfecv_scoring,
    }
    (output_folder / "rfecv_selected_features.json").write_text(
        json.dumps(selection_metadata, indent=2)
    )

    multiclass_auc_str = (
        f"{summary['multiclass_macro_auc']:.3f}"
        if summary.get("multiclass_macro_auc") is not None
        else "N/A"
    )
    print(
        f"[{subset_name}] Macro F1: {summary['macro_f1']:.3f} | "
        f"Binary ROC-AUC: {summary['binary_roc_auc']:.3f} | "
        f"Multiclass Macro ROC-AUC: {multiclass_auc_str}"
    )

    rfecv_scores = None
    if hasattr(rfecv_selector, "cv_results_"):
        scores = rfecv_selector.cv_results_.get("mean_test_score")
        if scores is not None:
            rfecv_scores = [float(val) for val in scores]
    elif hasattr(rfecv_selector, "grid_scores_"):
        scores = getattr(rfecv_selector, "grid_scores_", None)
        if scores is not None:
            rfecv_scores = [float(val) for val in np.atleast_1d(scores)]

    aggregate_summary = {
        "subset_summary": summary,
        "subset_output_dir": summary.get("output_dir"),
        "rfecv_selected_feature_count": len(selected_columns),
        "rfecv_selected_features": selected_columns,
        "rfecv_step": step_value,
        "rfecv_min_features_requested": args.rfecv_min_features,
        "rfecv_min_features_used": min_features,
        "rfecv_scoring": args.rfecv_scoring,
        "rfecv_support_mask": rfecv_selector.support_.tolist(),
        "rfecv_ranking": rfecv_selector.ranking_.tolist(),
        "rfecv_mean_test_scores": rfecv_scores,
    }
    (output_folder / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2)
    )
    print("Completed hierarchical training with RFECV-selected features.")
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
    default_out = os.path.join(home, "mepram_data", "outputs" , "hierarchical_optuna")

    parser = argparse.ArgumentParser(
        description="Run hierarchical Optuna search for resultado_hemo."
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
        "--target",
        "-t",
        type=str,
        default="resultado_hemo",
        help="Target column to model in the hierarchy.",
    )
    parser.add_argument(
        "--negative-label",
        "-n",
        type=str,
        default="NEGATIVE",
        help="Label considered negative in the binary gate.",
    )
    parser.add_argument(
        "--na-perc-limit",
        "-na",
        type=float,
        default=0.052,
        help="Minimum NA percentage for columns in data.",
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
        help="Optuna trials for the binary gate.",
    )
    parser.add_argument(
        "--multiclass-trials",
        "-mtrials",
        type=int,
        default=500,
        help="Optuna trials for the multiclass head.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="StratifiedKFold splits for both studies.",
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
        help="RFECV step – if <1 treated as fraction, otherwise rounded to an integer count (default: 10%% of features).",
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
    return parser


def main() -> None:
    """Entry point: parse CLI arguments, run training, and report elapsed time.

    Measures wall-clock time from start to finish and prints the total elapsed minutes to stdout for SLURM job logs.
    """
    start = time.start = time.time()
    print("Checking parsed args...")
    parser = build_arg_parser()
    args = parser.parse_args()
    print("Parsed args: ", args)
    print("Starting script...")
    run_training(args)
    print(" ")
    end = time.time()
    print("ELAPSED TIME (minutes): ", (end - start)/60)


if __name__ == "__main__":
    main()
