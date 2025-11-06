#!/usr/bin/env python3
"""
Hierarchical Optuna training script for resultado_hemo.

This script reproduces the notebook workflow:
  1. Load and preprocess the merged dataset.
  2. Run RFE to create predefined feature subsets (15, 20, 40 columns by default).
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
from typing import Dict, List, Sequence, Tuple
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
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

TARGET_REMOVE = ["sepsis", "resultado_hemo", "all_cult_org", "infected_yes_no"]

DELETE_COLUMNS = ["qsofa", "vasopresores", "hipotension", "freq_bacteria", "freq_bac_foco", "Unnamed: 0", "person_id", "fecha_ingreso_urgencias", "fecha_ingreso_urgencias_x", "shock_septico", "sintoma_nan", "fecha_nacimiento", "codigo_postal", "center", "dag"]

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

MINOR_CLASSES_TO_DROP = {
    "Enterococcus",
    "_Fungi",
    "_Other bacteria",
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def safe_drop_columns(df, columns):
    for col in columns:
        try:
            df = df.drop(columns=col)
        except KeyError:
            print(f"Warning: Column {col} not found in DataFrame. Skipping drop.")
    return df

def load_processed_dataframe(csv_path: Path, cols_to_delete: list) -> pd.DataFrame:
    """Load the merged dataframe and apply the focus mapping/filter."""
    df = pd.read_csv(csv_path)
    if "foco" in df.columns:
        df = df.copy()
        df["foco"] = df["foco"].map(FOCUS_MAP).fillna(df["foco"])
        df = df[~df["foco"].isin(FOCUS_TO_EXCLUDE)]
    df = safe_drop_columns(df=df, columns=cols_to_delete)
    return df


def resample_positive_classes(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance positive classes with SMOTE or ROS depending on sample counts."""
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


def perform_rfe_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_counts: Sequence[int],
    *,
    random_state: int,
) -> Dict[int, List[str]]:
    """Run RFE for the requested feature counts and return the selected columns."""
    if X.empty:
        raise ValueError("Cannot run RFE on an empty feature matrix.")

    available_features = X.shape[1]
    unique_counts = sorted({int(count) for count in feature_counts})
    if not unique_counts:
        raise ValueError("At least one feature count must be provided for RFE.")

    invalid_counts = [count for count in unique_counts if count <= 0 or count > available_features]
    if invalid_counts:
        raise ValueError(
            f"RFE feature counts {invalid_counts} are invalid for a dataset with "
            f"{available_features} available features."
        )

    step_size = max(1, int(0.1 * available_features))
    if step_size >= available_features:
        step_size = max(1, available_features - 1)

    selected_feature_sets: Dict[int, List[str]] = {}
    for count in unique_counts:
        rfe = RFE(
            estimator=RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
            n_features_to_select=count,
            step=step_size,
        )
        rfe.fit(X, y)
        selected_feature_sets[count] = X.columns[rfe.support_].tolist()

    return selected_feature_sets

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
    """Tune the binary gate and choose the best probability threshold."""
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
    """Tune the multiclass positive head with stratified CV and SMOTE."""
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
    """Train the hierarchical model using the provided feature subset."""
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


def parse_feature_counts(raw_counts: str) -> List[int]:
    """Parse a comma separated string of feature counts into a clean list."""
    if isinstance(raw_counts, (list, tuple)):
        counts = [int(count) for count in raw_counts]
    else:
        try:
            counts = [
                int(part.strip())
                for part in str(raw_counts).split(",")
                if part.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "RFE feature counts must be integers separated by commas."
            ) from exc

    if not counts:
        raise ValueError("At least one RFE feature count must be provided.")

    ordered_counts: List[int] = []
    seen = set()
    for count in counts:
        if count <= 0:
            raise ValueError("RFE feature counts must be positive integers.")
        if count not in seen:
            ordered_counts.append(count)
            seen.add(count)

    return ordered_counts


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("SELECTED ARGS: ", args)
    cols_to_delete = DELETE_COLUMNS
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
    feature_counts = parse_feature_counts(args.rfe_feature_counts)
    print("Starting RFE...")
    feature_sets = perform_rfe_feature_selection(
        X_train_full, y_train_full, feature_counts, random_state=args.random_state
    )
    print("Finished RFE")
    output_folder = Path(str(args.output_dir) +  "_" + str(TODAY))
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "rfe_feature_sets.json").write_text(
        json.dumps(
            {f"rfe_{count}": columns for count, columns in feature_sets.items()},
            indent=2,
        )
    )

    aggregate_summary = {
        "requested_feature_counts": feature_counts,
        "subsets": [],
    }
    for count in feature_counts:
        print(f"Training with {count} features: ", feature_sets[count])
        selected_columns = feature_sets[count]
        subset_name = f"rfe_{count}"
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
        aggregate_summary["subsets"].append(summary)
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

    (output_folder / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2)
    )
    print("Completed hierarchical training for all RFE subsets.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
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
        "--rfe-feature-counts",
        type=str,
        default="15,20,40",
        help="Comma separated list of feature counts to evaluate with RFE before Optuna.",
    )
    return parser


def main() -> None:
    start = time.start = time.time()
    print("Checking parsed args...")
    parser = build_arg_parser()
    args = parser.parse_args()
    print("Starting script...")
    run_training(args)
    print(" ")
    end = time.time()
    print("ELAPSED TIME (minutes): ", (end - start)/60)


if __name__ == "__main__":
    main()
