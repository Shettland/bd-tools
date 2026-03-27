#!/usr/bin/env python3
"""
Feature Reduction Analysis Script

Trains the binary etiology model (CatBoost or LightGBM) with progressively
more features, ordered by RFECV ranking from a reference model.

This produces a "score vs number of features" curve to find the minimum
feature set needed for a practical deployment (e.g. a clinical form).

Usage:
    python feature_reduction_analysis.py \
        --database-file /mnt/mepram_data/df_merged_full_multilabel_grouped.csv \
        --reference-model /mnt/outputs/binary_optuna_catb_20260204114358 \
        --output-dir /mnt/outputs/feature_reduction_catb \
        --binary-model catb \
        --feature-counts 5,8,10,12,15,18,20,25,30,40,50,75,100,150,188 \
        --binary-trials 500
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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Configuration (reused from hierarchical_model_train_binary_only.py)
# ---------------------------------------------------------------------------

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
TODAY = datetime.today().strftime("%Y%m%d%H%M%S")

FOCUS_MAP = {
    1: "pulmonar", 2: "intraabdominal", 3: "biliar", 4: "urinario",
    5: "cardiovascular", 6: "piel", 7: "sistema nervioso central",
    8: "cateter venoso", 9: "vías altas respiratorias", 10: "osteoarticular",
    11: "genital", 12: "desconocido",
}

TARGET_REMOVE = [
    "sepsis", "resultado_hemo", "resultado_hemo_grouped", "all_cult_org",
    "infected_yes_no", "bmr_etiologia", "fenotipo_resistencia",
    "fenotipo_resistencia_grouped", "resistente_cefalosporina",
]

DELETE_COLUMNS = [
    "qsofa", "vasopresores", "hipotension", "freq_bacteria", "freq_bac_foco",
    "Unnamed: 0", "person_id", "fecha_ingreso_urgencias",
    "fecha_ingreso_urgencias_x", "shock_septico", "sintoma_nan",
    "fecha_nacimiento", "codigo_postal", "center", "dag", "ultima_fecha",
    "mujer_gestante",
]

FOCUS_TO_EXCLUDE = {
    "piel", "osteoarticular", "biliar", "genital",
    "sistema nervioso central", "cateter venoso",
    "vías altas respiratorias", "cardiovascular",
}


# ---------------------------------------------------------------------------
# Data preparation (reused)
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
            pass
    return df


def load_processed_dataframe(csv_path, cols_to_delete, target):
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
        df_copy[binary_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df_copy[binary_cols]).astype(int)
    if continuous_cols:
        df_copy[continuous_cols] = KNNImputer(n_neighbors=5, weights="distance").fit_transform(df_copy[continuous_cols])
    if len(categorical_cols) > 0:
        df_copy[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df_copy[categorical_cols])
        df_copy[categorical_cols] = df_copy[categorical_cols].astype(str)
    if categorical_numeric_cols:
        df_copy[categorical_numeric_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df_copy[categorical_numeric_cols])
        df_copy[categorical_numeric_cols] = df_copy[categorical_numeric_cols].astype(int)

    for col in exclude_cols:
        if col in loaded_df.columns:
            df_copy[col] = loaded_df[col]
    return df_copy


def compute_balanced_sample_weight(labels, base_sample_weight=None):
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


# ---------------------------------------------------------------------------
# Feature ordering from reference model
# ---------------------------------------------------------------------------

def load_feature_ordering(reference_model_dir: Path, all_feature_cols: List[str]) -> List[str]:
    """
    Load RFECV rankings from a reference model and return features
    ordered by importance (rank 1 first).
    """
    rfecv_file = reference_model_dir / "rfecv_selected_features.json"
    if not rfecv_file.exists():
        raise FileNotFoundError(f"RFECV file not found: {rfecv_file}")

    with open(rfecv_file) as f:
        rfecv_data = json.load(f)

    binary_data = rfecv_data.get("binary", {})
    ref_features = binary_data.get("selected_features", [])
    rankings = binary_data.get("rfecv_ranking", binary_data.get("ranking", []))

    if not ref_features or not rankings:
        raise ValueError("No features or rankings found in reference model.")

    # Build ranking map: feature -> rank
    rank_map = {}
    for feat, rank in zip(ref_features, rankings[:len(ref_features)]):
        rank_map[feat] = rank

    # Order current features by their rank in the reference model
    # Features not in reference get max_rank + 1
    max_rank = max(rankings[:len(ref_features)]) if rankings else 999
    ordered = sorted(
        all_feature_cols,
        key=lambda f: rank_map.get(f, max_rank + 1)
    )

    return ordered


# ---------------------------------------------------------------------------
# Optuna optimisation (simplified from original)
# ---------------------------------------------------------------------------

def optimise_and_evaluate(
    X_train, y_train, X_test, y_test,
    sample_weight_train, sample_weight_test,
    model_type, n_trials, n_splits, random_state,
):
    """Run Optuna optimisation and evaluate on holdout set."""
    label_encoder = LabelEncoder()
    y_train_enc = pd.Series(
        label_encoder.fit_transform(y_train), index=y_train.index, name="target"
    )
    y_test_enc = pd.Series(
        label_encoder.transform(y_test), index=y_test.index, name="target"
    )

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    gate_weight = compute_balanced_sample_weight(y_train_enc, base_sample_weight=sample_weight_train)

    # Scale pos weight
    pos_mask = y_train_enc == 1
    pos_w = float(gate_weight.loc[pos_mask].sum())
    neg_w = float(gate_weight.loc[~pos_mask].sum())
    scale_pos_weight = neg_w / pos_w if pos_w else 1.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        if model_type == "catb":
            params = {
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
        elif model_type == "lgbm":
            params = {
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
                "verbosity": -1,
            }
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        threshold = trial.suggest_float("threshold", 0.1, 0.9)
        scores = []

        for tr_idx, va_idx in skf.split(X_train_scaled, y_train_enc):
            X_tr = X_train_scaled.iloc[tr_idx]
            X_va = X_train_scaled.iloc[va_idx]
            y_tr = y_train_enc.iloc[tr_idx]
            y_va = y_train_enc.iloc[va_idx]

            w_tr = gate_weight.loc[X_tr.index].to_numpy()
            w_va = gate_weight.loc[X_va.index].to_numpy()

            if model_type == "catb":
                model = CatBoostClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_va, y_va),
                    early_stopping_rounds=max(20, int(0.05 * params["iterations"])),
                    verbose=False,
                )
            else:
                model_cls = {"lgbm": LGBMClassifier}[model_type]
                model = model_cls(**params)
                model.fit(X_tr, y_tr)

            probas = model.predict_proba(X_va)[:, 1]
            preds = (probas >= threshold).astype(int)
            score = f1_score(y_va, preds, average="macro")
            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=N_CPUS, gc_after_trial=True)

    # Retrain best model on full training data
    best_params = study.best_trial.params.copy()
    best_threshold = best_params.pop("threshold")

    if model_type == "catb":
        final_model = CatBoostClassifier(**best_params)
        final_model.fit(X_train_scaled, y_train_enc, verbose=False)
    elif model_type == "lgbm":
        final_model = LGBMClassifier(**best_params)
        final_model.fit(X_train_scaled, y_train_enc)

    # Evaluate on holdout
    test_probas = final_model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probas >= best_threshold).astype(int)

    test_weights = None
    if sample_weight_test is not None:
        test_weights = sample_weight_test.loc[X_test.index].to_numpy()

    macro_f1 = f1_score(y_test_enc, test_preds, average="macro", sample_weight=test_weights)
    roc_auc = roc_auc_score(y_test_enc, test_probas, sample_weight=test_weights)
    report = classification_report(
        y_test_enc, test_preds,
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0, sample_weight=test_weights,
    )
    conf_matrix = confusion_matrix(y_test_enc, test_preds, labels=[0, 1], sample_weight=test_weights)

    # Per-class F1
    per_class_f1 = f1_score(y_test_enc, test_preds, average=None, sample_weight=test_weights)

    # Feature importance from the final model
    if model_type == "catb":
        feat_imp = final_model.get_feature_importance()
    elif model_type == "lgbm":
        feat_imp = final_model.feature_importances_
    feat_imp_dict = dict(zip(X_train.columns, feat_imp.tolist()))

    return {
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
        "best_threshold": float(best_threshold),
        "best_cv_score": float(study.best_value),
        "best_params": best_params,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "class_names": [str(c) for c in label_encoder.classes_],
        "feature_importances": feat_imp_dict,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_analysis(args):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"Feature Reduction Analysis - {args.binary_model.upper()}")
    print(f"Feature counts to test: {args.feature_counts}")
    print(f"Optuna trials per run: {args.binary_trials}")
    print(f"CPUs: {N_CPUS}")
    print("=" * 80)

    # --- Load and preprocess data (same as training script) ---
    cols_to_delete = list(DELETE_COLUMNS)
    keep_targets = {args.binary_target, args.weight_column}
    cols_to_delete.extend([x for x in TARGET_REMOVE if x not in keep_targets])

    df = load_processed_dataframe(args.database_file, cols_to_delete, args.binary_target)

    working_df = df.copy()
    working_df[args.weight_column] = pd.to_numeric(working_df[args.weight_column], errors="coerce")
    working_df = working_df.dropna(subset=[args.binary_target, args.weight_column])
    working_df = working_df[working_df[args.weight_column] > 0]

    exclude_cols = {args.binary_target, args.weight_column}
    feature_cols = [col for col in working_df.columns if col not in exclude_cols]

    # Drop high-NA columns
    for col in list(feature_cols):
        if working_df[col].isna().mean() > args.na_perc_limit:
            print(f"  Dropped {col} (>{args.na_perc_limit*100:.0f}% NA)")
            feature_cols.remove(col)
            working_df = working_df.drop(columns=col)

    # Impute
    working_df = impute_missing_values(working_df, exclude_cols)

    # Dummies for categorical
    feature_df = working_df[feature_cols]
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
        feature_df.columns = feature_df.columns.str.replace("[^0-9a-zA-Z_]+", "_", regex=True)

    binary_target = working_df.loc[feature_df.index, args.binary_target]
    weight_series = working_df.loc[feature_df.index, args.weight_column]

    # Train/test split (same random state for reproducibility)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        feature_df, binary_target, weight_series,
        test_size=args.test_size, random_state=args.random_state,
        stratify=binary_target,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Available features: {X_train.shape[1]}")

    # --- Get feature ordering from reference model ---
    all_features = X_train.columns.tolist()

    if args.reference_model:
        print(f"\nLoading feature ordering from: {args.reference_model}")
        ordered_features = load_feature_ordering(Path(args.reference_model), all_features)
    else:
        # Fallback: quick RF importance-based ordering
        print("\nNo reference model. Computing RF feature importance for ordering...")
        rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=args.random_state, n_jobs=N_CPUS,
        )
        rf.fit(X_train, y_train)
        imp_order = np.argsort(rf.feature_importances_)[::-1]
        ordered_features = [all_features[i] for i in imp_order]

    print(f"Top 10 features in ordering:")
    for i, feat in enumerate(ordered_features[:10], 1):
        print(f"  {i:2d}. {feat}")

    # --- Output directory ---
    output_dir = Path(str(args.output_dir) + "_" + TODAY)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the feature ordering
    (output_dir / "feature_ordering.json").write_text(
        json.dumps({"ordered_features": ordered_features}, indent=2)
    )

    # --- Run analysis for each feature count ---
    results = []
    max_available = len(ordered_features)

    for k in args.feature_counts:
        if k > max_available:
            print(f"\nSkipping k={k} (only {max_available} features available)")
            continue

        selected = ordered_features[:k]
        X_train_k = X_train[selected]
        X_test_k = X_test[selected]

        print(f"\n{'='*80}")
        print(f"Training with top {k} features...")
        print(f"{'='*80}")
        start = time.time()

        result = optimise_and_evaluate(
            X_train_k, y_train, X_test_k, y_test,
            w_train, w_test,
            model_type=args.binary_model,
            n_trials=args.binary_trials,
            n_splits=args.cv_splits,
            random_state=args.random_state,
        )
        elapsed = (time.time() - start) / 60

        result["n_features"] = k
        result["features"] = selected
        result["elapsed_minutes"] = elapsed
        results.append(result)

        # Save per-step results
        step_dir = output_dir / f"top_{k}_features"
        step_dir.mkdir(exist_ok=True)
        (step_dir / "summary.json").write_text(json.dumps({
            "n_features": k,
            "macro_f1": result["macro_f1"],
            "roc_auc": result["roc_auc"],
            "best_cv_score": result["best_cv_score"],
            "best_threshold": result["best_threshold"],
            "best_params": result["best_params"],
            "classification_report": result["classification_report"],
            "confusion_matrix": result["confusion_matrix"],
            "per_class_f1": result["per_class_f1"],
            "class_names": result["class_names"],
            "features": selected,
            "elapsed_minutes": elapsed,
        }, indent=2))
        (step_dir / "feature_importances.json").write_text(
            json.dumps(result["feature_importances"], indent=2)
        )

        print(f"  k={k:3d} | Macro F1: {result['macro_f1']:.4f} | "
              f"ROC AUC: {result['roc_auc']:.4f} | "
              f"CV best: {result['best_cv_score']:.4f} | "
              f"Time: {elapsed:.1f}min")

    # --- Save combined results ---
    curve_data = [{
        "n_features": r["n_features"],
        "macro_f1": r["macro_f1"],
        "roc_auc": r["roc_auc"],
        "best_cv_score": r["best_cv_score"],
        "best_threshold": r["best_threshold"],
        "per_class_f1": r["per_class_f1"],
        "class_names": r["class_names"],
        "elapsed_minutes": r["elapsed_minutes"],
    } for r in results]

    (output_dir / "feature_reduction_curve.json").write_text(
        json.dumps(curve_data, indent=2)
    )

    # Also save as CSV for easy plotting
    curve_df = pd.DataFrame(curve_data)
    curve_df.to_csv(output_dir / "feature_reduction_curve.csv", index=False)

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("FEATURE REDUCTION SUMMARY")
    print("=" * 80)
    print(f"{'Features':>10} | {'Macro F1':>10} | {'ROC AUC':>10} | {'CV Score':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['n_features']:>10} | {r['macro_f1']:>10.4f} | "
              f"{r['roc_auc']:>10.4f} | {r['best_cv_score']:>10.4f}")

    # Find knee point (biggest drop per feature removed)
    if len(results) >= 2:
        best_f1 = max(r["macro_f1"] for r in results)
        print(f"\nBest Macro F1: {best_f1:.4f}")
        for r in results:
            pct_of_best = r["macro_f1"] / best_f1 * 100
            print(f"  k={r['n_features']:3d}: {r['macro_f1']:.4f} "
                  f"({pct_of_best:.1f}% of best)")
            if pct_of_best >= 95:
                print(f"    ^ >= 95% of best performance")

    print(f"\nResults saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Feature reduction analysis: score vs number of features."
    )
    parser.add_argument("--database-file", "-db", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument(
        "--reference-model", type=Path, default=None,
        help="Path to a reference model directory with rfecv_selected_features.json "
             "for feature ordering. If not provided, uses RF importance.",
    )
    parser.add_argument(
        "--binary-target", type=str, default="resultado_hemo_grouped",
        help="Binary target column (default: resultado_hemo_grouped).",
    )
    parser.add_argument(
        "--binary-model", type=str, choices=["catb", "lgbm"], default="catb",
    )
    parser.add_argument(
        "--weight-column", type=str, default="sample_weight",
    )
    parser.add_argument(
        "--feature-counts", type=str,
        default="5,8,10,12,15,18,20,25,30,40,50,75,100,150,188",
        help="Comma-separated list of feature counts to test.",
    )
    parser.add_argument("--binary-trials", "-btrials", type=int, default=500)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=99)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--na-perc-limit", type=float, default=0.052)
    return parser


def main():
    """Entry point: parse CLI arguments, run training, and report elapsed time.

    Measures wall-clock time from start to finish and prints the total elapsed minutes to stdout for SLURM job logs.
    """
    start = time.time()
    parser = build_parser()
    args = parser.parse_args()

    # Parse feature counts
    args.feature_counts = sorted([int(x.strip()) for x in args.feature_counts.split(",")])

    run_analysis(args)

    elapsed = (time.time() - start) / 60
    print(f"\nTotal elapsed time: {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
