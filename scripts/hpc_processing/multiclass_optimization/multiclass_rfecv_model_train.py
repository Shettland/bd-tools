#!/usr/bin/env python3
"""
Multiclass-only Optuna training script (no binary gate) for resultado_hemo-style targets.

This script reuses the preprocessing and RFECV flow from the hierarchical trainer, but
trains a single multiclass model directly:
  1. Load and preprocess the merged dataset.
  2. Use RFECV to select an optimal feature subset automatically.
  3. Optimise an LGBM multiclass head with Optuna + SMOTE/ROS.
  4. Train, evaluate on a hold-out set, and persist artefacts.
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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

MINOR_CLASSES_TO_DROP = {
    "Enterococcus",
    "_Fungi",
    "_Other bacteria",
}


def sanitize_feature_names(columns: pd.Index) -> pd.Index:
    """Make feature names LightGBM-safe (no special JSON chars) and unique."""
    cleaned = columns.str.replace("[^0-9a-zA-Z_]+", "_", regex=True)
    new_names = []
    seen: Dict[str, int] = {}
    for name in cleaned:
        base = name or "feature"
        if base[0].isdigit():
            base = f"f_{base}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        new_names.append(base if count == 0 else f"{base}__{count}")
    return pd.Index(new_names)


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


def resample_classes(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance classes with SMOTE or ROS depending on sample counts."""
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

def impute_missing_values(loaded_df, exclude_cols):
    """Impute missing values in the dataframe, avoid imputing values in target_cols."""
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
    """Run RFECV once and return the selected columns alongside the fitted selector."""
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
        n_jobs=1,
    )
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=skf,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=1,
    )
    selector.fit(X, y)
    selected_columns = X.columns.tolist() #X.columns[selector.support_].tolist()
    return selected_columns, selector


# ---------------------------------------------------------------------------
# Optuna optimisation
# ---------------------------------------------------------------------------

def optimise_multiclass_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    n_trials: int,
    random_state: int,
    use_stacking: bool,
    model_type: str,
    meta_model,
) -> Tuple[Dict[str, object], optuna.study.Study]:
    """Tune the multiclass model with stratified CV and SMOTE/ROS."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    num_classes = len(np.unique(y))

    def stacking_objective(trial: optuna.Trial) -> float:
        params_lgr = {
            "C": trial.suggest_float("meta_lr_C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("meta_lr_penalty", ["l2"]),
            "solver": "lbfgs",
            "max_iter": 5000,
            "multi_class": "auto",
        }

        params_lgbm = {
            "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("lgbm_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_int("lgbm_depth", 3, 10),
            "min_child_samples": trial.suggest_int("lgbm_min_child", 10, 60),
            "subsample": trial.suggest_float("lgbm_subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("lgbm_l1", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lgbm_l2", 1e-8, 10.0, log=True),
        }

        params_xgb = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("xgb_estimators", 300, 1500, step=100),
            "max_depth": trial.suggest_int("xgb_depth", 3, 10),
            "min_child_weight": trial.suggest_float("xgb_child_weight", 1.0, 8.0),
            "subsample": trial.suggest_float("xgb_subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.7, 1.0),
            "gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("xgb_l1", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("xgb_l2", 1e-8, 10.0, log=True),
            "eval_metric": "mlogloss",
            "n_jobs": 1,
        }

        scores = []
        for train_idx, valid_idx in skf.split(X, y):
            X_tr = X.iloc[train_idx]
            X_va = X.iloc[valid_idx]
            y_tr = y.iloc[train_idx]
            y_va = y.iloc[valid_idx]

            X_tr_bal, y_tr_bal = resample_classes(X_tr, y_tr, random_state=random_state)
            base_models = [
                ("lgr", LogisticRegression(**params_lgr, random_state=random_state, n_jobs= 1)),
                ("xgb", XGBClassifier(**params_xgb)),
                ("lgbm", LGBMClassifier(objective="multiclass",
                    num_class=num_classes,
                    random_state=random_state,
                    n_jobs=1,
                    verbose=-1,
                    **params_lgbm,))
                ]
            model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5,
                n_jobs=1,
                passthrough=False
            )

            model.fit(X_tr_bal, y_tr_bal)

            y_pred = model.predict(X_va)
            scores.append(f1_score(y_va, y_pred, average="macro"))

        return float(np.mean(scores))
    
    def objective(trial: optuna.Trial) -> float:
        if model_type == "catb":
            params_catb = {
                "iterations": trial.suggest_int("catb_iterations", 300, 2000),
                "learning_rate": trial.suggest_float("catb_learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("catb_depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("catb_l2_leaf_reg", 1e-3, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("catb_bagging_temperature", 0.0, 1.0),
                "border_count": trial.suggest_int("catb_border_count", 32, 255),
                "random_strength": trial.suggest_float("catb_random_strength", 0.0, 2.0),
                "auto_class_weights": "Balanced",  # handles class imbalance
                "loss_function": "MultiClass",
                "verbose": False,
                "random_state": 42,
                "classes_count": num_classes
            }
        elif model_type == "lgbm":
            params_lgbm = {
                "learning_rate": trial.suggest_float("lgbm_learning_rate", 1e-3, 0.3, log=True),
                "num_iterations": trial.suggest_int("lgbm_num_iterations", 300, 4000),
                "num_leaves": trial.suggest_int("lgbm_num_leaves", 8, 256),
                "max_depth": trial.suggest_int("lgbm_max_depth", 1, 16),
                "min_data_in_leaf": trial.suggest_int("lgbm_min_data_in_leaf", 5, 200),
                "min_sum_hessian_in_leaf": trial.suggest_float("lgbm_min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),
                "lambda_l1": trial.suggest_float("lgbm_lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lgbm_lambda_l2", 1e-8, 10.0, log=True),
                "min_gain_to_split": trial.suggest_float("lgbm_min_gain_to_split", 0.0, 1.0),
                "feature_fraction": trial.suggest_float("lgbm_feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("lgbm_bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("lgbm_bagging_freq", 1, 10),
                "max_bin": trial.suggest_int("lgbm_max_bin", 128, 512),
                "extra_trees": trial.suggest_categorical("lgbm_extra_trees", [True, False]),
                "objective": "multiclass",
                "num_class": num_classes,
                "boosting_type": "gbdt",
                "verbosity": -1,
                "force_col_wise": True,
                "class_weight": "balanced",
            }
        scores = []
        for train_idx, valid_idx in skf.split(X, y):
            X_tr = X.iloc[train_idx]
            X_va = X.iloc[valid_idx]
            y_tr = y.iloc[train_idx]
            y_va = y.iloc[valid_idx]
            if model_type == "catb":
                model = CatBoostClassifier(**params_catb)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_va, y_va),
                    early_stopping_rounds=50,
                    verbose=False
                )
            elif model_type == "lgbm":
                model = LGBMClassifier(**params_lgbm)
                model.fit(X_tr, y_tr)
            else:
                raise ValueError(f"invalid model_type: {model_type}")

            y_pred = model.predict(X_va)
            scores.append(f1_score(y_va, y_pred, average="macro"))

        return float(np.mean(scores))
    
    study = optuna.create_study(direction="maximize")
    if use_stacking:
        study.optimize(stacking_objective, n_trials=n_trials, n_jobs=N_CPUS, gc_after_trial=True)
    else:
        study.optimize(objective, n_trials=n_trials, n_jobs=N_CPUS, gc_after_trial=True)

    best_params = study.best_trial.params.copy()
    return best_params, study

def build_final_stacking_model(best_params, random_state, num_classes, meta_model):
    params_lgr = {k.replace("lr_", ""): v for k, v in best_params.items() if k.startswith("lr_")}
    params_lgbm = {k.replace("lgbm_", ""): v for k, v in best_params.items() if k.startswith("lgbm_")}
    params_xgb = {k.replace("xgb_", ""): v for k, v in best_params.items() if k.startswith("xgb_")}
    final_base_models = [
        (
            "lgr", LogisticRegression(
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
                objective="multiclass",
                num_class=num_classes,
                random_state=random_state,
                n_jobs=1,
                verbose=-1,
            )
        ),
        (
            "xgb",
            XGBClassifier(
                **params_xgb,
                objective="multi:softprob",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=1,
            )
        ),
    ]
    multiclass_model = StackingClassifier(
        estimators=final_base_models,
        final_estimator=meta_model,
        passthrough=False,
    )
    return multiclass_model

def build_multiclass_model(model_type, best_params):
    models_dict = {
        "lgbm": LGBMClassifier,
        "rf": RandomForestClassifier,
        "xgb": XGBClassifier,
        "catb": CatBoostClassifier
    }
    return models_dict[model_type](**best_params)

# ---------------------------------------------------------------------------
# Training with one feature subset
# ---------------------------------------------------------------------------

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
    """Train the multiclass model using the provided feature subset."""
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

    # Label encoding
    label_encoder = LabelEncoder()
    y_train_enc = pd.Series(
        label_encoder.fit_transform(y_train_full),
        index=y_train_full.index,
        name="encoded_target",
    )
    y_test_enc = label_encoder.transform(y_test_full)
    meta_model = LogisticRegression(max_iter=5000, multi_class="auto", n_jobs=1)
    # ------------------------------------------------------------------
    # Multiclass optimisation + training
    # ------------------------------------------------------------------
    multiclass_params, multiclass_study = optimise_multiclass_model(
        X_train_scaled,
        y_train_enc,
        n_splits=args.cv_splits,
        n_trials=args.multiclass_trials,
        random_state=args.random_state,
        use_stacking=args.use_stacking,
        model_type=args.multiclass_model,
        meta_model=meta_model
    )
    num_classes = len(np.unique(y_train_enc))
    multiclass_params = {
        k.replace(args.multiclass_model + "_", ""): v 
        for k, v in multiclass_params.items() 
        if k.startswith(args.multiclass_model)
    }
    X_train_bal, y_train_bal = resample_classes(
        X_train_scaled, y_train_enc, random_state=args.random_state
    )
    if args.use_stacking:
        multiclass_model = build_final_stacking_model(
            best_params=multiclass_params,
            num_classes=num_classes,
            meta_model=meta_model,
            random_state=args.random_state
        )
    else:
        multiclass_model = build_multiclass_model(args.multiclass_model, multiclass_params)
    multiclass_model.fit(X_train_bal, y_train_bal)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    proba_test = multiclass_model.predict_proba(X_test_scaled)
    pred_idx = np.argmax(proba_test, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_idx)

    all_labels = label_encoder.classes_.tolist()
    hierarchical_conf = proba_test.max(axis=1)
    multiclass_conf_df = confusion_matrix(
        y_test_full,
        pred_labels,
        labels=all_labels,
    )
    multiclass_conf_df = pd.DataFrame(
        multiclass_conf_df,
        index=[f"true_{lbl}" for lbl in all_labels],
        columns=[f"pred_{lbl}" for lbl in all_labels],
    )

    multiclass_report = classification_report(
        y_test_full,
        pred_labels,
        labels=all_labels,
        target_names=all_labels,
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_test_full,
        pred_labels,
        labels=all_labels,
        average="macro",
    )
    try:
        macro_auc = roc_auc_score(
            y_test_enc,
            proba_test,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        macro_auc = None

    probability_payload = []
    for idx in range(len(X_test_scaled)):
        proba_map = {cls: float(prob) for cls, prob in zip(all_labels, proba_test[idx])}
        probability_payload.append({"multiclass": proba_map})

    results_df = pd.DataFrame(
        {
            "true_label": y_test_full.reset_index(drop=True),
            "pred_label": pred_labels,
            "confidence": hierarchical_conf,
        }
    )
    results_df["proba"] = probability_payload

    macro_f1 = float(macro_f1)
    if macro_auc is not None:
        macro_auc = float(macro_auc)

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    results_df.to_csv(subset_output_dir / "predictions.csv", index=False)
    multiclass_conf_df.to_csv(subset_output_dir / "multiclass_confusion_matrix.csv")
    (subset_output_dir / "multiclass_report.txt").write_text(multiclass_report)

    summary = {
        "subset_name": subset_name,
        "feature_count": len(selected_columns),
        "selected_features": selected_columns,
        "multiclass_best_params": multiclass_params,
        "macro_f1": macro_f1,
        "multiclass_macro_auc": macro_auc,
        "label_encoder_classes": all_labels,
        "multiclass_trials": args.multiclass_trials,
        "output_dir": str(subset_output_dir),
    }
    (subset_output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    trials_dir = subset_output_dir / "optuna_trials"
    trials_dir.mkdir(exist_ok=True)
    multiclass_study.trials_dataframe().to_csv(
        trials_dir / "multiclass_trials.csv",
        index=False,
    )

    # Final console summary for SLURM logs
    print(f"[{subset_name}] Multiclass best params:", multiclass_params)
    print(f"[{subset_name}] Hold-out macro F1:", f"{macro_f1:.3f}")
    if macro_auc is not None:
        print(f"[{subset_name}] Hold-out multiclass macro ROC-AUC:", f"{macro_auc:.3f}")
    else:
        print(f"[{subset_name}] Hold-out multiclass macro ROC-AUC: N/A")

    return summary


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
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
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns
    categorical_cols = [x for x in categorical_cols if x != args.target]
    if categorical_cols:
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=False)
    feature_df = impute_missing_values(feature_df, [args.target])
    # Clean feature names to avoid LightGBM JSON issues
    feature_df.columns = sanitize_feature_names(feature_df.columns)

    target_series = feature_df[args.target]

    feature_df = feature_df.drop(columns=args.target)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        feature_df,
        target_series,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target_series,
    )
    print("Finished train/test split")
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
    selected_columns = X_test_full.columns.tolist()
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
    print("Completed multiclass training with RFECV-selected features.")
    print(f"Results saved in {output_folder}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    home = Path.cwd()
    default_db = os.path.join(home, "mepram_data", "df_merged_full.csv")
    default_out = os.path.join(home, "mepram_data", "outputs", "multiclass_optuna")

    parser = argparse.ArgumentParser(
        description="Run Optuna + RFECV search for a single multiclass model (no binary gate)."
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
        help="Target column to model.",
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
        "--multiclass-trials",
        "-mtrials",
        type=int,
        default=500,
        help="Optuna trials for the multiclass model.",
    )
    parser.add_argument(
        "--multiclass-model",
        type=str,
        choices=["xgb", "lgbm", "rf", "catb"],
        default="lgbm",
        help="Multiclass estimator to optimise.",
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
        help="Wether to use stacking or not",
    )
    return parser


def main() -> None:
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
