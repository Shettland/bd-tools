# HPC Processing Scripts

Machine learning pipelines for hierarchical clinical outcome prediction (etiology, antibiotic resistance phenotype). All training scripts are designed for SLURM/non-interactive execution and write timestamped output directories.

---

## Directory Structure

```
hpc_processing/
├── binary_optimization/                 # Two-level binary-gate + optional multiclass or binary head
├── cascade_modelling/                   # Three-level cascaded and independent hierarchies
├── multiclass_optimization/             # Standalone multiclass classifiers
├── multilabel_optimization/             # Multi-label resistance phenotype classifiers
└── analyse_hierarchical_results.ipynb   # Post-run analysis notebook
```

---

## Common Output Structure

Every training script writes results under `--output-dir/rfecv_<subset>/`:

| File | Description |
|---|---|
| `binary_confusion_matrix.csv` | Normalised binary confusion matrix |
| `multiclass_confusion_matrix.csv` | Normalised multiclass confusion matrix |
| `binary_predictions.csv` / `hierarchical_predictions.csv` | Per-sample test-set predictions |
| `summary.json` | Key metrics (ROC-AUC, macro F1, selected features, …) |
| `multilabel_metrics.json` | Multi-label specific metrics (micro/macro F1, exact match, coverage, P@k, R@k) |

---

## binary_optimization/

Two-stage models: a binary gate classifies **NEGATIVE vs POSITIVE** (BMR etiology), and optionally a multiclass head classifies the resistance phenotype among positive samples. All scripts use RFECV feature selection and Optuna hyperparameter tuning.

### `hierarchical_model_train.py`
Full two-level hierarchy: binary gate + multiclass head for `fenotipo_resistencia`.

```bash
python hierarchical_model_train.py \
  --database-file data/merged.csv \
  --output-dir outputs/run_001 \
  --binary-model lgbm \
  --multiclass-model xgb \
  --binary-trials 100 \
  --multiclass-trials 100 \
  --cv-splits 5
```

### `hierarchical_model_train_binary_head.py`
Variant of the above that prioritises optimising the binary gate. Same interface.

### `hierarchical_model_train_binary_only.py`
Binary-only classifier (no multiclass head). Supports stacked ensembles (RF + LGBM + XGB → LogisticRegression meta-learner).

```bash
python hierarchical_model_train_binary_only.py \
  --database-file data/merged.csv \
  --output-dir outputs/binary_run \
  --binary-model lgbm
```

### `hierarchical_model_train_rfecv.py`
Two-level hierarchy targeting `resultado_hemo` (blood culture outcome) with an explicit RFECV history logged to `summary.json`.

### `binary_model_cefalosporina_dropother.py`
Binary-only classifier specifically for cefalosporin resistance. Automatically drops low-frequency foci and the "other" category before training.

```bash
python binary_model_cefalosporina_dropother.py \
  --database-file data/merged.csv \
  --output-dir outputs/cef_run \
  --binary-model lgbm \
  --binary-trials 150 \
  --weight-column sample_weight
```

### `feature_reduction_analysis.py`
Sweeps over a list of feature counts to find the minimal clinically practical feature set. Uses a previously trained RFECV model for feature ranking, then re-trains at each count with Optuna.

```bash
python feature_reduction_analysis.py \
  --database-file data/merged.csv \
  --reference-model outputs/best_run/rfecv_0/rfecv_selector.pkl \
  --output-dir outputs/feature_sweep \
  --feature-counts 5,10,15,20,30,50,75,100 \
  --binary-trials 80
```

---

## cascade_modelling/

Three-level stacked pipelines:
- **Level 1** — Sepsis (binary, all patients)
- **Level 2** — Blood culture outcome / `resultado_hemo` (multiclass, all patients)
- **Level 3** — Antibiotic resistance / `resistente_cefalosporina` (binary or multiclass, positive patients only)

Outside each level-subfolder theres an `aggregate_summary.json` which holds all the results for each level, including the args used as input for the script, selected features during RFECV, and evaluation metrics.

### `hierarchical_cascade_three_level.py`
Full OOF (out-of-fold) cascade: each level appends its predicted probabilities as new features for the next level, preventing label leakage. Uses CatBoost, LGBM, and XGB with SHAP-based RFECV at each level.

```bash
python hierarchical_cascade_three_level.py \
  --database-file data/merged.csv \
  --output-dir outputs/cascade_run \
  --binary-trials 100 \
  --cv-splits 5
```

Outputs: `level1_sepsis/`, `level2_hemo/`, `level3_cef/` subdirs + `aggregate_summary.json`.

### `hierarchical_cascade_three_level_cefmult.py`
Variant where Level 3 is **multiclass** (`resistente_cefalosporina_multi`) instead of binary. Same interface.

### `hierarchical_cascade_three_level_copytest.py`
Development/testing copy of the three-level cascade. Not intended for production runs.

### `hierarchical_independent_three_level.py`
Three-level hierarchy where each level is trained **independently** on the original features only (no OOF propagation). Useful as a baseline comparison against the cascade approach.

```bash
python hierarchical_independent_three_level.py \
  --database-file data/merged.csv \
  --output-dir outputs/independent_run
```

### `shared_features_analysis.py`
Same as previous cascade trainings but runs a single joint RFECV across all three levels simultaneously to find the most optimal set of training features for all targets at once instead of an individual set of features for each target, eliminating features by weighted-mean importance. Then each level is tuned with Optuna on the shared feature set.

```bash
python shared_features_analysis.py \
  --database-file data/merged.csv \
  --output-dir outputs/shared_run \
  --level-weights 1 1 2   # double weight to Level 3
```

### `visualize_cascade_results.ipynb`
Notebook for reviewing and ranking cascade runs. Set `ROOT_FOLDER` to the outputs directory, optionally filter by `limit_date` or `TAG` \(folder name prefix\) and run all cells to get a ranked comparison table and per-level metrics.

---

## multiclass_optimization/

### `calibrated_model_2level.py`
Two-level hierarchy (binary gate + multiclass head) where both classifiers are wrapped with `CalibratedClassifierCV`. Produces well-calibrated probability scores, which improves threshold-based decision-making.

```bash
python calibrated_model_2level.py \
  --database-file data/merged.csv \
  --output-dir outputs/calibrated_run
```

### `multiclass_rfecv_model_train.py`
Direct multiclass classifier for `resultado_hemo`-style targets — **no binary gate**. Applies RFECV and SMOTE/ROS for class imbalance. Use when only the multiclass prediction is needed.

```bash
python multiclass_rfecv_model_train.py \
  --database-file data/merged.csv \
  --output-dir outputs/multiclass_run
```

---

## multilabel_optimization/

Classifiers where the resistance phenotype target is **multi-label** (a patient can have multiple resistance profiles simultaneously). Predictions are pipe-delimited strings or multi-hot vectors.

### `hierarchical_model_train_bmr_fenotipo_rfecv_multilabel.py`
Binary gate + multi-label head. Use `--multilabel` to activate multi-label mode; otherwise falls back to multiclass. Per-label thresholds are tuned post-training.

```bash
python hierarchical_model_train_bmr_fenotipo_rfecv_multilabel.py \
  --database-file data/merged.csv \
  --output-dir outputs/multilabel_run \
  --multilabel
```

### `model_train_bmr_multilabel_catboost.py`
Multi-label `fenotipo_resistencia` using CatBoost, which handles categorical features natively without one-hot encoding. Generally faster on high-cardinality categoricals.

```bash
python model_train_bmr_multilabel_catboost.py \
  --database-file data/merged.csv \
  --output-dir outputs/catboost_run \
  --multiclass-target fenotipo_resistencia
```

### `model_train_bmr_multilabel_only.py`
Multi-label classifier with no binary gate — all patients go directly to the multi-label head. Supports LGBM, XGB, and CatBoost backends.

```bash
python model_train_bmr_multilabel_only.py \
  --database-file data/merged.csv \
  --output-dir outputs/multilabel_only_run
```

---

## Notebooks

### `analyse_hierarchical_results.ipynb`
Post-run analysis for binary, multiclass, hierarchical, and multi-label runs. Point `results_root` at an output directory and run the relevant cell:

| Cell | Purpose |
|---|---|
| 1 | Single run — auto-detects mode, plots confusion matrix heatmap |
| 2 | Batch scan — compares binary ROC-AUC across multiple runs |
| 3 | Binary-only run — confusion matrix with raw counts |
| 4 | Hierarchical run — binary + multiclass side-by-side with metrics |
| 5 | Inspect `summary` dict |
| 6 | Multi-label dashboard — binary CM, per-label recall, top labels, label count histogram |
| 7 | Reference: infection focus (`foco`) integer → string label map |

---

## SLURM Job Scripts (`.sbatch`)

Each subfolder contains a `.sbatch` file that wraps its corresponding Python script for cluster submission. All jobs run inside a **Singularity container** (`env_singularity_img.sif`) mounted at `/mnt`, with the scratch working directory bound to that mount point so paths inside the container use `/mnt/...`.

### Common resource settings

| Setting | Typical value | Notes |
|---|---|---|
| `--partition` | `long_idx` | Used across all jobs |
| `--time` | `47:00:00` | Just under the 48 h partition limit |
| `--ntasks` | `1` | Single-process; parallelism comes from `--cpus-per-task` |
| `--output` / `--error` | `logs/%x_%j.out` | Written to a `logs/` dir relative to `--chdir` |
| `SLURM_CPUS_PER_TASK` | passed into container via `--env` | Scripts read this to set internal thread/job budgets |

### Per-script summary

#### `run_feature_reduction.sbatch` (root level)
Launches `feature_reduction_analysis.py`. Sweeps 15 feature-count checkpoints (`5` → `188`) using a reference RFECV model for feature ranking. Uses 6 CPUs, 8 GB RAM.

Key variables to edit before submitting:
```bash
MODEL_TYPE="catb"          # model backend: catb | lgbm | xgb
TRIALS=500                 # Optuna trials per feature-count checkpoint
FEAT_COUNTS="5,8,10,..."   # comma-separated list of counts to evaluate
REF_MODEL="..."            # path to a previously trained RFECV output dir
```

#### `binary_optimization/run_training_bmr_binary_stacking.sbatch`
Launches `hierarchical_model_train_binary_only.py` with stacking enabled (`--use-stacking`). Targets `resistente_cefalosporina` with a RF binary model and 2000 Optuna trials. Uses 6 CPUs, 24 GB RAM.

#### `cascade_modelling/run_cascade_train_shaprfecv.sbatch`
Launches the three-level cascade with SHAP-based RFECV. Only 2 CPUs requested — the script internally manages thread budgets from `SLURM_CPUS_PER_TASK` — but 64 GB RAM to accommodate the OOF feature matrix and SHAP computation.

Key variables:
```bash
model_type="catb"   # model backend for all three levels
max_f=1000          # max features allowed after RFECV elimination
```

#### `multiclass_optimization/run_training_bmr_multiclass_only.sbatch`
Launches `multiclass_rfecv_model_train.py` in multiclass-only mode (no binary gate) targeting `resistente_cefalosporina` with 3000 Optuna trials. Uses 6 CPUs, 24 GB RAM.

Key variable:
```bash
model_type="lgbm"   # model backend: lgbm | xgb | catb
```

#### `multilabel_optimization/run_training_bmr_multilabel.sbatch`
Launches `hierarchical_model_train_bmr_fenotipo_rfecv_multilabel.py` in multi-label mode. Binary head uses RF (1000 trials), multiclass head uses XGB (1000 trials), with a prediction threshold of `0.15` and comma delimiter for multi-label targets. Uses 6 CPUs, 24 GB RAM.

### Submitting a job

```bash
# From the cluster scratch directory that contains env_singularity_img.sif
cd /scratch/bi/TESTS/YOUR_FOLDER/
sbatch /path/to/run_training_bmr_binary_stacking.sbatch
```

Logs land in `logs/<job-name>_<job-id>.out` relative to the `--chdir` path defined in the script.

---

## Key Design Patterns

- **RFECV** — Recursive feature elimination with CV is used in all training scripts to select a minimal, generalisable feature subset.
- **Optuna** — Bayesian hyperparameter optimisation with stratified K-fold CV.
- **SMOTE / RandomOverSampler** — Applied to the minority class before training to address label imbalance in some scripts.
- **OOF cascade** — In cascade scripts, Level N+1 uses out-of-fold predictions from Level N as additional features to prevent leakage.
- **Sample weights** — Optional `--weight-column` allows cohort-specific weighting to correct for sampling bias.
- **SLURM-ready** — All scripts are non-interactive and write self-contained, timestamped output directories suitable for parallel job arrays.
