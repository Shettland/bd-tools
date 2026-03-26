#!/bin/bash
#SBATCH --job-name=feat_reduction
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=47:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --partition=long_idx
#SBATCH --chdir /scratch/bi/TESTS/pmata_mepram_tests/

module load singularity

# --- Configuration ---
MODEL_TYPE="catb"
TRIALS=500
# Feature counts to evaluate (comma-separated, ascending)
FEAT_COUNTS="5,8,10,12,15,18,20,25,30,40,50,75,100,150,188"

# Reference model for feature ordering (best CatBoost etiology model)
REF_MODEL="/mnt/outputs/binary_optuna_catb_20260204114358"

singularity exec \
    --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK \
    --containall \
    --bind ./:/mnt \
    env_singularity_img.sif \
    python3 /mnt/feature_reduction_analysis.py \
        -db /mnt/mepram_data/df_merged_full_multilabel_grouped.csv \
        -o /mnt/outputs/feature_reduction_${MODEL_TYPE} \
        --reference-model ${REF_MODEL} \
        --binary-model ${MODEL_TYPE} \
        --binary-target resultado_hemo_grouped \
        --feature-counts ${FEAT_COUNTS} \
        -btrials ${TRIALS}
