#!/bin/bash
#SBATCH --job-name=ml_training_test
#SBATCH --output=logs/%x_%j_multilabel.out
#SBATCH --error=logs/%x_%j_multilabel.err
#SBATCH --time=47:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --partition=long_idx
#SBATCH --chdir /scratch/bi/TESTS/pmata_mepram_tests/bmr_phenotest

module load singularity

singularity exec --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK --containall --bind ./:/mnt env_singularity_img.sif python3 /mnt/hierarchical_model_train_bmr_fenotipo_rfecv_multilabel.py --multilabel --multilabel-delimiter ',' --multilabel-threshold 0.15 -db /mnt/mepram_data/df_resist_bmr_grouped_cefalosporinas.csv -o /mnt/outputs/hierarchical_optuna_bmr_multilabel_g2 --binary-model rf --multiclass-model xgb -btrials 1000 -mtrials 1000 --binary-target bmr_etiologia --multiclass-target fenotipo_resistencia --no-impute
