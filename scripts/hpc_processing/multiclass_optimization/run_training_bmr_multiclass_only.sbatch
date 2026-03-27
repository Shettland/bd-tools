#!/bin/bash
#SBATCH --job-name=mltc_bmr_ml_training_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=47:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --partition=long_idx
#SBATCH --chdir /scratch/bi/TESTS/pmata_mepram_tests/bmr_phenotest

module load singularity

model_type="lgbm"

singularity exec --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK --containall --bind ./:/mnt env_singularity_img.sif python3 /mnt/multiclass_rfecv_model_train.py -db /mnt/mepram_data/df_merged_full_multilabel_grouped.csv -o /mnt/outputs/multiclass_optuna_norfecv_${model_type} --multiclass-model ${model_type} --target resistente_cefalosporina -mtrials 3000
