#!/bin/bash
#SBATCH --job-name=cascade_training_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=47:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --partition=long_idx
#SBATCH --chdir /scratch/bi/TESTS/pmata_mepram_tests/cascade_modelling/

module load singularity

model_type="catb"
max_f=1000

singularity exec --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK --env SLURM_JOB_ID=$SLURM_JOB_ID --containall --bind ./:/mnt env_singularity_img.sif \
    python3 /mnt/hierarchical_cascade_three_level_shaprfecv.py \
    -db /mnt/mepram_data/df_merged_full_multilabel_grouped_old.csv \
    -o /mnt/outputs/cascade_three_level_shaprfecv_olddb_${model_type}_${max_f}f \
    --model-type ${model_type} \
    --binary-trials 500 \
    --max-features $max_f
