#!/bin/bash

# This file was created by Josefine Busch

#SBATCH --job-name=dna_pret
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
##SBATCH --nodelist=kiwinode02
#SBATCH --time=120:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal
#SBATCH --output=../../sbatch_output/dnabert2_modern/pretraining/with_hf_tokenizer/scenario2/%j_%x.out
#SBATCH --error=../../sbatch_output/dnabert2_modern/pretraining/with_hf_tokenizer/scenario2/%j_%x.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
# conda activate bert24
# conda activate bens_tokenizer_modernbert2 #-> weird cpu and precision error see 12629
# conda activate modernbert_experiment
conda activate mb_hs_v5

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

composer main.py yamls/main/v10c_hf_dna_mosaic-bert-base-uncased.yaml

# Log completion time
echo "Job finished at $(date)"