#!/bin/bash

#SBATCH --job-name=dna_pret
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal
#SBATCH --output=../../sbatch_output/dnabert2_modern/pretraining/c4/%j_%x.out
#SBATCH --error=../../sbatch_output/dnabert2_modern/pretraining/c4/%j_%x.err

##SBATCH --mail-type=ALL
##SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
# conda activate bert24
# conda activate bens_tokenizer_modernbert2 #-> weird cpu and precision error see 12629
conda activate modernbert_experiment

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

composer main.py yamls/main/jb_mosaic-bert-base-uncased.yaml

# Log completion time
echo "Job finished at $(date)"