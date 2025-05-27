#!/bin/bash

#SBATCH --job-name=hs_tokenizer_training
#SBATCH --nodes=1
##SBATCH --nodelist=kiwinode02
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=493G
##SBATCH --mem-per-cpu=250G
#SBATCH --qos=normal
#SBATCH --output=../../sbatch_output/dnabert2_modern/bens_tokenizer/%j_%x.out
#SBATCH --error=../../sbatch_output/dnabert2_modern/bens_tokenizer/%j_%x.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
conda activate bens_tokenizer_39_less_output

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

srun python helix_swap_train.py \
    --tokenizer_training_data /scratch/jbusch/ma/data/dnabert_2_pretrain/train_sub5M.txt \
    --tokenizer_save_path /scratch/jbusch/ma/tokenizer/hs_tokenizer/fixed_encode_decode/sub5M_tokenizer.json

echo "Job finished at $(date)"