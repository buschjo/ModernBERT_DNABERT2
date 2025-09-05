#!/bin/bash

# This file was created by Josefine Busch

#SBATCH --job-name=decompress
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal
#SBATCH --output=../../sbatch_output/dnabert2_modern/decompress_streaming_dataset/%j_%x.out
#SBATCH --error=../../sbatch_output/dnabert2_modern/decompress_streaming_dataset/%j_%x.err

##SBATCH --mail-type=ALL
##SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
conda activate bert24

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

python src/data/mds_conversion.py --decompress --data_path /scratch/jbusch/ma/data/dnabert_2_pretrain/streaming_test --read_split val --write_split val_decompressed
# python src/data/mds_conversion.py --decompress --data_path /scratch/jbusch/ma/data/dnabert_2_pretrain/streaming_test --read_split train --write_split train_decompressed

# Log completion time
echo "Job finished at $(date)"