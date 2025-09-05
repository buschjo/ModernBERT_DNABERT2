#!/bin/bash

#SBATCH --job-name=mds_conversion
#SBATCH --nodes=1
##SBATCH --nodelist=kiwinode02
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --qos=normal
#SBATCH --output=../../../../sbatch_output/dnabert2_modern/convert_data/%j_%x.out
#SBATCH --error=../../../../sbatch_output/dnabert2_modern/convert_data/%j_%x.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
conda activate ma_notebooks

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

srun python convert_pretraining_data_to_mds.py \
    --pretraining_data_path /scratch/jbusch/ma/data/dnabert_2_pretrain/dev.csv \
    --target_path /scratch/jbusch/ma/data/dnabert_2_pretrain/mds_data/dev/

echo "Job finished at $(date)"