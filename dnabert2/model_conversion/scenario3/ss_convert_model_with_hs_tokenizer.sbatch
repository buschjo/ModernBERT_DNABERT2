#!/bin/bash

#SBATCH --job-name=convert_modernbert_to_hf
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
##SBATCH --nodelist=kiwinode02
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal
#SBATCH --output=/home/cluster_home/jbusch/ma/sbatch_output/dnabert2_modern/convert_to_hf/scenario3/%j_%x.out
#SBATCH --error=/home/cluster_home/jbusch/ma/sbatch_output/dnabert2_modern/convert_to_hf/scenario3/%j_%x.err


##SBATCH --mail-type=ALL
##SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
conda activate mb_conversion_hs #for hs tokenizer base model

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

python ../convert_mb_ckpt_to_hf_model.py --output-name HF_HS_5M_NRC_150K_199_v6c \
	--output-dir /scratch/jbusch/ma/models/modern_bert/converted_scenario2 \
	--input-checkpoint /scratch/jbusch/ma/models/modern_bert/pretraining_scenario3/MB_HS_5M_NRC_150K_199_v6c/ckpt/ep18-ba150000-rank0.pt \
	--unk-token-id 0 \
	--cls-token-id 1 \
	--sep-token-id 2 \
	--pad-token-id 3 \
	--mask-token-id 4 \
	--max-length 1000 \

# Log completion time
echo "Job finished at $(date)"