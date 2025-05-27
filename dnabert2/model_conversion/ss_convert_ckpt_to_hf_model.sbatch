#!/bin/bash

#SBATCH --job-name=convert_modernbert_to_hf
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
##SBATCH --nodelist=kiwinode02
#SBATCH --time=00:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal
#SBATCH --output=/home/cluster_home/jbusch/ma/sbatch_output/dnabert2_modern/convert_to_hf/%j_%x.out
#SBATCH --error=/home/cluster_home/jbusch/ma/sbatch_output/dnabert2_modern/convert_to_hf/%j_%x.err


##SBATCH --mail-type=ALL
##SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
# conda activate mb_conversion #for hf tokenizer base model
conda activate mb_conversion_hs #for hs tokenizer base model

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

# python convert_to_hf.py --output-name hf_tokenizer5M_v4 \
# 	--output-dir /scratch/jbusch/ma/models/modern_bert/converted_to_hf \
# 	--input-checkpoint /scratch/jbusch/ma/models/modern_bert/hf_tokenizer_5M_v5/ckpt/ep18-ba150000-rank0.pt \
# 	--cls-token-id 3 \
# 	--sep-token-id 1 \
# 	--pad-token-id 2 \
# 	--mask-token-id 4 \
# 	--max-length 128 \


python convert_to_hf.py --output-name hs_tokenizer5M_v5 \
	--output-dir /scratch/jbusch/ma/models/modern_bert/converted_to_hf \
	--input-checkpoint /scratch/jbusch/ma/models/modern_bert/hs_tokenizer5M_v5/ckpt/ep18-ba150000-rank0.pt \
	--cls-token-id 1 \
	--sep-token-id 2 \
	--pad-token-id 3 \
	--mask-token-id 4 \
	--max-length 30 \

# Log completion time
echo "Job finished at $(date)"