#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 8:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=eval_output_%j.log      # Standard output file (%j will be replaced with job ID)
#SBATCH --error=eval_error_%j.log        # Standard error file (%j will be replaced with job ID)

# load conda
module load anaconda3/2024.10-1
module load cuda/12.4.0

# activate environment
conda activate vcc

cell-eval run -ap your_adata_file_here \
 -ar your_reference_adata_file_here \
 --num-threads 64 --profile full -o your_output_directory_here
