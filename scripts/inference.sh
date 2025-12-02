#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 8:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=inference_output_%j.log      # Standard output file (%j will be replaced with job ID)
#SBATCH --error=inference_error_%j.log        # Standard error file (%j will be replaced with job ID)

# load conda
module load anaconda3/2024.10-1
module load cuda/12.4.0

# activate environment
conda activate vcc

# execute Inference
uv run state tx infer \
  --output your_output_file_here \
  --model-dir your_model_directory_here \
  --checkpoint your_trained_model_checkpoint_here \
  --adata your_adata_file_here \
  --embed-key X_state \
  --pert-col target_gene
