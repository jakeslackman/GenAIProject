#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 8:00:00
#SBATCH --gpus=h100-80:1
#SBATCH --output=embed_output_%j.log
#SBATCH --error=embed_error_%j.log

# Load modules
module load anaconda3/2024.10-1
module load cuda/12.4.0

# Activate environment
conda activate vcc
nvidia-smi

# Embed for training input
FILES="competition_train k562_gwps rpe1 jurkat k562 hepg2"

for file in $FILES; do  
  echo "Embedding ${file}.h5..."
  uv run state emb transform \
    --model-folder your_model_folder_here \
    --checkpoint your_checkpoint_here \
    --input your_input_file_here \
    --output your_output_file_here
done
echo "Embedding completed for: $FILES"

# Embed for validation template input
uv run state emb transform \
  --model-folder your_model_folder_here \
  --checkpoint your_checkpoint_here \
  --input your_input_file_here \
  --output your_output_file_here
