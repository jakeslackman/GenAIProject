#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=st_output_%j.log      # Standard output file (%j will be replaced with job ID)
#SBATCH --error=st_error_%j.log        # Standard error file (%j will be replaced with job ID)

# load conda
module load anaconda3/2024.10-1
module load cuda/12.4.0

# activate environment
conda activate vcc

export WANDB_API_KEY=your_wandb_api_key_here
nvidia-smi

# execute training with Gene Adapter (ESM2 + scGenePT embeddings) + ST-Tahoe pre-trained weights
# Gene Adapter fuses ESM2 perturbation features with scGenePT gene embeddings
uv run state tx train \
  data.kwargs.toml_config_path=your_toml_config_path_here \
  data.kwargs.num_workers=8 \
  data.kwargs.output_space="gene" \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file=your_perturbation_features_file_here \
  training.max_steps=40000 \
  training.ckpt_every_n_steps=10000 \
  training.lr=1e-5 \
  model=your_model_name_here \
  model.kwargs.batch_encoder=true \
  model.kwargs.cell_set_len=128 \
  model.kwargs.init_from=your_pretrained_model_checkpoint_here \
  wandb.project=your_wandb_project_here \
  wandb.entity=your_wandb_entity_here \
  output_dir=your_output_directory_here \
  name=your_experiment_name_here \
  # model.kwargs.use_gene_adapter=true \
  # model.kwargs.gene_embeddings_file="/ocean/projects/cis250160p/dkim17/competition/GO_C_gene_embeddings-gpt3.5-ada-concat.pickle" \
  # model.kwargs.gene_emb_dim=1536 \
