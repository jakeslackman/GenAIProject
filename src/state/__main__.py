import argparse as ap
import os
import tempfile
from pathlib import Path

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from ._cli import (
    add_arguments_emb,
    add_arguments_tx,
    run_emb_fit,
    run_emb_transform,
    run_emb_query,
    run_emb_preprocess,
    run_emb_eval,
    run_tx_infer,
    run_tx_predict,
    run_tx_preprocess_infer,
    run_tx_preprocess_train,
    run_tx_train,
)


def get_args() -> ap.Namespace:
    """Parse known args and return remaining args for Hydra overrides"""
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_arguments_emb(subparsers.add_parser("emb"))
    add_arguments_tx(subparsers.add_parser("tx"))

    # Use parse_known_args to get both known args and remaining args
    return parser.parse_args()


def create_embedded_configs():
    """Create embedded configuration files that are always available"""
    # Define the embedded configurations as strings
    configs = {
        "config.yaml": """# This is a template used in the application to generating the config file for
# training tasks
defaults:
  - data: perturbation
  - model: state
  - training: default
  - wandb: default
  - _self_
  

# output_dir must be an absolute path (so that launch scripts are fully descriptive)
name: debug
output_dir: ./debugging
use_wandb: true
overwrite: false
return_adatas: false
pred_adata_path: null
true_adata_path: null

# don't save hydra output
hydra:
  output_subdir: null
  run:
    dir: .
  job_logging:
    formatters:
      simple:
        format: "[%(levelname)s] %(message)s"  # Simple format for logging
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        level: INFO
        stream: ext://sys.stdout
    root:
      level: INFO
    loggers:
      __main__:
        level: DEBUG
        handlers: [console]
        propagate: false
""",
    "state-defaults.yaml": """# Template for state embedding model
experiment:
  name: vci_pretrain_${loss.name}_${model.nhead}_${model.nlayers}
  local: local
  compiled: false
  deaware: false
  profile:
    enable_profiler: false
    profile_steps:
    - 10
    - 100
    max_steps: 110
  num_epochs: 16
  num_nodes: 1
  num_gpus_per_node: 1
  port: 12400
  val_check_interval: 1000
  limit_val_batches: 100
  ddp_timeout: 3600
  checkpoint:
    path: /scratch/ctc/ML/vci/checkpoint/pretrain
    save_top_k: 4
    monitor: trainer/train_loss
    every_n_train_steps: 1000
wandb:
  enable: true
  project: vci
embeddings:
  current: esm2-cellxgene
  esm2-cellxgene:
    all_embeddings: /large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt
    ds_emb_mapping: /large_storage/ctc/datasets/vci/training/gene_embidx_mapping.torch
    valid_genes_masks: null
    size: 5120
    num: 19790
  esm2-cellxgene-basecamp-tahoe:
    all_embeddings: /large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt
    ds_emb_mapping: /large_storage/ctc/datasets/updated1_gene_embidx_mapping_tahoe_basecamp_cellxgene.torch
    valid_genes_masks: null
    size: 5120
    num: 19790
  esm2-cellxgene-tahoe:
    all_embeddings: /large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt
    ds_emb_mapping: /large_storage/ctc/datasets/updated1_gene_embidx_mapping_tahoe_basecamp_cellxgene.torch
    valid_genes_masks: null
    size: 5120
    num: 19790
  evo2-scbasecamp:
    all_embeddings: /large_storage/ctc/projects/vci/scbasecamp/Evo2/all_species_Evo2.torch
    ds_emb_mapping: /large_storage/ctc/projects/vci/scbasecamp/Evo2/dataset_emb_idx_Evo2_fixed.torch
    valid_genes_masks: /large_storage/ctc/projects/vci/scbasecamp/Evo2/valid_gene_index_Evo2.torch
    size: 4096
    num: 503178
  esm2-scbasecamp:
    all_embeddings: /large_storage/ctc/projects/vci/scbasecamp/ESM2/all_species_ESM2.torch
    ds_emb_mapping: /large_storage/ctc/projects/vci/scbasecamp/ESM2/dataset_emb_idx_ESM2.torch
    valid_genes_masks: /large_storage/ctc/projects/vci/scbasecamp/ESM2/valid_gene_index_ESM2.torch
    size: 1280
    num: 503178
  esm2_3B-scbasecamp:
    all_embeddings: /large_storage/ctc/projects/vci/scbasecamp/ESM2_3B/all_species.torch
    ds_emb_mapping: /large_storage/ctc/projects/vci/scbasecamp/ESM2_3B/dataset_emb_idx.torch
    valid_genes_masks: /large_storage/ctc/projects/vci/scbasecamp/ESM2_3B/valid_gene_index.torch
    size: 2560
    num: 503178
  esm2_3B-scbasecamp_cellxgene:
    all_embeddings: /large_storage/ctc/projects/vci/scbasecamp/ESM2_3B/all_species.torch
    ds_emb_mapping: /home/alishbaimran/scbasecamp/dataset_emb_idx_ESM2_copy.torch
    valid_genes_masks: /home/alishbaimran/scbasecamp/valid_gene_index.torch
    size: 2560
    num: 503178
  cellxgene_test:
    all_embeddings: /large_storage/ctc/userspace/aadduri/cellxgene_test_profile/all_embeddings_cellxgene_test.pt
    ds_emb_mapping: /large_storage/ctc/userspace/aadduri/cellxgene_test_profile/ds_emb_mapping_cellxgene_test.torch
    valid_genes_masks: /large_storage/ctc/userspace/aadduri/cellxgene_test_profile/valid_genes_masks_cellxgene_test.torch
    size: 5120
    num: 19790
validations:
  diff_exp:
    enable: false
    eval_interval_multiple: 10
    obs_pert_col: gene
    obs_filter_label: non-targeting
    top_k_rank: 200
    method: null
    dataset: /large_storage/ctc/datasets/cellxgene/processed/rpe1_top5000_variable.h5ad
    dataset_name: rpe1_top5000_variable
  perturbation:
    enable: false
    eval_interval_multiple: 10
    pert_col: gene
    ctrl_label: non-targeting
    dataset: /large_storage/ctc/datasets/cellxgene/processed/rpe1_top5000_variable.h5ad
    dataset_name: rpe1_top5000_variable
dataset:
  name: vci
  seed: 42
  num_train_workers: 16
  num_val_workers: 4
  current: cellxgene
  cellxgene:
    data_dir: /large_experiments/goodarzilab/mohsen/cellxgene/processed
    ds_type: h5ad
    filter: false
    train: /scratch/ctc/ML/uce/h5ad_train_dataset.csv
    val: /scratch/ctc/ML/uce/h5ad_val_dataset.csv
    num_datasets: 1139
  scbasecamp:
    ds_type: filtered_h5ad
    train: /home/alishbaimran/scbasecamp/scbasecamp_all.csv
    val: /home/alishbaimran/scbasecamp/scbasecamp_all.csv
    filter: true
    filter_by_species: null
  scbasecamp-cellxgene:
    ds_type: filtered_h5ad
    train: /home/alishbaimran/scbasecamp/scBasecamp_cellxgene_all.csv
    val: /home/alishbaimran/scbasecamp/scBasecamp_cellxgene_all.csv
    filter: true
    filter_by_species: null
  scbasecamp-cellxgene-tahoe-filtered:
    ds_type: filtered_h5ad
    train: /large_storage/ctc/userspace/rohankshah/19kfilt_combined_train.csv
    val: /large_storage/ctc/userspace/rohankshah/19kfilt_combined_val.csv
    filter: true
    filter_by_species: null
    num_datasets: 14420
  scbasecamp-cellxgene-tahoe:
    ds_type: h5ad
    train: /large_storage/ctc/datasets/scbasecamp_filtered_tahoe_cellxgene_train.csv
    val: /large_storage/ctc/datasets/scbasecamp_filtered_tahoe_cellxgene_val.csv
    filter: false
    filter_by_species: null
    num_datasets: 15700
  cellxgene-tahoe:
    ds_type: filtered_h5ad
    train: /large_storage/ctc/datasets/tahoe_cellxgene_train.csv
    val: /large_storage/ctc/datasets/tahoe_cellxgene_val.csv
    filter: true
    filter_by_species: null
    num_datasets: 1139
  tahoe:
    ds_type: filtered_h5ad
    train: /scratch/ctc/ML/uce/full_train_datasets.csv
    val: /scratch/ctc/ML/uce/full_train_datasets.csv
    filter: true
    valid_genes_masks: null
  tahoe-h5ad:
    ds_type: filtered_h5ad
    train: /scratch/ctc/ML/uce/h5ad_train_dataset_tahoe.csv
    val: /scratch/ctc/ML/uce/h5ad_val_dataset_tahoe.csv
    filter: true
    valid_genes_masks: null
  pad_length: 2048
  pad_token_idx: 0
  cls_token_idx: 3
  chrom_token_right_idx: 2
  P: 512
  'N': 512
  S: 512
  num_cells: 36238464
  overrides:
    rpe1_top5000_variable: /large_storage/ctc/datasets/vci/validation/rpe1_top5000_variable.h5ad
  cellxgene_test:
    ds_type: h5ad
    train: /large_storage/ctc/userspace/aadduri/cellxgene_test_profile/train_cellxgene_test.csv
    val: /large_storage/ctc/userspace/aadduri/cellxgene_test_profile/val_cellxgene_test.csv
    filter: false
    num_datasets: 1105
tokenizer:
  token_dim: 5120
model:
  name: vci
  batch_size: 128
  emsize: 512
  d_hid: 1024
  nhead: 16
  nlayers: 8
  dropout: 0.1
  output_dim: 512
  use_flash_attention: true
  rda: true
  counts: true
  dataset_correction: true
  ema: false
  ema_decay: 0.999
  ema_update_interval: 1000
  sample_rda: false
  batch_tabular_loss: false
  num_downsample: 1
  variable_masking: true
task:
  mask: 0.2
optimizer:
  max_lr: 1.0e-05
  weight_decay: 0.01
  start: 0.33
  end: 1.0
  max_grad_norm: 0.8
  gradient_accumulation_steps: 8
  reset_lr_on_restart: false
  zclip: false
loss:
  name: tabular
  apply_normalization: false
  kernel: energy
  uniformity: false
""",
        "data/perturbation.yaml": """name: PerturbationDataModule
kwargs:
  toml_config_path: null
  embed_key: null
  output_space: all
  pert_rep: onehot
  basal_rep: sample
  num_workers: 12
  pin_memory: true
  n_basal_samples: 1
  basal_mapping_strategy: random
  should_yield_control_cells: true
  batch_col: gem_group
  pert_col: gene
  cell_type_key: cell_type
  control_pert: DMSO_TF
  map_controls: true # for a control cell, should we use it as the target (learn identity) or sample a control?
  perturbation_features_file: null
  store_raw_basal: false
  int_counts: false
  barcode: true
output_dir: null
debug: true
""",
        "model/state.yaml": """name: state
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512
  blur: 0.05
  hidden_dim: 696      # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 4
  n_decoder_layers: 4
  predict_residual: True
  softplus: True
  freeze_pert: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  batch_encoder: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False
  distributional_loss: energy
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      max_position_embeddings: ${model.kwargs.cell_set_len}
      hidden_size: ${model.kwargs.hidden_dim}
      intermediate_size: 2784
      num_hidden_layers: 8
      num_attention_heads: 12
      num_key_value_heads: 12
      head_dim: 58
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
""",
        "training/default.yaml": """wandb_track: false
weight_decay: 0.0005
batch_size: 16
lr: 1e-4
max_steps: 40000
train_seed: 42
val_freq: 2000
ckpt_every_n_steps: 2000
gradient_clip_val: 10 # 0 means no clipping
loss_fn: mse
devices: 1  # Number of GPUs to use for training
strategy: auto  # DDP strategy for multi-GPU training
""",
        "wandb/default.yaml": """# Generic wandb configuration
# Users should customize these values for their own use
entity: your_entity_name
project: state
local_wandb_dir: ./wandb_logs
tags: []
""",
        "wandb/abhinav.yaml": """entity: arcinstitute
project: vci1
local_wandb_dir: /large_storage/ctc/userspace/aadduri/wandb_dir/vci1
tags: []
""",
        "data/default.yaml": """name: 
kwargs:
  embed_key: X_uce
  embed_size: null
  pert_rep: onehot
  basal_rep: sample
  only_keep_perts_with_expression: false # only keep perturbations for which expression data is available
  esm_perts_only: false
  n_basal_samples: 1
  sampling_random_state: 42
  split_random_state: 42
  normalize: true
  pseudobulk: false
  load_from_path: null
  test_cell_type:
  dataloader_preprocess: null
  k562_rpe1_name: replogle_k562_rpe1_filtered
  jurkat_name: replogle_jurkat_filtered
  hepg2_name: replogle_hepg2_filtered
output_dir: null
debug: true
""",
        "training/scgpt.yaml": """max_steps: 250000
train_seed: 42
val_freq: 5000
test_freq: 9000
gradient_clip_val: 10 # 0 means no clipping

lr: 5e-5
wd: 4e-7
step_size_lr: 25
do_clip_grad: false
batch_size: 256
""",
        "training/scvi.yaml": """max_steps: 250000
train_seed: 42
val_freq: 5000
test_freq: 9000
gradient_clip_val: 10 # 0 means no clipping

n_epochs_kl_warmup: 1e4
lr: 5e-4
wd: 4e-7
step_size_lr: 25
do_clip_grad: false
batch_size: 2048
""",
        "training/cpa.yaml": """max_steps: 250000
train_seed: 42
val_freq: 5000
test_freq: 9000
gradient_clip_val: 10 # 0 means no clipping

n_epochs_kl_warmup: null
n_steps_adv_warmup: 50000
n_steps_pretrain_ae: 50000
adv_steps: null
reg_adv: 15.0
pen_adv: 20.0
lr: 5e-4
wd: 4e-7
adv_lr: 5e-4
adv_wd: 4e-7
step_size_lr: 25
do_clip_grad: false
adv_loss: "cce"
batch_size: 2048
""",
        "model/state_sm.yaml": """name: state
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 128
  blur: 0.05
  hidden_dim: 672  # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 4
  n_decoder_layers: 4
  predict_residual: True
  softplus: True
  freeze_pert: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  batch_encoder: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False
  use_basal_projection: False
  distributional_loss: energy
  gene_decoder_bool: False
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      max_position_embeddings: ${model.kwargs.cell_set_len}
      hidden_size: ${model.kwargs.hidden_dim}
      intermediate_size: 2688
      num_hidden_layers: 4
      num_attention_heads: 8
      num_key_value_heads: 8
      head_dim: 84
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
""",
        "model/state_lg.yaml": """name: state
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512
  blur: 0.05
  hidden_dim: 1488      # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 4
  n_decoder_layers: 4
  predict_residual: True
  softplus: True
  freeze_pert: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  decoder_loss_weight: 1.0
  batch_encoder: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False
  use_basal_projection: False
  distributional_loss: energy
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      max_position_embeddings: ${model.kwargs.cell_set_len}
      hidden_size: ${model.kwargs.hidden_dim}
      intermediate_size: 5952
      num_hidden_layers: 6
      num_attention_heads: 12
      num_key_value_heads: 12
      head_dim: 124
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
""",
        "model/pertsets.yaml": """name: PertSets
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512 # how many cells to group together into a single set of cells
  extra_tokens: 1  # configurable buffer for confidence/special tokens
  decoder_hidden_dims: [1024, 1024, 512]
  blur: 0.05
  hidden_dim: 328 # hidden dimension going into the transformer backbone
  loss: energy
  confidence_token: False # if true, model tries to predict its own confidence
  n_encoder_layers: 4 # number of MLP layers for pert, basal encoders
  n_decoder_layers: 4
  predict_residual: True # if true, predicts the residual in embedding space to the basal cells
  freeze_pert_backbone: False # if true, the perturbation model is frozen
  finetune_vci_decoder: False # if true, the pretrained state decoder is used in finetuning
  residual_decoder: False # if true, the pretrained state decoder is used in finetuning
  batch_encoder: False # if true, batch variables are used
  nb_decoder: False # if true, use a negative binomial decoder
  decoder_loss_weight: 1.0
  use_basal_projection: False
  mask_attn: False # if true, mask the attention
  distributional_loss: energy
  regularization: 0.0
  init_from: null # initial checkpoint to start the model
  transformer_backbone_key: GPT2
  transformer_backbone_kwargs:
      max_position_embeddings: ${model.kwargs.cell_set_len} # llama
      n_positions: ${model.kwargs.cell_set_len} # gpt2
      hidden_size: ${model.kwargs.hidden_dim} # llama
      n_embd: ${model.kwargs.hidden_dim} # gpt2
      n_layer: 8
      n_head: 8
      resid_pdrop: 0.0
      embd_pdrop: 0.0
      attn_pdrop: 0.0
      use_cache: false
""",
        "model/tahoe_best.yaml": """name: PertSets
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512
  blur: 0.05
  hidden_dim: 1440      # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 4
  n_decoder_layers: 4
  predict_residual: True
  softplus: True
  freeze_pert: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  batch_encoder: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False
  distributional_loss: energy
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      max_position_embeddings: ${model.kwargs.cell_set_len}
      hidden_size: ${model.kwargs.hidden_dim}
      intermediate_size: 4416
      num_hidden_layers: 4
      num_attention_heads: 12
      num_key_value_heads: 12
      head_dim: 120
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
""",
        "model/scgpt-genetic.yaml": """name: scGPT-genetic
checkpoint: null
device: cuda

kwargs:
  hidden_dim: 256 # not used
  pad_token: "<pad>"
  special_tokens:
    - "<pad>"
    - "<cls>"
    - "<eoc>"
  
  pad_value: 0
  pert_pad_id: 2

  include_zero_gene: "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
  max_seq_len: 1536

  do_MLM: true  # whether to use masked language modeling, currently it is always on.
  do_CLS: false  # celltype classification objective
  do_CCE: false  # Contrastive cell embedding objective
  do_MVC: false  # Masked value prediction for cell embedding
  do_ECS: false  # Elastic cell similarity objective
  cell_emb_style: "cls"
  mvc_decoder_style: "inner product, detach"
  use_amp: true
  pretrained_path: "/large_storage/goodarzilab/userspace/mohsen/scGPT/scGPT_human/"
  load_param_prefixes:
    - "encoder"
    - "value_encoder"
    - "transformer_encoder"

  # settings for the model
  embsize: 512  # embedding dimension
  d_hid: 512  # dimension of the feedforward network model in nn.TransformerEncoder
  nlayers: 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  nhead: 8  # number of heads in nn.MultiheadAttention
  n_layers_cls: 3
  dropout: 0.2  # dropout probability
  use_fast_transformer: true  # whether to use fast transformer

  expr_transform: none
  perturbation_type: genetic
  seed: 2025
  cell_sentence_len: 2048
  nb_decoder: false
""",
        "model/scgpt-chemical.yaml": """name: scGPT-chemical
checkpoint: null
device: cuda

kwargs:
  hidden_dim: 256 # not used
  pad_token: "<pad>"
  special_tokens:
    - "<pad>"
    - "<cls>"
    - "<eoc>"
  
  pad_value: 0
  pert_pad_id: 2

  include_zero_gene: "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
  max_seq_len: 1536

  do_MLM: true  # whether to use masked language modeling, currently it is always on.
  do_CLS: false  # celltype classification objective
  do_CCE: false  # Contrastive cell embedding objective
  do_MVC: false  # Masked value prediction for cell embedding
  do_ECS: false  # Elastic cell similarity objective
  cell_emb_style: "cls"
  mvc_decoder_style: "inner product, detach"
  use_amp: true
  pretrained_path: "/large_storage/goodarzilab/userspace/mohsen/scGPT/scGPT_human/"
  load_param_prefixes:
    - "encoder"
    - "value_encoder"
    - "transformer_encoder"

  # settings for the model
  embsize: 512  # embedding dimension
  d_hid: 512  # dimension of the feedforward network model in nn.TransformerEncoder
  nlayers: 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  nhead: 8  # number of heads in nn.MultiheadAttention
  n_layers_cls: 3
  dropout: 0.2  # dropout probability
  use_fast_transformer: true  # whether to use fast transformer

  expr_transform: none
  perturbation_type: chemical
  seed: 2025
  cell_sentence_len: 2048
  nb_decoder: false
""",
        "model/scvi.yaml": """name: scVI
checkpoint: null
device: cuda

kwargs:
  n_latent: 84
  recon_loss: zinb
  pert_embeddings: null
  hidden_dim: 256 # not used
  n_hidden_encoder: 512
  n_layers_encoder: 2
  n_hidden_decoder: 512
  n_layers_decoder: 2
  use_batch_norm: both
  use_layer_norm: none
  dropout_rate_encoder: 0.1
  dropout_rate_decoder: 0.1
  expr_transform: none
  seed: 2025
  cell_sentence_len: 512
  nb_decoder: false
""",
        "model/cpa.yaml": """name: CPA
checkpoint: null
device: cuda

kwargs:
  n_latent: 84
  recon_loss: gauss
  pert_embeddings: null
  hidden_dim: 256 # not used
  n_hidden_encoder: 1024
  n_layers_encoder: 5
  n_hidden_decoder: 1024
  n_layers_decoder: 4
  use_batch_norm: decoder
  use_layer_norm: encoder
  dropout_rate_encoder: 0.2
  dropout_rate_decoder: 0.2
  n_hidden_adv: 128
  n_layers_adv: 3
  use_norm_adv: batch
  dropout_rate_adv: 0.25
  variational: False
  expr_transform: none
  seed: 2025
  cell_sentence_len: 512
  nb_decoder: false
""",
        "model/perturb_mean.yaml": """name: perturb_mean
checkpoint: null
device: cuda

kwargs:
  hidden_dim: 256 # not used
  pert_rep: onehot
  basal_rep: sample
  n_basal_samples: 1
  sampling_random_state: 42
  split_random_state: 42
  normalize: true
  pseudobulk: false
  load_from_path: null
  test_cell_type:
  dataloader_preprocess: null
  k562_rpe1_name: replogle_k562_rpe1_filtered
  jurkat_name: replogle_jurkat_filtered
  hepg2_name: replogle_hepg2_filtered
  output_dir: null
  debug: true
""",
        "model/context_mean.yaml": """name: context_mean
checkpoint: null
device: cuda

kwargs:
  hidden_dim: 256 # not used
  pert_rep: onehot
  basal_rep: sample
  n_basal_samples: 1
  sampling_random_state: 42
  split_random_state: 42
  normalize: true
  pseudobulk: false
  load_from_path: null
  test_cell_type:
  dataloader_preprocess: null
  k562_rpe1_name: replogle_k562_rpe1_filtered
  jurkat_name: replogle_jurkat_filtered
  hepg2_name: replogle_hepg2_filtered
  output_dir: null
  debug: true
""",
        "model/celltypemean.yaml": """name: celltypemean
checkpoint: null
device: cuda

kwargs:
  hidden_dim: 256 # not used
  pert_rep: onehot
  basal_rep: sample
  n_basal_samples: 1
  sampling_random_state: 42
  split_random_state: 42
  normalize: true
  pseudobulk: false
  load_from_path: null
  test_cell_type:
  dataloader_preprocess: null
  k562_rpe1_name: replogle_k562_rpe1_filtered
  jurkat_name: replogle_jurkat_filtered
  hepg2_name: replogle_hepg2_filtered
  output_dir: null
  debug: true
""",
        "model/globalsimplesum.yaml": """name: globalsimplesum
checkpoint: null
device: cuda

kwargs:
  hidden_dim: 256 # not used
  pert_rep: onehot
  basal_rep: sample
  n_basal_samples: 1
  sampling_random_state: 42
  split_random_state: 42
  normalize: true
  pseudobulk: false
  load_from_path: null
  test_cell_type:
  dataloader_preprocess: null
  k562_rpe1_name: replogle_k562_rpe1_filtered
  jurkat_name: replogle_jurkat_filtered
  hepg2_name: replogle_hepg2_filtered
  output_dir: null
  debug: true
""",
        "model/embedsum.yaml": """name: embedsum
checkpoint: null
device: cuda

kwargs:
  hidden_dim: 256 # not used
  pert_rep: onehot
  basal_rep: sample
  n_basal_samples: 1
  sampling_random_state: 42
  split_random_state: 42
  normalize: true
  pseudobulk: false
  load_from_path: null
  test_cell_type:
  dataloader_preprocess: null
  k562_rpe1_name: replogle_k562_rpe1_filtered
  jurkat_name: replogle_jurkat_filtered
  hepg2_name: replogle_hepg2_filtered
  output_dir: null
  debug: true
""",
        "model/decoder_only.yaml": """name: decoder_only
checkpoint: null
device: cuda

kwargs:
  hidden_dim: 256 # not used
  pert_rep: onehot
  basal_rep: sample
  n_basal_samples: 1
  sampling_random_state: 42
  split_random_state: 42
  normalize: true
  pseudobulk: false
  load_from_path: null
  test_cell_type:
  dataloader_preprocess: null
  k562_rpe1_name: replogle_k562_rpe1_filtered
  jurkat_name: replogle_jurkat_filtered
  hepg2_name: replogle_hepg2_filtered
  output_dir: null
  debug: true
""",
        "model/old_neuralot.yaml": """name: old_neuralot
checkpoint: null
device: cuda

kwargs:
  hidden_dim: 256 # not used
  pert_rep: onehot
  basal_rep: sample
  n_basal_samples: 1
  sampling_random_state: 42
  split_random_state: 42
  normalize: true
  pseudobulk: false
  load_from_path: null
  test_cell_type:
  dataloader_preprocess: null
  k562_rpe1_name: replogle_k562_rpe1_filtered
  jurkat_name: replogle_jurkat_filtered
  hepg2_name: replogle_hepg2_filtered
  output_dir: null
  debug: true
""",
        "model/tahoe_llama_62089464.yaml": """name: PertSets
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512
  blur: 0.05
  hidden_dim: 1440      # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 4
  n_decoder_layers: 4
  predict_residual: True
  softplus: True
  freeze_pert: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  batch_encoder: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False
  distributional_loss: energy
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      max_position_embeddings: ${model.kwargs.cell_set_len}
      hidden_size: ${model.kwargs.hidden_dim}
      intermediate_size: 4416
      num_hidden_layers: 4
      num_attention_heads: 12
      num_key_value_heads: 12
      head_dim: 120
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
""",
        "model/tahoe_llama_212693232.yaml": """name: PertSets
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512
  blur: 0.05
  hidden_dim: 1440      # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 4
  n_decoder_layers: 4
  predict_residual: True
  softplus: True
  freeze_pert: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  decoder_loss_weight: 1.0
  batch_encoder: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False
  use_basal_projection: False
  distributional_loss: energy
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      max_position_embeddings: ${model.kwargs.cell_set_len}
      hidden_size: ${model.kwargs.hidden_dim}
      intermediate_size: 4416
      num_hidden_layers: 4
      num_attention_heads: 12
      num_key_value_heads: 12
      head_dim: 120
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
"""
    }
    
    # Create configs in the module directory with a unique prefix
    import os
    module_dir = Path(__file__).parent
    config_dir = module_dir / f"state_configs_{os.getpid()}"
    
    # Write all config files
    for config_name, config_content in configs.items():
        config_path = config_dir / config_name
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)
    
    return str(config_dir)


def load_hydra_config(method: str, overrides: list[str] = None) -> DictConfig:
    """Load Hydra config with optional overrides"""
    if overrides is None:
        overrides = []

    # Create embedded configs in the module directory
    config_dir = create_embedded_configs()
    
    try:
        # Get just the directory name for relative path
        config_dir_name = Path(config_dir).name
        
        # Initialize Hydra with the configs directory
        with initialize(version_base=None, config_path=config_dir_name):
            match method:
                case "emb":
                    cfg = compose(config_name="state-defaults", overrides=overrides)
                case "tx":
                    cfg = compose(config_name="config", overrides=overrides)
                case _:
                    raise ValueError(f"Unknown method: {method}")
        return cfg
    finally:
        # Clean up config directory
        import shutil
        shutil.rmtree(config_dir, ignore_errors=True)


def show_hydra_help(method: str):
    """Show Hydra configuration help with all parameters"""
    from omegaconf import OmegaConf
    
    # Load the default config to show structure
    cfg = load_hydra_config(method)
    
    print("Hydra Configuration Help")
    print("=" * 50)
    print(f"Configuration for method: {method}")
    print()
    print("Full configuration structure:")
    print(OmegaConf.to_yaml(cfg))
    print()
    print("Usage examples:")
    print("  Override single parameter:")
    print(f"    uv run state tx train data.batch_size=64")
    print()
    print("  Override nested parameter:")
    print(f"    uv run state tx train model.kwargs.hidden_dim=512")
    print()
    print("  Override multiple parameters:")
    print(f"    uv run state tx train data.batch_size=64 training.lr=0.001")
    print()
    print("  Change config group:")
    print(f"    uv run state tx train data=custom_data model=custom_model")
    print()
    print("Available config groups:")
    print("  data: perturbation")
    print("  model: pertsets, state_sm")
    print("  training: default")
    print("  wandb: default")
    
    exit(0)


def main():
    args = get_args()

    match args.command:
        case "emb":
            match args.subcommand:
                case "fit":
                    cfg = load_hydra_config("emb", args.hydra_overrides)
                    run_emb_fit(cfg, args)
                case "transform":
                    run_emb_transform(args)
                case "query":
                    run_emb_query(args)
                case "preprocess":
                    run_emb_preprocess(args)
                case "eval":
                    run_emb_eval(args)
        case "tx":
            match args.subcommand:
                case "train":
                    if hasattr(args, 'help') and args.help:
                        # Show Hydra configuration help
                        show_hydra_help("tx")
                    else:
                        # Load Hydra config with overrides for sets training
                        cfg = load_hydra_config("tx", args.hydra_overrides)
                        run_tx_train(cfg)
                case "predict":
                    # For now, predict uses argparse and not hydra
                    run_tx_predict(args)
                case "infer":
                    # Run inference using argparse, similar to predict
                    run_tx_infer(args)
                case "preprocess_train":
                    # Run preprocessing using argparse
                    run_tx_preprocess_train(args.adata, args.output, args.num_hvgs)
                case "preprocess_infer":
                    # Run inference preprocessing using argparse
                    run_tx_preprocess_infer(args.adata, args.output, args.control_condition, args.pert_col, args.seed)


if __name__ == "__main__":
    main()
