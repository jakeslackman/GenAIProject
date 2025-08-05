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
  - model: pertsets
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
        "state-defaults.yaml": """# Default configuration for state embeddings
defaults:
  - _self_

# Model configuration
model:
  name: "state"
  kwargs:
    hidden_dim: 512
    num_layers: 6
    num_heads: 8
    dropout: 0.1
    activation: "gelu"

# Data configuration
data:
  batch_size: 32
  num_workers: 4
  kwargs:
    transform: "log-normalize"
    train_task: "perturbation"

# Training configuration
training:
  max_epochs: 100
  lr: 0.001
  weight_decay: 0.0001
  train_seed: 42

# Output configuration
output_dir: "./outputs"
name: "default_run"
overwrite: false

# Logging configuration
use_wandb: false
""",
        "data/perturbation.yaml": """# Data configuration for perturbation tasks
kwargs:
  toml_config_path: null
  num_workers: 4
  batch_col: "batch"
  pert_col: "perturbation"
  cell_type_key: "cell_type"
  control_pert: "control"
  perturbation_features_file: null
  transform: "log-normalize"
  train_task: "perturbation"
  dataset_cls: "PerturbationDataset"
""",
        "model/pertsets.yaml": """# Model configuration for perturbation sets
name: "state_sm"
kwargs:
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1
  activation: "gelu"
  cell_set_len: 1
  transformer_backbone_kwargs:
    n_positions: 1024
    max_position_embeddings: 1024
""",
        "model/state_sm.yaml": """# Model configuration for state_sm
name: "state_sm"
kwargs:
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1
  activation: "gelu"
  cell_set_len: 1
  transformer_backbone_kwargs:
    n_positions: 1024
    max_position_embeddings: 1024
""",
        "training/default.yaml": """# Default training configuration
max_steps: 10000
ckpt_every_n_steps: 1000
train_seed: 42
batch_size: 32
lr: 0.001
weight_decay: 0.0001
max_epochs: 100
""",
        "wandb/default.yaml": """# Generic wandb configuration
# Users should customize these values for their own use
entity: your_entity_name
project: state
local_wandb_dir: ./wandb_logs
tags: []
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
