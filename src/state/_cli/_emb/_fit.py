import argparse as ap


def add_arguments_fit(parser: ap.ArgumentParser):
    """Add arguments for embedding training CLI."""
    parser.add_argument("--conf", type=str, default=None, help="Path to config YAML file")
    parser.add_argument(
        "hydra_overrides", nargs="*", help="Hydra configuration overrides (e.g., embeddings.current=esm2-cellxgene)"
    )


def run_emb_fit(cfg, args):
    """
    Run state training with the provided config and overrides.
    """
    import logging
    import os
    import sys

    from omegaconf import OmegaConf

    from ...emb.train.trainer import main as trainer_main

    log = logging.getLogger(__name__)

    # Load the base configuration
    if args.conf:
        cfg = OmegaConf.load(args.conf)

    # Process the remaining command line arguments as overrides
    if args.hydra_overrides:
        overrides = OmegaConf.from_dotlist(args.hydra_overrides)
        cfg = OmegaConf.merge(cfg, overrides)

    # Validate required configuration
    if cfg.embeddings.current is None:
        log.error("Gene embeddings are required for training. Please set 'embeddings.current'")
        sys.exit(1)

    if cfg.dataset.current is None:
        log.error("Please set the desired dataset to 'dataset.current'")
        sys.exit(1)

    # Set environment variables for distributed training
    os.environ["MASTER_PORT"] = str(cfg.experiment.port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # localhost for single-node multi-GPU
    
    # WAR: Workaround for sbatch failing when --ntasks-per-node is set.
    # lightning expects this to be set.
    os.environ["SLURM_NTASKS_PER_NODE"] = str(cfg.experiment.num_gpus_per_node)
    
    # Additional environment variables for PyTorch distributed
    if cfg.experiment.num_gpus_per_node > 1:
        os.environ["WORLD_SIZE"] = str(cfg.experiment.num_gpus_per_node * cfg.experiment.num_nodes)
        # Set a higher timeout for distributed initialization
        os.environ["NCCL_TIMEOUT_MS"] = str(cfg.experiment.get("ddp_timeout", 3600) * 1000)

    log.info(f"*************** Training {cfg.experiment.name} ***************")
    log.info(OmegaConf.to_yaml(cfg))

    # Execute the main training logic
    trainer_main(cfg)
