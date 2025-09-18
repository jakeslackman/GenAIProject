import argparse as ap


def add_arguments_predict(parser: ap.ArgumentParser):
    """
    CLI for evaluation using cell-eval metrics.
    """

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )

    parser.add_argument(
        "--test-time-finetune",
        type=int,
        default=0,
        help="If >0, run test-time fine-tuning for the specified number of epochs on only control cells.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "minimal", "de", "anndata"],
        help="run all metrics, minimal, only de metrics, or only output adatas",
    )

    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="If set, only run prediction without evaluation metrics.",
    )

    parser.add_argument(
        "--shared-only",
        action="store_true",
        help=("If set, restrict predictions/evaluation to perturbations shared between train and test (train ∩ test)."),
    )

    parser.add_argument(
        "--eval-train-data",
        action="store_true",
        help="If set, evaluate the model on the training data rather than on the test data.",
    )

    # Optional: apply directional shift on a chosen index using control distributions
    parser.add_argument(
        "--shift-index",
        type=int,
        default=None,
        help="If set, apply a ±2σ shift to this index across core_cells using control distributions.",
    )
    parser.add_argument(
        "--shift-direction",
        type=str,
        default=None,
        choices=["up", "down"],
        help="Direction for the 2σ shift applied to --shift-index. Requires --shift-index.",
    )

    parser.add_argument(
        "--test-time-heat-map",
        action="store_true",
        help="If set, run test-time heat map analysis with position upregulation.",
    )
    parser.add_argument(
        "--heatmap-output-path",
        type=str,
        default=None,
        help="Path to save the matplotlib heatmap visualization. If not provided, defaults to <results-dir>/position_upregulation_heatmap.png",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results. If not provided, defaults to <output-dir>/eval_<checkpoint>",
    )


def run_tx_predict(args: ap.ArgumentParser):
    import logging
    import os
    import sys

    import anndata
    import lightning.pytorch as pl
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    import json
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Cell-eval for metrics computation
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype
    from cell_load.data_modules import PerturbationDataModule
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.multiprocessing.set_sharing_strategy("file_system")

    def run_test_time_finetune(model, dataloader, ft_epochs, control_pert, device):
        """
        Perform test-time fine-tuning on only control cells.
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        logger.info(f"Starting test-time fine-tuning for {ft_epochs} epoch(s) on control cells only.")
        for epoch in range(ft_epochs):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"Finetune epoch {epoch + 1}/{ft_epochs}", leave=True)
            for batch in pbar:
                # Check if this batch contains control cells
                first_pert = (
                    batch["pert_name"][0] if isinstance(batch["pert_name"], list) else batch["pert_name"][0].item()
                )
                if first_pert != control_pert:
                    continue

                # Move batch data to device
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                optimizer.zero_grad()
                loss = model.training_step(batch, batch_idx=0, padded=False)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            logger.info(f"Finetune epoch {epoch + 1}/{ft_epochs}, mean loss: {mean_loss}")
        model.eval()

    def load_config(cfg_path: str) -> dict:
        """Load config from the YAML file that was dumped during training."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 2. Find run output directory & load data module
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    data_module_path = os.path.join(run_output_dir, "data_module.torch")
    if not os.path.exists(data_module_path):
        raise FileNotFoundError(f"Could not find data module at {data_module_path}?")
    data_module = PerturbationDataModule.load_state(data_module_path)
    data_module.setup(stage="test")
    logger.info("Loaded data module from %s", data_module_path)

    # Seed everything
    pl.seed_everything(cfg["training"]["train_seed"])

    # 3. Load the trained model
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    # Determine model class and load
    model_class_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]

    # Import the correct model class
    if model_class_name.lower() == "embedsum":
        from state.tx.models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_class_name.lower() == "old_neuralot":
        from state.tx.models.old_neural_ot import OldNeuralOTPerturbationModel

        ModelClass = OldNeuralOTPerturbationModel
    elif model_class_name.lower() in ["neuralot", "pertsets", "state"]:
        from state.tx.models.state_transition import StateTransitionPerturbationModel

        ModelClass = StateTransitionPerturbationModel

    elif model_class_name.lower() in ["globalsimplesum", "perturb_mean"]:
        from state.tx.models.perturb_mean import PerturbMeanPerturbationModel

        ModelClass = PerturbMeanPerturbationModel
    elif model_class_name.lower() in ["celltypemean", "context_mean"]:
        from state.tx.models.context_mean import ContextMeanPerturbationModel

        ModelClass = ContextMeanPerturbationModel
    elif model_class_name.lower() == "decoder_only":
        from state.tx.models.decoder_only import DecoderOnlyPerturbationModel

        ModelClass = DecoderOnlyPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }

    model = ModelClass.load_from_checkpoint(checkpoint_path, **model_init_kwargs)
    model.eval()
    logger.info("Model loaded successfully.")

    # 4. Test-time fine-tuning if requested
    data_module.batch_size = 1
    if args.test_time_finetune > 0:
        control_pert = data_module.get_control_pert()
        if args.eval_train_data:
            test_loader = data_module.train_dataloader(test=True)
        else:
            test_loader = data_module.test_dataloader()

        run_test_time_finetune(
            model, test_loader, args.test_time_finetune, control_pert, device=next(model.parameters()).device
        )
        logger.info("Test-time fine-tuning complete.")

    # 5. Run inference on test set
    data_module.setup(stage="test")
    if args.eval_train_data:
        scan_loader = data_module.train_dataloader(test=True)
    else:
        scan_loader = data_module.test_dataloader()

    if scan_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    logger.info("Preparing a fixed batch of 64 control cells (core_cells) and enumerating perturbations...")

    # Helper to normalize values to python lists
    def _to_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, torch.Tensor):
            try:
                return [x.item() if x.dim() == 0 else x for x in value]
            except Exception:
                return value.tolist()
        return [value]

    control_pert = data_module.get_control_pert()

    # Collect unique perturbation names from the loader without running the model
    unique_perts = []
    seen_perts = set()
    for batch in scan_loader:
        names = _to_list(batch.get("pert_name", []))
        for n in names:
            if isinstance(n, torch.Tensor):
                try:
                    n = n.item()
                except Exception:
                    n = str(n)
            if n not in seen_perts:
                seen_perts.add(n)
                unique_perts.append(n)

    if control_pert in seen_perts:
        logger.info(f"Found {len(unique_perts)} total perturbations (including control '{control_pert}').")
    else:
        logger.warning("Control perturbation not observed in test loader perturbation names.")

    # Build a single fixed batch of exactly 64 control cells
    target_core_n = 64
    core_cells = None
    accum = {}

    def _append_field(store, key, value):
        if key not in store:
            store[key] = []
        store[key].append(value)

    # Iterate again to collect control cells only
    for batch in scan_loader:
        names = _to_list(batch.get("pert_name", []))
        # Build a mask for control entries when possible
        mask = None
        if len(names) > 0:
            mask = torch.tensor([str(x) == str(control_pert) for x in names], dtype=torch.bool)
            if mask.sum().item() == 0:
                continue
        else:
            # If no names provided in batch, skip (cannot verify control)
            continue

        # Slice each tensor field by mask and accumulate until we have 64
        current_count = 0 if "_count" not in accum else accum["_count"]
        take = min(target_core_n - current_count, int(mask.sum().item()))
        if take <= 0:
            break

        # Identify keys to carry forward; prefer tensors and essential metadata
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                try:
                    vsel = v[mask][:take].detach().clone()
                except Exception:
                    # fallback: try first dimension slice
                    vsel = v[:take].detach().clone()
                _append_field(accum, k, vsel)
            else:
                # For non-tensor fields, convert to list and slice by mask when possible
                vals = _to_list(v)
                try:
                    selected_vals = [vals[i] for i, m in enumerate(mask.tolist()) if m][:take]
                except Exception:
                    selected_vals = vals[:take]
                _append_field(accum, k, selected_vals)

        accum["_count"] = current_count + take
        if accum["_count"] >= target_core_n:
            break

    if accum.get("_count", 0) < target_core_n:
        raise RuntimeError(f"Could not assemble {target_core_n} control cells for core_cells; gathered {accum.get('_count', 0)}.")

    # Collate accumulated pieces into a single batch dict of length 64
    core_cells = {}
    for k, parts in accum.items():
        if k == "_count":
            continue
        if len(parts) == 1:
            val = parts[0]
        else:
            if isinstance(parts[0], torch.Tensor):
                val = torch.cat(parts, dim=0)
            else:
                merged = []
                for p in parts:
                    merged.extend(_to_list(p))
                val = merged
        # Ensure final length == 64
        if isinstance(val, torch.Tensor):
            core_cells[k] = val[:target_core_n]
        else:
            core_cells[k] = _to_list(val)[:target_core_n]

    logger.info(f"Constructed core_cells batch with size {target_core_n}.")

    # Compute distributions for each position across ALL control cells in the test loader
    # Strategy: determine a 2D vector key from the first batch, then aggregate all control rows
    vector_key_candidates = ["ctrl_cell_emb", "pert_cell_emb", "X"]
    dist_source_key = None
    # Find key by peeking one batch
    for b in scan_loader:
        for cand in vector_key_candidates:
            if cand in b and isinstance(b[cand], torch.Tensor) and b[cand].dim() == 2:
                dist_source_key = cand
                break
        if dist_source_key is None:
            # fallback: any 2D tensor
            for k, v in b.items():
                if isinstance(v, torch.Tensor) and v.dim() == 2:
                    dist_source_key = k
                    break
        # break after first batch inspected
        break
    if dist_source_key is None:
        raise RuntimeError("Could not find a 2D tensor in test loader batches to compute per-dimension distributions.")

    # Aggregate all control rows for the chosen key
    control_rows = []
    for batch in scan_loader:
        names = _to_list(batch.get("pert_name", []))
        if len(names) == 0:
            continue
        mask = torch.tensor([str(x) == str(control_pert) for x in names], dtype=torch.bool)
        if mask.sum().item() == 0:
            continue
        vec = batch.get(dist_source_key, None)
        if isinstance(vec, torch.Tensor) and vec.dim() == 2:
            try:
                control_rows.append(vec[mask].detach().cpu().float())
            except Exception:
                # fallback: take leading rows equal to mask sum
                take = int(mask.sum().item())
                if take > 0:
                    control_rows.append(vec[:take].detach().cpu().float())

    if len(control_rows) == 0:
        raise RuntimeError("No control rows found to compute distributions.")

    control_vectors_all = torch.cat(control_rows, dim=0)  # [Nc, D]
    D = control_vectors_all.shape[1]
    if D != 2000:
        logger.warning(f"Expected vector dimension 2000; found {D}. Proceeding with {D} dimensions.")

    control_mean = control_vectors_all.mean(dim=0)
    control_std = control_vectors_all.std(dim=0, unbiased=False).clamp_min(1e-8)

    # Save distributions to results directory later; keep in scope for optional shifting
    distributions = {
        "key": dist_source_key,
        "mean": control_mean.numpy(),
        "std": control_std.numpy(),
        "dim": int(D),
        "num_cells": int(control_vectors_all.shape[0]),
    }

    def apply_shift_to_core_cells(index: int, upregulate: bool):
        """Apply ±2σ shift at a single index across all vectors in core_cells.

        - index: integer in [0, D)
        - upregulate: True for +2σ, False for -2σ
        Operates in-place on the tensor stored at distributions['key'] inside core_cells.
        """
        nonlocal core_cells, distributions
        if index < 0 or index >= distributions["dim"]:
            raise ValueError(f"Index {index} is out of bounds for dimension {distributions['dim']}")
        shift_value = (2.0 if upregulate else -2.0) * float(distributions["std"][index])
        key = distributions["key"]
        tensor = core_cells[key]
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
            raise RuntimeError(f"Core cell field '{key}' is not a 2D tensor")
        tensor[:, index] = tensor[:, index] + shift_value
        core_cells[key] = tensor

    # Optionally apply shift based on CLI flags before running inference
    if args.shift_index is not None:
        if args.shift_direction is None:
            raise ValueError("--shift-direction is required when --shift-index is provided")
        apply_shift_to_core_cells(index=int(args.shift_index), upregulate=(args.shift_direction == "up"))
        logger.info(f"Applied 2σ {'up' if args.shift_direction=='up' else 'down'} shift at index {int(args.shift_index)} across core_cells")

    # Prepare output arrays sized by num_perts * 64
    # Keep all perturbations including control to be explicit
    perts_order = list(unique_perts)
    num_cells = len(perts_order) * target_core_n
    output_dim = var_dims["output_dim"]
    gene_dim = var_dims["gene_dim"]
    hvg_dim = var_dims["hvg_dim"]

    logger.info("Generating predictions: one forward pass per perturbation on core_cells...")
    device = next(model.parameters()).device

    # Phase 1: Normal inference on all perturbations
    final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
    final_reals = np.empty((num_cells, output_dim), dtype=np.float32)
    
    # Phase 2: Store normal predictions for distance computation
    normal_preds_per_pert = {}  # pert_name -> [64, output_dim] array

    store_raw_expression = (
        data_module.embed_key is not None
        and data_module.embed_key != "X_hvg"
        and cfg["data"]["kwargs"]["output_space"] == "gene"
    ) or (data_module.embed_key is not None and cfg["data"]["kwargs"]["output_space"] == "all")

    final_X_hvg = None
    final_pert_cell_counts_preds = None
    if store_raw_expression:
        # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
        if cfg["data"]["kwargs"]["output_space"] == "gene":
            final_X_hvg = np.empty((num_cells, hvg_dim), dtype=np.float32)
            final_pert_cell_counts_preds = np.empty((num_cells, hvg_dim), dtype=np.float32)
        if cfg["data"]["kwargs"]["output_space"] == "all":
            final_X_hvg = np.empty((num_cells, gene_dim), dtype=np.float32)
            final_pert_cell_counts_preds = np.empty((num_cells, gene_dim), dtype=np.float32)

    current_idx = 0

    # Initialize aggregation variables directly
    all_pert_names = []
    all_celltypes = []
    all_gem_groups = []
    all_pert_barcodes = []
    all_ctrl_barcodes = []

    with torch.no_grad():
        for p_idx, pert in enumerate(tqdm(perts_order, desc="Predicting", unit="pert")):
            # Build a batch by copying core_cells and swapping perturbation
            batch = {}
            for k, v in core_cells.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                else:
                    batch[k] = list(v)

            # Overwrite perturbation fields to target pert
            if "pert_name" in batch:
                batch["pert_name"] = [pert for _ in range(target_core_n)]
            # Best-effort: update any index fields if present and mapping exists
            try:
                if "pert_idx" in batch and hasattr(data_module, "get_pert_index"):
                    idx_val = int(data_module.get_pert_index(pert))
                    batch["pert_idx"] = torch.tensor([idx_val] * target_core_n, device=device)
            except Exception:
                pass

            batch_preds = model.predict_step(batch, p_idx, padded=False)

            # Extract metadata and data directly from batch_preds
            # Handle pert_name
            batch_pert_names = []
            if isinstance(batch_preds["pert_name"], list):
                all_pert_names.extend(batch_preds["pert_name"])
                batch_pert_names = batch_preds["pert_name"]
            else:
                all_pert_names.append(batch_preds["pert_name"])
                batch_pert_names = [batch_preds["pert_name"]]

            if "pert_cell_barcode" in batch_preds:
                if isinstance(batch_preds["pert_cell_barcode"], list):
                    all_pert_barcodes.extend(batch_preds["pert_cell_barcode"])
                    all_ctrl_barcodes.extend(batch_preds.get("ctrl_cell_barcode", [None] * len(batch_preds["pert_cell_barcode"])) )
                else:
                    all_pert_barcodes.append(batch_preds["pert_cell_barcode"])
                    all_ctrl_barcodes.append(batch_preds.get("ctrl_cell_barcode", None))

            # Handle celltype_name
            if isinstance(batch_preds["celltype_name"], list):
                all_celltypes.extend(batch_preds["celltype_name"])
            else:
                all_celltypes.append(batch_preds["celltype_name"])

            # Handle gem_group
            if isinstance(batch_preds["batch"], list):
                all_gem_groups.extend([str(x) for x in batch_preds["batch"]])
            elif isinstance(batch_preds["batch"], torch.Tensor):
                all_gem_groups.extend([str(x) for x in batch_preds["batch"].cpu().numpy()])
            else:
                all_gem_groups.append(str(batch_preds["batch"]))

            batch_pred_np = batch_preds["preds"].detach().cpu().numpy().astype(np.float32)
            batch_real_np = batch_preds["pert_cell_emb"].detach().cpu().numpy().astype(np.float32)
            batch_size = batch_pred_np.shape[0]
            final_preds[current_idx : current_idx + batch_size, :] = batch_pred_np
            final_reals[current_idx : current_idx + batch_size, :] = batch_real_np
            
            # Store normal predictions for this perturbation for distance computation
            normal_preds_per_pert[pert] = batch_pred_np.copy()
            
            current_idx += batch_size

            # Handle X_hvg for HVG space ground truth
            if final_X_hvg is not None:
                batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                final_X_hvg[current_idx - batch_size : current_idx, :] = batch_real_gene_np

            # Handle decoded gene predictions if available
            if final_pert_cell_counts_preds is not None:
                batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                final_pert_cell_counts_preds[current_idx - batch_size : current_idx, :] = batch_gene_pred_np

    logger.info("Phase 1 complete: Normal inference on all perturbations.")
    
    # Phase 2: Run inference with GO MF pathway groups upregulated (only if requested)
    if args.test_time_heat_map:
        logger.info("Phase 2: Loading GO MF pathway annotations and running pathway-based upregulation...")
        
        # Load gene annotations
        import pickle
        with open('/home/dhruvgautam/gene_annotations_1_2000.pkl', 'rb') as f:
            gene_annotations = pickle.load(f)
        
        # Group genes by GO MF pathways
        from collections import defaultdict
        pathway_to_genes = defaultdict(list)
        
        for idx, data in gene_annotations.items():
            mf_paths = data['go_cc_paths']
            if mf_paths:  # If gene has MF pathways
                pathways = mf_paths.split(';')
                for pathway in pathways:
                    # Convert 1-indexed to 0-indexed
                    pathway_to_genes[pathway].append(idx - 1)
        
        # Filter out pathways with too few genes (less than 3) to avoid noise
        filtered_pathways = {pathway: genes for pathway, genes in pathway_to_genes.items() if len(genes) >= 3}
        
        logger.info(f"Found {len(pathway_to_genes)} total GO MF pathways")
        logger.info(f"Using {len(filtered_pathways)} pathways with 3+ genes for upregulation")
        
        # Initialize heatmap array: [num_pathways, num_perturbations]
        num_pathways = len(filtered_pathways)
        heatmap_distances = np.zeros((num_pathways, len(perts_order)), dtype=np.float32)
        pathway_names = list(filtered_pathways.keys())
        
        # Create a copy of core_cells for upregulation experiments
        original_core_cells = {}
        for k, v in core_cells.items():
            if isinstance(v, torch.Tensor):
                original_core_cells[k] = v.clone()
            else:
                original_core_cells[k] = v.copy() if hasattr(v, 'copy') else v
        
        def apply_pathway_shift_to_core_cells(gene_indices: list, upregulate: bool):
            """Apply ±2σ shift to multiple gene indices across all vectors in core_cells.
            
            - gene_indices: list of 0-indexed gene positions
            - upregulate: True for +2σ, False for -2σ
            Operates in-place on the tensor stored at distributions['key'] inside core_cells.
            """
            nonlocal core_cells, distributions
            key = distributions['key']
            tensor = core_cells[key]
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
                raise RuntimeError(f"Core cell field '{key}' is not a 2D tensor")
            
            for idx in gene_indices:
                if 0 <= idx < distributions["dim"]:
                    shift_value = (2.0 if upregulate else -2.0) * float(distributions["std"][idx])
                    tensor[:, idx] = tensor[:, idx] + shift_value
        
        with torch.no_grad():
            for pathway_idx, (pathway_name, gene_indices) in enumerate(tqdm(filtered_pathways.items(), desc="Upregulating pathways", unit="pathway")):
                # Apply upregulation to all genes in this pathway
                apply_pathway_shift_to_core_cells(gene_indices, upregulate=True)
                
                # Run inference for all perturbations with this pathway upregulated
                for p_idx, pert in enumerate(perts_order):
                    # Build batch by copying upregulated core_cells
                    batch = {}
                    for k, v in core_cells.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                        else:
                            batch[k] = list(v)
                    
                    # Overwrite perturbation fields
                    if "pert_name" in batch:
                        batch["pert_name"] = [pert for _ in range(target_core_n)]
                    try:
                        if "pert_idx" in batch and hasattr(data_module, "get_pert_index"):
                            idx_val = int(data_module.get_pert_index(pert))
                            batch["pert_idx"] = torch.tensor([idx_val] * target_core_n, device=device)
                    except Exception:
                        pass
                    
                    # Get predictions with upregulated pathway
                    batch_preds = model.predict_step(batch, p_idx, padded=False)
                    upregulated_preds = batch_preds["preds"].detach().cpu().numpy().astype(np.float32)
                    
                    # Compute euclidean distance between normal and upregulated predictions
                    normal_preds = normal_preds_per_pert[pert]  # [64, output_dim]
                    distance = np.linalg.norm(upregulated_preds - normal_preds, axis=1).mean()  # Mean across 64 cells
                    heatmap_distances[pathway_idx, p_idx] = distance
                
                # Restore original core_cells for next pathway
                for k, v in original_core_cells.items():
                    if isinstance(v, torch.Tensor):
                        core_cells[k] = v.clone()
                    else:
                        core_cells[k] = v.copy() if hasattr(v, 'copy') else v
        
        logger.info("Phase 2 complete: Upregulated inference for all GO MF pathways.")
        
        # Save heatmap data
        try:
            # Determine results directory
            if args.results_dir is not None:
                results_dir = args.results_dir
            else:
                results_dir = os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint))
            os.makedirs(results_dir, exist_ok=True)
            
            heatmap_path = os.path.join(results_dir, "go_cc_pathway_upregulation_heatmap.npy")
            np.save(heatmap_path, heatmap_distances)
            
            # Save pathway information
            pathway_info_path = os.path.join(results_dir, "go_cc_pathways_info.json")
            pathway_info = {
                "pathway_names": pathway_names,
                "pathway_to_genes": {pathway: genes for pathway, genes in filtered_pathways.items()},
                "total_pathways": len(pathway_to_genes),
                "filtered_pathways": len(filtered_pathways),
                "min_genes_per_pathway": 3
            }
            with open(pathway_info_path, "w") as f:
                json.dump(pathway_info, f, indent=2)
            
            # Save metadata for the heatmap
            heatmap_meta = {
                "shape": [num_pathways, len(perts_order)],
                "description": "Euclidean distance heatmap: rows=GO MF pathways, cols=perturbations",
                "perturbations": perts_order,
                "pathway_names": pathway_names,
                "distance_type": "mean_euclidean_norm_across_64_cells",
                "upregulation": "2_std_deviation_shift_per_pathway_group"
            }
            heatmap_meta_path = os.path.join(results_dir, "go_cc_pathway_upregulation_heatmap.meta.json")
            with open(heatmap_meta_path, "w") as f:
                json.dump(heatmap_meta, f, indent=2)
            
            logger.info(f"Saved GO MF pathway upregulation heatmap to {heatmap_path}")
            logger.info(f"Heatmap shape: {heatmap_distances.shape} (pathways x perturbations)")
        except Exception as e:
            logger.warning(f"Failed to save heatmap data: {e}")
        
        # Create and save matplotlib heatmap visualization
        try:
            # Determine output path for heatmap image
            if args.heatmap_output_path is not None:
                heatmap_img_path = args.heatmap_output_path
            else:
                heatmap_img_path = os.path.join(results_dir, "go_cc_pathway_upregulation_heatmap.png")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(heatmap_img_path), exist_ok=True)
            
            # Create the heatmap with appropriate size
            fig_width = max(12, len(perts_order) * 0.3)
            fig_height = max(8, num_pathways * 0.05)  # Smaller height per pathway since we have fewer rows
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create heatmap with proper labels
            im = ax.imshow(heatmap_distances, cmap='viridis', aspect='auto')
            
            # Set labels and title
            ax.set_xlabel('Perturbations')
            ax.set_ylabel('GO MF Pathways')
            ax.set_title('GO MF Pathway Upregulation Impact Heatmap\n(Euclidean Distance from Normal Predictions)')
            
            # Set x-axis labels (perturbations)
            ax.set_xticks(range(len(perts_order)))
            ax.set_xticklabels(perts_order, rotation=45, ha='right', fontsize=8)
            
            # Set y-axis labels (pathways) - show pathway names, truncated if too long
            ax.set_yticks(range(num_pathways))
            truncated_pathway_names = []
            for pathway_name in pathway_names:
                # Remove GOMF_ prefix and truncate long names
                clean_name = pathway_name.replace('GOMF_', '')
                if len(clean_name) > 30:
                    clean_name = clean_name[:27] + '...'
                truncated_pathway_names.append(clean_name)
            ax.set_yticklabels(truncated_pathway_names, fontsize=6)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mean Euclidean Distance', rotation=270, labelpad=20)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(heatmap_img_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
            
            logger.info(f"Saved GO MF pathway heatmap visualization to {heatmap_img_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create heatmap visualization: {e}")
    else:
        logger.info("Skipping heatmap analysis (--test-time-heat-map not set)")

    logger.info("Creating anndatas from predictions from manual loop...")

    # Build pandas DataFrame for obs and var
    df_dict = {
        data_module.pert_col: all_pert_names,
        data_module.cell_type_key: all_celltypes,
        data_module.batch_col: all_gem_groups,
    }

    if len(all_pert_barcodes) > 0:
        df_dict["pert_cell_barcode"] = all_pert_barcodes
        df_dict["ctrl_cell_barcode"] = all_ctrl_barcodes

    obs = pd.DataFrame(df_dict)

    gene_names = var_dims["gene_names"]
    var = pd.DataFrame({"gene_names": gene_names})

    if final_X_hvg is not None:
        if len(gene_names) != final_pert_cell_counts_preds.shape[1]:
            gene_names = np.load(
                "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
            )
            var = pd.DataFrame({"gene_names": gene_names})

        # Create adata for predictions - using the decoded gene expression values
        adata_pred = anndata.AnnData(X=final_pert_cell_counts_preds, obs=obs, var=var)
        # Create adata for real - using the true gene expression values
        adata_real = anndata.AnnData(X=final_X_hvg, obs=obs, var=var)

        # add the embedding predictions
        adata_pred.obsm[data_module.embed_key] = final_preds
        adata_real.obsm[data_module.embed_key] = final_reals
        logger.info(f"Added predicted embeddings to adata.obsm['{data_module.embed_key}']")
    else:
        # if len(gene_names) != final_preds.shape[1]:
        #     gene_names = np.load(
        #         "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
        #     )
        #     var = pd.DataFrame({"gene_names": gene_names})

        # Create adata for predictions - model was trained on gene expression space already
        # adata_pred = anndata.AnnData(X=final_preds, obs=obs, var=var)
        adata_pred = anndata.AnnData(X=final_preds, obs=obs)
        # Create adata for real - using the true gene expression values
        # adata_real = anndata.AnnData(X=final_reals, obs=obs, var=var)
        adata_real = anndata.AnnData(X=final_reals, obs=obs)

    # Optionally filter to perturbations seen in at least one training context
    if args.shared_only:
        try:
            shared_perts = data_module.get_shared_perturbations()
            if len(shared_perts) == 0:
                logger.warning("No shared perturbations between train and test; skipping filtering.")
            else:
                logger.info(
                    "Filtering to %d shared perturbations present in train ∩ test.",
                    len(shared_perts),
                )
                mask = adata_pred.obs[data_module.pert_col].isin(shared_perts)
                before_n = adata_pred.n_obs
                adata_pred = adata_pred[mask].copy()
                adata_real = adata_real[mask].copy()
                logger.info(
                    "Filtered cells: %d -> %d (kept only seen perturbations)",
                    before_n,
                    adata_pred.n_obs,
                )
        except Exception as e:
            logger.warning(
                "Failed to filter by shared perturbations (%s). Proceeding without filter.",
                str(e),
            )

    # Save the AnnData objects
    results_dir = os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint))
    os.makedirs(results_dir, exist_ok=True)
    adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
    adata_real_path = os.path.join(results_dir, "adata_real.h5ad")

    adata_pred.write_h5ad(adata_pred_path)
    adata_real.write_h5ad(adata_real_path)

    logger.info(f"Saved adata_pred to {adata_pred_path}")
    logger.info(f"Saved adata_real to {adata_real_path}")

    # Save per-dimension control-cell distributions for reproducibility
    try:
        dist_out = {
            "key": distributions["key"],
            "dim": distributions["dim"],
            "num_cells": distributions["num_cells"],
        }
        dist_out_path = os.path.join(results_dir, "control_distributions.meta.json")
        with open(dist_out_path, "w") as f:
            json.dump(dist_out, f)
        np.save(os.path.join(results_dir, "control_mean.npy"), distributions["mean"])  # [D]
        np.save(os.path.join(results_dir, "control_std.npy"), distributions["std"])    # [D]
        logger.info("Saved control-cell per-dimension mean/std distributions")
    except Exception as e:
        logger.warning(f"Failed to save control-cell distributions: {e}")

    if not args.predict_only:
        # 6. Compute metrics using cell-eval
        logger.info("Computing metrics using cell-eval...")

        control_pert = data_module.get_control_pert()

        ct_split_real = split_anndata_on_celltype(adata=adata_real, celltype_col=data_module.cell_type_key)
        ct_split_pred = split_anndata_on_celltype(adata=adata_pred, celltype_col=data_module.cell_type_key)

        assert len(ct_split_real) == len(ct_split_pred), (
            f"Number of celltypes in real and pred anndata must match: {len(ct_split_real)} != {len(ct_split_pred)}"
        )

        pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)
        for ct in ct_split_real.keys():
            real_ct = ct_split_real[ct]
            pred_ct = ct_split_pred[ct]

            evaluator = MetricsEvaluator(
                adata_pred=pred_ct,
                adata_real=real_ct,
                control_pert=control_pert,
                pert_col=data_module.pert_col,
                outdir=results_dir,
                prefix=ct,
                pdex_kwargs=pdex_kwargs,
                batch_size=2048,
            )

            evaluator.compute(
                profile=args.profile,
                metric_configs={
                    "discrimination_score": {
                        "embed_key": data_module.embed_key,
                    }
                    if data_module.embed_key and data_module.embed_key != "X_hvg"
                    else {},
                    "pearson_edistance": {
                        "embed_key": data_module.embed_key,
                        "n_jobs": -1,  # set to all available cores
                    }
                    if data_module.embed_key and data_module.embed_key != "X_hvg"
                    else {
                        "n_jobs": -1,
                    },
                }
                if data_module.embed_key and data_module.embed_key != "X_hvg"
                else {},
                skip_metrics=["pearson_edistance", "clustering_agreement"],
            )
