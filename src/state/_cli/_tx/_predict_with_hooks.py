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
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Custom directory to save results. If not provided, defaults to <output-dir>/eval_<checkpoint-name>/",
    )

    # Attention head ablation (optional)
    parser.add_argument(
        "--attention-head-layer",
        type=int,
        default=None,
        help="Layer index to remove attention head contributions from. Requires --attention-head-index or --attention-head-indices.",
    )
    parser.add_argument(
        "--attention-head-index",
        type=int,
        default=None,
        help="Index of attention head to remove contributions from (0-based). Requires --attention-head-layer.",
    )
    parser.add_argument(
        "--attention-head-indices",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated list of head indices to remove (0-based). Requires --attention-head-layer.",
    )

    # Full layer ablation (optional)
    parser.add_argument(
        "--ablate-layer",
        type=int,
        default=None,
        help="Ablate a whole transformer layer by zeroing its output (0-based).",
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

    # Cell-eval for metrics computation
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype
    from cell_load.data_modules import PerturbationDataModule
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.multiprocessing.set_sharing_strategy("file_system")

    # -----------------------
    # Attention Head Removal Utilities
    # -----------------------
    def create_head_removal_forward_hook(head_index):
        """Create a forward hook that zeros out contributions from a specific attention head.

        Best-effort fallback: if exact head metadata is unavailable, infer number of heads from hidden size divisors.
        """

        def head_removal_hook(module, input, output):
            # Tuple outputs (e.g., (attn_output, attn_weights, ...))
            if isinstance(output, tuple) and len(output) >= 1:
                main_output = output[0]
                if main_output is not None and hasattr(main_output, "dim") and main_output.dim() == 3:
                    hidden_dim = main_output.shape[-1]
                    num_heads = getattr(module, "num_heads", getattr(module, "num_attention_heads", None))
                    if num_heads is None or num_heads <= 1:
                        candidates = [32, 24, 16, 12, 8, 4, 2]
                        num_heads = next((h for h in candidates if hidden_dim % h == 0), None) or 1
                    if num_heads > 1 and 0 <= head_index < num_heads:
                        head_dim = hidden_dim // num_heads
                        start_idx = head_index * head_dim
                        end_idx = min(start_idx + head_dim, hidden_dim)
                        main_output[:, :, start_idx:end_idx] = 0.0
                        logger.info(f"[hook:fallback] Zeroed head {head_index} on post-proj slice {start_idx}:{end_idx}")
                return (main_output,) + tuple(output[1:])

            # Direct tensor outputs
            if hasattr(output, "dim") and output.dim() == 3:
                hidden_dim = output.shape[-1]
                num_heads = getattr(module, "num_heads", getattr(module, "num_attention_heads", None))
                if num_heads is None or num_heads <= 1:
                    candidates = [32, 24, 16, 12, 8, 4, 2]
                    num_heads = next((h for h in candidates if hidden_dim % h == 0), None) or 1
                if num_heads > 1 and 0 <= head_index < num_heads:
                    head_dim = hidden_dim // num_heads
                    start_idx = head_index * head_dim
                    end_idx = min(start_idx + head_dim, hidden_dim)
                    output[:, :, start_idx:end_idx] = 0.0
                    logger.info(f"[hook:fallback] Zeroed head {head_index} on direct slice {start_idx}:{end_idx}")
                return output

            return output

        return head_removal_hook

    def apply_head_removal_patch(model, layer_idx: int, head_idx: int):
        """Apply hooks to remove contributions from a specific attention head on a given layer.

        Attempts to register a pre-hook on the attention projection when available; otherwise falls back to
        a forward hook on the attention module output.
        """
        hooks = []

        transformer = getattr(model, "transformer_backbone", None)
        if transformer is None:
            logger.warning("Model does not expose 'transformer_backbone'; skipping attention head removal.")
            return hooks

        target_module = None
        if hasattr(transformer, "h") and len(transformer.h) > layer_idx:
            layer = transformer.h[layer_idx]
            target_module = getattr(layer, "attn", None)
        elif hasattr(transformer, "layers") and len(transformer.layers) > layer_idx:
            layer = transformer.layers[layer_idx]
            target_module = getattr(layer, "self_attn", None)
        else:
            logger.warning(f"Could not find transformer layer {layer_idx}; skipping head removal.")
            return hooks

        if target_module is None:
            logger.warning(f"Layer {layer_idx} has no attention module; skipping head removal.")
            return hooks

        proj_module = None
        for attr in ["c_proj", "o_proj", "out_proj"]:
            if hasattr(target_module, attr):
                proj_module = getattr(target_module, attr)
                if proj_module is not None:
                    break

        if proj_module is not None:
            parent_attn_module = target_module

            def pre_zero_hook(module, inputs):
                if not inputs or inputs[0] is None:
                    return inputs
                x = inputs[0]
                if not hasattr(x, "shape") or x is None or x.dim() < 2:
                    return inputs

                hidden_dim = x.shape[-1]
                num_heads_local = getattr(parent_attn_module, "num_heads", getattr(parent_attn_module, "num_attention_heads", None))
                if num_heads_local is None or num_heads_local <= 1:
                    candidates = [32, 24, 16, 12, 8, 4, 2]
                    num_heads_local = next((h for h in candidates if hidden_dim % h == 0), None) or 1

                if num_heads_local > 1 and 0 <= head_idx < num_heads_local:
                    head_dim = hidden_dim // num_heads_local
                    start_idx = head_idx * head_dim
                    end_idx = min(start_idx + head_dim, hidden_dim)
                    x = x.clone()
                    x[..., start_idx:end_idx] = 0.0
                    logger.info(
                        f"Applied pre-proj head zeroing: layer {layer_idx}, head {head_idx}, slice {start_idx}:{end_idx}, shape {tuple(x.shape)}"
                    )
                    return (x,)
                return inputs

            hook = proj_module.register_forward_pre_hook(pre_zero_hook)
            hooks.append(hook)
            logger.info(f"Applied head removal pre-hook on projection: layer {layer_idx}, head {head_idx}")
        else:
            hook_fn = create_head_removal_forward_hook(head_idx)
            hook = target_module.register_forward_hook(hook_fn)
            hooks.append(hook)
            logger.info(f"Applied head removal fallback hook on attn output: layer {layer_idx}, head {head_idx}")

        return hooks

    def apply_multi_head_removal_patches(model, layer_idx: int, head_indices: list):
        """Apply removal hooks for multiple heads on a given layer."""
        all_hooks = []
        unique_heads = sorted(set(int(h) for h in head_indices))
        for h in unique_heads:
            all_hooks.extend(apply_head_removal_patch(model, layer_idx=layer_idx, head_idx=h))
        return all_hooks

    def apply_layer_ablation_patch(model, layer_idx: int):
        """Apply a forward hook to ablate (zero) the output of a transformer layer.

        Tries common backbone layouts: transformer_backbone.h[layer] or .layers[layer]. If the layer's forward
        returns a tuple, only the first tensor is zeroed and the rest are passed through.
        """
        hooks = []

        transformer = getattr(model, "transformer_backbone", None)
        if transformer is None:
            logger.warning("Model does not expose 'transformer_backbone'; skipping layer ablation.")
            return hooks

        target_layer = None
        if hasattr(transformer, "h") and len(transformer.h) > layer_idx:
            target_layer = transformer.h[layer_idx]
        elif hasattr(transformer, "layers") and len(transformer.layers) > layer_idx:
            target_layer = transformer.layers[layer_idx]
        else:
            logger.warning(f"Could not find transformer layer {layer_idx}; skipping layer ablation.")
            return hooks

        def zero_output_hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) >= 1:
                main = output[0]
                if hasattr(main, "zeros_like") or hasattr(main, "shape"):
                    main_zeros = torch.zeros_like(main)
                    return (main_zeros,) + tuple(output[1:])
                return output
            if hasattr(output, "zeros_like") or hasattr(output, "shape"):
                return torch.zeros_like(output)
            return output

        hook = target_layer.register_forward_hook(zero_output_hook)
        hooks.append(hook)
        logger.info(f"Applied full layer ablation hook on layer {layer_idx}")
        return hooks

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
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_class_name.lower() == "old_neuralot":
        from ...tx.models.old_neural_ot import OldNeuralOTPerturbationModel

        ModelClass = OldNeuralOTPerturbationModel
    elif model_class_name.lower() in ["neuralot", "pertsets", "state"]:
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        ModelClass = StateTransitionPerturbationModel

    elif model_class_name.lower() in ["globalsimplesum", "perturb_mean"]:
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        ModelClass = PerturbMeanPerturbationModel
    elif model_class_name.lower() in ["celltypemean", "context_mean"]:
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        ModelClass = ContextMeanPerturbationModel
    elif model_class_name.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

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

    # Validate and apply attention head and/or layer ablation
    if args.attention_head_index is not None or (args.attention_head_indices is not None and len(args.attention_head_indices) > 0):
        if args.attention_head_layer is None:
            raise ValueError("--attention-head-layer is required when specifying head indices")

    inference_hooks = []

    # Apply full layer ablation if requested (takes precedence over head removals)
    if args.ablate_layer is not None:
        if args.attention_head_index is not None or (args.attention_head_indices is not None and len(args.attention_head_indices) > 0):
            logger.info(
                "--ablate-layer set: ignoring provided --attention-head-index/--attention-head-indices and zeroing the layer output"
            )
        inference_hooks.extend(apply_layer_ablation_patch(model, layer_idx=args.ablate_layer))
    else:
        # Apply head removals if requested (support single or multiple)
        if args.attention_head_layer is not None and (
            args.attention_head_index is not None or (args.attention_head_indices is not None and len(args.attention_head_indices) > 0)
        ):
            heads = []
            if args.attention_head_index is not None:
                heads.append(args.attention_head_index)
            if args.attention_head_indices is not None:
                heads.extend(list(args.attention_head_indices))
            unique_heads = sorted(set(int(h) for h in heads))
            inference_hooks.extend(
                apply_multi_head_removal_patches(model, layer_idx=args.attention_head_layer, head_indices=unique_heads)
            )
            logger.info(
                f"Attention head removal enabled: layer {args.attention_head_layer}, heads {unique_heads}"
            )

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
        test_loader = data_module.train_dataloader(test=True)
    else:
        test_loader = data_module.test_dataloader()

    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    num_cells = test_loader.batch_sampler.tot_num
    output_dim = var_dims["output_dim"]
    gene_dim = var_dims["gene_dim"]
    hvg_dim = var_dims["hvg_dim"]

    logger.info("Generating predictions on test set using manual loop...")
    device = next(model.parameters()).device

    final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
    final_reals = np.empty((num_cells, output_dim), dtype=np.float32)

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
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
            # Move each tensor in the batch to the model's device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Get predictions
            batch_preds = model.predict_step(batch, batch_idx, padded=False)

            # Extract metadata and data directly from batch_preds
            # Handle pert_name
            if isinstance(batch_preds["pert_name"], list):
                all_pert_names.extend(batch_preds["pert_name"])
            else:
                all_pert_names.append(batch_preds["pert_name"])

            if "pert_cell_barcode" in batch_preds:
                if isinstance(batch_preds["pert_cell_barcode"], list):
                    all_pert_barcodes.extend(batch_preds["pert_cell_barcode"])
                    all_ctrl_barcodes.extend(batch_preds["ctrl_cell_barcode"])
                else:
                    all_pert_barcodes.append(batch_preds["pert_cell_barcode"])
                    all_ctrl_barcodes.append(batch_preds["ctrl_cell_barcode"])

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

            batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
            batch_real_np = batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)
            batch_size = batch_pred_np.shape[0]
            final_preds[current_idx : current_idx + batch_size, :] = batch_pred_np
            final_reals[current_idx : current_idx + batch_size, :] = batch_real_np
            current_idx += batch_size

            # Handle X_hvg for HVG space ground truth
            if final_X_hvg is not None:
                batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                final_X_hvg[current_idx - batch_size : current_idx, :] = batch_real_gene_np

            # Handle decoded gene predictions if available
            if final_pert_cell_counts_preds is not None:
                batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                final_pert_cell_counts_preds[current_idx - batch_size : current_idx, :] = batch_gene_pred_np

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
    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint))
    os.makedirs(results_dir, exist_ok=True)
    adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
    adata_real_path = os.path.join(results_dir, "adata_real.h5ad")

    adata_pred.write_h5ad(adata_pred_path)
    adata_real.write_h5ad(adata_real_path)

    logger.info(f"Saved adata_pred to {adata_pred_path}")
    logger.info(f"Saved adata_real to {adata_real_path}")

    # Cleanup inference hooks if applied (after inference is done)
    if 'inference_hooks' in locals() and inference_hooks:
        for hook in inference_hooks:
            try:
                hook.remove()
            except Exception:
                pass
        logger.info(f"Removed {len(inference_hooks)} registered inference hooks")

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
