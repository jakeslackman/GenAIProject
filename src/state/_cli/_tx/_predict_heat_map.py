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
        "--shift-batch-size",
        type=int,
        default=8,
        help="Number of position shifts to batch together during heat map generation (default: 8).",
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

    # Save final hidden states (optional)
    parser.add_argument(
        "--save-final-hidden",
        action="store_true",
        help="If set, capture and save the final transformer layer hidden states to adata_pred.obsm and a .npy file.",
    )
    parser.add_argument(
        "--final-hidden-key",
        type=str,
        default="X_final_hidden",
        help="Key under which to store final hidden states in adata_pred.obsm (default: 'X_final_hidden').",
    )
    parser.add_argument(
        "--final-hidden-pooling",
        type=str,
        default="flat",
        choices=["mean", "cls", "sum", "flat"],
        help=(
            "Pooling over sequence when hidden is 3D: 'mean'|'cls'|'sum'|'flat'. "
            "'flat' flattens [B,S,H] to [B,S*H] to preserve entire final layer."
        ),
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

    # Save all final layers (optional)
    parser.add_argument(
        "--save-all-final-layers",
        action="store_true",
        help="If set, capture and save all final transformer layer outputs to a .npy file.",
    )
    parser.add_argument(
        "--final-layers-output-path",
        type=str,
        default=None,
        help="Path to save the final layers .npy file. If not provided, defaults to <results-dir>/final_layers.npy",
    )

    # Metadata output (optional)
    parser.add_argument(
        "--final-hidden-metadata-path",
        type=str,
        default=None,
        help="Path to save JSON mapping of rows to perturbation/gene names for final hidden states.",
    )
    parser.add_argument(
        "--final-layers-metadata-path",
        type=str,
        default=None,
        help="Path to save JSON with 'layers' and 'perturbations' for final layers npy.",
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
    captured_hidden_batches = []
    captured_final_layer_attn_batches = []
    original_attn_impl = None  # Store original attention implementation

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

    # Optionally capture the final transformer hidden states via a forward hook
    if getattr(args, "save_final_hidden", False):
        # Prefer capturing the pre-projection representation feeding into project_out,
        # which should be [batch, hidden] and align with predict_step batch sizes
        proj_module = getattr(model, "project_out", None)
        capture_registered = False
        if proj_module is not None:
            target_mod = None
            if hasattr(proj_module, "__getitem__"):
                try:
                    target_mod = proj_module[0]
                except Exception:
                    target_mod = proj_module
            else:
                target_mod = proj_module

            def _capture_preproj_hook(module, inputs):
                if not inputs:
                    return None
                x = inputs[0]
                try:
                    captured_hidden_batches.append(x.detach().cpu())
                except Exception as e:
                    logger.warning(f"Failed to capture pre-projection hidden state: {e}")
                return None

            try:
                hook = target_mod.register_forward_pre_hook(_capture_preproj_hook)
                inference_hooks.append(hook)
                capture_registered = True
                logger.info("Registered pre-hook on project_out to capture pre-projection hidden states")
            except Exception as e:
                logger.warning(f"Could not register pre-hook on project_out: {e}")

        if not capture_registered:
            transformer = getattr(model, "transformer_backbone", None)
            if transformer is None:
                logger.warning("Model does not expose 'transformer_backbone'; cannot capture final hidden states.")
            else:
                target_layer = None
                if hasattr(transformer, "h") and len(transformer.h) > 0:
                    target_layer = transformer.h[-1]
                elif hasattr(transformer, "layers") and len(transformer.layers) > 0:
                    target_layer = transformer.layers[-1]

                if target_layer is None:
                    logger.warning("Could not find final transformer layer; skipping final hidden capture.")
                else:
                    def _capture_hidden_hook(module, inputs, output):
                        out = output[0] if isinstance(output, tuple) and len(output) >= 1 else output
                        if out is None:
                            return output
                        try:
                            out_cpu = out.detach()
                            if out_cpu.dim() == 3:
                                if args.final_hidden_pooling == "cls":
                                    pooled = out_cpu[:, 0, :]
                                elif args.final_hidden_pooling == "sum":
                                    pooled = out_cpu.sum(dim=1)
                                elif args.final_hidden_pooling == "flat":
                                    b, s, h = out_cpu.shape
                                    pooled = out_cpu.reshape(b, s * h)
                                else:
                                    pooled = out_cpu.mean(dim=1)
                            elif out_cpu.dim() == 2:
                                pooled = out_cpu
                            else:
                                return output
                            captured_hidden_batches.append(pooled.cpu())
                        except Exception as e:
                            logger.warning(f"Failed to capture final hidden state: {e}")
                        return output

                    hook = target_layer.register_forward_hook(_capture_hidden_hook)
                    inference_hooks.append(hook)
                    logger.info("Registered forward hook to capture final hidden states from last transformer layer")

    # Optionally capture self-attention from the final transformer layer
    if getattr(args, "save_all_final_layers", False):
        transformer = getattr(model, "transformer_backbone", None)
        if transformer is None:
            logger.warning("Model does not expose 'transformer_backbone'; cannot capture final-layer self-attention.")
        else:
            # Switch to eager attention implementation to support output_attentions
            if hasattr(transformer, 'config'):
                # Store original attention implementation
                original_attn_impl = getattr(transformer.config, '_attn_implementation', None)
                if original_attn_impl != 'eager':
                    transformer.config._attn_implementation = 'eager'
                    transformer._attn_implementation = 'eager'
                    logger.info(f"Switched attention implementation from '{original_attn_impl}' to 'eager' for attention capture")
                
                # Configure model to output attention weights
                transformer.config.output_attentions = True
                logger.info("Set transformer config output_attentions = True")
            
            # Also set on individual attention modules
            all_layers = []
            if hasattr(transformer, "h") and len(transformer.h) > 0:
                all_layers = transformer.h
            elif hasattr(transformer, "layers") and len(transformer.layers) > 0:
                all_layers = transformer.layers

            if len(all_layers) == 0:
                logger.warning("Could not find any transformer layers; skipping final-layer self-attention capture.")
            else:
                # Set output_attentions on all attention modules
                for layer in all_layers:
                    attn_module = getattr(layer, "attn", getattr(layer, "self_attn", None))
                    if attn_module is not None and hasattr(attn_module, 'output_attentions'):
                        attn_module.output_attentions = True
                
                final_layer_idx = len(all_layers) - 1
                final_layer = all_layers[final_layer_idx]
                attn_module = getattr(final_layer, "attn", getattr(final_layer, "self_attn", None))
                if attn_module is None:
                    logger.warning(f"Final layer {final_layer_idx} has no attention module; skipping self-attention capture.")
                else:
                    def _capture_final_layer_attn(module, inputs, output):
                        # Look for attention weights in the output tuple
                        # The model should now output (attn_output, attn_weights) when output_attentions=True
                        attn_weights = None
                        
                        if isinstance(output, tuple):
                            # Look through all outputs for 4D tensors that look like attention weights [B, H, S, S]
                            for i, out in enumerate(output):
                                if hasattr(out, 'shape') and len(out.shape) == 4:
                                    B, H, S1, S2 = out.shape
                                    if S1 == S2:  # Square matrix indicates attention weights
                                        attn_weights = out.detach().cpu()
                                        logger.debug(f"Found attention weights at output[{i}] with shape: {out.shape}")
                                        break
                        elif hasattr(output, 'shape') and len(output.shape) == 4:
                            # Single 4D tensor output
                            B, H, S1, S2 = output.shape
                            if S1 == S2:
                                attn_weights = output.detach().cpu()
                                logger.debug(f"Found attention weights with shape: {output.shape}")

                        if attn_weights is not None:
                            try:
                                captured_final_layer_attn_batches.append(attn_weights)
                                logger.debug(f"Captured attention weights with shape: {attn_weights.shape}")
                            except Exception as e:
                                logger.warning(f"Failed to capture attention weights: {e}")
                        else:
                            logger.warning("Could not find attention weights in attention module output")
                        return output

                    hook = attn_module.register_forward_hook(_capture_final_layer_attn)
                    inference_hooks.append(hook)
                    logger.info(f"Registered forward hook to capture QK attention matrices from final transformer layer {final_layer_idx}")

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

    def apply_batched_shifts_to_core_cells(indices: list, upregulate: bool = True):
        """Apply ±2σ shifts at multiple indices across all vectors in core_cells.
        
        Creates multiple copies of core_cells, each with a different position shifted.
        Returns a list of batched core_cells dicts.
        
        - indices: list of integers in [0, D)
        - upregulate: True for +2σ, False for -2σ
        """
        nonlocal core_cells, distributions
        key = distributions["key"]
        original_tensor = core_cells[key]
        
        batched_core_cells = []
        for idx in indices:
            if idx < 0 or idx >= distributions["dim"]:
                raise ValueError(f"Index {idx} is out of bounds for dimension {distributions['dim']}")
            
            # Create a copy of core_cells for this shift
            shifted_core_cells = {}
            for k, v in core_cells.items():
                if isinstance(v, torch.Tensor):
                    shifted_core_cells[k] = v.clone()
                else:
                    shifted_core_cells[k] = v.copy() if hasattr(v, 'copy') else v
            
            # Apply the shift to this copy
            shift_value = (2.0 if upregulate else -2.0) * float(distributions["std"][idx])
            shifted_tensor = shifted_core_cells[key]
            shifted_tensor[:, idx] = shifted_tensor[:, idx] + shift_value
            shifted_core_cells[key] = shifted_tensor
            
            batched_core_cells.append(shifted_core_cells)
        
        return batched_core_cells

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

    final_hidden = None
    final_layers_data = None
    attention_to_perturbation_map = []  # List to map attention matrices to perturbations
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

                # Clear any residual captures from prior iterations
            if getattr(args, "save_final_hidden", False):
                captured_hidden_batches.clear()
            if getattr(args, "save_all_final_layers", False):
                captured_final_layer_attn_batches.clear()

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

            # Capture final hidden states if enabled
            if getattr(args, "save_final_hidden", False) and len(captured_hidden_batches) > 0:
                try:
                    cat_hidden = torch.cat([t for t in captured_hidden_batches], dim=0)
                    batch_hidden_np = cat_hidden.cpu().numpy().astype(np.float32)
                    if final_hidden is None:
                        final_hidden = np.empty((num_cells, batch_hidden_np.shape[-1]), dtype=np.float32)
                    if batch_hidden_np.shape[0] != batch_size:
                        logger.warning(
                            f"Captured hidden rows {batch_hidden_np.shape[0]} != preds batch size {batch_size}; aligning by min."
                        )
                        min_b = min(batch_hidden_np.shape[0], batch_size)
                        final_hidden[current_idx - batch_size : current_idx - batch_size + min_b, :] = batch_hidden_np[:min_b]
                    else:
                        final_hidden[current_idx - batch_size : current_idx, :] = batch_hidden_np
                except Exception as e:
                    logger.warning(f"Failed to store captured final hidden states for perturbation {p_idx}: {e}")

            # Capture final-layer self-attention if enabled
            if getattr(args, "save_all_final_layers", False) and len(captured_final_layer_attn_batches) > 0:
                try:
                    # Combine captures for this batch along the batch dimension
                    # Handles either attention weights [B, H, S, S] or attention outputs [B, ...]
                    batch_attn_data = torch.cat([t for t in captured_final_layer_attn_batches], dim=0)
                    batch_attn_np = batch_attn_data.cpu().numpy().astype(np.float32)

                    if final_layers_data is None:
                        # Initialize target array with shape [num_cells, H, max_seq_len, max_seq_len]
                        # Use the first batch to determine head dimension and max sequence length
                        B, H, S, _ = batch_attn_np.shape
                        max_seq_len = S  # Will be updated as we see larger sequences
                        final_layers_data = np.zeros((num_cells, H, max_seq_len, max_seq_len), dtype=np.float32)
                        logger.info(f"Initialized final_layers_data with shape: {final_layers_data.shape}")

                    # Get current dimensions
                    B, H, S, _ = batch_attn_np.shape
                    
                    # Update max sequence length if needed
                    if S > final_layers_data.shape[2]:
                        # Resize the array to accommodate larger sequences
                        old_shape = final_layers_data.shape
                        new_final_layers_data = np.zeros((num_cells, H, S, S), dtype=np.float32)
                        new_final_layers_data[:, :, :old_shape[2], :old_shape[3]] = final_layers_data
                        final_layers_data = new_final_layers_data
                        logger.info(f"Resized final_layers_data from {old_shape} to {final_layers_data.shape}")

                    # Pad the current batch attention matrices to match the target shape
                    if S < final_layers_data.shape[2]:
                        # Pad with zeros to match the target sequence length
                        padded_batch = np.zeros((B, H, final_layers_data.shape[2], final_layers_data.shape[3]), dtype=np.float32)
                        padded_batch[:, :, :S, :S] = batch_attn_np
                        batch_attn_np = padded_batch
                        logger.debug(f"Padded attention matrices from {S}x{S} to {final_layers_data.shape[2]}x{final_layers_data.shape[3]}")

                    if batch_attn_np.shape[0] != batch_size:
                        logger.warning(
                            f"Captured final-layer attention rows {batch_attn_np.shape[0]} != preds batch size {batch_size}; aligning by min."
                        )
                        min_b = min(batch_attn_np.shape[0], batch_size)
                        final_layers_data[current_idx - batch_size : current_idx - batch_size + min_b, ...] = batch_attn_np[:min_b]
                        # Map attention matrices to perturbations for this batch
                        for i in range(min_b):
                            attention_to_perturbation_map.append({
                                "tensor_index": current_idx - batch_size + i,
                                "perturbation": batch_pert_names[i] if i < len(batch_pert_names) else "unknown",
                                "batch_idx": p_idx,
                                "sequence_length": S
                            })
                    else:
                        final_layers_data[current_idx - batch_size : current_idx, ...] = batch_attn_np
                        # Map attention matrices to perturbations for this batch
                        for i in range(batch_size):
                            attention_to_perturbation_map.append({
                                "tensor_index": current_idx - batch_size + i,
                                "perturbation": batch_pert_names[i] if i < len(batch_pert_names) else "unknown",
                                "batch_idx": p_idx,
                                "sequence_length": S
                            })
                except Exception as e:
                    logger.warning(f"Failed to store captured final-layer self-attention for perturbation {p_idx}: {e}")

            # Handle X_hvg for HVG space ground truth
            if final_X_hvg is not None:
                batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                final_X_hvg[current_idx - batch_size : current_idx, :] = batch_real_gene_np

            # Handle decoded gene predictions if available
            if final_pert_cell_counts_preds is not None:
                batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                final_pert_cell_counts_preds[current_idx - batch_size : current_idx, :] = batch_gene_pred_np

    logger.info("Phase 1 complete: Normal inference on all perturbations.")
    
    # Phase 2: Run inference with each position upregulated (only if requested)
    if args.test_time_heat_map:
        logger.info(f"Phase 2: Running inference with each of {D} positions upregulated...")
        logger.info(f"Using batch size {args.shift_batch_size} for position shifts")
    
        # Initialize heatmap array: [num_positions, num_perturbations]
        heatmap_distances = np.zeros((D, len(perts_order)), dtype=np.float32)
        
        # Create batches of position indices
        position_batches = []
        for i in range(0, D, args.shift_batch_size):
            batch_indices = list(range(i, min(i + args.shift_batch_size, D)))
            position_batches.append(batch_indices)
        
        with torch.no_grad():
            for batch_idx, pos_indices in enumerate(tqdm(position_batches, desc="Batched position shifts", unit="batch")):
                # Create batched shifted core_cells for all positions in this batch
                batched_shifted_core_cells = apply_batched_shifts_to_core_cells(pos_indices, upregulate=True)
                
                # Run inference for all perturbations with each shifted position
                for p_idx, pert in enumerate(perts_order):
                    # Collect predictions for all shifts in this batch
                    batch_upregulated_preds = []
                    
                    for shift_idx, shifted_core_cells in enumerate(batched_shifted_core_cells):
                        # Build batch by copying this shifted core_cells
                        batch = {}
                        for k, v in shifted_core_cells.items():
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
                        
                        # Get predictions with upregulated position
                        batch_preds = model.predict_step(batch, p_idx, padded=False)
                        upregulated_preds = batch_preds["preds"].detach().cpu().numpy().astype(np.float32)
                        batch_upregulated_preds.append(upregulated_preds)
                    
                    # Compute distances for all positions in this batch
                    normal_preds = normal_preds_per_pert[pert]  # [64, output_dim]
                    for shift_idx, upregulated_preds in enumerate(batch_upregulated_preds):
                        pos_idx = pos_indices[shift_idx]
                        distance = np.linalg.norm(upregulated_preds - normal_preds, axis=1).mean()  # Mean across 64 cells
                        heatmap_distances[pos_idx, p_idx] = distance
        
        logger.info("Phase 2 complete: Batched upregulated inference for all positions.")
        
        # Save heatmap data
        try:
            heatmap_path = os.path.join(results_dir, "position_upregulation_heatmap.npy")
            np.save(heatmap_path, heatmap_distances)
            
            # Save metadata for the heatmap
            heatmap_meta = {
                "shape": [int(D), len(perts_order)],
                "description": "Euclidean distance heatmap: rows=positions (0-1999), cols=perturbations",
                "perturbations": perts_order,
                "distance_type": "mean_euclidean_norm_across_64_cells",
                "upregulation": "2_std_deviation_shift"
            }
            heatmap_meta_path = os.path.join(results_dir, "position_upregulation_heatmap.meta.json")
            with open(heatmap_meta_path, "w") as f:
                json.dump(heatmap_meta, f)
            
            logger.info(f"Saved position upregulation heatmap to {heatmap_path}")
            logger.info(f"Heatmap shape: {heatmap_distances.shape} (positions x perturbations)")
        except Exception as e:
            logger.warning(f"Failed to save heatmap data: {e}")
        
        # Create and save matplotlib heatmap visualization
        try:
            # Determine output path for heatmap image
            if args.heatmap_output_path is not None:
                heatmap_img_path = args.heatmap_output_path
            else:
                heatmap_img_path = os.path.join(results_dir, "position_upregulation_heatmap.png")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(heatmap_img_path), exist_ok=True)
            
            # Create the heatmap
            fig, ax = plt.subplots(figsize=(max(12, len(perts_order) * 0.3), max(8, D * 0.01)))
            
            # Create heatmap with proper labels
            im = ax.imshow(heatmap_distances, cmap='viridis', aspect='auto')
            
            # Set labels and title
            ax.set_xlabel('Perturbations')
            ax.set_ylabel('Vector Positions')
            ax.set_title('Position Upregulation Impact Heatmap\n(Euclidean Distance from Normal Predictions)')
            
            # Set x-axis labels (perturbations)
            ax.set_xticks(range(len(perts_order)))
            ax.set_xticklabels(perts_order, rotation=45, ha='right', fontsize=8)
            
            # Set y-axis labels (positions) - show every 200th position to avoid overcrowding
            y_ticks = range(0, D, max(1, D // 20))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(i) for i in y_ticks], fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mean Euclidean Distance', rotation=270, labelpad=20)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(heatmap_img_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
            
            logger.info(f"Saved heatmap visualization to {heatmap_img_path}")
            
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

    # Attach final hidden states if captured
    if getattr(args, "save_final_hidden", False) and final_hidden is not None:
        try:
            adata_pred.obsm[args.final_hidden_key] = final_hidden
            logger.info(f"Added final hidden states to adata_pred.obsm['{args.final_hidden_key}'] with shape {final_hidden.shape}")
        except Exception as e:
            logger.warning(f"Failed to add final hidden states to adata: {e}")

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

    # Optionally dump final hidden states as .npy for convenience
    if getattr(args, "save_final_hidden", False) and final_hidden is not None:
        try:
            hidden_npy_path = os.path.join(results_dir, f"{args.final_hidden_key}.npy")
            np.save(hidden_npy_path, final_hidden)
            logger.info(f"Saved final hidden states to {hidden_npy_path}")

            # Save metadata mapping rows to perturbation names
            try:
                if getattr(args, "final_hidden_metadata_path", None) is not None:
                    hidden_meta_path = args.final_hidden_metadata_path
                else:
                    hidden_meta_path = os.path.join(results_dir, f"{args.final_hidden_key}.meta.json")

                parent = os.path.dirname(hidden_meta_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)

                hidden_metadata = {
                    "perturbations": list(all_pert_names),
                    "pert_col": data_module.pert_col,
                    "array_path": hidden_npy_path,
                    "shape": list(final_hidden.shape),
                }
                with open(hidden_meta_path, "w") as f:
                    json.dump(hidden_metadata, f)
                logger.info(f"Saved final hidden metadata to {hidden_meta_path}")
            except Exception as e:
                logger.warning(f"Failed to save final hidden metadata JSON: {e}")
        except Exception as e:
            logger.warning(f"Failed to save final hidden states .npy: {e}")

    # Optionally save final-layer self-attention data as .npy
    if getattr(args, "save_all_final_layers", False) and final_layers_data is not None:
        try:
            if args.final_layers_output_path is not None:
                final_layers_npy_path = args.final_layers_output_path
            else:
                final_layers_npy_path = os.path.join(results_dir, "final_layers.npy")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(final_layers_npy_path), exist_ok=True)
            
            np.save(final_layers_npy_path, final_layers_data)
            logger.info(f"Saved QK attention matrices to {final_layers_npy_path} with shape {final_layers_data.shape}")

            # Save metadata with final layer index and perturbation names
            try:
                if getattr(args, "final_layers_metadata_path", None) is not None:
                    layers_meta_path = args.final_layers_metadata_path
                else:
                    layers_meta_path = os.path.join(results_dir, "final_layers.meta.json")

                parent = os.path.dirname(layers_meta_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)

                # Determine final layer index for metadata
                final_layer_index = None
                transformer = getattr(model, "transformer_backbone", None)
                if transformer is not None:
                    if hasattr(transformer, "h") and len(transformer.h) > 0:
                        final_layer_index = len(transformer.h) - 1
                    elif hasattr(transformer, "layers") and len(transformer.layers) > 0:
                        final_layer_index = len(transformer.layers) - 1

                layers_metadata = {
                    "layer": int(final_layer_index) if final_layer_index is not None else None,
                    "kind": "qk_attention_matrices",
                    "description": "QK attention weight matrices from final transformer layer [B, H, S, S]. Smaller sequences are padded with zeros.",
                    "perturbations": list(all_pert_names),
                    "array_path": final_layers_npy_path,
                    "shape": [int(x) for x in list(final_layers_data.shape)],
                    "padding_info": "Sequences shorter than max_seq_len are padded with zeros in the last two dimensions",
                    "attention_to_perturbation_map": attention_to_perturbation_map,
                    "mapping_info": "Each entry maps tensor_index to perturbation name, batch_idx, and original sequence_length"
                }
                with open(layers_meta_path, "w") as f:
                    json.dump(layers_metadata, f)
                logger.info(f"Saved QK attention matrices metadata to {layers_meta_path}")
                logger.info(f"Created {len(attention_to_perturbation_map)} attention-to-perturbation mappings")
            except Exception as e:
                logger.warning(f"Failed to save final layers metadata JSON: {e}")
        except Exception as e:
            logger.warning(f"Failed to save final layers .npy: {e}")

    # Cleanup inference hooks if applied (after inference is done)
    if 'inference_hooks' in locals() and inference_hooks:
        for hook in inference_hooks:
            try:
                hook.remove()
            except Exception:
                pass
        logger.info(f"Removed {len(inference_hooks)} registered inference hooks")
    
    # Restore original attention implementation if it was changed
    if getattr(args, "save_all_final_layers", False) and original_attn_impl is not None:
        transformer = getattr(model, "transformer_backbone", None)
        if transformer is not None and hasattr(transformer, 'config'):
            if original_attn_impl != 'eager':
                # Restore original implementation
                transformer.config._attn_implementation = original_attn_impl
                transformer._attn_implementation = original_attn_impl
                logger.info(f"Restored attention implementation to '{original_attn_impl}'")

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
