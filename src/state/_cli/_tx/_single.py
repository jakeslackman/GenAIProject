import argparse as ap


def add_arguments_single(parser: ap.ArgumentParser) -> None:
    """CLI for single-pass perturbation analysis on a target cell line."""

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Path to the output_dir containing the config.yaml file that was saved during training."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename relative to the output directory (default: last.ckpt).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "minimal", "de", "anndata"],
        help="Evaluation profile to run after inference.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip metric computation and only run inference.",
    )
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Restrict outputs to perturbations present in both train and test sets.",
    )
    parser.add_argument(
        "--eval-train-data",
        action="store_true",
        help="Evaluate the model on the training data instead of the test data (ignored with --core-cells-path).",
    )
    parser.add_argument(
        "--target-cell-type",
        type=str,
        default=None,
        help=(
            "Optional cell type to construct the base core cells for single perturbations."
            " Ignored when --core-cells-path is provided."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results. Defaults to <output-dir>/eval_<checkpoint>.",
    )
    parser.add_argument(
        "--first-pass-only",
        action="store_true",
        help="Run only the first round of inference and save per-perturbation predictions to a NumPy file.",
    )
    parser.add_argument(
        "--core-cells-path",
        type=str,
        default=None,
        help=(
            "Path to a NumPy .npy file containing a serialized core_cells dictionary to use instead of"
            " constructing a new control batch."
        ),
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help=(
            "Path to a TOML data configuration file to override the data paths in the loaded data module."
        ),
    )


def run_tx_single(args: ap.ArgumentParser, *, phase_one_only: bool = False) -> None:
    import logging
    import os
    import sys

    import anndata
    import lightning.pytorch as pl
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from tqdm import tqdm

    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype
    from cell_load.data_modules import PerturbationDataModule

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    phase_one_only = phase_one_only or getattr(args, "first_pass_only", False)

    def _prepare_for_serialization(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().copy()
        if isinstance(obj, dict):
            return {k: _prepare_for_serialization(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_prepare_for_serialization(v) for v in obj]
        return obj

    def _save_numpy_snapshot(obj, path, description=None):
        serializable = _prepare_for_serialization(obj)
        try:
            np.save(path, serializable, allow_pickle=True)
            if description:
                logger.info("Saved %s to %s", description, path)
            else:
                logger.info("Saved snapshot to %s", path)
        except Exception as exc:
            logger.warning("Failed to save %s to %s: %s", description or "snapshot", path, exc)

    def _to_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, torch.Tensor):
            try:
                return [x.item() if x.dim() == 0 else x for x in value]
            except Exception:
                return value.tolist()
        if isinstance(value, (tuple, set)):
            return list(value)
        if value is None:
            return []
        return [value]

    def _normalize_field(values, length, filler=None):
        items = list(_to_list(values))
        if len(items) == 1 and length > 1:
            items = items * length
        if len(items) < length:
            items.extend([filler] * (length - len(items)))
        elif len(items) > length:
            items = items[:length]
        return items

    def _resolve_celltype_key(batch, module):
        candidate_keys = []
        base_key = getattr(module, "cell_type_key", None)
        if base_key:
            candidate_keys.append(base_key)
        alias_keys = getattr(module, "cell_type_key_aliases", None)
        if isinstance(alias_keys, (list, tuple)):
            candidate_keys.extend(alias_keys)
        alias_keys_alt = getattr(module, "celltype_key_aliases", None)
        if isinstance(alias_keys_alt, (list, tuple)):
            candidate_keys.extend(alias_keys_alt)
        candidate_keys.extend([
            "celltype_name",
            "cell_type",
            "celltype",
            "cell_line",
        ])
        seen = set()
        ordered_candidates = []
        for key in candidate_keys:
            if not key or key in seen:
                continue
            seen.add(key)
            ordered_candidates.append(key)
            if key in batch:
                return key, ordered_candidates
        return None, ordered_candidates

    torch.multiprocessing.set_sharing_strategy("file_system")

    config_path = os.path.join(args.output_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    logger.info("Loaded config from %s", config_path)

    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    if not os.path.isabs(run_output_dir):
        run_output_dir = os.path.abspath(run_output_dir)
    if not os.path.exists(run_output_dir):
        inferred_run_dir = args.output_dir
        if os.path.exists(inferred_run_dir):
            logger.warning(
                "Run directory %s not found; falling back to config directory %s",
                run_output_dir,
                inferred_run_dir,
            )
            run_output_dir = inferred_run_dir
        else:
            raise FileNotFoundError(
                "Could not resolve run directory. Checked: %s and %s" % (run_output_dir, inferred_run_dir)
            )

    # Override data paths if --data-config is provided
    # We need to modify the saved state before loading the data module
    if getattr(args, "data_config", None) is not None:
        import tempfile
        
        logger.info("Overriding data paths with config from %s", args.data_config)
        
        data_module_path = os.path.join(run_output_dir, "data_module.torch")
        if not os.path.exists(data_module_path):
            raise FileNotFoundError(f"Could not find data module at {data_module_path}")
        
        # Load the raw state dict
        state_dict = torch.load(data_module_path, weights_only=False)
        
        # Override the toml_config_path in the state dict
        state_dict['toml_config_path'] = os.path.abspath(args.data_config)
        logger.info("Updated toml_config_path to: %s", state_dict['toml_config_path'])
        
        # Save to a temporary location in a writable directory
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.torch', delete=False) as tmp:
            temp_path = tmp.name
        torch.save(state_dict, temp_path)
        logger.info("Saved modified state to temp file: %s", temp_path)
        
        # Load the data module from the modified state
        data_module = PerturbationDataModule.load_state(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
    else:
        data_module_path = os.path.join(run_output_dir, "data_module.torch")
        if not os.path.exists(data_module_path):
            raise FileNotFoundError(f"Could not find data module at {data_module_path}")
        data_module = PerturbationDataModule.load_state(data_module_path)
    
    # Only setup data module if we need to load cell data (not using preassembled core cells)
    custom_core_cells_path = getattr(args, "core_cells_path", None)
    if custom_core_cells_path is None:
        data_module.setup(stage="test")
        logger.info("Loaded data module from %s", data_module_path)
    else:
        logger.info("Loaded data module configuration from %s (skipping data setup for preassembled core cells)", data_module_path)

    pl.seed_everything(cfg["training"]["train_seed"])

    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}. Specify --checkpoint with a valid file."
        )
    logger.info("Loading model from %s", checkpoint_path)

    model_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]
    
    # Get var_dims from checkpoint when using preassembled core cells to avoid loading data
    if custom_core_cells_path is not None:
        logger.info("Reading var_dims from checkpoint to avoid loading data")
        ckpt_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        var_dims = {}
        
        # Extract dimensions from model state dict
        if 'state_dict' in ckpt_state:
            state_dict = ckpt_state['state_dict']
            # Try to infer dimensions from model weights
            for key in state_dict.keys():
                if 'pert_emb' in key or 'perturbation_encoder' in key:
                    if 'weight' in key:
                        var_dims['pert_dim'] = state_dict[key].shape[0] if state_dict[key].dim() > 1 else state_dict[key].shape[0]
                if 'cell_encoder' in key or 'encoder.0' in key:
                    if 'weight' in key and 'input_dim' not in var_dims:
                        var_dims['input_dim'] = state_dict[key].shape[1] if state_dict[key].dim() > 1 else state_dict[key].shape[0]
                if 'decoder' in key or 'cell_decoder' in key:
                    if 'weight' in key and 'output_dim' not in var_dims:
                        # Output dim is usually the last layer of decoder
                        if 'decoder.weight' in key or 'cell_decoder.weight' in key:
                            var_dims['output_dim'] = state_dict[key].shape[0]
        
        # Also check hyper_parameters in checkpoint
        if 'hyper_parameters' in ckpt_state:
            hp = ckpt_state['hyper_parameters']
            for dim_key in ['input_dim', 'output_dim', 'pert_dim', 'gene_dim', 'hvg_dim', 'hidden_dim']:
                if dim_key in hp:
                    var_dims[dim_key] = hp[dim_key]
        
        logger.info("Extracted var_dims from checkpoint: %s", var_dims)
        
        if not var_dims or 'input_dim' not in var_dims or 'output_dim' not in var_dims or 'pert_dim' not in var_dims:
            raise RuntimeError(f"Could not extract required dimensions from checkpoint. Got: {var_dims}")
    else:
        var_dims = data_module.get_var_dims()

    if model_name.lower() == "embedsum":
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_name.lower() == "old_neuralot":
        from ...tx.models.old_neural_ot import OldNeuralOTPerturbationModel

        ModelClass = OldNeuralOTPerturbationModel
    elif model_name.lower() in {"neuralot", "pertsets", "state"}:
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        ModelClass = StateTransitionPerturbationModel
    elif model_name.lower() in {"globalsimplesum", "perturb_mean"}:
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        ModelClass = PerturbMeanPerturbationModel
    elif model_name.lower() in {"celltypemean", "context_mean"}:
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        ModelClass = ContextMeanPerturbationModel
    elif model_name.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

        ModelClass = DecoderOnlyPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_name}")

    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }

    for optional_key in ("gene_dim", "hvg_dim"):
        optional_value = var_dims.get(optional_key)
        if optional_value is not None:
            model_init_kwargs[optional_key] = optional_value

    if "hidden_dim" in var_dims and "hidden_dim" not in model_init_kwargs:
        model_init_kwargs["hidden_dim"] = var_dims["hidden_dim"]

    model = ModelClass.load_from_checkpoint(
        checkpoint_path,
        **model_init_kwargs,
    )
    model.eval()
    logger.info("Model loaded successfully.")

    results_dir_default = (
        args.results_dir
        if args.results_dir is not None
        else os.path.join(args.output_dir, f"eval_{os.path.basename(args.checkpoint)}")
    )

    data_module.batch_size = 1
    target_celltype = getattr(args, "target_cell_type")
    # custom_core_cells_path was already loaded earlier to skip data setup
    if custom_core_cells_path is not None:
        target_celltype = None

    def _create_filtered_loader(module):
        use_preassembled_core = custom_core_cells_path is not None
        eval_train = bool(getattr(args, "eval_train_data", False)) and not use_preassembled_core

        base_loader = (
            module.train_dataloader(test=True)
            if eval_train
            else module.test_dataloader()
        )

        celltype_key, attempted = _resolve_celltype_key({}, module)

        def _generator():
            found_target = False
            for batch in base_loader:
                if use_preassembled_core:
                    found_target = True
                    yield batch
                    continue

                if target_celltype is None:
                    found_target = True
                    yield batch
                    continue

                key = celltype_key
                if key is None:
                    key, attempted_keys = _resolve_celltype_key(batch, module)
                    if key is None:
                        available_keys = [k for k in batch.keys() if isinstance(k, str)]
                        available_preview = ", ".join(sorted(available_keys)[:10])
                        raise ValueError(
                            "--target-cell-type requested filtering but none of the expected keys (%s) were present."
                            " Available batch keys: %s%s"
                            % (
                                ", ".join(attempted_keys) if attempted_keys else "none",
                                available_preview,
                                "..." if len(available_keys) > 10 else "",
                            )
                        )

                celltypes = _to_list(batch[key])
                mask_values = [str(ct).lower() == target_celltype.lower() for ct in celltypes]
                if not mask_values or not any(mask_values):
                    continue

                mask = torch.tensor(mask_values, dtype=torch.bool)
                filtered = {}
                for batch_key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        mask_device = mask.to(value.device)
                        selected = value[mask_device]
                        if selected.shape[0] == 0:
                            continue
                        filtered[batch_key] = selected
                    else:
                        vals = _to_list(value)
                        selected = [vals[idx] for idx, keep in enumerate(mask_values) if keep]
                        if not selected:
                            continue
                        filtered[batch_key] = selected
                if filtered:
                    found_target = True
                    yield filtered

            if target_celltype and not found_target:
                raise ValueError(
                    f"Target cell type '{target_celltype}' not found in any batches for evaluation."
                )

        return _generator()

    # Only create eval_loader if we're not using preassembled core cells
    if custom_core_cells_path is None:
        eval_loader = _create_filtered_loader(data_module)
    else:
        eval_loader = None

    logger.info("Preparing core cells batch and enumerating perturbations...")

    # Get control_pert - when using preassembled core cells, get it from the data_module attributes
    if custom_core_cells_path is not None:
        control_pert = data_module.control_pert
        logger.info("Using control_pert from data_module config: %s", control_pert)
    else:
        control_pert = data_module.get_control_pert()
    
    # Load pert_onehot_map early to get perturbations without loading data
    pert_onehot_map = getattr(data_module, "pert_onehot_map", None)
    if pert_onehot_map is None:
        map_path = os.path.join(run_output_dir, "pert_onehot_map.pt")
        if os.path.exists(map_path):
            pert_onehot_map = torch.load(map_path, weights_only=False)
            logger.info("Loaded pert_onehot_map from %s", map_path)
        else:
            logger.warning("pert_onehot_map.pt not found at %s", map_path)
            pert_onehot_map = {}
    
    # When using preassembled core cells, get perturbations from pert_onehot_map (no data loading!)
    # Otherwise enumerate from the filtered eval_loader
    if custom_core_cells_path is not None and pert_onehot_map:
        # Use pert_onehot_map to get all perturbations without loading data
        unique_perts = list(pert_onehot_map.keys())
        logger.info("Enumerating %d perturbations from pert_onehot_map (using preassembled core cells)", len(unique_perts))
    else:
        # Enumerate from dataloader (original behavior)
        unique_perts = []
        seen_perts = set()
        for batch in eval_loader:
            names = _to_list(batch.get("pert_name", []))
            for name in names:
                name_value = name.item() if isinstance(name, torch.Tensor) else str(name)
                if name_value not in seen_perts:
                    seen_perts.add(name_value)
                    unique_perts.append(name_value)
        if not unique_perts:
            raise RuntimeError("No perturbations found in the provided dataloader.")

    target_core_n_default = 256

    def _load_core_cells_from_path(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Core cells file not found: {path}")

        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("Expected a .npy file containing a serialized dictionary; received an .npz archive.")
        if isinstance(loaded, np.ndarray) and loaded.dtype == object:
            if loaded.shape == ():
                loaded = loaded.item()
        if not isinstance(loaded, dict):
            raise ValueError(
                "Serialized core cells must be a dictionary mapping field names to arrays/tensors;"
                f" received type {type(loaded).__name__}."
            )

        converted = {}
        inferred_length = None
        for key, value in loaded.items():
            if isinstance(value, torch.Tensor):
                tensor = value.clone().detach()
            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            else:
                tensor = None

            if tensor is not None:
                if tensor.dim() == 0:
                    converted[key] = tensor.item()
                    continue
                if inferred_length is None:
                    inferred_length = tensor.shape[0]
                else:
                    if tensor.shape[0] != inferred_length:
                        raise ValueError(
                            f"Mismatched leading dimensions in core cells: expected {inferred_length},"
                            f" received {tensor.shape[0]} for key '{key}'."
                        )
                converted[key] = tensor
            else:
                values_list = _to_list(value)
                if inferred_length is None:
                    inferred_length = len(values_list)
                elif len(values_list) != inferred_length:
                    raise ValueError(
                        f"Mismatched list length in core cells: expected {inferred_length},"
                        f" received {len(values_list)} for key '{key}'."
                    )
                converted[key] = values_list

        if inferred_length is None or inferred_length == 0:
            raise ValueError("Loaded core cells did not contain any batched entries.")

        for key, value in list(converted.items()):
            if isinstance(value, torch.Tensor):
                converted[key] = value[:inferred_length].clone()
            else:
                converted[key] = value[:inferred_length]

        return converted, inferred_length

    if custom_core_cells_path:
        core_cells, target_core_n = _load_core_cells_from_path(os.path.abspath(custom_core_cells_path))
        
        # Map latent_embedding to ctrl_cell_emb if needed for model compatibility
        if 'latent_embedding' in core_cells and 'ctrl_cell_emb' not in core_cells:
            core_cells['ctrl_cell_emb'] = core_cells['latent_embedding']
            logger.info("Mapped latent_embedding to ctrl_cell_emb for model compatibility")
        
        # Map plate to batch if needed for model compatibility
        if 'plate' in core_cells and 'batch' not in core_cells:
            plate_data = core_cells['plate']
            # Check if plate is a list of strings (tensor representations)
            if isinstance(plate_data, list) and len(plate_data) > 0:
                if isinstance(plate_data[0], str) and 'tensor' in plate_data[0].lower():
                    # Parse string tensor representations
                    import re
                    parsed_tensors = []
                    for plate_str in plate_data:
                        # Extract the numbers from the string representation
                        numbers = re.findall(r'[-+]?\d*\.?\d+', plate_str)
                        parsed_tensors.append(torch.tensor([float(n) for n in numbers]))
                    core_cells['batch'] = torch.stack(parsed_tensors)
                    logger.info("Parsed %d plate strings to batch tensor with shape %s", len(parsed_tensors), core_cells['batch'].shape)
                elif isinstance(plate_data[0], torch.Tensor):
                    core_cells['batch'] = torch.stack(plate_data)
                    logger.info("Stacked %d plate tensors to batch tensor", len(plate_data))
                else:
                    # Assume it's batch indices
                    core_cells['batch'] = torch.tensor(plate_data)
                    logger.info("Converted plate list to batch tensor")
            elif isinstance(plate_data, torch.Tensor):
                core_cells['batch'] = plate_data
                logger.info("Using plate tensor as batch")
            else:
                logger.warning("Could not convert plate to batch, type: %s", type(plate_data))
        
        logger.info(
            "Loaded custom core_cells batch with size %d from %s.",
            target_core_n,
            custom_core_cells_path,
        )
    else:
        eval_loader = _create_filtered_loader(data_module)
        target_core_n = target_core_n_default
        accum = {}

        def _append_field(store, key, value):
            if key not in store:
                store[key] = []
            store[key].append(value)

        for batch in eval_loader:
            names = _to_list(batch.get("pert_name", []))
            if not names:
                continue
            mask = torch.tensor([str(item) == str(control_pert) for item in names], dtype=torch.bool)
            if mask.sum().item() == 0:
                continue

            current_count = accum.get("_count", 0)
            take = min(target_core_n - current_count, int(mask.sum().item()))
            if take <= 0:
                break

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    mask_device = mask.to(value.device)
                    selected = value[mask_device][:take].detach().clone()
                    _append_field(accum, key, selected)
                else:
                    vals = _to_list(value)
                    selected_vals = [vals[idx] for idx, keep in enumerate(mask.tolist()) if keep][:take]
                    _append_field(accum, key, selected_vals)

            accum["_count"] = current_count + take
            if accum["_count"] >= target_core_n:
                break

        if accum.get("_count", 0) < target_core_n:
            raise RuntimeError(
                f"Could not assemble {target_core_n} control cells; gathered only {accum.get('_count', 0)}."
            )

        core_cells = {}
        for key, parts in accum.items():
            if key == "_count":
                continue
            if len(parts) == 1:
                val = parts[0]
            else:
                if isinstance(parts[0], torch.Tensor):
                    val = torch.cat(parts, dim=0)
                else:
                    merged = []
                    for part in parts:
                        merged.extend(_to_list(part))
                    val = merged
            core_cells[key] = (
                val[:target_core_n]
                if isinstance(val, torch.Tensor)
                else _to_list(val)[:target_core_n]
            )

        logger.info("Constructed core_cells batch with size %d.", target_core_n)

    os.makedirs(results_dir_default, exist_ok=True)
    baseline_path = os.path.join(results_dir_default, "core_cells_baseline.npy")
    _save_numpy_snapshot(core_cells, baseline_path, "baseline core_cells batch")

    perts_order = list(unique_perts)
    num_perts = len(perts_order)
    output_dim = var_dims["output_dim"]
    gene_dim = var_dims.get("gene_dim", 0)
    hvg_dim = var_dims.get("hvg_dim", 0)

    logger.info("Running first-pass predictions across %d perturbations...", num_perts)

    first_pass_preds = np.empty((num_perts, target_core_n, output_dim), dtype=np.float32)
    first_pass_real = np.empty((num_perts, target_core_n, output_dim), dtype=np.float32)

    embed_key = getattr(data_module, "embed_key", None) or "latent_embedding"
    output_space = cfg["data"]["kwargs"].get("output_space", "embedding")
    store_counts = output_space in {"gene", "all"}

    first_pass_counts = None
    first_pass_counts_pred = None
    if store_counts:
        feature_dim = hvg_dim if output_space == "gene" and hvg_dim else gene_dim
        if feature_dim > 0:
            first_pass_counts = np.empty((num_perts, target_core_n, feature_dim), dtype=np.float32)
            first_pass_counts_pred = np.empty((num_perts, target_core_n, feature_dim), dtype=np.float32)
        else:
            store_counts = False

    metadata = {
        "pert_name": [],
        "celltype_name": [],
        "batch": [],
        "pert_cell_barcode": [],
        "ctrl_cell_barcode": [],
    }

    device = next(model.parameters()).device

    # pert_onehot_map was already loaded earlier for perturbation enumeration
    if not pert_onehot_map:
        logger.warning("No pert_onehot_map available; will use zero embeddings for perturbations")
        pert_onehot_map = {}

    def _prepare_pert_emb(pert_name, length):
        vec = pert_onehot_map.get(pert_name)
        if vec is None and control_pert in pert_onehot_map:
            vec = pert_onehot_map[control_pert]
        if vec is None:
            pert_dim = getattr(model, "pert_dim", var_dims.get("pert_dim", 0))
            if pert_dim <= 0:
                raise RuntimeError("pert_dim is undefined; cannot create perturbation embedding")
            vec = torch.zeros(pert_dim)
        return vec.float().unsqueeze(0).repeat(length, 1).to(device)

    with torch.no_grad():
        for p_idx, pert in enumerate(tqdm(perts_order, desc="First pass", unit="pert")):
            batch = {}
            for key, value in core_cells.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.clone().to(device)
                else:
                    batch[key] = list(value)

            if "pert_name" in batch:
                batch["pert_name"] = [pert for _ in range(target_core_n)]
            if "pert_idx" in batch and hasattr(data_module, "get_pert_index"):
                try:
                    idx_val = int(data_module.get_pert_index(pert))
                    batch["pert_idx"] = torch.tensor([idx_val] * target_core_n, device=device)
                except Exception:
                    pass

            batch["pert_emb"] = _prepare_pert_emb(pert, target_core_n)

            batch_preds = model.predict_step(batch, p_idx, padded=False)

            batch_size = batch_preds["preds"].shape[0]
            metadata["pert_name"].extend(_normalize_field(batch_preds.get("pert_name", pert), batch_size, pert))
            metadata["celltype_name"].extend(
                _normalize_field(
                    batch_preds.get("celltype_name"),
                    batch_size,
                    target_celltype,
                )
            )
            metadata["batch"].extend(
                [None if b is None else str(b) for b in _normalize_field(batch_preds.get("batch"), batch_size)]
            )
            metadata["pert_cell_barcode"].extend(
                _normalize_field(batch_preds.get("pert_cell_barcode"), batch_size)
            )
            metadata["ctrl_cell_barcode"].extend(
                _normalize_field(batch_preds.get("ctrl_cell_barcode"), batch_size)
            )

            batch_pred_np = batch_preds["preds"].detach().cpu().numpy().astype(np.float32)
            
            # When using preassembled core cells, pert_cell_emb may not exist (double perturbation case)
            # Use the input ctrl_cell_emb as the "real" baseline in that case
            if batch_preds.get("pert_cell_emb") is not None:
                batch_real_np = batch_preds["pert_cell_emb"].detach().cpu().numpy().astype(np.float32)
            else:
                # Use ctrl_cell_emb (the input embeddings) as the baseline
                batch_real_np = batch["ctrl_cell_emb"].detach().cpu().numpy().astype(np.float32)
                logger.info("Using ctrl_cell_emb as baseline (pert_cell_emb not available)")

            first_pass_preds[p_idx, :, :] = batch_pred_np
            first_pass_real[p_idx, :, :] = batch_real_np

            if store_counts and first_pass_counts is not None and batch_preds.get("pert_cell_counts") is not None:
                counts_np = batch_preds["pert_cell_counts"].detach().cpu().numpy().astype(np.float32)
                first_pass_counts[p_idx, :, :] = counts_np

            if (
                store_counts
                and first_pass_counts_pred is not None
                and batch_preds.get("pert_cell_counts_preds") is not None
            ):
                counts_pred_np = batch_preds["pert_cell_counts_preds"].detach().cpu().numpy().astype(np.float32)
                first_pass_counts_pred[p_idx, :, :] = counts_pred_np

    logger.info("First pass complete across %d perturbations.", num_perts)

    metadata_df = pd.DataFrame(metadata)
    if metadata_df.empty:
        raise RuntimeError("No metadata collected during first pass; cannot proceed.")

    pert_col = getattr(data_module, "pert_col", None) or "perturbation"
    cell_type_col = getattr(data_module, "cell_type_key", None) or "cell_type"
    batch_col = getattr(data_module, "batch_col", None) or "batch"

    obs_df = pd.DataFrame(
        {
            pert_col: metadata_df["pert_name"],
            cell_type_col: metadata_df["celltype_name"],
            batch_col: metadata_df["batch"],
        }
    )
    if metadata_df["pert_cell_barcode"].notna().any():
        obs_df["pert_cell_barcode"] = metadata_df["pert_cell_barcode"]
    if metadata_df["ctrl_cell_barcode"].notna().any():
        obs_df["ctrl_cell_barcode"] = metadata_df["ctrl_cell_barcode"]

    first_pass_pred_flat = first_pass_preds.reshape(num_perts * target_core_n, output_dim)
    first_pass_real_flat = first_pass_real.reshape(num_perts * target_core_n, output_dim)

    if store_counts and first_pass_counts is not None and first_pass_counts_pred is not None:
        feature_dim = first_pass_counts.shape[-1]
        gene_names = var_dims.get("gene_names")
        if gene_names is not None and len(gene_names) == feature_dim:
            var_index = pd.Index([str(name) for name in gene_names], name="gene")
        else:
            var_index = pd.Index([f"feature_{idx}" for idx in range(feature_dim)], name="feature")
        var_df = pd.DataFrame(index=var_index)

        pred_X = first_pass_counts_pred.reshape(num_perts * target_core_n, feature_dim)
        real_X = first_pass_counts.reshape(num_perts * target_core_n, feature_dim)
    else:
        var_index = pd.Index([f"embedding_{idx}" for idx in range(output_dim)], name="embedding")
        var_df = pd.DataFrame(index=var_index)
        pred_X = first_pass_pred_flat
        real_X = first_pass_real_flat

    first_pass_pred_adata = anndata.AnnData(X=pred_X, obs=obs_df.copy(), var=var_df.copy())
    first_pass_real_adata = anndata.AnnData(X=real_X, obs=obs_df.copy(), var=var_df.copy())
    first_pass_pred_adata.obsm[embed_key] = first_pass_pred_flat
    first_pass_real_adata.obsm[embed_key] = first_pass_real_flat

    first_pass_pred_path = os.path.join(results_dir_default, "first_pass_preds.h5ad")
    first_pass_real_path = os.path.join(results_dir_default, "first_pass_real.h5ad")
    first_pass_pred_adata.write_h5ad(first_pass_pred_path)
    first_pass_real_adata.write_h5ad(first_pass_real_path)
    logger.info("Saved first-pass predicted adata to %s", first_pass_pred_path)
    logger.info("Saved first-pass real adata to %s", first_pass_real_path)

    np.save(os.path.join(results_dir_default, "first_pass_preds.npy"), first_pass_preds)
    np.save(os.path.join(results_dir_default, "first_pass_real.npy"), first_pass_real)
    if first_pass_counts is not None:
        np.save(os.path.join(results_dir_default, "first_pass_counts.npy"), first_pass_counts)
    if first_pass_counts_pred is not None:
        np.save(os.path.join(results_dir_default, "first_pass_counts_pred.npy"), first_pass_counts_pred)

    if phase_one_only:
        logger.info("Phase one complete; skipping metrics as requested.")
        return

    if args.predict_only:
        return

    if cell_type_col not in first_pass_real_adata.obs.columns:
        logger.warning(
            "Cell type column '%s' not found in observations; skipping metric computation.",
            cell_type_col,
        )
        return

    control_pert = data_module.get_control_pert()
    ct_split_real = split_anndata_on_celltype(
        adata=first_pass_real_adata,
        celltype_col=cell_type_col,
    )
    ct_split_pred = split_anndata_on_celltype(
        adata=first_pass_pred_adata,
        celltype_col=cell_type_col,
    )

    if len(ct_split_real) != len(ct_split_pred):
        logger.warning(
            "Number of celltypes in real and predicted AnnData objects differ (%d vs %d); skipping metrics.",
            len(ct_split_real),
            len(ct_split_pred),
        )
        return

    pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)
    for celltype in ct_split_real.keys():
        real_ct = ct_split_real[celltype]
        pred_ct = ct_split_pred[celltype]

        metric_configs = {}
        if data_module.embed_key and data_module.embed_key != "X_hvg":
                metric_configs = {
                "discrimination_score": {"embed_key": embed_key},
                "pearson_edistance": {"embed_key": embed_key, "n_jobs": -1},
            }
        else:
            metric_configs = {"pearson_edistance": {"n_jobs": -1}}

        evaluator = MetricsEvaluator(
            adata_pred=pred_ct,
            adata_real=real_ct,
            control_pert=control_pert,
            pert_col=pert_col,
            outdir=results_dir_default,
            prefix=str(celltype),
            pdex_kwargs=pdex_kwargs,
            batch_size=2048,
        )
        evaluator.compute(
            profile=args.profile,
            metric_configs=metric_configs,
            skip_metrics=["pearson_edistance", "clustering_agreement"],
        )


def save_core_cells_real_preds(args: ap.ArgumentParser) -> None:
    """Run only phase one of the pipeline and persist real core-cell embeddings per perturbation."""
    return run_tx_single(args, phase_one_only=True)
