"""
Modified data loading utilities for gene embeddings integration.
This module provides hooks to inject gene embeddings into the data pipeline.
"""

import logging
from typing import Optional, List
import torch
import numpy as np

logger = logging.getLogger(__name__)


def add_gene_embeddings_to_batch(
    batch: dict,
    gene_embedding_loader,
    pert_onehot_map: Optional[dict] = None,
) -> dict:
    """
    Add gene embeddings to a batch based on perturbation names.
    
    Args:
        batch: Batch dictionary from dataloader
        gene_embedding_loader: GeneEmbeddingLoader instance
        pert_onehot_map: Mapping from perturbation names to indices (optional)
        
    Returns:
        Modified batch with gene_emb field added
    """
    gene_names = []
    
    # Method 1: Direct pert_name field
    if "pert_name" in batch:
        pert_names = batch["pert_name"]
        if isinstance(pert_names, (list, tuple)):
            gene_names = list(pert_names)
        elif isinstance(pert_names, torch.Tensor):
            # If it's tensor of strings (some datasets)
            gene_names = pert_names.tolist()
        elif isinstance(pert_names, np.ndarray):
            gene_names = pert_names.tolist()
    
    # Method 2: Decode from pert_emb one-hot using pert_onehot_map
    elif "pert_emb" in batch and pert_onehot_map is not None:
        pert_emb = batch["pert_emb"]
        if isinstance(pert_emb, torch.Tensor):
            # Get indices from one-hot
            pert_indices = pert_emb.argmax(dim=-1).cpu().numpy()
            # Reverse map from onehot_map
            idx_to_gene = {v: k for k, v in pert_onehot_map.items()}
            gene_names = [idx_to_gene.get(int(idx), "UNKNOWN") for idx in pert_indices.flatten()]
    
    if not gene_names:
        logger.warning("Could not extract gene names from batch")
        return batch
    
    # Get embeddings
    gene_embeddings = gene_embedding_loader.get_embeddings_batch(gene_names)
    
    # Add to batch - same shape as pert_emb
    batch["gene_emb"] = torch.from_numpy(gene_embeddings).float()
    
    return batch


class GeneEmbeddingDataLoader:
    """
    Wrapper around a DataLoader that adds gene embeddings to batches.
    """
    
    def __init__(
        self,
        base_dataloader,
        gene_embedding_loader,
        pert_onehot_map: Optional[dict] = None,
    ):
        """
        Initialize wrapper.
        
        Args:
            base_dataloader: Original DataLoader
            gene_embedding_loader: GeneEmbeddingLoader instance
            pert_onehot_map: Optional mapping from gene names to indices
        """
        self.base_dataloader = base_dataloader
        self.gene_embedding_loader = gene_embedding_loader
        self.pert_onehot_map = pert_onehot_map
    
    def __iter__(self):
        for batch in self.base_dataloader:
            # Add gene embeddings to batch
            batch = add_gene_embeddings_to_batch(
                batch,
                self.gene_embedding_loader,
                self.pert_onehot_map,
            )
            yield batch
    
    def __len__(self):
        return len(self.base_dataloader)
    
    # Delegate attribute access to base dataloader
    def __getattr__(self, name):
        return getattr(self.base_dataloader, name)


def wrap_datamodule_with_gene_embeddings(
    data_module,
    gene_embedding_loader,
):
    """
    Wrap a PerturbationDataModule to add gene embeddings to all dataloaders.
    
    Args:
        data_module: PerturbationDataModule instance
        gene_embedding_loader: GeneEmbeddingLoader instance
        
    Returns:
        Modified data module
    """
    # Store original dataloader methods
    original_train_dataloader = data_module.train_dataloader
    original_val_dataloader = data_module.val_dataloader
    original_test_dataloader = data_module.test_dataloader
    
    # Get pert_onehot_map
    pert_onehot_map = getattr(data_module, 'pert_onehot_map', None)
    
    # Wrap train_dataloader
    def wrapped_train_dataloader(*args, **kwargs):
        base_dl = original_train_dataloader(*args, **kwargs)
        return GeneEmbeddingDataLoader(base_dl, gene_embedding_loader, pert_onehot_map)
    
    # Wrap val_dataloader
    def wrapped_val_dataloader(*args, **kwargs):
        base_dl = original_val_dataloader(*args, **kwargs)
        return GeneEmbeddingDataLoader(base_dl, gene_embedding_loader, pert_onehot_map)
    
    # Wrap test_dataloader
    def wrapped_test_dataloader(*args, **kwargs):
        base_dl = original_test_dataloader(*args, **kwargs)
        return GeneEmbeddingDataLoader(base_dl, gene_embedding_loader, pert_onehot_map)
    
    # Replace methods
    data_module.train_dataloader = wrapped_train_dataloader
    data_module.val_dataloader = wrapped_val_dataloader
    data_module.test_dataloader = wrapped_test_dataloader
    
    # Store loader reference
    data_module.gene_embedding_loader = gene_embedding_loader
    
    logger.info("Successfully wrapped data module with gene embedding functionality")
    
    return data_module
