"""
Gene embeddings loader and utilities for incorporating gene embeddings into batches.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class GeneEmbeddingLoader:
    """
    Loads and manages gene embeddings from a pickle file.
    Provides zero-padding for genes not found in the embedding dictionary.
    """
    
    def __init__(self, embedding_file: str, embedding_dim: int = 1536):
        """
        Initialize the gene embedding loader.
        
        Args:
            embedding_file: Path to pickle file containing gene embeddings
            embedding_dim: Dimension of gene embeddings (default: 1536 for scGenePT gene embeddings)
        """
        self.embedding_file = Path(embedding_file)
        self.embedding_dim = embedding_dim
        self.embeddings_dict = self._load_embeddings()
        self.zero_embedding = np.zeros(embedding_dim, dtype=np.float32)
        
        logger.info(f"Loaded {len(self.embeddings_dict)} gene embeddings from {embedding_file}")
    
    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load gene embeddings from pickle file."""
        if not self.embedding_file.exists():
            raise FileNotFoundError(f"Gene embeddings file not found: {self.embedding_file}")
        
        with open(self.embedding_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Convert to numpy arrays if needed
        for key in list(embeddings.keys()):
            if not isinstance(embeddings[key], np.ndarray):
                embeddings[key] = np.array(embeddings[key], dtype=np.float32)
        
        return embeddings
    
    def get_embedding(self, gene_name: str) -> np.ndarray:
        """
        Get embedding for a single gene.
        Returns zero vector if gene not found.
        
        Args:
            gene_name: Name of the gene
            
        Returns:
            Embedding vector of shape [embedding_dim]
        """
        if gene_name in self.embeddings_dict:
            return self.embeddings_dict[gene_name]
        else:
            # Return zero padding for unknown genes
            return self.zero_embedding.copy()
    
    def get_embeddings_batch(self, gene_names: list) -> np.ndarray:
        """
        Get embeddings for a list of genes.
        
        Args:
            gene_names: List of gene names
            
        Returns:
            Array of shape [len(gene_names), embedding_dim]
        """
        embeddings = []
        for gene_name in gene_names:
            embeddings.append(self.get_embedding(gene_name))
        return np.stack(embeddings, axis=0)
    
    def get_embedding_tensor(self, gene_name: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get embedding as PyTorch tensor.
        
        Args:
            gene_name: Name of the gene
            device: Device to place tensor on
            
        Returns:
            Tensor of shape [embedding_dim]
        """
        embedding = self.get_embedding(gene_name)
        tensor = torch.from_numpy(embedding)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    def get_embeddings_batch_tensor(
        self, 
        gene_names: list, 
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get embeddings for a list of genes as PyTorch tensor.
        
        Args:
            gene_names: List of gene names
            device: Device to place tensor on
            
        Returns:
            Tensor of shape [len(gene_names), embedding_dim]
        """
        embeddings = self.get_embeddings_batch(gene_names)
        tensor = torch.from_numpy(embeddings)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


def create_gene_embedding_collate_fn(
    base_collate_fn,
    gene_embedding_loader: Optional[GeneEmbeddingLoader] = None,
    pert_col: str = "target_gene",
):
    """
    Create a custom collate function that adds gene embeddings to batches.
    
    Args:
        base_collate_fn: Original collate function from data module
        gene_embedding_loader: GeneEmbeddingLoader instance
        pert_col: Column name for perturbation (gene) names
        
    Returns:
        Modified collate function that includes gene embeddings
    """
    def collate_with_gene_embeddings(batch_list):
        batch = base_collate_fn(batch_list)
        if gene_embedding_loader is None:
            return batch
        
        if "pert_name" in batch:
            pert_names = batch["pert_name"]
            
            if isinstance(pert_names, (list, tuple)):
                gene_names = pert_names
            else:
                logger.warning("Cannot extract gene names from batch, skipping gene embeddings")
                return batch
            
            gene_embeddings = gene_embedding_loader.get_embeddings_batch(gene_names)    
            batch["gene_emb"] = torch.from_numpy(gene_embeddings).float()
        else:
            logger.warning("'pert_name' not found in batch, cannot add gene embeddings")
        
        return batch
    
    return collate_with_gene_embeddings


def inject_gene_embeddings_into_datamodule(
    data_module,
    gene_embeddings_file: str,
    gene_emb_dim: int = 1536,
):
    """
    Inject gene embedding functionality into an existing data module.
    
    Args:
        data_module: PerturbationDataModule instance
        gene_embeddings_file: Path to gene embeddings pickle file
        gene_emb_dim: Dimension of gene embeddings
    """
    gene_loader = GeneEmbeddingLoader(gene_embeddings_file, gene_emb_dim)
    data_module.gene_embedding_loader = gene_loader
    original_collate_fn = data_module.collate_fn if hasattr(data_module, 'collate_fn') else None
    
    if original_collate_fn is not None:
        data_module.collate_fn = create_gene_embedding_collate_fn(
            original_collate_fn,
            gene_loader,
        )
        logger.info("Gene embedding collate function injected into data module")
    else:
        logger.warning("Data module has no collate_fn attribute, gene embeddings may not work")
    
    return data_module
