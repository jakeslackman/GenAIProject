"""Utilities for working with precomputed gene expression bins."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch


@dataclass(frozen=True)
class GeneBins:
    """Gene expression binning information.

    Attributes:
        num_genes: Number of genes (G)
        num_bins: Total number of bins including bin_0
        pos_ends: [G, K] tensor of upper bounds for positive bins (K = num_bins - 1)
        modes: [G, num_bins] tensor of per-bin modes (bin_0 included)
        medians: [G, num_bins] tensor of per-bin medians (bin_0 included)
        valid: [G] boolean tensor indicating genes with at least one positive bin
    """

    num_genes: int
    num_bins: int
    pos_ends: torch.Tensor
    modes: torch.Tensor
    medians: torch.Tensor
    valid: torch.Tensor


def load_gene_bins(gene_bin_file: str | Path) -> GeneBins:
    """Load gene bins from a pickle file produced by the binning script.

    Args:
        gene_bin_file: Path to pickle file containing per-gene bin tuples

    Returns:
        A ``GeneBins`` object with tensors stored on CPU.
    """

    gene_bin_file = Path(gene_bin_file)
    with gene_bin_file.open("rb") as fh:
        bins_dict: Dict[int, list[tuple[float, float, float, float, float]]] = pickle.load(fh)

    if not bins_dict:
        raise ValueError(f"Empty bins file: {gene_bin_file}")

    num_genes = len(bins_dict)
    bin_counts = {len(bins) for bins in bins_dict.values()}
    if len(bin_counts) != 1:
        raise ValueError(f"Inconsistent number of bins across genes: {sorted(bin_counts)}")

    num_bins = bin_counts.pop()
    if num_bins < 1:
        raise ValueError(f"Expected at least 1 bin (bin_0), got {num_bins}")

    num_pos_bins = num_bins - 1
    pos_ends = torch.zeros((num_genes, num_pos_bins), dtype=torch.float32)
    modes = torch.zeros((num_genes, num_bins), dtype=torch.float32)
    medians = torch.zeros((num_genes, num_bins), dtype=torch.float32)
    valid = torch.zeros(num_genes, dtype=torch.bool)

    for gene_idx in range(num_genes):
        if gene_idx not in bins_dict:
            raise ValueError(f"Missing gene index {gene_idx} in bins file")

        bins = bins_dict[gene_idx]
        if len(bins) != num_bins:
            raise ValueError(
                f"Gene {gene_idx} has {len(bins)} bins but expected {num_bins}"
            )

        has_positive = False
        for bin_idx, (_, end, mode, median, _) in enumerate(bins):
            modes[gene_idx, bin_idx] = mode
            medians[gene_idx, bin_idx] = median
            if bin_idx > 0:
                pos_bin_idx = bin_idx - 1
                pos_ends[gene_idx, pos_bin_idx] = end
                if end > 0.0 and not has_positive:
                    has_positive = True

        valid[gene_idx] = has_positive

    return GeneBins(
        num_genes=num_genes,
        num_bins=num_bins,
        pos_ends=pos_ends,
        modes=modes,
        medians=medians,
        valid=valid,
    )


def compute_bin_indices(
    expr: torch.Tensor,
    pos_ends: torch.Tensor,
    valid: torch.Tensor,
    zero_threshold: float,
    chunk_size: int | None = 100000,
) -> torch.Tensor:
    """Map expression tensor to per-gene bin indices.

    Args:
        expr: Expression values of shape [..., G]
        pos_ends: [G, K] tensor with positive bin upper bounds
        valid: [G] boolean tensor marking genes with positive bins
        zero_threshold: Values <= threshold map to bin_0
        chunk_size: Optional chunk size for memory-friendly processing

    Returns:
        Long tensor of bin indices with shape matching ``expr``.
    """

    original_shape = expr.shape
    device = expr.device

    pos_ends = pos_ends.to(device)
    valid = valid.to(device)

    g = expr.shape[-1]
    if pos_ends.shape[0] != g:
        raise ValueError(f"pos_ends has {pos_ends.shape[0]} genes but expr has {g}")
    if valid.shape[0] != g:
        raise ValueError(f"valid has {valid.shape[0]} genes but expr has {g}")

    expr_flat = expr.reshape(-1, g)
    n_cells = expr_flat.shape[0]

    k = pos_ends.shape[1]
    if k == 0:
        return torch.zeros_like(expr_flat, dtype=torch.long).reshape(original_shape)

    def _compute_chunk(chunk: torch.Tensor) -> torch.Tensor:
        chunk_bins = torch.zeros((chunk.shape[0], g), device=device, dtype=torch.long)
        for gene_idx in range(g):
            boundaries = pos_ends[gene_idx].contiguous()
            # searchsorted expects ascending boundaries; returns index in [0, K]
            gene_bins = torch.searchsorted(boundaries, chunk[:, gene_idx].contiguous(), right=False)
            chunk_bins[:, gene_idx] = gene_bins + 1

        chunk_bins = torch.clamp(chunk_bins, min=1, max=k)
        zero_mask = chunk <= zero_threshold
        chunk_bins = torch.where(zero_mask, torch.zeros_like(chunk_bins), chunk_bins)
        if valid is not None:
            chunk_bins = torch.where(
                valid.unsqueeze(0),
                chunk_bins,
                torch.zeros_like(chunk_bins),
            )
        return chunk_bins

    if chunk_size is None or n_cells <= chunk_size:
        bin_idx = _compute_chunk(expr_flat)
    else:
        outputs = []
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            outputs.append(_compute_chunk(expr_flat[start:end]))
        bin_idx = torch.cat(outputs, dim=0)

    return bin_idx.reshape(original_shape)

