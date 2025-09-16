#!/usr/bin/env python3
"""
Script to create discrete rankings from differential expression data.

This script processes DE data by:
1. Loading adata to get gene symbol mappings
2. Converting numeric feature IDs to gene symbols
3. Filtering by FDR threshold (default 0.05)
4. Ranking by fold_change (absolute value)
5. Saving all data for the filtered and ranked perturbations
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import anndata

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_adata(filepath: str) -> anndata.AnnData:
    """
    Load AnnData object from h5ad file.
    
    Args:
        filepath: Path to the h5ad file
        
    Returns:
        AnnData object
    """
    logger.info(f"Loading AnnData from {filepath}")
    adata = anndata.read_h5ad(filepath)
    logger.info(f"Loaded adata with shape {adata.shape}")
    return adata


def create_gene_symbol_mapping(adata: anndata.AnnData) -> Dict[str, str]:
    """
    Create mapping from feature indices to gene symbols.
    
    Args:
        adata: AnnData object
        
    Returns:
        Dictionary mapping feature indices (as strings) to gene symbols
    """
    logger.info("Creating gene symbol mapping from adata")
    
    # Get gene symbols from adata.var
    if 'gene_symbols' in adata.var.columns:
        gene_symbols = adata.var['gene_symbols']
    else:
        # Fallback to var_names if gene_symbols not available
        gene_symbols = adata.var_names
        logger.warning("No 'gene_symbols' column found in adata.var, using var_names instead")
    
    # Create mapping from feature index to gene symbol
    mapping = {}
    for idx, symbol in enumerate(gene_symbols):
        mapping[str(idx)] = str(symbol)
    
    logger.info(f"Created mapping for {len(mapping)} genes")
    return mapping


def load_de_data(filepath: str) -> pd.DataFrame:
    """
    Load differential expression data from CSV file.
    
    Args:
        filepath: Path to the CSV file containing DE data
        
    Returns:
        DataFrame with DE data
    """
    logger.info(f"Loading DE data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def convert_feature_ids_to_gene_symbols(df: pd.DataFrame, gene_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Convert numeric feature IDs to gene symbols using the provided mapping.
    
    Args:
        df: DataFrame with DE data containing 'feature' column
        gene_mapping: Dictionary mapping feature indices to gene symbols
        
    Returns:
        DataFrame with feature IDs converted to gene symbols
    """
    logger.info("Converting feature IDs to gene symbols")
    
    # Create a copy to avoid modifying the original
    df_converted = df.copy()
    
    # Convert feature column to string for mapping
    df_converted['feature'] = df_converted['feature'].astype(str)
    
    # Apply the mapping
    df_converted['feature'] = df_converted['feature'].map(gene_mapping)
    
    # Check for unmapped features
    unmapped = df_converted['feature'].isna().sum()
    if unmapped > 0:
        logger.warning(f"Found {unmapped} features that could not be mapped to gene symbols")
        # Remove rows with unmapped features
        df_converted = df_converted.dropna(subset=['feature'])
        logger.info(f"Removed {unmapped} unmapped features, {len(df_converted)} rows remaining")
    
    # Count unique gene symbols
    unique_genes = df_converted['feature'].nunique()
    logger.info(f"Converted to {unique_genes} unique gene symbols")
    
    return df_converted


def filter_by_fdr_func(df: pd.DataFrame, fdr_threshold: float = 0.05) -> pd.DataFrame:
    """
    Filter data by FDR threshold.
    
    Args:
        df: DataFrame with DE data
        fdr_threshold: FDR threshold for filtering (default 0.05)
        
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Filtering by FDR threshold: {fdr_threshold}")
    filtered_df = df[df['fdr'] <= fdr_threshold].copy()
    logger.info(f"After FDR filtering: {len(filtered_df)} rows")
    return filtered_df


def rank_by_fold_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank genes by absolute fold change within each perturbation.
    
    Args:
        df: DataFrame with DE data
        
    Returns:
        DataFrame with ranking information added
    """
    logger.info("Ranking by absolute fold change")
    
    # Calculate absolute fold change for ranking
    df['abs_fold_change'] = np.abs(df['fold_change'])
    
    # Group by target (perturbation) and rank by absolute fold change
    df['rank'] = df.groupby('target')['abs_fold_change'].rank(method='dense', ascending=False)
    
    # Sort by target and rank
    df_sorted = df.sort_values(['target', 'rank']).copy()
    
    logger.info(f"Ranking complete. Found {df_sorted['target'].nunique()} unique perturbations")
    return df_sorted


def process_perturbation_data(df: pd.DataFrame, perturbation: str) -> pd.DataFrame:
    """
    Process data for a specific perturbation.
    
    Args:
        df: DataFrame with DE data
        perturbation: Name of the perturbation to process
        
    Returns:
        DataFrame filtered for the specific perturbation
    """
    pert_data = df[df['target'] == perturbation].copy()
    logger.info(f"Processing perturbation {perturbation}: {len(pert_data)} genes")
    return pert_data


def save_ranked_data(df: pd.DataFrame, output_path: str, perturbation: str = None):
    """
    Save ranked data to CSV file.
    
    Args:
        df: DataFrame with ranked data
        output_path: Path to save the output file
        perturbation: Optional perturbation name for filename
    """
    if perturbation:
        # Create filename with perturbation name
        base_path = Path(output_path)
        filename = f"{base_path.stem}_{perturbation}{base_path.suffix}"
        output_file = base_path.parent / filename
    else:
        output_file = output_path
    
    logger.info(f"Saving ranked data to {output_file}")
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} rows to {output_file}")


def create_discrete_rankings(input_file: str, 
                           output_file: str, 
                           adata_file: str = None,
                           fdr_threshold: float = 0.05,
                           filter_by_fdr: bool = True,
                           perturbation: str = None) -> pd.DataFrame:
    """
    Main function to create discrete rankings from DE data.
    
    Args:
        input_file: Path to input CSV file with DE data
        output_file: Path to output CSV file for ranked data
        adata_file: Path to h5ad file for gene symbol mapping (optional)
        fdr_threshold: FDR threshold for filtering (default 0.05)
        filter_by_fdr: Whether to filter by FDR threshold (default True)
        perturbation: Optional specific perturbation to process
        
    Returns:
        DataFrame with ranked data
    """
    # Load data
    df = load_de_data(input_file)
    
    # Load adata and convert feature IDs to gene symbols if adata_file provided
    if adata_file:
        adata = load_adata(adata_file)
        gene_mapping = create_gene_symbol_mapping(adata)
        df = convert_feature_ids_to_gene_symbols(df, gene_mapping)
    
    # Filter by FDR if requested
    if filter_by_fdr:
        logger.info(f"Filtering by FDR threshold: {fdr_threshold}")
        filtered_df = filter_by_fdr_func(df, fdr_threshold)
        
        if len(filtered_df) == 0:
            logger.warning("No data passed FDR filtering threshold")
            return pd.DataFrame()
    else:
        logger.info("Skipping FDR filtering - using all data")
        filtered_df = df
    
    # Rank by fold change
    ranked_df = rank_by_fold_change(filtered_df)
    
    # Process specific perturbation if provided
    if perturbation:
        if perturbation not in ranked_df['target'].unique():
            logger.error(f"Perturbation '{perturbation}' not found in data")
            logger.info(f"Available perturbations: {sorted(ranked_df['target'].unique())}")
            return pd.DataFrame()
        
        ranked_df = process_perturbation_data(ranked_df, perturbation)
    
    # Save results
    save_ranked_data(ranked_df, output_file, perturbation)
    
    # Print summary statistics
    print_summary_stats(ranked_df, perturbation)
    
    return ranked_df


def print_summary_stats(df: pd.DataFrame, perturbation: str = None):
    """
    Print summary statistics for the ranked data.
    
    Args:
        df: DataFrame with ranked data
        perturbation: Optional perturbation name
    """
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    if perturbation:
        print(f"Perturbation: {perturbation}")
        print(f"Number of genes: {len(df)}")
        print(f"Mean fold change: {df['fold_change'].mean():.4f}")
        print(f"Median fold change: {df['fold_change'].median():.4f}")
        print(f"Max fold change: {df['fold_change'].max():.4f}")
        print(f"Min fold change: {df['fold_change'].min():.4f}")
    else:
        print(f"Total perturbations: {df['target'].nunique()}")
        print(f"Total genes: {len(df)}")
        print(f"Mean genes per perturbation: {len(df) / df['target'].nunique():.1f}")
        
        # Show top 10 perturbations by number of genes
        pert_counts = df['target'].value_counts().head(10)
        print(f"\nTop 10 perturbations by gene count:")
        for pert, count in pert_counts.items():
            print(f"  {pert}: {count} genes")
    
    print("="*50)


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Create discrete rankings from DE data')
    parser.add_argument('input_file', help='Path to input CSV file with DE data')
    parser.add_argument('output_file', help='Path to output CSV file for ranked data')
    parser.add_argument('--adata-file', type=str, default=None,
                       help='Path to h5ad file for gene symbol mapping (optional)')
    parser.add_argument('--fdr-threshold', type=float, default=0.05, 
                       help='FDR threshold for filtering (default: 0.05)')
    parser.add_argument('--filter-by-fdr', action='store_true', default=True,
                       help='Filter data by FDR threshold (default: True)')
    parser.add_argument('--no-filter-by-fdr', action='store_false', dest='filter_by_fdr',
                       help='Skip FDR filtering and use all data')
    parser.add_argument('--perturbation', type=str, default=None,
                       help='Specific perturbation to process (optional)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        return
    
    # Check if adata file exists if provided
    if args.adata_file and not Path(args.adata_file).exists():
        logger.error(f"AnnData file not found: {args.adata_file}")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the data
    ranked_data = create_discrete_rankings(
        input_file=args.input_file,
        output_file=args.output_file,
        adata_file=args.adata_file,
        fdr_threshold=args.fdr_threshold,
        filter_by_fdr=False,
        perturbation=args.perturbation
    )
    
    if len(ranked_data) > 0:
        logger.info("Processing completed successfully!")
    else:
        logger.error("No data was processed. Please check your input and parameters.")


if __name__ == "__main__":
    main() 