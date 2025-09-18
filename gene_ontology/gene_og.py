import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pickle
import toml

import pandas as pd
import anndata as ad


def load_go_gene_sets(go_dir: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    Load GO gene sets from MSigDB JSON exports.

    Returns a nested mapping:
      collection_key -> gene_symbol_upper -> set of pathway names

    Where collection_key is one of: "bp", "cc", "mf".
    """
    collection_files = {
        "bp": go_dir / "c5.go.bp.v2025.1.Hs.json",
        "cc": go_dir / "c5.go.cc.v2025.1.Hs.json",
        "mf": go_dir / "c5.go.mf.v2025.1.Hs.json",
    }

    collection_to_gene_to_paths: Dict[str, Dict[str, Set[str]]] = {
        key: {} for key in collection_files.keys()
    }

    for collection_key, json_path in collection_files.items():
        if not json_path.exists():
            raise FileNotFoundError(f"Missing GO file: {json_path}")

        # File structure is a single JSON object: { pathway_name: { geneSymbols: [...] , ... }, ... }
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gene_to_paths: Dict[str, Set[str]] = collection_to_gene_to_paths[collection_key]

        for pathway_name, meta in data.items():
            genes: List[str] = meta.get("geneSymbols", [])
            for gene_symbol in genes:
                gene_upper = str(gene_symbol).strip().upper()
                if not gene_upper:
                    continue
                if gene_upper not in gene_to_paths:
                    gene_to_paths[gene_upper] = set()
                gene_to_paths[gene_upper].add(pathway_name)

    return collection_to_gene_to_paths


def detect_gene_column(columns: List[str]) -> str:
    """Heuristically detect the gene symbol column in a ranked CSV."""
    candidates = [
        "feature"
    ]
    lower_to_original: Dict[str, str] = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    # Fall back to first column
    return columns[0]


def build_annotation_maps(
    gene_series: pd.Series,
    go_maps: Dict[str, Dict[str, Set[str]]],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    For a Series of gene symbols, build 8 Series for:
      bp_paths, bp_count, cc_paths, cc_count, mf_paths, mf_count, any_paths, any_count
    """
    def annot_for_gene(gene_value: object) -> Tuple[str, int, str, int, str, int, str, int]:
        if pd.isna(gene_value):
            return "", 0, "", 0, "", 0, "", 0
        gene_upper = str(gene_value).strip().upper()
        bp = go_maps["bp"].get(gene_upper, set())
        cc = go_maps["cc"].get(gene_upper, set())
        mf = go_maps["mf"].get(gene_upper, set())
        any_paths: Set[str] = set().union(bp, cc, mf)
        bp_list = sorted(bp)
        cc_list = sorted(cc)
        mf_list = sorted(mf)
        any_list = sorted(any_paths)
        return (
            ";".join(bp_list),
            len(bp_list),
            ";".join(cc_list),
            len(cc_list),
            ";".join(mf_list),
            len(mf_list),
            ";".join(any_list),
            len(any_list),
        )

    annots: List[Tuple[str, int, str, int, str, int, str, int]] = [
        annot_for_gene(g) for g in gene_series
    ]
    df_ann = pd.DataFrame(
        annots,
        columns=[
            "go_bp_paths",
            "go_bp_count",
            "go_cc_paths",
            "go_cc_count",
            "go_mf_paths",
            "go_mf_count",
            "go_any_paths",
            "go_any_count",
        ],
        index=gene_series.index,
    )
    return (
        df_ann["go_bp_paths"],
        df_ann["go_bp_count"],
        df_ann["go_cc_paths"],
        df_ann["go_cc_count"],
        df_ann["go_mf_paths"],
        df_ann["go_mf_count"],
        df_ann["go_any_paths"],
        df_ann["go_any_count"],
    )


def annotate_ranked_file(
    input_csv: Path,
    output_csv: Path,
    go_maps: Dict[str, Dict[str, Set[str]]],
    gene_mapping: Dict[str, str],
    limit_rows: Optional[int] = None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Read CSV. Use python engine for safety; tolerate extra spaces.
    df = pd.read_csv(
        input_csv,
        engine="python",
        dtype=str,
        skipinitialspace=True,
    )

    if limit_rows is not None and limit_rows > 0:
        df = df.head(limit_rows)

    if df.empty:
        df.to_csv(output_csv, index=False)
        return

    # Enforce annotating by feature only (do not annotate target)
    if "feature" not in df.columns:
        raise ValueError("Expected 'feature' column in input CSV; cannot annotate without it.")

    # Map feature indices (as strings) to gene symbols using the provided AnnData mapping
    feature_series: pd.Series = df["feature"].astype(str)
    gene_series: pd.Series = feature_series.map(gene_mapping).fillna(feature_series)
    # Insert human-readable gene symbol right after the feature column
    try:
        feature_idx = df.columns.get_loc("feature")
        df.insert(feature_idx + 1, "feature_gene_symbol", gene_series)
    except Exception:
        # Fallback: append at the end if any issue determining location
        df.insert(len(df.columns), "feature_gene_symbol", gene_series)

    (
        go_bp_paths,
        go_bp_count,
        go_cc_paths,
        go_cc_count,
        go_mf_paths,
        go_mf_count,
        go_any_paths,
        go_any_count,
    ) = build_annotation_maps(gene_series, go_maps)

    df.insert(len(df.columns), "go_bp_paths", go_bp_paths)
    df.insert(len(df.columns), "go_bp_count", go_bp_count)
    df.insert(len(df.columns), "go_cc_paths", go_cc_paths)
    df.insert(len(df.columns), "go_cc_count", go_cc_count)
    df.insert(len(df.columns), "go_mf_paths", go_mf_paths)
    df.insert(len(df.columns), "go_mf_count", go_mf_count)
    df.insert(len(df.columns), "go_any_paths", go_any_paths)
    df.insert(len(df.columns), "go_any_count", go_any_count)

    # Alias total pathways as go_total_count for convenience/clarity
    df.insert(len(df.columns), "go_total_count", go_any_count)

    # Compute interaction features relative to fold/percent change, if present
    go_total_numeric = pd.to_numeric(go_any_count, errors="coerce")

    if "fold_change" in df.columns:
        fold_change_numeric = pd.to_numeric(df["fold_change"], errors="coerce")
        df.insert(
            len(df.columns),
            "fold_change_x_go_total",
            fold_change_numeric * go_total_numeric,
        )
    else:
        df.insert(len(df.columns), "fold_change_x_go_total", pd.Series([pd.NA] * len(df), index=df.index))

    if "percent_change" in df.columns:
        percent_change_numeric = pd.to_numeric(df["percent_change"], errors="coerce")
        df.insert(
            len(df.columns),
            "percent_change_x_go_total",
            percent_change_numeric * go_total_numeric,
        )
    else:
        df.insert(len(df.columns), "percent_change_x_go_total", pd.Series([pd.NA] * len(df), index=df.index))

    df.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate ranked gene CSVs with GO pathways")
    parser.add_argument(
        "--go_dir",
        type=Path,
        default=Path("/home/dhruvgautam/state/gene_ontology"),
        help="Directory containing GO JSON files",
    )
    parser.add_argument(
        "--adata_path",
        type=Path,
        default=Path("/large_storage/ctc/userspace/aadduri/revisions/replogle_nogwps_state_sm_batch/rpe1/eval_step=60000.ckpt/adata_real.h5ad"),
        help="Path to .h5ad file used to map feature indices to gene symbols",
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("/home/dhruvgautam/state/gene_ontology/rpe1_pred_de_ranked.csv"),
        help="Optional single CSV file to annotate (only this file will be processed)",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/dhruvgautam/state/gene_ontology"),
        help="Directory with ranked CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/dhruvgautam/state/gene_ontology"),
        help="Directory to write annotated CSV files",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("/home/dhruvgautam/state/gene_ontology/annotated_rpe1_pred_ranked.csv"),
        help="Optional explicit output CSV path (used only with --input_csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="When --input_csv is provided, limit processing to the first N rows",
    )
    args = parser.parse_args()

    go_maps = load_go_gene_sets(args.go_dir)
    # Build feature-index -> gene-symbol mapping from AnnData
    if not args.adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {args.adata_path}")
    adata = ad.read_h5ad(args.adata_path)

    # Determine gene symbols for each feature index
    if "gene_symbols" in adata.var.columns:
        gene_symbols_series = adata.var["gene_symbols"]
    elif "gene_names" in adata.var.columns:
        gene_symbols_series = adata.var["gene_names"]
    else:
        # Try to locate a sidecar var_dims.pkl containing 'gene_names'
        gene_symbols_series = None
        possible_pkl_paths: List[Path] = []
        try:
            # Check a few likely locations relative to the adata path
            possible_pkl_paths.append(args.adata_path.parent / "var_dims.pkl")
            if len(args.adata_path.parents) > 1:
                possible_pkl_paths.append(args.adata_path.parents[1] / "var_dims.pkl")
            if len(args.adata_path.parents) > 2:
                possible_pkl_paths.append(args.adata_path.parents[2] / "var_dims.pkl")
        except Exception:
            pass

        sidecar_gene_names: Optional[List[str]] = None
        for pkl_path in possible_pkl_paths:
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        meta = pickle.load(f)
                    if isinstance(meta, dict) and "gene_names" in meta:
                        names = meta["gene_names"]
                        if isinstance(names, list) and len(names) == adata.n_vars:
                            sidecar_gene_names = names
                            print(f"Using gene names from sidecar: {pkl_path}")
                            break
                        else:
                            print(
                                f"Found {pkl_path} with 'gene_names' of length {len(names) if isinstance(names, list) else 'unknown'} not matching n_vars={adata.n_vars}; skipping"
                            )
                except Exception as e:
                    print(f"Warning: Failed to read {pkl_path}: {e}")

        if sidecar_gene_names is not None:
            gene_symbols_series = pd.Series(sidecar_gene_names, index=adata.var_names)
        else:
            # Fallback to var_names as a last resort
            gene_symbols_series = adata.var_names
            print("Warning: No gene names found in adata.var and no valid var_dims.pkl; using var_names indices instead")

    gene_mapping: Dict[str, str] = {}
    for idx, symbol in enumerate(gene_symbols_series):
        gene_mapping[str(idx)] = str(symbol)

    if args.input_csv is not None:
        input_csv: Path = args.input_csv
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        output_csv: Path = args.output_csv if args.output_csv is not None else (args.output_dir / input_csv.name)
        annotate_ranked_file(input_csv, output_csv, go_maps, gene_mapping, limit_rows=args.limit)
        print(f"Annotated {input_csv} to {output_csv}")
    else:
        # Iterate CSVs in input_dir (non-recursive), skip subdirectories (e.g., ranked_old)
        for entry in sorted(args.input_dir.iterdir()):
            if entry.is_dir():
                # Skip subdirectories by default
                continue
            if entry.suffix.lower() != ".csv":
                continue
            output_csv = args.output_dir / entry.name
            annotate_ranked_file(entry, output_csv, go_maps, gene_mapping)
    print(f"Annotated {len(args.input_csv)} CSV files and {len(args.input_dir)} directories.")


if __name__ == "__main__":
    main()


