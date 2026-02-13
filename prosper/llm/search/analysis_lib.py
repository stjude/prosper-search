"""Analysis library for PROSPER LLM integration.

Provides functions for loading similarity matrices, metadata, and building
LLM-ready JSON payloads from slide search results.

Adapted from the original COMET-LLM analysis_lib.py for the PROSPER pipeline.
"""

import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MIN_SITE_COUNT = 40
TOPK = 10


def load_results_new(sim_in: str) -> pd.DataFrame:
    """Load a similarity/distance matrix from pickle or CSV.

    The matrix should be square with slide IDs as both index and columns.
    If the input is the PROSPER embedding matrix (non-square), it will
    compute pairwise L1 distances automatically.

    Args:
        sim_in: Path to similarity matrix pickle or CSV file.

    Returns:
        Square DataFrame with slide IDs as index and columns.
    """
    try:
        sim = pd.read_pickle(sim_in)
        sim.index = sim.index.astype(str)
        sim.columns = sim.columns.astype(str)
    except Exception:
        sim = pd.read_csv(sim_in, index_col=0)
        sim.index = sim.index.astype(str)
        sim.columns = sim.columns.astype(str)

    # Check if it's a square similarity matrix or an embedding matrix
    if sim.shape[0] != sim.shape[1]:
        print(f"Non-square matrix detected ({sim.shape}). Computing pairwise L1 distances...")
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(sim.values, sim.values, metric='cityblock')
        sim = pd.DataFrame(dist_matrix, index=sim.index, columns=sim.index)
        print(f"Computed distance matrix: {sim.shape}")

    return sim


def load_metadata(
    metadata_path: str,
    min_site_count: int = MIN_SITE_COUNT,
    disease_name: str = 'tumor_group',
    slide_id_col: Optional[str] = None,
    disease_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load and filter metadata from CSV.

    Handles both PROSPER-format CSVs (Tumor.group, Slide.Scan.File) and
    standard format (tumor_group, slide_id).

    Args:
        metadata_path: Path to metadata CSV.
        min_site_count: Minimum number of slides per disease group to keep.
        disease_name: Target column name for disease labels (internal).
        slide_id_col: Override column name for slide IDs.
        disease_col: Override column name for disease labels.

    Returns:
        Filtered metadata DataFrame with 'slide_id' and 'tumor_group' columns.
    """
    all_metadata = pd.read_csv(metadata_path)

    # Auto-detect column names
    if slide_id_col is None:
        if 'slide_id' in all_metadata.columns:
            slide_id_col = 'slide_id'
        elif 'Slide.Scan.File' in all_metadata.columns:
            slide_id_col = 'Slide.Scan.File'
        else:
            raise ValueError(
                f"Cannot detect slide ID column. Available: {list(all_metadata.columns)}"
            )

    if disease_col is None:
        if 'tumor_group' in all_metadata.columns:
            disease_col = 'tumor_group'
        elif 'Tumor.group' in all_metadata.columns:
            disease_col = 'Tumor.group'
        else:
            raise ValueError(
                f"Cannot detect disease column. Available: {list(all_metadata.columns)}"
            )

    # Standardize column names
    all_metadata = all_metadata.rename(columns={
        slide_id_col: 'slide_id',
        disease_col: 'tumor_group',
    })

    all_metadata = all_metadata.dropna(subset=['slide_id'])
    all_metadata['tumor_group'] = all_metadata['tumor_group'].apply(
        lambda x: str(x).upper().replace(' ', '_')
    )

    # Filter by minimum site count
    populated_sites = all_metadata['tumor_group'].value_counts()
    populated_sites = list(populated_sites[populated_sites >= min_site_count].index)
    all_metadata = all_metadata[all_metadata['tumor_group'].isin(populated_sites)]

    # Ensure slide_id is string
    all_metadata['slide_id'] = all_metadata['slide_id'].astype(int).astype(str)

    print(f"Size of metadata: {len(all_metadata)}")
    return all_metadata


def top_k_neighbors(
    sim_matrix: pd.DataFrame,
    slide_id: str,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Return top-k neighbor slide IDs with their scores (lower is better).

    Assumes sim_matrix is square with slide IDs as both index and columns.
    Excludes the query slide itself if present.

    Args:
        sim_matrix: Square DataFrame of pairwise distances/similarities.
        slide_id: Query slide ID.
        k: Number of neighbors to return.

    Returns:
        List of (neighbor_slide_id, distance_score) tuples.
    """
    if slide_id not in sim_matrix.index:
        raise ValueError(f"Slide {slide_id} not found in similarity matrix")
    scores_series = sim_matrix.loc[slide_id]
    # Exclude self
    scores_series = scores_series.drop(labels=[slide_id], errors='ignore')
    # Lower score better: sort ascending
    top = scores_series.sort_values(ascending=True).head(k)
    return list(zip(top.index.tolist(), top.values.tolist()))


def enrich_metadata(
    metadata: pd.DataFrame,
    slide_ids: List[str],
    disease_col: str = 'tumor_group',
) -> Dict[str, dict]:
    """Extract rich metadata rows for given slide IDs.

    Args:
        metadata: Metadata DataFrame with 'slide_id' column.
        slide_ids: List of slide IDs to extract metadata for.
        disease_col: Disease column name.

    Returns:
        Mapping of slide_id -> metadata dict (all columns).
    """
    subset = metadata[metadata['slide_id'].isin(slide_ids)].copy()
    records = {}
    for _, row in subset.iterrows():
        sid = str(row['slide_id'])
        records[sid] = row.to_dict()
    return records


def build_json_for_slide(
    sim_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    slide_id: str,
    k: int = 5,
    disease_col: str = 'tumor_group',
) -> dict:
    """Construct JSON payload for a single query slide including standardized metadata.

    Adds the following metadata fields (snake_case) for both the query slide and each neighbor:
      age, sex, site, tumor_group, evaluation, sample_type, treatment,
      discrepant, diagnosis, met, ped_adult

    The similarity score (lower is better) is retained as 'similarity'.
    Any value equal to 'Data not available' (case-insensitive) or blank is converted to None.

    Args:
        sim_matrix: Square DataFrame of pairwise distances.
        metadata: Metadata DataFrame with 'slide_id' and optional clinical columns.
        slide_id: Query slide ID.
        k: Number of neighbors.
        disease_col: Disease column name.

    Returns:
        Dictionary with 'current_case', 'neighbors', and 'schema_version' keys.
    """
    METADATA_COLUMN_MAP = {
        'Comp.Bio.ID': 'comp_bio_id',
        'Age.at.Collection': 'age',
        'Sex': 'sex',
        'Site': 'site',
        'tumor_group': 'tumor_group',
        'Evaluation': 'evaluation',
        'Sample.Type': 'sample_type',
        'Treatment': 'treatment',
        'Discrepant': 'discrepant',
        'Diagnosis': 'diagnosis',
        'Met': 'met',
        'PedAdult': 'ped_adult',
    }

    def clean_value(v):
        if v is None:
            return None
        vs = str(v).strip()
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        if vs == '' or vs.lower().startswith('data not available') or vs.lower() == 'nan':
            return None
        return v

    def standardize_row(row_dict: dict) -> dict:
        std = {}
        for source_col, target_key in METADATA_COLUMN_MAP.items():
            if source_col in row_dict:
                std[target_key] = clean_value(row_dict.get(source_col))
            else:
                std[target_key] = None
        return std

    neighbors = top_k_neighbors(sim_matrix, slide_id, k=k)
    all_ids = [slide_id] + [n[0] for n in neighbors]
    enriched = enrich_metadata(metadata, all_ids, disease_col=disease_col)

    # Query slide metadata
    query_row = enriched.get(str(slide_id), {})
    query_std = standardize_row(query_row)
    case_id = query_std.pop('comp_bio_id', None) or str(slide_id)
    current_case = {'case_id': case_id}
    current_case.update(query_std)

    # Neighbor entries with metadata
    neighbor_entries = []
    for sid, score in neighbors:
        row = enriched.get(str(sid), {})
        nbr_std = standardize_row(row)
        nbr_case_id = nbr_std.pop('comp_bio_id', None) or str(sid)
        entry = {
            'case_id': nbr_case_id,
            'similarity': float(score),
        }
        entry.update(nbr_std)
        neighbor_entries.append(entry)

    return {
        'current_case': current_case,
        'neighbors': neighbor_entries,
        'schema_version': 1,
    }


def filter_data_new(
    sim: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Index, pd.DataFrame]:
    """Filter similarity matrix and metadata to keep only shared slides.

    Removes MISC slides and slides not present in both sim and metadata.

    Args:
        sim: Square similarity matrix DataFrame.
        metadata: Metadata DataFrame.

    Returns:
        Tuple of (filtered_sim, slides, filtered_metadata).
    """
    slides = sim.index
    print("Slides before filtering:", len(slides))

    slides = slides[slides != None]  # noqa: E711
    slides = slides[np.isin(slides, metadata['slide_id'])]
    print("Slides after filtering:", len(slides))

    sim_filtered = sim.loc[slides, slides]
    metadata = metadata[metadata['slide_id'].isin(slides)].drop_duplicates(subset=['slide_id'])

    # Remove MISC slides
    misc_slides = metadata.loc[metadata['tumor_group'] == "MISC", 'slide_id'].tolist()
    print(f"Number of MISC slides: {len(misc_slides)}")
    metadata = metadata[metadata['tumor_group'] != "MISC"]
    sim_filtered = sim_filtered.drop(index=misc_slides, columns=misc_slides, errors='ignore')

    print(f"Size of metadata after filtering: {len(metadata)}")
    slides = sim_filtered.index
    return sim_filtered, slides, metadata
