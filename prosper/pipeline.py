"""
PROSPER Pipeline - Main orchestrator for the complete workflow.

This module provides the unified pipeline interface for processing
whole slide images and performing morphological similarity search.
"""

import json
import os
import pickle
from glob import glob
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ProsperConfig
from .preprocessing import MosaicGenerator
from .embeddings import (
    get_patch_embedding_extractor,
    get_slide_embedding_generator,
    list_patch_embedding_models,
    list_slide_embedding_methods,
)
from .search import SlideSearchEngine, ROISearchEngine


@dataclass
class ProsperResults:
    """Container for PROSPER pipeline results."""
    query_slide_ids: List[str]
    slide_search_results: Optional[Dict] = None
    roi_search_results: Optional[Dict] = None
    visualizations: Optional[Dict] = None
    disease_probabilities: Optional[Dict] = None  # Bayesian disease probability estimates
    llm_summaries: Optional[Dict] = None  # LLM-generated summaries per slide
    
    def to_dataframe(self, search_type: str = 'slide') -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if search_type == 'slide' and self.slide_search_results:
            rows = []
            for slide_id, results in self.slide_search_results.items():
                for r in results:
                    row = r if isinstance(r, dict) else r.to_dict()
                    rows.append(row)
            return pd.DataFrame(rows)
        elif search_type == 'roi' and self.roi_search_results:
            rows = []
            for slide_id, patch_results in self.roi_search_results.items():
                for patch_res in patch_results:
                    for r in patch_res:
                        row = r if isinstance(r, dict) else r.to_dict()
                        rows.append(row)
            return pd.DataFrame(rows)
        return pd.DataFrame()
    
    def save(self, output_path: str):
        """Save results to file."""
        with open(output_path, 'wb') as f:
            pickle.dump({
                'query_slide_ids': self.query_slide_ids,
                'slide_search_results': self.slide_search_results,
                'roi_search_results': self.roi_search_results,
                'visualizations': self.visualizations,
                'disease_probabilities': self.disease_probabilities,
                'llm_summaries': self.llm_summaries,
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> "ProsperResults":
        """Load results from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)


# Backward compatibility aliases
PePathResults = ProsperResults


class ProsperPipeline:
    """
    Main pipeline for PROSPER - Pediatric Retrieval for Oncology Slides, Pathology Evidence, and Reporting.
    
    This class orchestrates the complete workflow from raw whole slide images
    to morphological similarity search results.
    
    Supports multiple patch embedding models (Virchow2, UNI2, GigaPath) and
    slide similarity formulations (Percentile, BoB, OT).
    
    Example usage:
        ```python
        from prosper import ProsperPipeline, ProsperConfig
        
        # Initialize with configuration
        config = ProsperConfig(
            slide_path="/path/to/slides",
            output_dir="/path/to/output",
            device="cuda",
            patch_model="virchow2",
            slide_embedding_method="percentile"
        )
        pipeline = ProsperPipeline(config)
        
        # Run complete pipeline
        results = pipeline.run()
        
        # Or run steps individually
        pipeline.generate_mosaics()
        pipeline.extract_patch_embeddings()
        pipeline.generate_slide_embeddings()
        results = pipeline.search()
        ```
    """
    
    def __init__(self, config: Union[ProsperConfig, str, dict]):
        """
        Initialize PROSPER pipeline.
        
        Args:
            config: ProsperConfig object, path to config YAML, or config dictionary
        """
        if isinstance(config, str):
            self.config = ProsperConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = ProsperConfig.from_dict(config)
        else:
            self.config = config
        
        # Initialize components (lazy loading)
        self._mosaic_generator = None
        self._patch_extractor = None
        self._slide_generator = None
        self._slide_search = None
        self._roi_search = None
        self._visualizer = None
        
        # State tracking
        self.mosaics: Dict[str, List] = {}
        self.patch_embeddings: Dict[str, np.ndarray] = {}
        self.slide_embeddings: Dict[str, np.ndarray] = {}
        
        if self.config.verbose:
            print(f"PROSPER Pipeline initialized")
            print(f"  Patch model: {self.config.patch_model}")
            print(f"  Slide embedding method: {self.config.slide_embedding_method}")
    
    @property
    def mosaic_generator(self) -> MosaicGenerator:
        if self._mosaic_generator is None:
            self._mosaic_generator = MosaicGenerator(
                base_patchsize=self.config.base_patchsize,
                percentage_selected=self.config.percentage_selected,
                tissue_threshold=self.config.tissue_threshold,
                n_workers=self.config.n_cpus,
                verbose=self.config.verbose
            )
        return self._mosaic_generator
    
    @property
    def patch_extractor(self):
        """Get patch embedding extractor based on config."""
        if self._patch_extractor is None:
            self._patch_extractor = get_patch_embedding_extractor(
                model_name=self.config.patch_model,
                device=self.config.device,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                verbose=self.config.verbose
            )
        return self._patch_extractor
    
    @property
    def slide_generator(self):
        """Get slide embedding generator based on config."""
        if self._slide_generator is None:
            method = self.config.slide_embedding_method
            
            kwargs = {
                'feature_dim': self.config.feature_dim,
                'n_workers': self.config.n_cpus,
                'verbose': self.config.verbose,
            }
            
            if method == 'percentile':
                kwargs['percentiles'] = self.config.percentiles
            elif method == 'bob':
                kwargs['barcode_dim'] = self.config.bob_barcode_dim
            elif method == 'ot':
                kwargs['metric'] = self.config.ot_metric
                kwargs['reg'] = self.config.ot_reg
            
            self._slide_generator = get_slide_embedding_generator(
                method_name=method,
                **kwargs
            )
        return self._slide_generator
    
    @property
    def slide_search(self) -> SlideSearchEngine:
        if self._slide_search is None:
            metric_map = {'l1': 'cityblock', 'l2': 'euclidean', 'cosine': 'cosine'}
            self._slide_search = SlideSearchEngine(
                distance_metric=metric_map.get(self.config.distance_metric, 'cityblock'),
                top_k=self.config.top_k,
                verbose=self.config.verbose
            )
        return self._slide_search
    
    @property
    def roi_search(self) -> ROISearchEngine:
        if self._roi_search is None:
            self._roi_search = ROISearchEngine(
                top_k=self.config.top_k,
                n_workers=self.config.n_cpus,
                verbose=self.config.verbose
            )
        return self._roi_search
    
    @property
    def visualizer(self):
        """Get OT visualizer if visualization is enabled."""
        if self._visualizer is None and self.config.generate_visualizations:
            from .visualization import OTVisualizer
            self._visualizer = OTVisualizer(
                patch_size=self.config.base_patchsize,
                slide_level=self.config.visualization_slide_level,
                alpha=self.config.visualization_alpha,
                output_dir=self.config.visualizations_dir
            )
        return self._visualizer
    
    def get_slide_paths(self) -> List[str]:
        """Get list of slide paths to process."""
        slide_paths = sorted(glob(os.path.join(self.config.slide_path, '*.svs')))
        
        if self.config.smoketest:
            slide_paths = slide_paths[:5]
        
        return slide_paths
    
    def generate_mosaics(self, force_recompute: bool = None) -> Dict[str, List]:
        """
        Step 1: Generate mosaics from whole slide images.
        
        Args:
            force_recompute: Override config force_recompute setting
        
        Returns:
            Dictionary mapping slide IDs to mosaic patches
        """
        if self.config.verbose:
            print("=" * 60)
            print("Step 1: Generating Mosaics")
            print("=" * 60)
        
        force = force_recompute if force_recompute is not None else self.config.force_recompute
        slide_paths = self.get_slide_paths()
        
        self.mosaics = self.mosaic_generator.process_slides(
            slide_paths=slide_paths,
            output_dir=self.config.mosaic_dir,
            force_recompute=force
        )
        
        if self.config.save_intermediate:
            self.mosaic_generator.save_mosaics(self.mosaics, self.config.mosaic_file)
        
        return self.mosaics
    
    def extract_patch_embeddings(self, force_recompute: bool = None) -> Dict[str, np.ndarray]:
        """
        Step 2: Extract patch-level embeddings using the configured model.
        
        Args:
            force_recompute: Override config force_recompute setting
        
        Returns:
            Dictionary mapping slide IDs to embedding arrays
        """
        if self.config.verbose:
            print("=" * 60)
            print(f"Step 2: Extracting Patch Embeddings (model: {self.config.patch_model})")
            print("=" * 60)
        
        force = force_recompute if force_recompute is not None else self.config.force_recompute
        
        if not self.mosaics:
            if os.path.exists(self.config.mosaic_file):
                self.mosaics = MosaicGenerator.load_mosaics(self.config.mosaic_file)
            else:
                raise ValueError("No mosaics available. Run generate_mosaics() first.")
        
        self.patch_embeddings = self.patch_extractor.process_slides(
            slide_dir=self.config.slide_path,
            mosaics=self.mosaics,
            output_dir=self.config.patch_embeddings_dir,
            base_patchsize=self.config.base_patchsize,
            force_recompute=force
        )
        
        return self.patch_embeddings
    
    def generate_slide_embeddings(self, force_recompute: bool = None) -> Dict[str, np.ndarray]:
        """
        Step 3: Generate slide-level embeddings from patch embeddings.
        
        Args:
            force_recompute: Override config force_recompute setting
        
        Returns:
            Dictionary mapping slide IDs to slide embeddings
        """
        if self.config.verbose:
            print("=" * 60)
            print(f"Step 3: Generating Slide Embeddings (method: {self.config.slide_embedding_method})")
            print("=" * 60)
        
        force = force_recompute if force_recompute is not None else self.config.force_recompute
        
        patch_emb_tmp = os.path.join(self.config.patch_embeddings_dir, "tmp")
        
        self.slide_embeddings = self.slide_generator.generate_embeddings(
            patch_embeddings_dir=patch_emb_tmp,
            output_dir=self.config.slide_embeddings_dir,
            force_recompute=force
        )
        
        return self.slide_embeddings
    
    def search(
        self,
        search_type: str = 'slide',
        reference_db_path: str = None
    ) -> ProsperResults:
        """
        Step 4: Perform morphological similarity search.
        
        Args:
            search_type: Type of search ('slide' or 'roi')
            reference_db_path: Optional path to reference database
        
        Returns:
            ProsperResults object containing search results
        """
        if self.config.verbose:
            print("=" * 60)
            print(f"Step 4: Performing {search_type.upper()} Search")
            print("=" * 60)
        
        ref_path = reference_db_path or self.config.reference_db_path
        
        if search_type == 'slide':
            return self._slide_level_search(ref_path)
        elif search_type == 'roi':
            return self._roi_level_search(ref_path)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def _slide_level_search(self, reference_db_path: str = None) -> ProsperResults:
        """Perform slide-level search."""
        if reference_db_path:
            self.slide_search.load_reference_database(embeddings_dir=reference_db_path)
        else:
            if not self.slide_embeddings:
                self.slide_embeddings = self.slide_generator.load_all_embeddings(
                    self.config.slide_embeddings_dir
                )
            
            slide_ids = sorted(self.slide_embeddings.keys())
            embeddings = self.slide_embeddings
            
            if isinstance(list(embeddings.values())[0], dict):
                if self.config.slide_embedding_method == 'ot':
                    return self._ot_based_search()
                else:
                    matrix = np.stack([embeddings[sid]['int_array'] for sid in slide_ids])
            else:
                matrix = np.stack([embeddings[sid].flatten() for sid in slide_ids])
            
            self.slide_search.reference_matrix = matrix
            self.slide_search.reference_slide_ids = slide_ids
        
        results = self.slide_search.batch_search(self.slide_embeddings)
        
        output_path = os.path.join(
            self.config.search_results_dir,
            "slide_search_results.pkl"
        )
        self.slide_search.save_results(results, output_path)
        
        return ProsperResults(
            query_slide_ids=list(self.slide_embeddings.keys()),
            slide_search_results=results
        )
    
    def _ot_based_search(self) -> ProsperResults:
        """Perform OT-based slide search."""
        from .embeddings.slide_embedding import OTSlideEmbeddingGenerator
        
        if not isinstance(self.slide_generator, OTSlideEmbeddingGenerator):
            raise ValueError("OT search requires OT slide embedding generator")
        
        output_dir = os.path.join(self.config.search_results_dir, "ot_similarities")
        similarities = self.slide_generator.compute_pairwise_similarities(
            slide_data=self.slide_embeddings,
            output_dir=output_dir
        )
        
        results = {}
        for query_id, sim_list in similarities.items():
            sorted_sims = sorted(sim_list, key=lambda x: x[1])[:self.config.top_k]
            results[query_id] = [
                {'query_slide_id': query_id, 'hit_slide_id': hit_id, 'distance': dist, 'rank': i+1}
                for i, (hit_id, dist) in enumerate(sorted_sims)
            ]
        
        visualizations = None
        if self.config.generate_visualizations and self.visualizer:
            visualizations = self._generate_ot_visualizations(results)
        
        return ProsperResults(
            query_slide_ids=list(self.slide_embeddings.keys()),
            slide_search_results=results,
            visualizations=visualizations
        )
    
    def _generate_ot_visualizations(self, results: Dict) -> Dict:
        """Generate OT-based visualizations for top matches."""
        visualizations = {}
        
        for query_id, matches in results.items():
            if not matches:
                continue
            
            top_match = matches[0]
            hit_id = top_match['hit_slide_id']
            
            query_data = self.slide_embeddings[query_id]
            hit_data = self.slide_embeddings[hit_id]
            
            ot_result = self.slide_generator.compute_detailed_similarity(
                query_data, hit_data
            )
            
            mosaic_query = self.mosaics.get(query_id, query_data.get('mosaic', []))
            mosaic_hit = self.mosaics.get(hit_id, hit_data.get('mosaic', []))
            
            slide_a_path = os.path.join(self.config.slide_path, f"{query_id}.svs")
            slide_b_path = os.path.join(self.config.slide_path, f"{hit_id}.svs")
            
            if os.path.exists(slide_a_path) and os.path.exists(slide_b_path):
                vis_result = self.visualizer.visualize_similarity(
                    slide_a_path, slide_b_path,
                    ot_result, mosaic_query, mosaic_hit,
                    output_dir=os.path.join(self.config.visualizations_dir, query_id)
                )
                visualizations[query_id] = vis_result
        
        return visualizations
    
    def _roi_level_search(self, reference_db_path: str = None) -> ProsperResults:
        """Perform ROI-level search."""
        patch_emb_tmp = os.path.join(self.config.patch_embeddings_dir, "tmp")
        ref_path = reference_db_path or patch_emb_tmp
        self.roi_search.load_reference_database(ref_path)
        
        all_results = {}
        for slide_id, embeddings in tqdm(self.patch_embeddings.items(), 
                                         desc="ROI search", 
                                         disable=not self.config.verbose):
            mosaic = self.mosaics.get(slide_id)
            all_results[slide_id] = self.roi_search.search(embeddings, slide_id, mosaic)
        
        output_path = os.path.join(
            self.config.search_results_dir,
            "roi_search_results.pkl"
        )
        self.roi_search.save_results(all_results, output_path)
        
        return ProsperResults(
            query_slide_ids=list(self.patch_embeddings.keys()),
            roi_search_results=all_results
        )
    
    def compute_disease_probabilities(
        self,
        distance_matrix: np.ndarray = None,
        class_labels: List[str] = None,
        slide_ids: List[str] = None,
        metadata_path: str = None,
        n_neighbors: int = None,
        temperature: float = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Step 5: Compute Bayesian disease probabilities from distance matrix.
        
        Uses exponential-weighted nearest neighbors to estimate the probability
        distribution over diseases for each query slide.
        
        Args:
            distance_matrix: Precomputed distance matrix (n_slides x n_slides).
                If None, computed from slide embeddings.
            class_labels: Disease labels for each slide. If None, loaded from metadata.
            slide_ids: Slide IDs corresponding to matrix rows/columns.
            metadata_path: Path to metadata CSV (used if class_labels is None).
            n_neighbors: Number of nearest neighbors (default: config.bayesian_n_neighbors).
            temperature: Temperature parameter (default: config.bayesian_temperature).
        
        Returns:
            Dictionary mapping slide_id -> {disease: probability} (non-zero only).
        """
        from .bayesian import compute_disease_probs
        from scipy.spatial.distance import cdist
        
        n_neighbors = n_neighbors or self.config.bayesian_n_neighbors
        temperature = temperature or self.config.bayesian_temperature
        
        if self.config.verbose:
            print("=" * 60)
            print(f"Step 5: Computing Bayesian Disease Probabilities")
            print(f"  n_neighbors={n_neighbors}, temperature={temperature}")
            print("=" * 60)
        
        # Build distance matrix from slide embeddings if not provided
        if distance_matrix is None:
            if not self.slide_embeddings:
                self.slide_embeddings = self.slide_generator.load_all_embeddings(
                    self.config.slide_embeddings_dir
                )
            
            slide_ids = sorted(self.slide_embeddings.keys())
            embeddings = self.slide_embeddings
            
            if isinstance(list(embeddings.values())[0], dict):
                matrix = np.stack([embeddings[sid]['int_array'] for sid in slide_ids])
            else:
                matrix = np.stack([embeddings[sid].flatten() for sid in slide_ids])
            
            metric_map = {'l1': 'cityblock', 'l2': 'euclidean', 'cosine': 'cosine'}
            metric = metric_map.get(self.config.distance_metric, 'cityblock')
            
            if self.config.verbose:
                print(f"  Computing pairwise distances ({metric})...")
            distance_matrix = cdist(matrix, matrix, metric=metric)
        
        if slide_ids is None:
            raise ValueError("slide_ids must be provided or derivable from slide_embeddings")
        
        # Load class labels from metadata if not provided
        if class_labels is None:
            if metadata_path is None:
                metadata_path = os.path.join(
                    os.path.dirname(self.config.output_dir), "data",
                    "final_scan_id_disease_map_rich_data.csv"
                )
            
            if self.config.verbose:
                print(f"  Loading metadata from: {metadata_path}")
            meta = pd.read_csv(metadata_path)
            
            # Auto-detect columns
            if 'Slide.Scan.File' in meta.columns:
                meta = meta.rename(columns={
                    'Slide.Scan.File': 'slide_id', 'Tumor.group': 'tumor_group'
                })
            meta['slide_id'] = meta['slide_id'].astype(int).astype(str)
            meta = meta.dropna(subset=['tumor_group']).set_index('slide_id')
            
            class_labels = [meta.loc[sid, 'tumor_group'] if sid in meta.index else 'UNKNOWN'
                          for sid in slide_ids]
        
        # Compute probabilities
        list_of_dicts, df = compute_disease_probs(
            distance_matrix=distance_matrix,
            class_labels=class_labels,
            n_neighbors=n_neighbors,
            T=temperature,
            test_slide_ids=slide_ids,
        )
        
        # Store as dict of non-zero probabilities per slide
        disease_probs = {}
        for sid, probs in zip(slide_ids, list_of_dicts):
            nonzero = {d: round(p, 6) for d, p in probs.items() if p > 1e-8}
            disease_probs[sid] = nonzero
        
        # Save to output
        bayesian_dir = os.path.join(self.config.output_dir, "bayesian")
        os.makedirs(bayesian_dir, exist_ok=True)
        
        csv_path = os.path.join(bayesian_dir, "disease_probabilities.csv")
        df.to_csv(csv_path)
        
        if self.config.verbose:
            print(f"  Saved disease probability matrix: {csv_path}")
            print(f"  Computed probabilities for {len(disease_probs)} slides")
        
        return disease_probs
    
    def generate_llm_summary(
        self,
        slide_ids: List[str] = None,
        disease_probs: Dict[str, Dict[str, float]] = None,
        metadata_path: str = None,
        sim_matrix_path: str = None,
        k_neighbors: int = 5,
    ) -> Dict[str, str]:
        """
        Step 6: Generate LLM summaries for query slides.
        
        Builds a structured JSON payload for each slide (with neighbor metadata
        and Bayesian probabilities) and sends it to the LLM for grounded
        summarization with inline citations.
        
        Args:
            slide_ids: Slide IDs to summarize. If None, uses all query slides.
            disease_probs: Bayesian probabilities dict (slide_id -> {disease: prob}).
            metadata_path: Path to rich metadata CSV.
            sim_matrix_path: Path to similarity/embedding matrix pickle.
            k_neighbors: Number of neighbors for the LLM payload.
        
        Returns:
            Dictionary mapping slide_id -> summary text.
        """
        if self.config.verbose:
            print("=" * 60)
            print("Step 6: Generating LLM Summaries")
            print("=" * 60)
        
        try:
            from .llm import (
                load_results_new, load_metadata, build_json_for_slide,
                attempt_llm_summary_from_payload, MEDICAL_DISCLAIMER,
            )
        except ImportError as e:
            if self.config.verbose:
                print(f"  Warning: LLM dependencies not available: {e}")
            return {}
        
        # Determine paths
        if sim_matrix_path is None:
            sim_matrix_path = os.path.join(
                os.path.dirname(self.config.output_dir), "data",
                "final_slides_sampler_matrix.pkl"
            )
        if metadata_path is None:
            metadata_path = os.path.join(
                os.path.dirname(self.config.output_dir), "data",
                "final_scan_id_disease_map_rich_data.csv"
            )
        
        # Load similarity matrix and metadata
        if not os.path.exists(sim_matrix_path) or not os.path.exists(metadata_path):
            if self.config.verbose:
                print(f"  Warning: Required files not found:")
                if not os.path.exists(sim_matrix_path):
                    print(f"    - {sim_matrix_path}")
                if not os.path.exists(metadata_path):
                    print(f"    - {metadata_path}")
            return {}
        
        sim_matrix = load_results_new(sim_matrix_path)
        rich_metadata = load_metadata(metadata_path)
        
        if slide_ids is None:
            slide_ids = list(self.slide_embeddings.keys()) if self.slide_embeddings else []
        
        # Output directory
        llm_dir = os.path.join(self.config.output_dir, "llm")
        os.makedirs(llm_dir, exist_ok=True)
        
        summaries = {}
        for sid in slide_ids:
            if sid not in sim_matrix.index:
                if self.config.verbose:
                    print(f"  Skipping slide {sid}: not in similarity matrix")
                continue
            
            if self.config.verbose:
                print(f"\n  Processing slide {sid}...")
            
            # Build payload
            try:
                payload = build_json_for_slide(
                    sim_matrix, rich_metadata, slide_id=sid, k=k_neighbors
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"    Warning: Failed to build payload: {e}")
                continue
            
            # Attach Bayesian probabilities if available
            if disease_probs and sid in disease_probs:
                probs = disease_probs[sid]
                nonzero = {d: p for d, p in sorted(probs.items(), key=lambda x: -x[1]) if p > 1e-8}
                payload['probabilities'] = nonzero
            
            # Save payload
            payload_path = os.path.join(llm_dir, f"slide_{sid}.json")
            with open(payload_path, 'w') as f:
                json.dump(payload, f, indent=2)
            
            # Generate summary
            llm_text, accepted = attempt_llm_summary_from_payload(payload)
            
            if llm_text and not llm_text.startswith("ERROR:"):
                summaries[sid] = llm_text
                summary_path = os.path.join(llm_dir, f"summary_{sid}.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"{llm_text}\n\n{MEDICAL_DISCLAIMER}\n")
                if self.config.verbose:
                    print(f"    Summary generated ({len(llm_text.split())} words)")
            else:
                if self.config.verbose:
                    msg = llm_text if llm_text else "No API key available"
                    print(f"    Warning: LLM summary failed: {msg}")
        
        if self.config.verbose:
            print(f"\n  Generated {len(summaries)} summaries in: {llm_dir}")
        
        return summaries
    
    def run_core(self, search_type: str = 'slide') -> ProsperResults:
        """
        Run the core PROSPER pipeline (mosaic -> patch embed -> slide embed -> search).

        This does NOT run the optional Bayesian or LLM steps.  Use
        :meth:`run` for the full pipeline (backward compatible), or call
        :meth:`compute_disease_probabilities` / :meth:`generate_llm_summary`
        explicitly after this method.

        Args:
            search_type: Type of search to perform ('slide' or 'roi')

        Returns:
            ProsperResults object containing search results
        """
        if self.config.verbose:
            print("=" * 60)
            print("PROSPER Pipeline - Starting")
            print("=" * 60)
            print(f"Slide path: {self.config.slide_path}")
            print(f"Output directory: {self.config.output_dir}")
            print(f"Device: {self.config.device}")
            print(f"Patch embedding model: {self.config.patch_model}")
            print(f"Slide embedding method: {self.config.slide_embedding_method}")
            print("=" * 60)

        self.config.validate()

        self.generate_mosaics()
        self.extract_patch_embeddings()
        self.generate_slide_embeddings()
        results = self.search(search_type=search_type)

        if self.config.verbose:
            print("=" * 60)
            print("PROSPER Pipeline - Core steps complete")
            print("=" * 60)

        return results

    def run(self, search_type: str = 'slide') -> ProsperResults:
        """
        Run the complete PROSPER pipeline including optional Bayesian and LLM steps.

        This is the backward-compatible entry point that runs everything.
        For finer control, use :meth:`run_core` and call the optional steps
        individually.

        Args:
            search_type: Type of search to perform ('slide' or 'roi')

        Returns:
            ProsperResults object containing all results
        """
        results = self.run_core(search_type=search_type)

        # Step 5: Bayesian disease probabilities
        disease_probs = None
        try:
            disease_probs = self.compute_disease_probabilities()
            results.disease_probabilities = disease_probs
        except Exception as e:
            if self.config.verbose:
                print(f"  Warning: Bayesian disease probabilities skipped: {e}")

        # Step 6: LLM summarization
        try:
            summaries = self.generate_llm_summary(disease_probs=disease_probs)
            if summaries:
                results.llm_summaries = summaries
        except Exception as e:
            if self.config.verbose:
                print(f"  Warning: LLM summarization skipped: {e}")

        if self.config.verbose:
            print("=" * 60)
            print("PROSPER Pipeline - Complete")
            print("=" * 60)

        return results
    
    def process_single_slide(
        self,
        slide_path: str,
        reference_db_path: str = None,
        search_type: str = 'slide'
    ) -> ProsperResults:
        """
        Process a single slide and search against a reference database.
        
        Args:
            slide_path: Path to the slide file
            reference_db_path: Path to reference database
            search_type: Type of search ('slide' or 'roi')
        
        Returns:
            ProsperResults for this slide
        """
        slide_id = os.path.basename(slide_path).replace('.svs', '')
        
        if self.config.verbose:
            print(f"Processing slide: {slide_id}")
            print(f"Using patch model: {self.config.patch_model}")
        
        from .preprocessing.mosaic import process_slide
        mosaic = process_slide(
            slide_path,
            self.config.base_patchsize,
            self.config.percentage_selected,
            self.config.tissue_threshold
        )
        self.mosaics[slide_id] = mosaic
        
        embeddings = self.patch_extractor.extract_slide_embeddings(
            slide_path, mosaic, self.config.base_patchsize
        )
        self.patch_embeddings[slide_id] = embeddings
        
        slide_embedding = self.slide_generator.process_single_slide(
            embeddings, slide_id, mosaic
        )
        self.slide_embeddings[slide_id] = slide_embedding
        
        return self.search(search_type=search_type, reference_db_path=reference_db_path)
    
    @staticmethod
    def list_available_models() -> Dict[str, List[str]]:
        """List available patch models and slide similarity formulations."""
        return {
            'patch_models': list_patch_embedding_models(),
            'slide_embedding_methods': list_slide_embedding_methods()
        }
