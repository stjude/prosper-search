"""
Configuration management for PROSPER pipeline.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal


# Type definitions for model choices
PatchModelType = Literal["virchow2", "uni2", "uni", "gigapath"]
SlideEmbeddingMethodType = Literal["percentile", "bob", "ot"]
DistanceMetricType = Literal["l1", "l2", "cosine"]


@dataclass
class ProsperConfig:
    """
    Configuration class for PROSPER pipeline.
    
    Attributes:
        slide_path: Path to directory containing whole slide images (.svs files)
        output_dir: Directory for all output files
        base_patchsize: Base patch size for tiling slides (default: 512)
        percentage_selected: Percentage of mosaic tiles to select (default: 15)
        device: Device for model inference ('cuda' or 'cpu')
        batch_size: Batch size for feature extraction
        num_workers: Number of parallel workers
        top_k: Number of top similar slides to return
        reference_db_path: Path to pre-computed reference database (optional)
        
        # Model selection
        patch_model: Patch embedding model to use ('virchow2', 'uni2', 'uni', 'gigapath')
        slide_embedding_method: Slide embedding aggregation method ('percentile', 'bob', 'ot')
        
        # Visualization options
        generate_visualizations: Whether to generate OT visualization heatmaps
    """
    
    # Input/Output paths
    slide_path: str = ""
    output_dir: str = "./output"
    reference_db_path: Optional[str] = None
    
    # Processing parameters
    base_patchsize: int = 512
    percentage_selected: int = 15
    tissue_threshold: float = 0.7
    
    # Model selection - NEW
    patch_model: str = "virchow2"  # Options: virchow2, uni2, uni, gigapath
    slide_embedding_method: str = "percentile"  # Options: percentile, bob, ot
    
    # Model parameters
    device: str = "cuda"
    batch_size: int = 64
    model_name: str = "virchow2"  # Deprecated, use patch_model instead
    
    # Parallel processing
    num_workers: int = 8
    n_cpus: int = 32
    
    # Search parameters
    top_k: int = 10
    distance_metric: str = "l1"  # 'l1' or 'l2' or 'cosine'
    
    # Slide embedding parameters
    percentiles: Optional[List[int]] = None  # For percentile method
    feature_dim: int = 1280  # Dimension of features to use for slide embedding
    
    # OT-specific parameters
    ot_metric: str = "cosine"  # Distance metric for OT computation
    ot_reg: float = 0.0  # Regularization for entropic OT (0 = exact EMD)
    
    # BoB-specific parameters
    bob_barcode_dim: int = 64  # Dimension of binary barcodes
    
    # Bayesian disease probability parameters
    bayesian_n_neighbors: int = 20  # Number of nearest neighbors for probability estimation
    bayesian_temperature: float = 20.0  # Temperature for exponential weighting (larger = softer)
    
    # Visualization options - NEW
    generate_visualizations: bool = False
    visualization_output_dir: Optional[str] = None
    visualization_slide_level: int = 2  # OpenSlide level for visualization
    visualization_alpha: int = 100  # Alpha for heatmap overlay
    
    # Runtime options
    smoketest: bool = False
    force_recompute: bool = False
    save_intermediate: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Validate and setup configuration."""
        # Set default percentiles if not provided
        if self.percentiles is None:
            self.percentiles = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        
        # Set feature dimension based on model if not explicitly set
        self._set_feature_dim_for_model()
        
        # Sync model_name with patch_model for backward compatibility
        if self.model_name != self.patch_model and self.model_name != "virchow2":
            # User set model_name explicitly (backward compat)
            self.patch_model = self.model_name
        else:
            self.model_name = self.patch_model
        
        # Create output directory structure
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "mosaic"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "patch_embeddings"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "slide_embeddings"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "search_results"), exist_ok=True)
            
            if self.generate_visualizations:
                vis_dir = self.visualization_output_dir or os.path.join(self.output_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
    
    def _set_feature_dim_for_model(self):
        """Set appropriate feature dimension based on selected model."""
        # Only update if using default value
        if self.feature_dim == 1280:
            model_dims = {
                "virchow2": 1280,  # Class token dimension (full is 2560)
                "uni2": 1536,
                "uni": 1024,
                "gigapath": 1536,
            }
            self.feature_dim = model_dims.get(self.patch_model, 1280)
    
    @property
    def mosaic_dir(self) -> str:
        return os.path.join(self.output_dir, "mosaic")
    
    @property
    def mosaic_file(self) -> str:
        return os.path.join(self.mosaic_dir, "combined_mosaic.pkl")
    
    @property
    def patch_embeddings_dir(self) -> str:
        return os.path.join(self.output_dir, "patch_embeddings")
    
    @property
    def slide_embeddings_dir(self) -> str:
        return os.path.join(self.output_dir, "slide_embeddings")
    
    @property
    def search_results_dir(self) -> str:
        return os.path.join(self.output_dir, "search_results")
    
    @property
    def visualizations_dir(self) -> str:
        return self.visualization_output_dir or os.path.join(self.output_dir, "visualizations")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "ProsperConfig":
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ProsperConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate the configuration."""
        errors = []
        
        if not self.slide_path:
            errors.append("slide_path is required")
        elif not os.path.exists(self.slide_path):
            errors.append(f"slide_path does not exist: {self.slide_path}")
        
        if self.base_patchsize <= 0:
            errors.append("base_patchsize must be positive")
        
        if not 0 < self.percentage_selected <= 100:
            errors.append("percentage_selected must be between 0 and 100")
        
        if self.device not in ['cuda', 'cpu']:
            errors.append("device must be 'cuda' or 'cpu'")
        
        if self.distance_metric not in ['l1', 'l2', 'cosine']:
            errors.append("distance_metric must be 'l1', 'l2', or 'cosine'")
        
        # Validate model selection
        valid_patch_models = ['virchow2', 'uni2', 'uni', 'gigapath']
        if self.patch_model not in valid_patch_models:
            errors.append(f"patch_model must be one of {valid_patch_models}")
        
        valid_slide_methods = ['percentile', 'bob', 'ot']
        if self.slide_embedding_method not in valid_slide_methods:
            errors.append(f"slide_embedding_method must be one of {valid_slide_methods}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    def __repr__(self) -> str:
        lines = ["ProsperConfig("]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}={repr(value)},")
        lines.append(")")
        return "\n".join(lines)


# Backward compatibility aliases
COMETConfig = ProsperConfig
PePathConfig = ProsperConfig
