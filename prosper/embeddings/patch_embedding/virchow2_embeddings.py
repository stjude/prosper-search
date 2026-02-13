"""
Patch-level embedding extraction using Virchow2 foundation model.

This module handles the extraction of deep learning features from
individual patches using the Virchow2 pathology foundation model.
"""

from typing import Tuple, Any

import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from .base import BasePatchEmbeddingExtractor, register_patch_embedding


@register_patch_embedding("virchow2")
class Virchow2EmbeddingExtractor(BasePatchEmbeddingExtractor):
    """
    Extractor for patch-level embeddings using Virchow2 model.
    
    Virchow2 is a pathology foundation model from Paige AI that produces
    high-quality embeddings for histopathology images.
    
    Output embedding dimension: 2560 (1280 class token + 1280 mean patch tokens)
    """
    
    MODEL_NAME = "virchow2"
    EMBEDDING_DIM = 2560  # 1280 (class token) + 1280 (mean patch tokens)
    INPUT_SIZE = 224
    
    def __init__(
        self,
        device: str = 'cuda',
        batch_size: int = 64,
        num_workers: int = 8,
        verbose: bool = True,
        model_path: str = "hf-hub:paige-ai/Virchow2"
    ):
        """
        Initialize Virchow2 embedding extractor.
        
        Args:
            device: Device for inference
            batch_size: Batch size for processing
            num_workers: Number of workers
            verbose: Whether to print progress
            model_path: HuggingFace model hub identifier
        """
        super().__init__(device, batch_size, num_workers, verbose)
        self.model_path = model_path
    
    def load_model(self) -> Tuple[torch.nn.Module, Any]:
        """Load the Virchow2 pathology foundation model."""
        if self.verbose:
            print(f"Loading Virchow2 model on {self.device}...")
        
        model = timm.create_model(
            self.model_path,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        model = model.to(self.device)
        model.eval()
        
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        
        if self.verbose:
            print("Virchow2 model loaded successfully.")
        
        return model, transform
    
    def extract_features(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features from a batch of images using Virchow2.
        
        Virchow2 produces:
        - Class token at position 0
        - Patch tokens at positions 5 onwards (positions 1-4 are register tokens)
        
        The final embedding concatenates class token with mean of patch tokens.
        """
        with torch.no_grad():
            features = self.model(batch_tensor)
            class_token = features[:, 0]  # Shape: (B, 1280)
            patch_tokens = features[:, 5:]  # Shape: (B, num_patches, 1280)
            
            # Concatenate class token with mean of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            
            return embedding.cpu().numpy()


# Backward compatibility alias
PatchEmbeddingExtractor = Virchow2EmbeddingExtractor


def load_virchow_model(
    model_name: str = "hf-hub:paige-ai/Virchow2",
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, Any]:
    """
    Load the Virchow2 pathology foundation model.
    
    This function is provided for backward compatibility.
        
        Args:
        model_name: HuggingFace model hub identifier
        device: Device to load model on ('cuda' or 'cpu')
        
        Returns:
        Tuple of (model, transform)
    """
    extractor = Virchow2EmbeddingExtractor(
        device=device,
        model_path=model_name,
        verbose=True
    )
    return extractor.load_model()
