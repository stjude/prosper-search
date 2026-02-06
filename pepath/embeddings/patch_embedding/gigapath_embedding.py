"""
Patch-level embedding extraction using GigaPath foundation model.

This module handles the extraction of deep learning features from
individual patches using the GigaPath pathology foundation model
from Providence.
"""

from typing import Tuple, Any

import numpy as np
import torch
import timm
from torchvision import transforms

from .base import BasePatchEmbeddingExtractor, register_patch_embedding


@register_patch_embedding("gigapath")
class GigaPathEmbeddingExtractor(BasePatchEmbeddingExtractor):
    """
    Extractor for patch-level embeddings using GigaPath model.
    
    GigaPath is a pathology foundation model from Providence that produces
    high-quality embeddings for histopathology images at scale.
    
    Output embedding dimension: 1536
    """
    
    MODEL_NAME = "gigapath"
    EMBEDDING_DIM = 1536
    INPUT_SIZE = 224
    
    def __init__(
        self,
        device: str = 'cuda',
        batch_size: int = 64,
        num_workers: int = 8,
        verbose: bool = True,
        model_path: str = "hf_hub:prov-gigapath/prov-gigapath"
    ):
        """
        Initialize GigaPath embedding extractor.
        
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
        """Load the GigaPath pathology foundation model."""
        if self.verbose:
            print(f"Loading GigaPath model on {self.device}...")
        
        model = timm.create_model(
            self.model_path,
            pretrained=True
        )
        model = model.to(self.device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        if self.verbose:
            print("GigaPath model loaded successfully.")
        
        return model, transform
    
    def extract_features(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features from a batch of images using GigaPath.
        
        GigaPath directly outputs the embedding vector.
        """
        with torch.no_grad():
            features = self.model(batch_tensor)
            return features.cpu().numpy()


def load_gigapath_model(file_path="/cnvrg/model/", device='cuda'):
    """
    Load GigaPath model (backward compatibility function).
    
    Args:
        file_path: Not used (kept for API compatibility)
        device: Device for inference
    
    Returns:
        Tuple of (model, transform)
    """
    extractor = GigaPathEmbeddingExtractor(device=device, verbose=True)
    return extractor.load_model()


def process_slide_torch(slide_path, data, model, base_patchsize, transform, batch_size=128, device='cuda'):
    """
    Process a single slide and extract features (backward compatibility).
    
    This function is provided for backward compatibility with existing scripts.
    For new code, use GigaPathEmbeddingExtractor.extract_slide_embeddings() instead.
    """
    import os
    import openslide
    
    batch_size = 64
    slide_fn = os.path.join(slide_path, data[0])
    print(f"Processing slide: {slide_fn}")
    
    try:
        slide = openslide.OpenSlide(slide_fn)
    except:
        print(f"Failed to load slide {slide_fn}")
        return None
    
    mosaic = data[1]
    if mosaic is None:
        print(f"Couldn't load mosaic for slide {slide_fn}, continuing")
        return None
    
    num_patches = len(mosaic)
    feature_queue = []
    batch_images = []
    
    print(f"Processing {num_patches} patches in batches")
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for idx, patch in enumerate(mosaic):
            w, h = patch['wsi_loc']
            patch_region = slide.read_region((w, h), 0, (base_patchsize, base_patchsize)).convert('RGB')
            patch_tensor = transform(patch_region).to(device)
            batch_images.append(patch_tensor)
            
            if len(batch_images) == batch_size or idx == num_patches - 1:
                batch_tensor = torch.stack(batch_images).to(device)
                features = model(batch_tensor)
                feature_queue.extend(features.cpu().detach().numpy())
                batch_images.clear()
    
    if feature_queue:
        return np.array(feature_queue)
    return None
