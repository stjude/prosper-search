"""
Patch-level embedding extraction using UNI2 foundation model.

This module handles the extraction of deep learning features from
individual patches using the UNI2 (Universal pathology) foundation model
from Mahmood Lab.
"""

from typing import Tuple, Any

import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms

from .base import BasePatchEmbeddingExtractor, register_patch_embedding


@register_patch_embedding("uni2")
class UNI2EmbeddingExtractor(BasePatchEmbeddingExtractor):
    """
    Extractor for patch-level embeddings using UNI2 model.
    
    UNI2 (UNI2-h) is a pathology foundation model from Mahmood Lab that produces
    high-quality embeddings for histopathology images.
    
    Output embedding dimension: 1536
    """
    
    MODEL_NAME = "uni2"
    EMBEDDING_DIM = 1536
    INPUT_SIZE = 224
    
    def __init__(
        self,
        device: str = 'cuda',
        batch_size: int = 256,
        num_workers: int = 8,
        verbose: bool = True,
        model_path: str = "hf-hub:MahmoodLab/UNI2-h"
    ):
        """
        Initialize UNI2 embedding extractor.
        
        Args:
            device: Device for inference
            batch_size: Batch size for processing (default 256 as UNI2 is efficient)
            num_workers: Number of workers
            verbose: Whether to print progress
            model_path: HuggingFace model hub identifier
        """
        super().__init__(device, batch_size, num_workers, verbose)
        self.model_path = model_path
    
    def load_model(self) -> Tuple[torch.nn.Module, Any]:
        """Load the UNI2 pathology foundation model."""
        if self.verbose:
            print(f"Loading UNI2 model on {self.device}...")
        
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        
        model = timm.create_model(
            self.model_path,
            pretrained=True,
            **timm_kwargs
        )
        model = model.to(self.device)
        model.eval()
        
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        
        if self.verbose:
            print("UNI2 model loaded successfully.")
        
        return model, transform
    
    def extract_features(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features from a batch of images using UNI2.
        
        UNI2 directly outputs the embedding (no need for token manipulation).
        """
        with torch.no_grad():
            features = self.model(batch_tensor)
            return features.cpu().numpy()


@register_patch_embedding("uni")
class UNIEmbeddingExtractor(BasePatchEmbeddingExtractor):
    """
    Extractor for patch-level embeddings using UNI (v1) model.
    
    UNI is the original pathology foundation model from Mahmood Lab.
    
    Output embedding dimension: 1024
    """
    
    MODEL_NAME = "uni"
    EMBEDDING_DIM = 1024
    INPUT_SIZE = 224
    
    def __init__(
        self,
        device: str = 'cuda',
        batch_size: int = 256,
        num_workers: int = 8,
        verbose: bool = True,
        model_dir: str = "/cnvrg/model"
    ):
        """
        Initialize UNI embedding extractor.
        
        Args:
            device: Device for inference
            batch_size: Batch size for processing
            num_workers: Number of workers
            verbose: Whether to print progress
            model_dir: Directory containing the model weights
        """
        super().__init__(device, batch_size, num_workers, verbose)
        self.model_dir = model_dir
    
    def load_model(self) -> Tuple[torch.nn.Module, Any]:
        """Load the UNI pathology foundation model."""
        import os
        
        if self.verbose:
            print(f"Loading UNI model on {self.device}...")
        
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True
        )
        
        weights_path = os.path.join(self.model_dir, "pytorch_model.bin")
        model.load_state_dict(
            torch.load(weights_path, map_location=self.device),
            strict=True
        )
        model = model.to(self.device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        if self.verbose:
            print("UNI model loaded successfully.")
        
        return model, transform
    
    def extract_features(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """Extract features from a batch of images using UNI."""
        with torch.no_grad():
            features = self.model(batch_tensor)
            return features.cpu().numpy()


# Helper functions for backward compatibility
def load_uni_model(local_dir="/cnvrg/model", device='cuda'):
    """Load UNI model (backward compatibility)."""
    extractor = UNIEmbeddingExtractor(device=device, model_dir=local_dir, verbose=True)
    return extractor.load_model()


def load_univ2_model(local_dir="/cnvrg/model", device='cuda'):
    """Load UNI2 model (backward compatibility)."""
    extractor = UNI2EmbeddingExtractor(device=device, verbose=True)
    return extractor.load_model()


def process_slide_torch(slide_path, data, model, base_patchsize, transform, batch_size=128, device='cuda'):
    """
    Process a single slide and extract features (backward compatibility).
    
    This function is provided for backward compatibility with existing scripts.
    For new code, use UNI2EmbeddingExtractor.extract_slide_embeddings() instead.
    """
    import os
    import openslide
    
    batch_size = 256
    slide_fn = os.path.join(slide_path, data[0])
    
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
