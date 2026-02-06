"""
Tests for PePath pipeline.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pepath import PePathConfig, PePathPipeline


class TestPePathConfig:
    """Tests for PePathConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PePathConfig()
        assert config.base_patchsize == 512
        assert config.device in ['cuda', 'cpu']
        assert config.top_k == 10
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'slide_path': '/test/path',
            'output_dir': '/test/output',
            'batch_size': 32
        }
        config = PePathConfig.from_dict(config_dict)
        assert config.slide_path == '/test/path'
        assert config.batch_size == 32
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = PePathConfig(
            slide_path='/nonexistent/path',
            base_patchsize=-1
        )
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = PePathConfig(slide_path='/test')
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'slide_path' in config_dict


class TestPipeline:
    """Tests for PePathPipeline class."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        config = PePathConfig(slide_path='/test', output_dir='/test/output')
        pipeline = PePathPipeline(config)
        assert pipeline.config.slide_path == '/test'
    
    def test_pipeline_from_dict(self):
        """Test pipeline can be initialized from dict."""
        pipeline = PePathPipeline({
            'slide_path': '/test',
            'output_dir': '/test/output'
        })
        assert pipeline.config.slide_path == '/test'


class TestUtils:
    """Tests for utility functions."""
    
    def test_normalize_embeddings(self):
        """Test embedding normalization."""
        from pepath.utils import normalize_embeddings
        
        embeddings = np.random.randn(10, 128).astype(np.float32)
        
        # L2 normalization
        normalized = normalize_embeddings(embeddings, 'l2')
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(10))
    
    def test_get_slide_id(self):
        """Test slide ID extraction."""
        from pepath.utils import get_slide_id
        
        assert get_slide_id('/path/to/slide.svs') == 'slide'
        assert get_slide_id('/path/to/sample.ndpi') == 'sample'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
