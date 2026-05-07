#!/usr/bin/env python
"""
PROSPER - Pediatric Retrieval for Oncology Slides, Pathology Evidence, and Reporting
Setup script for installation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="prosper",
    version="1.0.0",
    author="Haoran Chen",
    author_email="haoran.chen@stjude.org",
    description="PROSPER: Pediatric Retrieval for Oncology Slides, Pathology Evidence, and Reporting",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/stjude/prosper-search",
    license="Apache-2.0",
    project_urls={
        "Bug Tracker": "https://github.com/stjude/prosper-search/issues",
        "Documentation": "https://github.com/stjude/prosper-search#readme",
        "Source Code": "https://github.com/stjude/prosper-search",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "openslide-python>=1.2.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "timm>=0.9.0",
        "huggingface-hub>=0.16.0",
        "POT>=0.9.0",
        "networkx>=2.6.0",
        "bitarray>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "colorcet>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prosper=prosper.cli:main",
            # Backward compatibility
            "pepath=prosper.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "prosper": ["*.yaml", "*.yml"],
    },
    keywords=[
        "pathology",
        "pediatric",
        "whole-slide-image",
        "deep-learning",
        "computer-vision",
        "medical-imaging",
        "similarity-search",
        "foundation-model",
        "virchow",
        "optimal-transport",
    ],
)
