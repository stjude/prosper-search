#!/usr/bin/env python3
"""
Download script for PePath data files.

This script downloads the required data files for running PePath demos and searches.
Data can be downloaded from St. Jude internal servers or external hosting (Zenodo).

Usage:
    python scripts/download_data.py --all           # Download everything (~14 GB)
    python scripts/download_data.py --essential    # Download essential files only (~10 GB)
    python scripts/download_data.py --patch-embeddings  # Download patch embeddings only
    python scripts/download_data.py --list         # List available files and sizes
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, Optional

# Data source configurations
INTERNAL_SOURCE = "REDACTED_INTERNAL_PATH/DNB_Core_Public/COMET_portal_files"
ZENODO_DOI = None  # To be added upon publication

# File manifest with sizes (in MB)
DATA_FILES = {
    "patch_embeddings": {
        "source_path": "COMET_final/patch_embeddings",
        "dest_name": "patch_embeddings",
        "size_mb": 9800,
        "is_dir": True,
        "essential": True,
        "description": "Pre-computed Virchow2 patch embeddings (2,921 slides)",
    },
    "sampler_matrix": {
        "source_path": "final_slides_sampler_matrix.pkl",
        "dest_name": "final_slides_sampler_matrix.pkl",
        "size_mb": 286,
        "is_dir": False,
        "essential": True,
        "description": "SAMPLER slide embeddings for all reference slides",
    },
    "mosaic_metadata": {
        "source_path": "mosaic_metadata.pkl",
        "dest_name": "mosaic_metadata.pkl",
        "size_mb": 1200,
        "is_dir": False,
        "essential": False,
        "description": "Mosaic cluster locations for heatmap visualization",
    },
    "thumbnails": {
        "source_path": "thumbnails",
        "dest_name": "thumbnails",
        "size_mb": 415,
        "is_dir": True,
        "essential": False,
        "description": "Slide thumbnails for heatmap overlays",
    },
    "sample_slide": {
        "source_path": "sample_slides/4261.svs",
        "dest_name": "4261.svs",
        "size_mb": 2300,
        "is_dir": False,
        "essential": False,
        "description": "Sample slide for demo notebook",
    },
}


def get_project_root() -> Path:
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def check_internal_access() -> bool:
    """Check if internal data source is accessible."""
    return os.path.exists(INTERNAL_SOURCE)


def format_size(size_mb: float) -> str:
    """Format size in human-readable format."""
    if size_mb >= 1000:
        return f"{size_mb / 1000:.1f} GB"
    return f"{size_mb:.0f} MB"


def list_files():
    """List available data files and their status."""
    data_dir = get_data_dir()
    internal_available = check_internal_access()
    
    print("\n📁 PePath Data Files")
    print("=" * 80)
    print(f"{'File':<25} {'Size':>10} {'Essential':>10} {'Status':>15}")
    print("-" * 80)
    
    total_size = 0
    total_missing = 0
    
    for name, info in DATA_FILES.items():
        dest_path = data_dir / info["dest_name"]
        exists = dest_path.exists()
        size = format_size(info["size_mb"])
        essential = "Yes" if info["essential"] else "No"
        status = "✅ Present" if exists else "❌ Missing"
        
        print(f"{info['dest_name']:<25} {size:>10} {essential:>10} {status:>15}")
        
        if not exists:
            total_size += info["size_mb"]
            total_missing += 1
    
    print("-" * 80)
    print(f"Missing: {total_missing} files, Total download: {format_size(total_size)}")
    
    if internal_available:
        print(f"\n✅ Internal source available: {INTERNAL_SOURCE}")
    else:
        print(f"\n⚠️  Internal source not available. Use Zenodo for external access.")
    
    print()


def copy_file(source: Path, dest: Path, is_dir: bool = False) -> bool:
    """Copy a file or directory with progress indication."""
    try:
        if is_dir:
            print(f"  Copying directory {source.name}...")
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
        else:
            print(f"  Copying {source.name}...")
            shutil.copy2(source, dest)
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def download_from_internal(file_key: str) -> bool:
    """Download a file from internal source."""
    if not check_internal_access():
        print(f"❌ Internal source not accessible: {INTERNAL_SOURCE}")
        return False
    
    info = DATA_FILES[file_key]
    source_path = Path(INTERNAL_SOURCE) / info["source_path"]
    dest_path = get_data_dir() / info["dest_name"]
    
    if not source_path.exists():
        print(f"❌ Source not found: {source_path}")
        return False
    
    print(f"\n📥 Downloading {file_key} ({format_size(info['size_mb'])})...")
    print(f"   {info['description']}")
    
    return copy_file(source_path, dest_path, info["is_dir"])


def download_from_zenodo(file_key: str) -> bool:
    """Download a file from Zenodo."""
    if ZENODO_DOI is None:
        print("❌ Zenodo download not yet available. Please use internal source.")
        print("   Zenodo hosting will be available upon publication.")
        return False
    
    # TODO: Implement Zenodo download when DOI is available
    print(f"Downloading {file_key} from Zenodo...")
    return False


def download_file(file_key: str, source: str = "auto") -> bool:
    """Download a specific file."""
    if file_key not in DATA_FILES:
        print(f"❌ Unknown file: {file_key}")
        print(f"   Available: {', '.join(DATA_FILES.keys())}")
        return False
    
    info = DATA_FILES[file_key]
    dest_path = get_data_dir() / info["dest_name"]
    
    if dest_path.exists():
        print(f"✅ {info['dest_name']} already exists, skipping.")
        return True
    
    if source == "auto":
        if check_internal_access():
            return download_from_internal(file_key)
        else:
            return download_from_zenodo(file_key)
    elif source == "internal":
        return download_from_internal(file_key)
    elif source == "zenodo":
        return download_from_zenodo(file_key)
    else:
        print(f"❌ Unknown source: {source}")
        return False


def download_all(source: str = "auto") -> bool:
    """Download all data files."""
    print("\n📥 Downloading all data files...")
    success = True
    for file_key in DATA_FILES:
        if not download_file(file_key, source):
            success = False
    return success


def download_essential(source: str = "auto") -> bool:
    """Download essential data files only."""
    print("\n📥 Downloading essential data files...")
    success = True
    for file_key, info in DATA_FILES.items():
        if info["essential"]:
            if not download_file(file_key, source):
                success = False
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download PePath data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_data.py --list              # Show available files
  python scripts/download_data.py --all               # Download everything
  python scripts/download_data.py --essential         # Download essentials only
  python scripts/download_data.py --patch-embeddings  # Download specific file
        """
    )
    
    parser.add_argument("--list", action="store_true",
                        help="List available files and their status")
    parser.add_argument("--all", action="store_true",
                        help="Download all data files (~14 GB)")
    parser.add_argument("--essential", action="store_true",
                        help="Download essential files only (~10 GB)")
    parser.add_argument("--source", choices=["auto", "internal", "zenodo"],
                        default="auto",
                        help="Data source (default: auto-detect)")
    
    # Individual file downloads
    parser.add_argument("--patch-embeddings", action="store_true",
                        help="Download patch embeddings (~10 GB)")
    parser.add_argument("--sampler-matrix", action="store_true",
                        help="Download SAMPLER slide embeddings (~286 MB)")
    parser.add_argument("--mosaic-metadata", action="store_true",
                        help="Download mosaic metadata (~1.2 GB)")
    parser.add_argument("--thumbnails", action="store_true",
                        help="Download slide thumbnails (~415 MB)")
    parser.add_argument("--sample-slide", action="store_true",
                        help="Download sample slide for demo (~2.3 GB)")
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    get_data_dir().mkdir(parents=True, exist_ok=True)
    
    if args.list:
        list_files()
        return
    
    # Handle specific file downloads
    file_flags = {
        "patch_embeddings": args.patch_embeddings,
        "sampler_matrix": args.sampler_matrix,
        "mosaic_metadata": args.mosaic_metadata,
        "thumbnails": args.thumbnails,
        "sample_slide": args.sample_slide,
    }
    
    any_specific = any(file_flags.values())
    
    if args.all:
        success = download_all(args.source)
    elif args.essential:
        success = download_essential(args.source)
    elif any_specific:
        success = True
        for file_key, should_download in file_flags.items():
            if should_download:
                if not download_file(file_key, args.source):
                    success = False
    else:
        # No arguments - show help and list
        parser.print_help()
        print()
        list_files()
        return
    
    if success:
        print("\n✅ Download complete!")
    else:
        print("\n⚠️  Some downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
