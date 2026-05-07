#!/usr/bin/env python3
"""
Script to extract slide dimensions from SVS files using multiprocessing.
Outputs a CSV with slide_id, width, height.
"""

import os
import sys
import csv
from glob import glob
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("Warning: openslide not installed. Install with: pip install openslide-python")


def get_slide_dimensions(slide_path):
    """
    Get width and height of a slide at level 0.
    
    Args:
        slide_path: Path to SVS file
    
    Returns:
        dict with slide_id, width, height, or error info
    """
    slide_id = os.path.basename(slide_path).replace(".svs", "")
    try:
        slide = openslide.OpenSlide(slide_path)
        width, height = slide.level_dimensions[0]
        slide.close()
        return {
            'slide_id': slide_id,
            'width': width,
            'height': height,
            'error': None
        }
    except Exception as e:
        return {
            'slide_id': slide_id,
            'width': None,
            'height': None,
            'error': str(e)
        }


def main():
    # Configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    slides_dir = os.environ.get("PROSPER_SLIDES_DIR", "./slides")
    output_csv = os.path.join(project_root, "data", "slide_dimensions.csv")
    num_workers = 10
    
    if not HAS_OPENSLIDE:
        print("openslide-python is required. Please install it.")
        sys.exit(1)
    
    # Find all SVS files
    print(f"Scanning for SVS files in: {slides_dir}")
    svs_files = glob(os.path.join(slides_dir, "**", "*.svs"), recursive=True)
    
    if len(svs_files) == 0:
        svs_files = glob(os.path.join(slides_dir, "*.svs"))
    
    print(f"Found {len(svs_files)} SVS files")
    print(f"Using {num_workers} worker processes")
    
    # Process with multiprocessing
    print("\nProcessing slides...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(get_slide_dimensions, svs_files),
            total=len(svs_files),
            desc="Reading dimensions"
        ))
    
    # Separate successful and failed
    successful = [r for r in results if r['error'] is None]
    failed = [r for r in results if r['error'] is not None]
    
    # Sort by slide_id
    successful.sort(key=lambda x: x['slide_id'])
    
    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['slide_id', 'width', 'height'])
        writer.writeheader()
        for r in successful:
            writer.writerow({
                'slide_id': r['slide_id'],
                'width': r['width'],
                'height': r['height']
            })
    
    print(f"\n✅ Saved dimensions to: {output_csv}")
    print(f"   Successfully processed: {len(successful)} slides")
    
    if failed:
        print(f"   Failed: {len(failed)} slides")
        for f in failed[:5]:
            print(f"     - {f['slide_id']}: {f['error']}")
        if len(failed) > 5:
            print(f"     ... and {len(failed) - 5} more")
    
    # Print summary stats
    if successful:
        widths = [r['width'] for r in successful]
        heights = [r['height'] for r in successful]
        print(f"\nDimension statistics:")
        print(f"   Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)//len(widths)}")
        print(f"   Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)//len(heights)}")


if __name__ == "__main__":
    main()
