#!/usr/bin/env python
"""
Compact Patch Embeddings Script

Converts patch embedding pickle files to a more compact numpy format.

Original format (pickle):
    (embeddings: np.ndarray[N, 1280], coords: list[dict{'wsi_loc': [x, y]}])
    
New format (compressed npz):
    embeddings: np.ndarray[N, 1280] float16 or int8 (quantized)
    coords: np.ndarray[N, 2] int32
    
Quantization mode saves ~50% more space with minimal precision loss.
"""

import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse


def quantize_to_int8(embeddings: np.ndarray) -> tuple:
    """
    Quantize float16 embeddings to int8 with per-feature min/max.
    
    Returns:
        quantized: np.ndarray int8
        scale: np.ndarray float32 (per feature)
        offset: np.ndarray float32 (per feature)
    """
    # Compute min/max per feature (column)
    fmin = embeddings.min(axis=0).astype(np.float32)
    fmax = embeddings.max(axis=0).astype(np.float32)
    
    # Avoid division by zero
    scale = (fmax - fmin)
    scale[scale == 0] = 1.0
    
    # Normalize to 0-255 and convert to uint8
    normalized = (embeddings.astype(np.float32) - fmin) / scale * 255.0
    quantized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return quantized, scale, fmin


def dequantize_from_int8(quantized: np.ndarray, scale: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Dequantize int8 embeddings back to float16.
    """
    return ((quantized.astype(np.float32) / 255.0 * scale) + offset).astype(np.float16)


def convert_to_compact(input_path: str, output_path: str, quantize: bool = False) -> dict:
    """
    Convert a single pickle file to compact npz format.
    
    Args:
        input_path: Path to input .pkl file
        output_path: Path to output .npz file
        quantize: If True, quantize embeddings to uint8
    
    Returns:
        dict with old_size, new_size, n_patches
    """
    # Load original data
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data[0]  # numpy array float16
    coords_list = data[1]  # List of dicts
    
    # Convert coordinates to numpy array
    coords = np.array([d['wsi_loc'] for d in coords_list], dtype=np.int32)
    
    if quantize:
        # Quantize embeddings to uint8
        quantized, scale, offset = quantize_to_int8(embeddings)
        np.savez_compressed(
            output_path,
            embeddings=quantized,
            coords=coords,
            scale=scale,
            offset=offset,
            quantized=np.array([True])
        )
    else:
        # Keep float16
        np.savez_compressed(output_path, embeddings=embeddings, coords=coords)
    
    old_size = os.path.getsize(input_path)
    new_size = os.path.getsize(output_path)
    
    return {
        'old_size': old_size,
        'new_size': new_size,
        'n_patches': len(coords)
    }


def load_compact_embeddings(filepath: str) -> tuple:
    """
    Load compact embeddings file.
    
    Returns:
        tuple: (embeddings: np.ndarray[N, 1280] float16, coords: np.ndarray[N, 2] int32)
    """
    data = np.load(filepath)
    
    # Check if quantized
    if 'quantized' in data and data['quantized'][0]:
        embeddings = dequantize_from_int8(data['embeddings'], data['scale'], data['offset'])
    else:
        embeddings = data['embeddings']
    
    return embeddings, data['coords']


def load_compact_as_original(filepath: str) -> tuple:
    """
    Load compact embeddings and convert back to original format.
    Useful for backward compatibility.
    
    Returns:
        tuple: (embeddings: np.ndarray, coords: list[dict])
    """
    embeddings, coords = load_compact_embeddings(filepath)
    coords_list = [{'wsi_loc': coord.tolist()} for coord in coords]
    return embeddings, coords_list


def main():
    parser = argparse.ArgumentParser(
        description="Convert patch embeddings to compact format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on 10 files with verification
  python compact_embeddings.py data/patch_embeddings --sample 10 --verify

  # Convert all files with quantization (best compression)
  python compact_embeddings.py data/patch_embeddings --quantize --in-place

  # Convert to new directory without quantization
  python compact_embeddings.py data/patch_embeddings --output-dir data/embeddings_compact
        """
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing .pkl embedding files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: input_dir + '_compact')"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Convert in place (delete original .pkl files after conversion)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize embeddings to uint8 (saves ~50%% more space, tiny precision loss)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion by comparing data"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process N files (for testing)"
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    
    if args.in_place:
        output_dir = input_dir
    else:
        output_dir = args.output_dir or (input_dir.rstrip('/') + '_compact')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all pickle files
    pkl_files = sorted(glob(os.path.join(input_dir, "*.pkl")))
    
    if args.sample:
        pkl_files = pkl_files[:args.sample]
    
    print(f"🔧 Converting {len(pkl_files)} files...")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Quantize: {args.quantize}")
    print()
    
    total_old = 0
    total_new = 0
    max_error = 0
    
    for pkl_path in tqdm(pkl_files, desc="Converting"):
        basename = os.path.basename(pkl_path).replace('.pkl', '')
        npz_path = os.path.join(output_dir, f"{basename}.npz")
        
        try:
            stats = convert_to_compact(pkl_path, npz_path, quantize=args.quantize)
            total_old += stats['old_size']
            total_new += stats['new_size']
            
            # Verify if requested
            if args.verify:
                # Load original
                with open(pkl_path, 'rb') as f:
                    orig = pickle.load(f)
                
                # Load converted
                embeddings, coords = load_compact_embeddings(npz_path)
                
                # Check embeddings match (with tolerance for quantized)
                if args.quantize:
                    error = np.abs(orig[0].astype(np.float32) - embeddings.astype(np.float32)).max()
                    max_error = max(max_error, error)
                    # Allow small quantization error
                    assert error < 0.1, f"Embeddings error too large: {basename} (max error: {error})"
                else:
                    assert np.allclose(orig[0], embeddings), f"Embeddings mismatch: {basename}"
                
                # Check coords match exactly
                orig_coords = np.array([d['wsi_loc'] for d in orig[1]])
                assert np.array_equal(orig_coords, coords), f"Coords mismatch: {basename}"
            
            # Delete original if in-place
            if args.in_place:
                os.remove(pkl_path)
                
        except Exception as e:
            print(f"\n❌ Error processing {basename}: {e}")
            continue
    
    # Summary
    reduction = (1 - total_new / total_old) * 100 if total_old > 0 else 0
    print()
    print("=" * 50)
    print(f"✅ Conversion complete!")
    print(f"   Original size: {total_old / 1e9:.2f} GB")
    print(f"   New size:      {total_new / 1e9:.2f} GB")
    print(f"   Reduction:     {reduction:.1f}%")
    if args.quantize and args.verify:
        print(f"   Max quant error: {max_error:.4f}")
    print("=" * 50)
    
    if not args.in_place:
        print(f"\n💡 To use the new format:")
        print(f"   from scripts.compact_embeddings import load_compact_embeddings")
        print(f"   embeddings, coords = load_compact_embeddings('file.npz')")


if __name__ == "__main__":
    main()
