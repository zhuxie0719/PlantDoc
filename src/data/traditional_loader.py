"""
Data loading utilities for traditional machine learning methods.
Loads images as numpy arrays (not tensors) for feature extraction.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_splits(split_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Load split JSON and return classes list and splits dict."""
    payload = json.loads(split_path.read_text())
    return payload["classes"], payload["splits"]


def load_images_for_split(
    split_name: str,
    split_entries: List[str],
    class_to_idx: Dict[str, int],
    processed_root: Path,
    target_size: Optional[Tuple[int, int]] = None,
    grayscale: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all images for a split as numpy arrays.
    
    Args:
        split_name: 'train', 'val', or 'test'
        split_entries: List of relative paths from split JSON
        class_to_idx: Mapping from class name to index
        processed_root: Root directory of processed images
        target_size: Optional (width, height) to resize. If None, use original size.
        grayscale: If True, convert to grayscale
        verbose: Show progress bar
    
    Returns:
        images: np.ndarray of shape (N, H, W, C) or (N, H, W) if grayscale
        labels: np.ndarray of shape (N,)
        file_paths: List of relative paths for reference
    """
    images = []
    labels = []
    file_paths = []
    
    iterator = tqdm(split_entries, desc=f"Loading {split_name}") if verbose else split_entries
    
    for entry in iterator:
        rel = Path(entry)
        class_name = rel.parent.name
        filename = rel.name
        
        img_path = processed_root / split_name / class_name / filename
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue
        
        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            
            # Resize if needed
            if target_size:
                img = img.resize(target_size, Image.Resampling.BICUBIC)
            
            # Convert to grayscale if needed
            if grayscale:
                img = img.convert("L")
            
            # Convert to numpy array
            img_array = np.asarray(img, dtype=np.uint8)
            images.append(img_array)
            labels.append(class_to_idx[class_name])
            file_paths.append(str(entry))
    
    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)
    
    return images, labels, file_paths


def load_all_splits(
    split_json_path: Path,
    processed_root: Path,
    target_size: Optional[Tuple[int, int]] = None,
    grayscale: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Load train, val, and test splits.
    
    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing
        (images, labels, file_paths) tuple.
    """
    classes, splits = load_splits(split_json_path)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    
    result = {}
    for split_name in ["train", "val", "test"]:
        images, labels, paths = load_images_for_split(
            split_name=split_name,
            split_entries=splits[split_name],
            class_to_idx=class_to_idx,
            processed_root=processed_root,
            target_size=target_size,
            grayscale=grayscale,
        )
        result[split_name] = (images, labels, paths)
    
    return result, classes







