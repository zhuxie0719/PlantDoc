"""
Feature extraction for traditional machine learning methods.
Includes HOG, SIFT/SURF, color histograms, and texture features (LBP).
"""
import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler

try:
    from skimage.feature import hog, local_binary_pattern
    from skimage import color
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Some features will be unavailable.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not available. SIFT/SURF features will be unavailable.")


def extract_hog_features(
    images: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    visualize: bool = False,
) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features.
    
    Args:
        images: Array of shape (N, H, W) or (N, H, W, C). If color, converts to grayscale.
        orientations: Number of orientation bins
        pixels_per_cell: Size of cells in pixels
        cells_per_block: Number of cells per block
        visualize: If True, also return visualization (not implemented for batch)
    
    Returns:
        features: Array of shape (N, feature_dim)
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image is required for HOG features")
    
    if len(images.shape) == 4:
        # Convert RGB to grayscale
        images = np.array([color.rgb2gray(img) for img in images])
    
    features = []
    for img in images:
        feat = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True,
        )
        features.append(feat)
    
    return np.array(features)


def extract_color_histogram(
    images: np.ndarray,
    bins: int = 32,
    channels: str = "rgb",
) -> np.ndarray:
    """
    Extract color histogram features.
    
    Args:
        images: Array of shape (N, H, W, 3) RGB images
        bins: Number of bins per channel
        channels: Which channels to use ('rgb', 'hsv', or 'lab')
    
    Returns:
        features: Array of shape (N, bins * num_channels)
    """
    if len(images.shape) != 4 or images.shape[3] != 3:
        raise ValueError("Expected RGB images of shape (N, H, W, 3)")
    
    features = []
    for img in images:
        if channels == "hsv":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) if CV2_AVAILABLE else img
        elif channels == "lab":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) if CV2_AVAILABLE else img
        
        hist_features = []
        for channel_idx in range(3):
            hist, _ = np.histogram(img[:, :, channel_idx], bins=bins, range=(0, 256))
            hist_features.extend(hist)
        
        features.append(hist_features)
    
    return np.array(features, dtype=np.float32)


def extract_lbp_features(
    images: np.ndarray,
    radius: int = 3,
    n_points: int = 24,
    method: str = "uniform",
) -> np.ndarray:
    """
    Extract LBP (Local Binary Pattern) texture features.
    
    Args:
        images: Array of shape (N, H, W) or (N, H, W, C). If color, converts to grayscale.
        radius: Radius of the circular neighborhood
        n_points: Number of points in the circular neighborhood
        method: Method for LBP computation ('default', 'ror', 'uniform', 'var')
    
    Returns:
        features: Array of shape (N, feature_dim) - histogram of LBP values
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image is required for LBP features")
    
    if len(images.shape) == 4:
        # Convert RGB to grayscale
        images = np.array([color.rgb2gray(img) for img in images])
        images = (images * 255).astype(np.uint8)
    
    features = []
    for img in images:
        lbp = local_binary_pattern(img, n_points, radius, method=method)
        # Compute histogram
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_points + 2,
            range=(0, n_points + 2),
        )
        # Normalize
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # Avoid division by zero
        features.append(hist)
    
    return np.array(features)


def extract_sift_features(
    images: np.ndarray,
    max_keypoints: int = 100,
    n_features: int = 128,
) -> np.ndarray:
    """
    Extract SIFT features using Bag-of-Visual-Words approach.
    
    Note: This is a simplified version. For full BOVW, you need to:
    1. Extract SIFT descriptors for all images
    2. Cluster descriptors (e.g., K-means) to create vocabulary
    3. Quantize each image's descriptors to vocabulary
    4. Create histogram
    
    This function returns a fixed-size feature vector per image.
    For now, it returns the mean and std of SIFT descriptors.
    
    Args:
        images: Array of shape (N, H, W) grayscale images
        max_keypoints: Maximum number of keypoints to extract
        n_features: SIFT descriptor dimension (always 128)
    
    Returns:
        features: Array of shape (N, n_features * 2) - mean and std of descriptors
    """
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is required for SIFT features")
    
    if len(images.shape) == 4:
        # Convert RGB to grayscale
        images = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images])
    
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    features = []
    
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        if descriptors is None or len(descriptors) == 0:
            # No keypoints found, return zero vector
            feat = np.zeros(n_features * 2, dtype=np.float32)
        else:
            # Use mean and std of descriptors as feature
            mean_desc = np.mean(descriptors, axis=0)
            std_desc = np.std(descriptors, axis=0)
            feat = np.concatenate([mean_desc, std_desc])
        
        features.append(feat)
    
    return np.array(features)


def extract_surf_features(
    images: np.ndarray,
    hessian_threshold: int = 400,
    max_keypoints: int = 100,
) -> np.ndarray:
    """
    Extract SURF features (similar to SIFT).
    
    Args:
        images: Array of shape (N, H, W) grayscale images
        hessian_threshold: Threshold for hessian keypoint detector
        max_keypoints: Maximum number of keypoints
    
    Returns:
        features: Array of shape (N, feature_dim)
    """
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is required for SURF features")
    
    try:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    except AttributeError:
        # OpenCV 4.x doesn't include SURF by default
        raise ImportError(
            "SURF is not available in this OpenCV build. "
            "You may need opencv-contrib-python."
        )
    
    if len(images.shape) == 4:
        images = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images])
    
    features = []
    for img in images:
        keypoints, descriptors = surf.detectAndCompute(img, None)
        
        if descriptors is None or len(descriptors) == 0:
            feat = np.zeros(64 * 2, dtype=np.float32)  # SURF default is 64-dim
        else:
            mean_desc = np.mean(descriptors, axis=0)
            std_desc = np.std(descriptors, axis=0)
            feat = np.concatenate([mean_desc, std_desc])
        
        features.append(feat)
    
    return np.array(features)


def combine_features(
    feature_dict: dict,
    normalize: bool = True,
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """
    Combine multiple feature types into a single feature matrix.
    
    Args:
        feature_dict: Dictionary mapping feature names to feature arrays
        normalize: If True, apply StandardScaler normalization
    
    Returns:
        combined_features: Array of shape (N, total_feature_dim)
        scaler: Fitted StandardScaler (if normalize=True) or None
    """
    feature_list = []
    for name, feat in feature_dict.items():
        if feat is not None and len(feat) > 0:
            feature_list.append(feat)
    
    if len(feature_list) == 0:
        raise ValueError("No valid features to combine")
    
    combined = np.hstack(feature_list)
    
    scaler = None
    if normalize:
        scaler = StandardScaler()
        combined = scaler.fit_transform(combined)
    
    return combined, scaler







