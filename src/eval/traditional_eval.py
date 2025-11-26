"""
Evaluation and visualization utilities for traditional ML methods.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 12),
    normalize: bool = True,
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: If True, normalize by row (show percentages)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
        label = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        label = "Confusion Matrix"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Truncate long class names for display
    display_names = [name[:20] + "..." if len(name) > 20 else name for name in class_names]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(label, fontsize=14, fontweight="bold")
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_feature_comparison(
    feature_names: List[str],
    accuracies: List[float],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot bar chart comparing different feature combinations.
    
    Args:
        feature_names: List of feature combination names
        accuracies: List of accuracies
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(feature_names, accuracies, color="steelblue")
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Feature Combination Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim([0, 1])
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.01, i, f"{acc:.3f}", va="center", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved feature comparison to {save_path}")
    
    plt.show()


def visualize_hog_features(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    save_path: Optional[Path] = None,
):
    """
    Visualize HOG features for a single image.
    
    Args:
        image: Input image (grayscale or RGB)
        orientations: HOG orientations
        pixels_per_cell: HOG cell size
        cells_per_block: HOG block size
        save_path: Path to save figure
    """
    try:
        from skimage.feature import hog
        from skimage import color, exposure
    except ImportError:
        print("scikit-image required for HOG visualization")
        return
    
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    fd, hog_image = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        feature_vector=True,
    )
    
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.axis("off")
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title("Input Image")
    
    ax2.axis("off")
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title("HOG Visualization")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved HOG visualization to {save_path}")
    
    plt.show()


def plot_classifier_comparison(
    classifier_names: List[str],
    metrics: Dict[str, List[float]],
    metric_name: str = "Accuracy",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot bar chart comparing different classifiers.
    
    Args:
        classifier_names: List of classifier names
        metrics: Dictionary mapping metric names to lists of values
        metric_name: Which metric to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    if metric_name not in metrics:
        raise ValueError(f"Metric '{metric_name}' not found in metrics dict")
    
    values = metrics[metric_name]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(classifier_names, values, color="coral")
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Classifier Comparison - {metric_name}", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved classifier comparison to {save_path}")
    
    plt.show()

