"""
Traditional machine learning classifiers for PlantDoc.
Includes SVM, Logistic Regression, and MLP.
"""
import numpy as np
from typing import Dict, Optional, Tuple
import pickle
from pathlib import Path

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: Optional[str] = "scale",
    class_weight: Optional[str] = "balanced",
    verbose: bool = False,
) -> SVC:
    """
    Train SVM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        kernel: 'rbf', 'linear', or 'poly'
        C: Regularization parameter
        gamma: Kernel coefficient ('scale', 'auto', or float)
        class_weight: 'balanced' to handle class imbalance
        verbose: Print training progress
    
    Returns:
        Trained SVM model
    """
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        verbose=verbose,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    penalty: str = "l2",
    max_iter: int = 1000,
    class_weight: Optional[str] = "balanced",
    solver: str = "lbfgs",
) -> LogisticRegression:
    """
    Train Logistic Regression classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        C: Inverse regularization strength
        penalty: 'l1' or 'l2'
        max_iter: Maximum iterations
        class_weight: 'balanced' to handle class imbalance
        solver: Optimization solver
    
    Returns:
        Trained Logistic Regression model
    """
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        max_iter=max_iter,
        class_weight=class_weight,
        solver=solver,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    activation: str = "relu",
    alpha: float = 0.0001,
    learning_rate: str = "adaptive",
    max_iter: int = 500,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
) -> MLPClassifier:
    """
    Train Multi-Layer Perceptron classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        hidden_layer_sizes: Tuple of hidden layer sizes
        activation: Activation function
        alpha: L2 regularization parameter
        learning_rate: Learning rate schedule
        max_iter: Maximum iterations
        early_stopping: Use early stopping
        validation_fraction: Fraction of data for validation
    
    Returns:
        Trained MLP model
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained sklearn model
        X: Features
        y: True labels
        class_names: List of class names
        verbose: Print detailed report
    
    Returns:
        Dictionary with metrics
    """
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y, y_pred, average="weighted", zero_division=0
    )
    
    cm = confusion_matrix(y, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": support,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, target_names=class_names, zero_division=0))
    
    return metrics


def save_model(model, path: Path):
    """Save model to pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path):
    """Load model from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)







