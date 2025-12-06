"""
Data loading utilities for Korean character recognition.

This module provides functions to load pre-extracted HOG features
and split datasets for training and testing.
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split

# Constants
DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_TEST_RATIO = 0.2


def get_class_names(feature_dir: str) -> List[str]:
    """
    Extract class names from feature directory.

    Args:
        feature_dir: Directory containing .npy feature files

    Returns:
        Sorted list of class names
    """
    class_names = sorted(
        [f[:-4] for f in os.listdir(feature_dir) if f.endswith(".npy")]
    )
    return class_names


def load_hog_features(
    feature_dir: str,
    selected_classes: Optional[List[str]] = None,
    max_samples_per_class: Optional[int] = None,
    shuffle: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load HOG features from pre-extracted .npy files.

    Args:
        feature_dir: Directory containing feature files
        selected_classes: Optional list of classes to load
        max_samples_per_class: Optional limit on samples per class
        shuffle: Whether to shuffle the data
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features, labels, class_names)
    """

    class_names = get_class_names(feature_dir)

    if selected_classes is not None:
        class_names = [c for c in class_names if c in selected_classes]

    X_list = []
    y_list = []

    for label_idx, cls in enumerate(class_names):
        path = os.path.join(feature_dir, f"{cls}.npy")
        if not os.path.exists(path):
            print(f"feature file not found for class {cls}: {path}")
            continue

        feats = np.load(path)

        if max_samples_per_class is not None and len(feats) > max_samples_per_class:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(len(feats), size=max_samples_per_class, replace=False)
            feats = feats[indices]

        labels = np.full(len(feats), label_idx, dtype=np.int64)

        X_list.append(feats)
        y_list.append(labels)

    if not X_list:
        raise ValueError(
            f"No features loaded from {feature_dir}. Check directory and files."
        )

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]

    return X, y, class_names


def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets.

    Args:
        X: Feature array
        y: Label array
        train_ratio: Proportion of training data
        test_ratio: Proportion of test data
        random_state: Random seed
        stratify: Whether to use stratified split

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """

    total = train_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    strat = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=(1.0 - train_ratio),
        random_state=random_state,
        stratify=strat,
    )

    return X_train, y_train, X_test, y_test
