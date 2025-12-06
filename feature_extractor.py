"""
Feature extraction utilities for Korean character recognition.

This module provides functions to extract HOG and Gabor features
from handwritten Korean character images.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Dict, Any

# Image preprocessing constants
IMG_SIZE = 64
NORMALIZATION_FACTOR = 255.0

# HOG feature parameters
DEFAULT_HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

# Gabor filter parameters
DEFAULT_GABOR_PARAMS = {
    "ksize": 7,
    "sigma": 4.0,
    "lambd": 10.0,
    "gamma": 0.5,
    "n_orientations": 4,
}

# Supported image extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def load_and_preprocess_image(path: str) -> Optional[np.ndarray]:
    """
    Load and preprocess an image for feature extraction.
    
    Args:
        path: Path to image file
        
    Returns:
        Preprocessed image array or None if loading fails
    """
    if not os.path.exists(path):
        return None
        
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / NORMALIZATION_FACTOR
    
    return img

def load_dataset(root_dir):
    X = []
    y = []
    labels = sorted(os.listdir(root_dir))

    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        folder = os.path.join(root_dir, label)
        if not os.path.isdir(folder):
            continue

        for file in tqdm(os.listdir(folder), desc=f"Loading {label}"):
            img_path = os.path.join(folder, file)
            img = load_and_preprocess_image(img_path)

            X.append(img)
            y.append(label_to_idx[label])

    return np.array(X), np.array(y), label_to_idx

def extract_hog_features(X: np.ndarray, hog_params: Optional[Dict] = None) -> np.ndarray:
    """
    Extract HOG features from image array.
    
    Args:
        X: Array of images
        hog_params: Optional HOG parameters
        
    Returns:
        Array of HOG features
    """
    if hog_params is None:
        hog_params = DEFAULT_HOG_PARAMS
        
    hog_features = []
    for img in tqdm(X, desc="Extracting HOG"):
        feature = hog(img, **hog_params)
        hog_features.append(feature)

    return np.array(hog_features)


def build_gabor_kernels(
    ksize: int = 7,
    sigma: float = 4.0,
    lambd: float = 10.0,
    gamma: float = 0.5,
    n_orientations: int = 4,
) -> List[np.ndarray]:
    """
    Build Gabor filter kernels with multiple orientations.
    
    Args:
        ksize: Kernel size
        sigma: Standard deviation of Gaussian envelope
        lambd: Wavelength of sinusoidal factor
        gamma: Spatial aspect ratio
        n_orientations: Number of orientations
        
    Returns:
        List of Gabor kernels
    """
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / n_orientations):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
        kernels.append(kern)
    return kernels


def extract_gabor_features(
    X: np.ndarray, gabor_params: Optional[Dict] = None
) -> np.ndarray:
    """
    Extract Gabor features from image array.
    
    Args:
        X: Array of images
        gabor_params: Optional Gabor parameters
        
    Returns:
        Array of Gabor features
    """
    if gabor_params is None:
        gabor_params = DEFAULT_GABOR_PARAMS
        
    kernels = build_gabor_kernels(**gabor_params)
    features = []

    for img in tqdm(X, desc="Extracting Gabor"):
        responses = [cv2.filter2D(img, cv2.CV_32F, k) for k in kernels]
        feat = np.concatenate([r.flatten() for r in responses])
        features.append(feat)

    return np.array(features)


def _iter_image_paths_for_class(
    basic_root: str, extended_root: str, cls_name: str
) -> List[str]:
    """
    Get all image paths for a given class from basic and extended datasets.
    
    Args:
        basic_root: Path to basic dataset root
        extended_root: Path to extended dataset root
        cls_name: Class name
        
    Returns:
        Sorted list of image paths
    """
    paths = []

    basic_cls_dir = os.path.join(basic_root, cls_name)
    extended_cls_dir = os.path.join(extended_root, cls_name)

    if os.path.isdir(basic_cls_dir):
        paths += [
            os.path.join(basic_cls_dir, f)
            for f in os.listdir(basic_cls_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

    if os.path.isdir(extended_cls_dir):
        paths += [
            os.path.join(extended_cls_dir, f)
            for f in os.listdir(extended_cls_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

    return sorted(paths)


def _extract_feature_for_image(img, extractor_name, hog_params=None, gabor_params=None, gabor_kernels=None):
    if extractor_name == "hog":
        if hog_params is None:
            hog_params = {}
        feature = hog(
            img,
            orientations=hog_params.get("orientations", 9),
            pixels_per_cell=hog_params.get("pixels_per_cell", (8, 8)),
            cells_per_block=hog_params.get("cells_per_block", (2, 2)),
            block_norm=hog_params.get("block_norm", "L2-Hys")
        )
        return feature

    elif extractor_name == "gabor":
        if gabor_params is None:
            gabor_params = {}
        if gabor_kernels is None:
            gabor_kernels = build_gabor_kernels(ksize=gabor_params.get("ksize", 7),
                                                sigma=gabor_params.get("sigma", 4.0),
                                                lambd=gabor_params.get("lambd", 10.0),gamma=gabor_params.get("gamma", 0.5),
                                                n_orientations=gabor_params.get("n_orientations", 4))
            
        responses = [cv2.filter2D(img, cv2.CV_32F, k) for k in gabor_kernels]
        feat = np.concatenate([r.flatten() for r in responses])
        return feat

    else:
        raise ValueError(f"Unknown extractor_name: {extractor_name}")


# def extract_and_save_per_class_features(basic_root, extended_root, output_root="features", extractors=("hog", "gabor"), hog_params=None, gabor_params=None):
#     os.makedirs(output_root, exist_ok=True)

#     classes = sorted([d for d in os.listdir(basic_root) if os.path.isdir(os.path.join(basic_root, d))])

#     print("Found classes:", classes)

#     for extractor_name in extractors:
#         print(f"\n=== Extractor: {extractor_name} ===")
#         feat_out_dir = os.path.join(output_root, extractor_name)
#         os.makedirs(feat_out_dir, exist_ok=True)

#         gabor_kernels = None
#         if extractor_name == "gabor":
#             if gabor_params is None:
#                 gabor_params = {}
#             gabor_kernels = build_gabor_kernels(ksize=gabor_params.get("ksize", 7), 
#                                                 sigma=gabor_params.get("sigma", 4.0), 
#                                                 lambd=gabor_params.get("lambd", 10.0), 
#                                                 gamma=gabor_params.get("gamma", 0.5), 
#                                                 n_orientations=gabor_params.get("n_orientations", 4))

#         for cls in classes:
#             img_paths = _iter_image_paths_for_class(basic_root, extended_root, cls)

#             if not img_paths:
#                 print(f"[{extractor_name}] Class {cls}: no images, skipping.")
#                 continue

#             feats = []
#             for p in tqdm(img_paths, desc=f"{extractor_name} | {cls}", ncols=80):
#                 img = load_and_preprocess_image(p)
#                 if img is None:
#                     continue
#                 feat = _extract_feature_for_image(
#                     img,
#                     extractor_name,
#                     hog_params=hog_params,
#                     gabor_params=gabor_params,
#                     gabor_kernels=gabor_kernels
#                 )
#                 feats.append(feat)

#             feats = np.array(feats, dtype=np.float32)
#             out_path = os.path.join(feat_out_dir, f"{cls}.npy")
#             np.save(out_path, feats)

#             print(f"[{extractor_name}] Class {cls}: saved {feats.shape} -> {out_path}")

# if __name__ == "__main__":
#     basic_root = "./archive/Hangul Database/Hangul Database"
#     extended_root = "./archive/Hangul Database Extended/Hangul Database Extended"

#     hog_params = {"orientations": 9,
#                   "pixels_per_cell": (8, 8),
#                   "cells_per_block": (2, 2),
#                   "block_norm": "L2-Hys"}

#     gabor_params = {"ksize": 7,
#                     "sigma": 4.0,
#                     "lambd": 10.0,
#                     "gamma": 0.5,
#                     "n_orientations": 4}

#     extract_and_save_per_class_features(basic_root,
#                                         extended_root,
#                                         output_root="features",
#                                         extractors=("hog", "gabor"),
#                                         hog_params=hog_params,
#                                         gabor_params=gabor_params)

def extract_hog_feature_for_image(
    img: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    """
    Extract HOG feature from a single image.
    
    Args:
        img: Input image
        orientations: Number of orientation bins
        pixels_per_cell: Size of cell
        cells_per_block: Number of cells in block
        block_norm: Block normalization method
        
    Returns:
        HOG feature vector
    """
    feature = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
    )
    return feature


def extract_and_save_hog_per_class(
    basic_root,
    extended_root,
    output_root="features",
    hog_params=None,
):

    if hog_params is None:
        hog_params = {}

    os.makedirs(output_root, exist_ok=True)

    hog_out_dir = os.path.join(output_root, "hog")
    os.makedirs(hog_out_dir, exist_ok=True)

    classes = sorted(
        [
            d
            for d in os.listdir(basic_root)
            if os.path.isdir(os.path.join(basic_root, d))
        ]
    )

    print("Found classes:", classes)

    print("\n=== Extractor: hog (resume enabled) ===")
    for cls in classes:
        out_path = os.path.join(hog_out_dir, f"{cls}.npy")

        if os.path.exists(out_path):
            print(f"[hog] Class {cls}: already exists at {out_path}, skip.")
            continue

        img_paths = _iter_image_paths_for_class(basic_root, extended_root, cls)
        if not img_paths:
            print(f"[hog] Class {cls}: no images found, skipping.")
            continue

        print(f"[hog] Class {cls}: {len(img_paths)} images")

        feats = []
        for p in tqdm(img_paths, desc=f"hog | {cls}", ncols=80):
            img = load_and_preprocess_image(p)
            if img is None:
                print(f"  [WARN] failed to load image: {p}")
                continue

            try:
                feat = extract_hog_feature_for_image(img, **hog_params)
                feats.append(feat)
            except Exception as e:
                print(f"  [ERROR] HOG failed at {p}: {e}")
                continue

        if len(feats) == 0:
            print(f"[hog] Class {cls}: no valid features extracted, skip saving.")
            continue

        feats = np.array(feats, dtype=np.float32)
        np.save(out_path, feats)

        print(f"[hog] Class {cls}: saved {feats.shape} -> {out_path}")


if __name__ == "__main__":
    basic_root = "archive/Hangul Database/Hangul Database"
    extended_root = "archive/Hangul Database Extended/Hangul Database Extended"

    hog_params = {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
    }

    extract_and_save_hog_per_class(
        basic_root=basic_root,
        extended_root=extended_root,
        output_root="features",
        hog_params=hog_params,
    )
