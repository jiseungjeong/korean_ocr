"""
Configuration and constants for Korean character recognition project.
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2

# PCA parameters
PCA_N_COMPONENTS = 256

# KNN parameters
KNN_N_NEIGHBORS = 7

# Dataset paths
BASIC_DATASET_PATH = "archive/Hangul Database/Hangul Database"
EXTENDED_DATASET_PATH = "archive/Hangul Database Extended/Hangul Database Extended"
FEATURES_PATH = "features/hog-extended"  # Using extended dataset (134,400 samples)
RESULTS_PATH = "results"
ERROR_CASES_PATH = "results/error_cases"

# HOG parameters (default)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

# Image parameters
IMG_SIZE = 64
NORMALIZATION_FACTOR = 255.0

# Visualization parameters
CONFUSION_MATRIX_SIZE = (20, 18)
PER_CLASS_ACC_SIZE = (16, 10)
DPI = 300
