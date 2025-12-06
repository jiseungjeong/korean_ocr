"""
Extract HOG features from individual Jamo images.

This script processes the hangul_characters_v1 dataset containing individual
Jamo images and extracts HOG features for training Jamo-level classifiers.
"""

import os
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm
from collections import defaultdict

from config import IMG_SIZE, NORMALIZATION_FACTOR

# Paths
JAMO_DATA_DIR = "data/hangul_characters_v1"
JAMO_FEATURES_DIR = "features/jamo"

# Jamo categorization
CHO_JAMOS = ["g", "gg", "n", "d", "r", "m", "b", "bb", "s", "ss", "j", "ch", "k", "t", "p", "h"]
JUNG_JAMOS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", 
              "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
JONG_JAMOS = ["", "k", "n", "ng", "l", "m", "s", "ss", "t"]  # Empty = no final consonant

print("=" * 80)
print("JAMO-LEVEL HOG FEATURE EXTRACTION")
print("=" * 80)

# Create output directories
os.makedirs(os.path.join(JAMO_FEATURES_DIR, "cho"), exist_ok=True)
os.makedirs(os.path.join(JAMO_FEATURES_DIR, "jung"), exist_ok=True)
os.makedirs(os.path.join(JAMO_FEATURES_DIR, "jong"), exist_ok=True)

print(f"\nSource: {JAMO_DATA_DIR}")
print(f"Output: {JAMO_FEATURES_DIR}")


def extract_hog_from_image(img_path):
    """Extract HOG features from a single image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / NORMALIZATION_FACTOR
    
    features = hog(
        img_norm,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    )
    return features


def categorize_jamo(jamo_name):
    """Categorize a Jamo into cho/jung/jong."""
    if jamo_name in CHO_JAMOS:
        return "cho"
    elif jamo_name in JUNG_JAMOS:
        return "jung"
    elif jamo_name in JONG_JAMOS:
        return "jong"
    else:
        # Try to infer from name patterns
        # Consonants at start positions are usually 초성
        if jamo_name in ["ng"]:  # Special case: ng is usually 종성
            return "jong"
        # All vowels are 중성
        return "jung"  # Default to jung for unknown


# Scan all files and group by Jamo
print("\n[1/4] Scanning Jamo image files...")
jamo_files = defaultdict(list)

all_files = [f for f in os.listdir(JAMO_DATA_DIR) if f.endswith('.jpg')]
print(f"Total files: {len(all_files)}")

for filename in all_files:
    # Parse filename: {jamo}_{rotation}_{sample}.jpg
    parts = filename.replace('.jpg', '').split('_')
    if len(parts) >= 3:
        jamo_name = parts[0]
        jamo_files[jamo_name].append(filename)

print(f"Unique Jamos: {len(jamo_files)}")
print(f"Sample Jamos: {list(jamo_files.keys())[:10]}")

# Extract features for each Jamo type
print("\n[2/4] Extracting HOG features for each Jamo...")

cho_features = {}
jung_features = {}
jong_features = {}

for jamo_name, file_list in tqdm(jamo_files.items(), desc="Processing Jamos"):
    category = categorize_jamo(jamo_name)
    
    # Extract features from all images of this Jamo
    features_list = []
    for filename in file_list:
        img_path = os.path.join(JAMO_DATA_DIR, filename)
        features = extract_hog_from_image(img_path)
        if features is not None:
            features_list.append(features)
    
    if len(features_list) == 0:
        continue
    
    # Stack features
    features_array = np.array(features_list)
    
    # Save based on category
    if category == "cho":
        cho_features[jamo_name] = features_array
    elif category == "jung":
        jung_features[jamo_name] = features_array
    elif category == "jong":
        jong_features[jamo_name] = features_array

print(f"\n초성 (Cho) Jamos: {len(cho_features)}")
print(f"중성 (Jung) Jamos: {len(jung_features)}")
print(f"종성 (Jong) Jamos: {len(jong_features)}")

# Save features
print("\n[3/4] Saving extracted features...")

for jamo_name, features in cho_features.items():
    output_path = os.path.join(JAMO_FEATURES_DIR, "cho", f"{jamo_name}.npy")
    np.save(output_path, features)
    print(f"  Saved: {output_path} - shape {features.shape}")

for jamo_name, features in jung_features.items():
    output_path = os.path.join(JAMO_FEATURES_DIR, "jung", f"{jamo_name}.npy")
    np.save(output_path, features)
    print(f"  Saved: {output_path} - shape {features.shape}")

for jamo_name, features in jong_features.items():
    output_path = os.path.join(JAMO_FEATURES_DIR, "jong", f"{jamo_name}.npy")
    np.save(output_path, features)
    print(f"  Saved: {output_path} - shape {features.shape}")

# Save Jamo lists for later use
print("\n[4/4] Saving Jamo metadata...")
np.save(os.path.join(JAMO_FEATURES_DIR, "cho_jamos.npy"), list(cho_features.keys()))
np.save(os.path.join(JAMO_FEATURES_DIR, "jung_jamos.npy"), list(jung_features.keys()))
np.save(os.path.join(JAMO_FEATURES_DIR, "jong_jamos.npy"), list(jong_features.keys()))

print("\n" + "=" * 80)
print("JAMO FEATURE EXTRACTION COMPLETE!")
print("=" * 80)
print(f"\nExtracted features:")
print(f"  초성: {len(cho_features)} types")
print(f"  중성: {len(jung_features)} types")
print(f"  종성: {len(jong_features)} types")
print(f"\nOutput directory: {JAMO_FEATURES_DIR}/")

