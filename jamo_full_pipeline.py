"""
Full Jamo-based Classification Pipeline.

This script implements the complete end-to-end Jamo-based approach:
1. Load complete character images
2. Segment into cho/jung/jong regions
3. Extract HOG from each region
4. Predict using trained Jamo classifiers
5. Combine predictions and evaluate
"""

import os
import time
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pandas as pd

from config import IMG_SIZE, NORMALIZATION_FACTOR, RANDOM_SEED
from romanization_mapping import romanization_to_romanized_jamos
from jamo_char_segmentation import segment_hangul_adaptive

print("=" * 80)
print("JAMO-BASED FULL PIPELINE EVALUATION")
print("=" * 80)

start_time = time.time()

# Paths
DATA_DIR = "archive/Hangul Database/Hangul Database"
JAMO_MODELS_DIR = "models/jamo"
RESULTS_DIR = "results/jamo_full"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load trained Jamo classifiers
print("\n[1/6] Loading trained Jamo classifiers...")

cho_clf = joblib.load(f"{JAMO_MODELS_DIR}/cho_classifier.pkl")
jung_clf = joblib.load(f"{JAMO_MODELS_DIR}/jung_classifier.pkl")
cho_names = list(np.load(f"{JAMO_MODELS_DIR}/cho_names.npy"))
jung_names = list(np.load(f"{JAMO_MODELS_DIR}/jung_names.npy"))

print(f"  초성 classifier: {len(cho_names)} classes")
print(f"  중성 classifier: {len(jung_names)} classes")
print(f"  종성 classifier: Not available (insufficient data)")

# Get all class names from dataset
print("\n[2/6] Scanning dataset...")

class_names = sorted([d for d in os.listdir(DATA_DIR) 
                     if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"  Total classes: {len(class_names)}")
print(f"  Sample classes: {class_names[:10]}")


def extract_hog_from_image(img):
    """Extract HOG features from an image."""
    if img is None or img.size == 0:
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


# Process test samples
print("\n[3/6] Processing test samples...")
print("  Strategy: Use 100 samples per class for testing")

all_true_cho = []
all_true_jung = []
all_pred_cho = []
all_pred_jung = []
all_class_labels = []

num_samples_per_class = 100
failed_segmentations = 0
missing_mappings = []

for class_idx, class_name in enumerate(tqdm(class_names, desc="Classes")):
    class_dir = os.path.join(DATA_DIR, class_name)
    
    # Get ground truth Jamo labels from romanization
    try:
        cho_true, jung_true, jong_true = romanization_to_romanized_jamos(class_name)
    except Exception as e:
        print(f"\n  Warning: Failed to map '{class_name}': {e}")
        missing_mappings.append(class_name)
        continue
    
    # Map to classifier indices
    if cho_true not in cho_names:
        # Try to find closest match or skip
        print(f"\n  Warning: 초성 '{cho_true}' not in trained classes for '{class_name}'")
        continue
    
    if jung_true not in jung_names:
        print(f"\n  Warning: 중성 '{jung_true}' not in trained classes for '{class_name}'")
        continue
    
    cho_true_idx = cho_names.index(cho_true)
    jung_true_idx = jung_names.index(jung_true)
    
    # Load images
    image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
    image_files = image_files[:num_samples_per_class]  # Limit to 100
    
    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        # Segment into cho/jung/jong regions
        try:
            cho_region, jung_region, jong_region = segment_hangul_adaptive(img)
        except Exception as e:
            failed_segmentations += 1
            continue
        
        # Extract HOG from each region
        cho_hog = extract_hog_from_image(cho_region)
        jung_hog = extract_hog_from_image(jung_region)
        
        if cho_hog is None or jung_hog is None:
            failed_segmentations += 1
            continue
        
        # Predict using Jamo classifiers
        cho_pred_idx = cho_clf.predict([cho_hog])[0]
        jung_pred_idx = jung_clf.predict([jung_hog])[0]
        
        # Store results
        all_true_cho.append(cho_true_idx)
        all_true_jung.append(jung_true_idx)
        all_pred_cho.append(cho_pred_idx)
        all_pred_jung.append(jung_pred_idx)
        all_class_labels.append(class_name)

print(f"\n  Total samples processed: {len(all_true_cho)}")
print(f"  Failed segmentations: {failed_segmentations}")
print(f"  Missing romanization mappings: {len(missing_mappings)}")

# Evaluate Jamo-level performance
print("\n[4/6] Evaluating Jamo-level accuracy...")

cho_accuracy = accuracy_score(all_true_cho, all_pred_cho)
jung_accuracy = accuracy_score(all_true_jung, all_pred_jung)

print(f"\n  초성 (Initial Consonant) Accuracy: {cho_accuracy:.4f} ({cho_accuracy*100:.2f}%)")
print(f"  중성 (Medial Vowel) Accuracy: {jung_accuracy:.4f} ({jung_accuracy*100:.2f}%)")

# Character-level accuracy (all Jamos must be correct)
print("\n[5/6] Evaluating character-level accuracy...")

correct_chars = 0
for i in range(len(all_true_cho)):
    if (all_true_cho[i] == all_pred_cho[i] and 
        all_true_jung[i] == all_pred_jung[i]):
        correct_chars += 1

char_accuracy = correct_chars / len(all_true_cho) if len(all_true_cho) > 0 else 0

print(f"\n  Character-level Accuracy: {char_accuracy:.4f} ({char_accuracy*100:.2f}%)")
print(f"  (Requires BOTH 초성 AND 중성 to be correct)")

# Compare with baseline
baseline_accuracy = 0.8467  # From character-level KNN

print(f"\n  Baseline (Character-level KNN): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"  Jamo-based approach: {char_accuracy:.4f} ({char_accuracy*100:.2f}%)")

improvement = char_accuracy - baseline_accuracy
if improvement > 0:
    print(f"  Improvement: +{improvement:.4f} ({improvement*100:.2f}%p)")
else:
    print(f"  Degradation: {improvement:.4f} ({improvement*100:.2f}%p)")

# Save results
print("\n[6/6] Saving results...")

results = {
    "Method": "Jamo-based Hierarchical Classification",
    "Cho_Accuracy": cho_accuracy,
    "Jung_Accuracy": jung_accuracy,
    "Char_Accuracy": char_accuracy,
    "Baseline_Accuracy": baseline_accuracy,
    "Improvement": improvement,
    "Total_Samples": len(all_true_cho),
    "Failed_Segmentations": failed_segmentations,
    "Missing_Mappings": len(missing_mappings),
}

df_results = pd.DataFrame([results])
df_results.to_csv(f"{RESULTS_DIR}/jamo_full_results.csv", index=False)
print(f"  Saved: {RESULTS_DIR}/jamo_full_results.csv")

# Save detailed predictions
df_predictions = pd.DataFrame({
    "class_name": all_class_labels,
    "true_cho": [cho_names[i] for i in all_true_cho],
    "pred_cho": [cho_names[i] for i in all_pred_cho],
    "true_jung": [jung_names[i] for i in all_true_jung],
    "pred_jung": [jung_names[i] for i in all_pred_jung],
    "correct": [(all_true_cho[i] == all_pred_cho[i] and 
                 all_true_jung[i] == all_pred_jung[i]) 
                for i in range(len(all_true_cho))]
})
df_predictions.to_csv(f"{RESULTS_DIR}/jamo_predictions.csv", index=False)
print(f"  Saved: {RESULTS_DIR}/jamo_predictions.csv")

print("\n" + "=" * 80)
print("JAMO-BASED PIPELINE EVALUATION COMPLETE!")
print("=" * 80)
print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} minutes")
print(f"\nFinal Results:")
print(f"  초성: {cho_accuracy*100:.2f}%")
print(f"  중성: {jung_accuracy*100:.2f}%")
print(f"  Character: {char_accuracy*100:.2f}%")
print(f"  vs Baseline: {improvement*100:+.2f}%p")

