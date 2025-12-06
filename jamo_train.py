"""
Jamo-based Hierarchical Classification for Korean Character Recognition.

This script implements a hierarchical approach that trains separate classifiers
for each Jamo component (초성/중성/종성) and combines their predictions.
"""

import os
import time
import numpy as np
import pandas as pd
from jamo import h2j, j2hcj, j2h
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import joblib

from config import (
    RANDOM_SEED,
    PCA_N_COMPONENTS,
    FEATURES_PATH,
    RESULTS_PATH,
)
from dataloader import load_hog_features, _train_test_split

print("=" * 80)
print("JAMO-BASED HIERARCHICAL CLASSIFICATION")
print("=" * 80)

start_time = time.time()

# Set seed
np.random.seed(RANDOM_SEED)

# Create output directory
jamo_models_dir = "models/jamo"
os.makedirs(jamo_models_dir, exist_ok=True)

print("\n[1/7] Loading HOG features...")
X, y, class_names = load_hog_features(
    feature_dir=FEATURES_PATH, shuffle=True, random_state=RANDOM_SEED
)
X_train, y_train, X_test, y_test = _train_test_split(
    X, y, random_state=RANDOM_SEED, stratify=True
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {len(class_names)}")

# Load pre-trained PCA
print("\n[2/7] Loading pre-trained PCA...")
pca_path = "models/pca.pkl"
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)
    print(f"  Loaded PCA from {pca_path}")
else:
    print(f"  PCA not found, training new one...")
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
    pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(
    f"PCA: {X_train_pca.shape[1]} components, "
    f"explained variance: {pca.explained_variance_ratio_.sum():.4f}"
)


# ============================================================================
# JAMO DECOMPOSITION AND LABEL CREATION
# ============================================================================

print("\n[3/7] Creating Jamo-level labels...")


def decompose_hangul(char):
    """Decompose Korean character into Jamo components."""
    try:
        jamo = j2hcj(h2j(char))
        cho = jamo[0] if len(jamo) > 0 else ""
        jung = jamo[1] if len(jamo) > 1 else ""
        jong = jamo[2] if len(jamo) > 2 else ""
        return cho, jung, jong
    except Exception as e:
        print(f"Warning: Failed to decompose '{char}': {e}")
        return "", "", ""


# Build Jamo vocabularies
cho_set = set()
jung_set = set()
jong_set = set()

for char in class_names:
    cho, jung, jong = decompose_hangul(char)
    cho_set.add(cho)
    jung_set.add(jung)
    jong_set.add(jong)

# Create index mappings (include empty string for all components)
cho_list = sorted(list(cho_set))  # Include empty if exists
jung_list = sorted(list(jung_set))  # Include empty if exists
jong_list = sorted(list(jong_set))  # Include empty

cho_to_idx = {cho: i for i, cho in enumerate(cho_list)}
jung_to_idx = {jung: i for i, jung in enumerate(jung_list)}
jong_to_idx = {jong: i for i, jong in enumerate(jong_list)}

idx_to_cho = {i: cho for cho, i in cho_to_idx.items()}
idx_to_jung = {i: jung for jung, i in jung_to_idx.items()}
idx_to_jong = {i: jong for jong, i in jong_to_idx.items()}

print(f"초성 classes: {len(cho_list)}")
print(f"중성 classes: {len(jung_list)}")
print(f"종성 classes: {len(jong_list)}")


def create_jamo_labels(y_labels, class_names_list):
    """Convert character labels to Jamo component labels."""
    y_cho, y_jung, y_jong = [], [], []

    for label in y_labels:
        char = class_names_list[label]
        cho, jung, jong = decompose_hangul(char)

        y_cho.append(cho_to_idx[cho])
        y_jung.append(jung_to_idx[jung])
        y_jong.append(jong_to_idx[jong])

    return np.array(y_cho), np.array(y_jung), np.array(y_jong)


# Create Jamo labels for train and test sets
y_train_cho, y_train_jung, y_train_jong = create_jamo_labels(y_train, class_names)
y_test_cho, y_test_jung, y_test_jong = create_jamo_labels(y_test, class_names)

print(
    f"Train - 초성: {y_train_cho.shape}, 중성: {y_train_jung.shape}, 종성: {y_train_jong.shape}"
)
print(
    f"Test  - 초성: {y_test_cho.shape}, 중성: {y_test_jung.shape}, 종성: {y_test_jong.shape}"
)

# ============================================================================
# TRAIN SEPARATE CLASSIFIERS FOR EACH JAMO COMPONENT
# ============================================================================

print("\n[4/7] Training Jamo-based classifiers...")

# Hyperparameter grid (same as main experiment)
param_grid_knn = {"n_neighbors": [3, 5, 7]}

# Train 초성 classifier
print("\n  Training 초성 (Initial Consonant) classifier...")
cho_start = time.time()
cho_knn = KNeighborsClassifier()
grid_cho = GridSearchCV(cho_knn, param_grid_knn, cv=3, n_jobs=-1, verbose=1)
grid_cho.fit(X_train_pca, y_train_cho)
print(f"  Best params (초성): {grid_cho.best_params_}")
print(f"  Best CV score (초성): {grid_cho.best_score_:.4f}")
print(f"  Time: {(time.time() - cho_start) / 60:.1f} min")

# Train 중성 classifier
print("\n  Training 중성 (Medial Vowel) classifier...")
jung_start = time.time()
jung_knn = KNeighborsClassifier()
grid_jung = GridSearchCV(jung_knn, param_grid_knn, cv=3, n_jobs=-1, verbose=1)
grid_jung.fit(X_train_pca, y_train_jung)
print(f"  Best params (중성): {grid_jung.best_params_}")
print(f"  Best CV score (중성): {grid_jung.best_score_:.4f}")
print(f"  Time: {(time.time() - jung_start) / 60:.1f} min")

# Train 종성 classifier
print("\n  Training 종성 (Final Consonant) classifier...")
jong_start = time.time()
jong_knn = KNeighborsClassifier()
grid_jong = GridSearchCV(jong_knn, param_grid_knn, cv=3, n_jobs=-1, verbose=1)
grid_jong.fit(X_train_pca, y_train_jong)
print(f"  Best params (종성): {grid_jong.best_params_}")
print(f"  Best CV score (종성): {grid_jong.best_score_:.4f}")
print(f"  Time: {(time.time() - jong_start) / 60:.1f} min")

# ============================================================================
# COMBINE JAMO PREDICTIONS TO FORM FINAL CHARACTER
# ============================================================================

print("\n[5/7] Combining Jamo predictions...")


def combine_jamo_predictions(cho_pred, jung_pred, jong_pred):
    """
    Combine Jamo component predictions to form final character.

    Args:
        cho_pred: Predicted 초성 indices
        jung_pred: Predicted 중성 indices
        jong_pred: Predicted 종성 indices

    Returns:
        Array of character label indices
    """
    final_predictions = []

    for cho_idx, jung_idx, jong_idx in zip(cho_pred, jung_pred, jong_pred):
        # Convert indices to Jamo strings
        cho_char = idx_to_cho[cho_idx]
        jung_char = idx_to_jung[jung_idx]
        jong_char = idx_to_jong[jong_idx]

        # Find matching character directly in class_names
        char_idx = find_closest_char(cho_char, jung_char, jong_char, class_names)
        final_predictions.append(char_idx)

    return np.array(final_predictions)


def find_closest_char(cho, jung, jong, class_names_list):
    """
    Find the character in class_names that best matches the given Jamo components.

    Matching priority:
    1. Exact match (cho, jung, jong all match)
    2. Match cho and jung, ignore jong
    3. Match cho only
    4. Return 0 as default
    """
    # Priority 1: Exact match
    for i, char in enumerate(class_names_list):
        char_cho, char_jung, char_jong = decompose_hangul(char)
        if char_cho == cho and char_jung == jung and char_jong == jong:
            return i

    # Priority 2: Match cho and jung (useful when jong prediction is wrong)
    for i, char in enumerate(class_names_list):
        char_cho, char_jung, _ = decompose_hangul(char)
        if char_cho == cho and char_jung == jung:
            return i

    # Priority 3: Match cho only (last resort before default)
    for i, char in enumerate(class_names_list):
        char_cho, _, _ = decompose_hangul(char)
        if char_cho == cho:
            return i

    # Default: return first class
    print(f"Warning: No match found for ({cho}, {jung}, {jong}), defaulting to class 0")
    return 0


# Get predictions on test set
y_pred_cho = grid_cho.predict(X_test_pca)
y_pred_jung = grid_jung.predict(X_test_pca)
y_pred_jong = grid_jong.predict(X_test_pca)

# Combine predictions
y_pred_jamo = combine_jamo_predictions(y_pred_cho, y_pred_jung, y_pred_jong)

# ============================================================================
# EVALUATION
# ============================================================================

print("\n[6/7] Evaluating Jamo-based approach...")

# Calculate accuracy
jamo_accuracy = accuracy_score(y_test, y_pred_jamo)
print(f"\nJamo-based Accuracy: {jamo_accuracy:.4f} ({jamo_accuracy * 100:.2f}%)")

# Component-level accuracies
cho_accuracy = accuracy_score(y_test_cho, y_pred_cho)
jung_accuracy = accuracy_score(y_test_jung, y_pred_jung)
jong_accuracy = accuracy_score(y_test_jong, y_pred_jong)

print(f"\nComponent Accuracies:")
print(f"  초성: {cho_accuracy:.4f} ({cho_accuracy * 100:.2f}%)")
print(f"  중성: {jung_accuracy:.4f} ({jung_accuracy * 100:.2f}%)")
print(f"  종성: {jong_accuracy:.4f} ({jong_accuracy * 100:.2f}%)")

# Compare with baseline (character-level KNN from main.ipynb)
baseline_accuracy = 0.8467  # From main.ipynb
improvement = jamo_accuracy - baseline_accuracy

print(f"\nComparison:")
print(
    f"  Character-level KNN (baseline): {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)"
)
print(
    f"  Jamo-based KNN (this approach): {jamo_accuracy:.4f} ({jamo_accuracy * 100:.2f}%)"
)
print(f"  Improvement: {improvement:+.4f} ({improvement * 100:+.2f}%p)")

# Analyze performance on previously confused pairs
print("\nPerformance on Previously Confused Pairs:")
confused_pairs = [
    ("yeo", "eo"),
    ("i", "eo"),
    ("ji", "gi"),
    ("deul", "reul"),
    ("jeong", "gyeong"),
]

for true_char, pred_char in confused_pairs:
    if true_char in class_names and pred_char in class_names:
        true_idx = class_names.index(true_char)
        # Count how many times this confusion occurred
        mask_true = y_test == true_idx
        if mask_true.sum() > 0:
            correct = (y_pred_jamo[mask_true] == true_idx).sum()
            total = mask_true.sum()
            acc = correct / total
            print(f"  {true_char}: {correct}/{total} = {acc:.2%}")

# ============================================================================
# SAVE MODELS AND RESULTS
# ============================================================================

print("\n[7/7] Saving models and results...")

# Save classifiers
joblib.dump(grid_cho.best_estimator_, f"{jamo_models_dir}/jamo_cho_knn.pkl")
joblib.dump(grid_jung.best_estimator_, f"{jamo_models_dir}/jamo_jung_knn.pkl")
joblib.dump(grid_jong.best_estimator_, f"{jamo_models_dir}/jamo_jong_knn.pkl")
print(f"  Saved: {jamo_models_dir}/jamo_*_knn.pkl")

# Save mappings
np.save(f"{jamo_models_dir}/cho_to_idx.npy", cho_to_idx)
np.save(f"{jamo_models_dir}/jung_to_idx.npy", jung_to_idx)
np.save(f"{jamo_models_dir}/jong_to_idx.npy", jong_to_idx)
print(f"  Saved: Jamo mappings")

# Save results
results = {
    "jamo_accuracy": jamo_accuracy,
    "baseline_accuracy": baseline_accuracy,
    "improvement": improvement,
    "cho_accuracy": cho_accuracy,
    "jung_accuracy": jung_accuracy,
    "jong_accuracy": jong_accuracy,
    "cho_best_params": grid_cho.best_params_,
    "jung_best_params": grid_jung.best_params_,
    "jong_best_params": grid_jong.best_params_,
}

results_df = pd.DataFrame([results])
results_path = os.path.join(RESULTS_PATH, "jamo_results.csv")
results_df.to_csv(results_path, index=False)
print(f"  Saved: {results_path}")

# ============================================================================
# SUMMARY
# ============================================================================

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("JAMO-BASED TRAINING COMPLETE!")
print("=" * 80)
print(f"\nTotal Time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
print(f"\nFinal Results:")
print(f"  Baseline (Character-level): {baseline_accuracy * 100:.2f}%")
print(f"  Jamo-based (This approach): {jamo_accuracy * 100:.2f}%")
print(f"  Improvement: {improvement * 100:+.2f}%p")

if jamo_accuracy > baseline_accuracy:
    print(f"\nSUCCESS! Jamo-based approach outperforms baseline!")
elif abs(jamo_accuracy - baseline_accuracy) < 0.01:
    print(f"\nComparable performance to baseline (within 1%)")
else:
    print(f"\nJamo-based approach underperforms baseline")

print("\nGenerated files:")
print(f"  - {jamo_models_dir}/jamo_cho_knn.pkl")
print(f"  - {jamo_models_dir}/jamo_jung_knn.pkl")
print(f"  - {jamo_models_dir}/jamo_jong_knn.pkl")
print(f"  - {results_path}")
print("=" * 80)
