"""
Train Jamo-level classifiers using extracted Jamo HOG features.

Strategy: Train separate classifiers for cho/jung/jong using individual Jamo images,
then apply them to complete character HOG features.
"""

import os
import time
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config import RANDOM_SEED

print("=" * 80)
print("JAMO CLASSIFIER TRAINING")
print("=" * 80)

start_time = time.time()
np.random.seed(RANDOM_SEED)

JAMO_FEATURES_DIR = "features/jamo"
JAMO_MODELS_DIR = "models/jamo"
os.makedirs(JAMO_MODELS_DIR, exist_ok=True)


def load_jamo_features(jamo_type):
    """
    Load all HOG features for a specific Jamo type (cho/jung/jong).

    Returns:
        X: Feature matrix
        y: Label vector
        jamo_names: List of Jamo names
    """
    features_dir = os.path.join(JAMO_FEATURES_DIR, jamo_type)

    if not os.path.exists(features_dir):
        print(f"Error: {features_dir} not found!")
        return None, None, None

    all_features = []
    all_labels = []
    jamo_names = []

    # Load each Jamo's features
    for filename in sorted(os.listdir(features_dir)):
        if not filename.endswith(".npy"):
            continue

        jamo_name = filename.replace(".npy", "")
        jamo_names.append(jamo_name)

        features = np.load(os.path.join(features_dir, filename))
        labels = np.full(len(features), len(jamo_names) - 1)  # Current index

        all_features.append(features)
        all_labels.append(labels)

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    return X, y, jamo_names


# Load features for each component
print("\n[1/4] Loading Jamo features...")

print("\n  Loading 초성 (Initial Consonants)...")
X_cho, y_cho, cho_names = load_jamo_features("cho")
print(f"  초성: {X_cho.shape}, Classes: {len(cho_names)}")
print(f"  Jamos: {cho_names}")

print("\n  Loading 중성 (Medial Vowels)...")
X_jung, y_jung, jung_names = load_jamo_features("jung")
print(f"  중성: {X_jung.shape}, Classes: {len(jung_names)}")
print(f"  Jamos: {jung_names}")

print("\n  Loading 종성 (Final Consonants)...")
X_jong, y_jong, jong_names = load_jamo_features("jong")
if X_jong is not None:
    print(f"  종성: {X_jong.shape}, Classes: {len(jong_names)}")
    print(f"  Jamos: {jong_names}")
else:
    print("  Warning: No 종성 features found. Will skip 종성 classifier.")

# Train/test split for each component
print("\n[2/4] Splitting train/test sets...")

X_cho_train, X_cho_test, y_cho_train, y_cho_test = train_test_split(
    X_cho, y_cho, test_size=0.2, random_state=RANDOM_SEED, stratify=y_cho
)
print(f"  초성 - Train: {X_cho_train.shape}, Test: {X_cho_test.shape}")

X_jung_train, X_jung_test, y_jung_train, y_jung_test = train_test_split(
    X_jung, y_jung, test_size=0.2, random_state=RANDOM_SEED, stratify=y_jung
)
print(f"  중성 - Train: {X_jung_train.shape}, Test: {X_jung_test.shape}")

if X_jong is not None and len(jong_names) > 1:
    X_jong_train, X_jong_test, y_jong_train, y_jong_test = train_test_split(
        X_jong, y_jong, test_size=0.2, random_state=RANDOM_SEED, stratify=y_jong
    )
    print(f"  종성 - Train: {X_jong_train.shape}, Test: {X_jong_test.shape}")
else:
    X_jong_train = X_jong_test = y_jong_train = y_jong_test = None
    print("  종성 - Skipped (insufficient classes)")

# Train classifiers
print("\n[3/4] Training Jamo classifiers...")

param_grid = {"n_neighbors": [3, 5, 7, 9]}

# Train 초성 classifier
print("\n  Training 초성 classifier...")
cho_start = time.time()
cho_knn = KNeighborsClassifier()
grid_cho = GridSearchCV(cho_knn, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_cho.fit(X_cho_train, y_cho_train)

y_cho_pred = grid_cho.predict(X_cho_test)
cho_acc = accuracy_score(y_cho_test, y_cho_pred)

print(f"  Best params: {grid_cho.best_params_}")
print(f"  CV score: {grid_cho.best_score_:.4f}")
print(f"  Test accuracy: {cho_acc:.4f}")
print(f"  Time: {(time.time() - cho_start) / 60:.1f} min")

# Train 중성 classifier
print("\n  Training 중성 classifier...")
jung_start = time.time()
jung_knn = KNeighborsClassifier()
grid_jung = GridSearchCV(jung_knn, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_jung.fit(X_jung_train, y_jung_train)

y_jung_pred = grid_jung.predict(X_jung_test)
jung_acc = accuracy_score(y_jung_test, y_jung_pred)

print(f"  Best params: {grid_jung.best_params_}")
print(f"  CV score: {grid_jung.best_score_:.4f}")
print(f"  Test accuracy: {jung_acc:.4f}")
print(f"  Time: {(time.time() - jung_start) / 60:.1f} min")

# Train 종성 classifier (if available)
if X_jong_train is not None:
    print("\n  Training 종성 classifier...")
    jong_start = time.time()
    jong_knn = KNeighborsClassifier()
    grid_jong = GridSearchCV(jong_knn, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_jong.fit(X_jong_train, y_jong_train)

    y_jong_pred = grid_jong.predict(X_jong_test)
    jong_acc = accuracy_score(y_jong_test, y_jong_pred)

    print(f"  Best params: {grid_jong.best_params_}")
    print(f"  CV score: {grid_jong.best_score_:.4f}")
    print(f"  Test accuracy: {jong_acc:.4f}")
    print(f"  Time: {(time.time() - jong_start) / 60:.1f} min")
else:
    grid_jong = None
    jong_acc = None

# Save models
print("\n[4/4] Saving trained classifiers...")

joblib.dump(grid_cho.best_estimator_, f"{JAMO_MODELS_DIR}/cho_classifier.pkl")
print(f"  Saved: {JAMO_MODELS_DIR}/cho_classifier.pkl")

joblib.dump(grid_jung.best_estimator_, f"{JAMO_MODELS_DIR}/jung_classifier.pkl")
print(f"  Saved: {JAMO_MODELS_DIR}/jung_classifier.pkl")

if grid_jong is not None:
    joblib.dump(grid_jong.best_estimator_, f"{JAMO_MODELS_DIR}/jong_classifier.pkl")
    print(f"  Saved: {JAMO_MODELS_DIR}/jong_classifier.pkl")

# Save Jamo name mappings
np.save(f"{JAMO_MODELS_DIR}/cho_names.npy", cho_names)
np.save(f"{JAMO_MODELS_DIR}/jung_names.npy", jung_names)
if jong_names:
    np.save(f"{JAMO_MODELS_DIR}/jong_names.npy", jong_names)

print("\n" + "=" * 80)
print("JAMO CLASSIFIER TRAINING COMPLETE!")
print("=" * 80)
print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} minutes")
print(f"\nResults:")
print(f"  초성 accuracy: {cho_acc:.4f} ({cho_acc * 100:.2f}%)")
print(f"  중성 accuracy: {jung_acc:.4f} ({jung_acc * 100:.2f}%)")
if jong_acc is not None:
    print(f"  종성 accuracy: {jong_acc:.4f} ({jong_acc * 100:.2f}%)")
else:
    print(f"  종성: Not trained (insufficient data)")
print("\nNext: Apply these classifiers to complete character images")
