"""
Error Analysis and Model Performance Evaluation.

This script performs confusion matrix generation and analyzes
misclassification patterns in Korean character recognition.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from config import (
    RANDOM_SEED,
    PCA_N_COMPONENTS,
    KNN_N_NEIGHBORS,
    FEATURES_PATH,
    RESULTS_PATH,
    CONFUSION_MATRIX_SIZE,
    PER_CLASS_ACC_SIZE,
    DPI,
)
from dataloader import load_hog_features, _train_test_split

# Set seed for reproducibility
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("PRIORITY 1: CONFUSION MATRIX & ERROR ANALYSIS")
print("=" * 80)

# Load HOG features
print("\n[1/6] Loading HOG features...")
X, y, class_names = load_hog_features(
    feature_dir=FEATURES_PATH,
    selected_classes=None,
    max_samples_per_class=None,
    shuffle=True,
    random_state=RANDOM_SEED,
)
print(f"Loaded: {X.shape[0]} samples, {len(class_names)} classes")
print(f"Feature dimension: {X.shape[1]}")
print(f"\nClass names: {class_names}")

# Train-test split
print("\n[2/6] Splitting data...")
X_train, y_train, X_test, y_test = _train_test_split(
    X, y, random_state=RANDOM_SEED, stratify=True
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# PCA
print("\n[3/6] Applying PCA...")
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")
print(f"Reduced dimension: {X_train_pca.shape[1]}")

# Train best model (KNN)
print(f"\n[4/6] Training KNN classifier (k={KNN_N_NEIGHBORS})...")
knn = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS)
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)

# Compute accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Generate confusion matrix
print("\n[5/6] Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

# Save confusion matrix visualization
os.makedirs(RESULTS_PATH, exist_ok=True)
plt.figure(figsize=CONFUSION_MATRIX_SIZE)
sns.heatmap(
    cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names
)
plt.title(f"Confusion Matrix - KNN (k={KNN_N_NEIGHBORS})", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
confusion_matrix_path = os.path.join(RESULTS_PATH, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=DPI, bbox_inches="tight")
print(f"Saved: {confusion_matrix_path}")

# Find most confused pairs
print("\n[6/6] Analyzing most confused character pairs...")
confused_pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            confused_pairs.append(
                {
                    "True Class": class_names[i],
                    "Predicted Class": class_names[j],
                    "Count": cm[i, j],
                    "Error Rate": cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0,
                }
            )

confused_df = pd.DataFrame(confused_pairs)
confused_df = confused_df.sort_values("Count", ascending=False)

print("\nTop 20 Most Confused Character Pairs:")
print(confused_df.head(20).to_string(index=False))

# Save to CSV
confused_pairs_path = os.path.join(RESULTS_PATH, "confused_pairs.csv")
confused_df.to_csv(confused_pairs_path, index=False)
print(f"\nSaved: {confused_pairs_path}")

# Per-class accuracy
print("\n" + "=" * 80)
print("PER-CLASS ACCURACY ANALYSIS")
print("=" * 80)

per_class_acc = []
for i, cls in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        acc = np.mean(y_pred[mask] == y_test[mask])
        per_class_acc.append(
            {
                "Class": cls,
                "Accuracy": acc,
                "Total Samples": mask.sum(),
                "Correct": np.sum(y_pred[mask] == y_test[mask]),
                "Wrong": np.sum(y_pred[mask] != y_test[mask]),
            }
        )

per_class_df = pd.DataFrame(per_class_acc)
per_class_df = per_class_df.sort_values("Accuracy")

print("\nWorst 10 Classes:")
print(per_class_df.head(10).to_string(index=False))

print("\nBest 10 Classes:")
print(per_class_df.tail(10).to_string(index=False))

# Save per-class accuracy
per_class_csv_path = os.path.join(RESULTS_PATH, "per_class_accuracy.csv")
per_class_df.to_csv(per_class_csv_path, index=False)
print(f"\nSaved: {per_class_csv_path}")

# Visualize per-class accuracy
plt.figure(figsize=PER_CLASS_ACC_SIZE)
colors = [
    "red" if acc < 0.7 else "orange" if acc < 0.8 else "green"
    for acc in per_class_df["Accuracy"]
]
plt.barh(range(len(per_class_df)), per_class_df["Accuracy"], color=colors)
plt.yticks(range(len(per_class_df)), per_class_df["Class"])
plt.xlabel("Accuracy", fontsize=12)
plt.ylabel("Character Class", fontsize=12)
plt.title("Per-Class Accuracy (Red < 0.7, Orange < 0.8, Green >= 0.8)", fontsize=14)
plt.axvline(x=0.8, color="gray", linestyle="--", linewidth=1)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
per_class_png_path = os.path.join(RESULTS_PATH, "per_class_accuracy.png")
plt.savefig(per_class_png_path, dpi=DPI, bbox_inches="tight")
print(f"Saved: {per_class_png_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - results/confusion_matrix.png")
print("  - results/confused_pairs.csv")
print("  - results/per_class_accuracy.csv")
print("  - results/per_class_accuracy.png")
print(
    "\nNext step: Run error_case_analysis.py to extract and visualize misclassified samples"
)
