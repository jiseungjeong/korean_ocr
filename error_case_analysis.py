"""
Error Case Visualization and Analysis
Extract and visualize misclassified samples with HOG features
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from dataloader import load_hog_features, _train_test_split
from feature_extractor import load_and_preprocess_image, extract_hog_feature_for_image
from skimage.feature import hog
from skimage import exposure

# Set seed
SEED = 42
np.random.seed(SEED)

print("=" * 80)
print("ERROR CASE VISUALIZATION")
print("=" * 80)

# Load features and train model
print("\n[1/5] Loading features and training model...")
feature_dir = "features/hog"
X, y, class_names = load_hog_features(
    feature_dir=feature_dir, shuffle=True, random_state=SEED
)
X_train, y_train, X_test, y_test = _train_test_split(
    X, y, train_ratio=0.8, test_ratio=0.2, random_state=SEED, stratify=True
)

pca = PCA(n_components=256, random_state=SEED)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)

# Find misclassified samples
print("\n[2/5] Finding misclassified samples...")
misclassified_indices = np.where(y_pred != y_test)[0]
print(
    f"Total misclassified: {len(misclassified_indices)} / {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.1f}%)"
)

# Create mapping from test indices to image paths
print("\n[3/5] Creating image mapping...")
base_dir = "archive/Hangul Database/Hangul Database"


def get_image_path_for_class(class_name, sample_idx):
    """Get image path for a given class and sample index"""
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_dir):
        return None
    files = sorted([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
    if sample_idx < len(files):
        return os.path.join(class_dir, files[sample_idx])
    return None


# Visualize top confused pairs
print("\n[4/5] Visualizing error cases...")
import pandas as pd

confused_df = pd.read_csv("results/confused_pairs.csv")
top_pairs = confused_df.head(10)

os.makedirs("results/error_cases", exist_ok=True)

for idx, row in top_pairs.iterrows():
    true_class = row["True Class"]
    pred_class = row["Predicted Class"]

    # Find examples of this error
    true_label = class_names.index(true_class)
    pred_label = class_names.index(pred_class)

    error_mask = (y_test == true_label) & (y_pred == pred_label)
    error_indices = np.where(error_mask)[0]

    if len(error_indices) == 0:
        continue

    # Visualize up to 3 examples
    num_examples = min(3, len(error_indices))

    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'Confused: "{true_class}" predicted as "{pred_class}" ({len(error_indices)} cases)',
        fontsize=16,
        y=0.995,
    )

    for i, err_idx in enumerate(error_indices[:num_examples]):
        # Get sample info
        test_sample_idx = err_idx

        # Load a sample image from this class (approximate)
        sample_num = test_sample_idx % 100  # Approximate mapping
        img_path = get_image_path_for_class(true_class, sample_num)

        if img_path and os.path.exists(img_path):
            # Load and display original image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (64, 64))

            # Display original
            axes[i, 0].imshow(img_resized, cmap="gray")
            axes[i, 0].set_title(f"Original\n(True: {true_class})")
            axes[i, 0].axis("off")

            # Compute and display HOG
            img_norm = img_resized.astype(np.float32) / 255.0
            fd, hog_image = hog(
                img_norm,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
                block_norm="L2-Hys",
            )
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

            axes[i, 1].imshow(hog_image_rescaled, cmap="hot")
            axes[i, 1].set_title(f"HOG Features\n(True: {true_class})")
            axes[i, 1].axis("off")

            # Load comparison image from predicted class
            comp_img_path = get_image_path_for_class(pred_class, sample_num)
            if comp_img_path and os.path.exists(comp_img_path):
                comp_img = cv2.imread(comp_img_path, cv2.IMREAD_GRAYSCALE)
                comp_img_resized = cv2.resize(comp_img, (64, 64))

                axes[i, 2].imshow(comp_img_resized, cmap="gray")
                axes[i, 2].set_title(f"Compare\n(Pred: {pred_class})")
                axes[i, 2].axis("off")

                # HOG of predicted class
                comp_img_norm = comp_img_resized.astype(np.float32) / 255.0
                comp_fd, comp_hog_image = hog(
                    comp_img_norm,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    visualize=True,
                    block_norm="L2-Hys",
                )
                comp_hog_rescaled = exposure.rescale_intensity(
                    comp_hog_image, in_range=(0, 10)
                )

                axes[i, 3].imshow(comp_hog_rescaled, cmap="hot")
                axes[i, 3].set_title(f"HOG Features\n(Pred: {pred_class})")
                axes[i, 3].axis("off")

    plt.tight_layout()
    filename = f"results/error_cases/{idx+1:02d}_{true_class}_to_{pred_class}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")

print("\n[5/5] Creating summary visualization...")
# Create a summary figure showing worst performing classes
per_class_df = pd.read_csv("results/per_class_accuracy.csv")
worst_classes = per_class_df.nsmallest(6, "Accuracy")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (_, row) in enumerate(worst_classes.iterrows()):
    class_name = row["Class"]
    accuracy = row["Accuracy"]

    # Load sample images from this class
    sample_imgs = []
    for i in range(3):
        img_path = get_image_path_for_class(class_name, i * 10)
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (64, 64))
            sample_imgs.append(img_resized)

    if sample_imgs:
        combined = np.hstack(sample_imgs)
        axes[idx].imshow(combined, cmap="gray")

    axes[idx].set_title(f"Class: {class_name}\nAccuracy: {accuracy:.1%}", fontsize=12)
    axes[idx].axis("off")

plt.suptitle("Worst Performing Classes", fontsize=16)
plt.tight_layout()
plt.savefig("results/worst_classes_samples.png", dpi=150, bbox_inches="tight")
print("  Saved: results/worst_classes_samples.png")

print("\n" + "=" * 80)
print("ERROR CASE ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print(f"  - results/error_cases/ ({len(top_pairs)} confusion pair visualizations)")
print("  - results/worst_classes_samples.png")
