# Handwritten Korean Character Recognition using Traditional ML

**CSE36301: Machine Learning — Final Project**  
**UNIST, 2025 Fall Semester**  
**Team: Group 14**

---

## Project Overview

This project tackles the **Optical Character Recognition (OCR)** problem for handwritten Korean characters using **classical machine learning approaches** without deep learning. We leverage the publicly available **Kaggle Handwritten Korean Characters Dataset** to develop an automated recognition system.

### Key Objectives

- Build a **traditional ML-based HOG-OCR model** for Korean character recognition
- Compare performance against **baseline deep learning model** (Korean Character Lightning Xception)
- Gain deep understanding of ML pipeline through hands-on implementation of preprocessing, feature extraction, training, and evaluation
- Demonstrate that classical ML can achieve competitive results on vision tasks

## Project Structure

```
pdp_ocr/
├── feature_extractor.py    # Feature extraction utilities (HOG, Gabor)
├── dataloader.py           # Data loading and preprocessing
├── main.ipynb              # Main training and evaluation notebook
└── README.md
```

## Problem Definition & Research Question

> **"Can computers recognize Korean handwriting without deep learning,  
> just as humans read handwritten text?"**

To answer this question, we built a **classical machine learning OCR pipeline** that relies on carefully designed feature extractors rather than automated deep feature learning.

### Processing Pipeline

| Stage | Description |
|-------|-------------|
| **Preprocessing** | Grayscale conversion, resize to 64×64 |
| **Feature Engineering** | HOG feature extraction (emphasizing contours/gradients), Gabor filters |
| **Dimensionality Reduction** | PCA (256 components, preserving 87% variance) |
| **Model Training** | Logistic Regression / SVM / KNN / Random Forest |
| **Hyperparameter Tuning** | GridSearchCV with 3-Fold Cross-Validation |
| **Evaluation** | Accuracy / Precision / Recall / F1 / ROC Curves |

### Key Idea

While CNNs automatically learn features, we **manually design features using HOG** and only train the classifier on top of them.

---

## Features

### Feature Extraction (`feature_extractor.py`)
- **HOG Features**: Extracts Histogram of Oriented Gradients features with configurable parameters
  - Orientations: 9
  - Pixels per cell: (8, 8)
  - Cells per block: (2, 2)
  - Block normalization: L2-Hys
- **Gabor Filters**: Multi-orientation Gabor filter banks for texture analysis
- **Image Preprocessing**: Grayscale conversion and resizing (64x64)
- **Batch Processing**: Class-wise feature extraction with resume capability

### Data Loading (`dataloader.py`)
- Load pre-extracted HOG features from `.npy` files
- Support for class selection and sampling
- Stratified train-test splitting
- Configurable data augmentation options

### Training Pipeline (`main.ipynb`)
- **Dimensionality Reduction**: PCA (256 components, ~87% variance explained)
- **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, ROC curves

### Machine Learning Models

| Algorithm | Purpose & Configuration | Characteristics |
|-----------|------------------------|-----------------|
| **Logistic Regression** | C ∈ {0.1, 1.0, 10.0} | Multinomial softmax classification |
| **K-Nearest Neighbors** | k ∈ {3, 5, 7} | Feature similarity-based |
| **Random Forest** | Trees ∈ {25, 50, 75} | Robust to noisy features |
| **SVM** | (Excluded from final report due to overfitting) | Margin-based decision boundary |

## Dataset Information

| Item | Description |
|------|-------------|
| **Dataset Name** | Handwritten Korean Characters |
| **Source** | Kaggle |
| **URL** | [https://www.kaggle.com/datasets/jkim289/handwritten-korean-characters](https://www.kaggle.com/datasets/jkim289/handwritten-korean-characters) |
| **Total Images** | 6,400 images |
| **Number of Classes** | 64 (Korean syllables/word-level labels) |
| **Format** | RGB images, varying sizes |
| **Train/Test Split** | 80% / 20% (stratified) |

### Data Characteristics

The dataset contains handwritten Korean characters with varying resolutions and writing pressures. This necessitates **aggressive normalization and feature extraction** to ensure consistent recognition performance.

### Dataset Structure

```
archive/
├── Hangul Database/
│   └── Hangul Database/
│       ├── class1/  (Korean syllable 1)
│       ├── class2/  (Korean syllable 2)
│       └── ...
└── Hangul Database Extended/
    └── Hangul Database Extended/
        ├── class1/  (Additional samples)
        ├── class2/
        └── ...
```

## Requirements

```
numpy
opencv-python (cv2)
scikit-image
scikit-learn
tqdm
pandas
matplotlib
seaborn
```

## Usage

### 1. Extract HOG Features

```python
from feature_extractor import extract_and_save_hog_per_class

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
```

### 2. Load Features and Train Models

```python
from dataloader import load_hog_features, _train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Load features
feature_dir = "features/hog"
X, y, class_names = load_hog_features(
    feature_dir=feature_dir,
    shuffle=True,
    random_state=42
)

# Split data
X_train, y_train, X_test, y_test = _train_test_split(
    X, y,
    train_ratio=0.8,
    test_ratio=0.2,
    random_state=42,
    stratify=True
)

# Apply PCA
pca = PCA(n_components=256, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train classifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_pca, y_train)

# Evaluate
accuracy = knn.score(X_test_pca, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### 3. Run Full Experiment

Open and run `main.ipynb` in Jupyter Notebook or Google Colab for the complete training and evaluation pipeline.

## Experimental Results

### Model Performance Summary

Based on experiments in `main.ipynb`:

| Model | Best Parameters | CV Accuracy | Test Accuracy |
|-------|----------------|-------------|---------------|
| **K-Nearest Neighbors** | k=7 | **82.27%** | (To be measured) |
| **Logistic Regression** | C=1.0 | 78.97% | - |
| **Random Forest** | n_estimators=75 | 62.66% | - |

### Key Achievement

**HOG → PCA → ML Classifier** pipeline achieves **over 80% accuracy**  
without any deep learning, demonstrating the power of classical ML approaches.

**Dataset Statistics**:
- Training samples: 107,520
- Test samples: 26,880
- Original feature dimension: 1,764 (HOG)
- Reduced dimension: 256 (PCA)

---

## Baseline Comparison

We compare our traditional ML approach against a state-of-the-art deep learning baseline:

| Item | Details |
|------|---------|
| **Model** | Xception CNN (Pretrained, PyTorch Lightning) |
| **Reference** | [Korean Character Lightning Xception](https://www.kaggle.com/code/stpeteishii/korean-character-lightning-xception) |
| **Test Accuracy** | **97.62%** |
| **Key Features** | Data augmentation + GPU training + Transfer learning |

### Research Goal

Analyze the **advantages and limitations** of traditional ML approaches compared to deep learning baselines, and identify opportunities for hybrid approaches.

## Feature Visualization

| Original Image | HOG Feature Map |
|----------------|-----------------|
| Korean handwritten character | Edge-enhanced representation |

HOG features emphasize **contour information**, making the structure of Korean characters (consonants and vowels) distinctly visible and separable.

---

## Key Insights

| Observation | Insight |
|-------------|---------|
| **HOG feature quality** is crucial for character structure recognition | Feature engineering is more important than model complexity |
| **PCA with 256 components** provides optimal trade-off | Dimensionality reduction without accuracy loss |
| **KNN achieves highest performance** | Korean character structure is well-suited for nearest neighbor classification |
| **Random Forest underperforms** | Tree-based models may struggle with high-dimensional visual features |

---

## Limitations & Future Work

### Current Limitations

| Limitation | Improvement Plan |
|------------|------------------|
| Limited handwriting style diversity | Apply Kaggle Extended Dataset for more variations |
| HOG feature constraints | Experiment with Feature Fusion (Gabor + LBP) |
| 15% performance gap vs. deep learning | Explore hybrid models (CNN features → ML classifier) |
| SVM overfitting issues | Re-tune kernel and margin parameters |

### Next Steps

- **Error Analysis**: Focus on confused classes (e.g., "ㅣ vs ㅓ", similar shapes)
- **Data Augmentation**: Enhance with noise, rotation, stroke intensity variations
- **Comparative Study**: Case-by-case performance comparison with Xception baseline
- **Interpretability**: Use Grad-CAM to understand differences in Korean structure learning
- **Hybrid Approach**: Combine CNN feature extraction with classical ML classifiers

---

## Key Technical Features

- **Modular Design**: Separate modules for feature extraction, data loading, and training  
- **Resume Capability**: Feature extraction can resume from interruption  
- **Multiple Classifiers**: Compare different ML algorithms  
- **Hyperparameter Tuning**: Automated grid search with cross-validation  
- **Visualization**: Performance comparison charts and ROC curves  
- **Efficient Processing**: Batch processing with progress bars  


## License

This project is for **educational and academic research purposes only**.  
Dataset license follows Kaggle original terms.

---

## Contributors

**Team: Group 14**  
**Course:** CSE36301 Machine Learning, UNIST  
**Semester:** 2025 Fall

- [@geonhokim1](https://github.com/geonhokim1)
- [@jiseungjeong](https://github.com/jiseungjeong)

---

## Summary

Traditional ML-based Korean OCR can achieve over 80% accuracy without deep learning. However, there's a clear performance improvement opportunity compared to deep learning baselines. Hybrid approaches combining classical feature engineering with modern deep learning may be the optimal strategy for future work.

