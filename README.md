# Handwritten Korean Character Recognition using Traditional ML

**CSE36301: Machine Learning — Final Project**  
**UNIST, Fall 2025**  
**Team: Group 2**

---

## Overview

This project builds a **Hangul OCR system using traditional machine learning** instead of deep learning. We focus on achieving reasonable accuracy while maintaining interpretability and conducting systematic error analysis to understand why certain methods succeed or fail.

### Key Achievements

- **84.65% accuracy** with HOG + PCA + KNN on 64 Hangul characters
- **Comprehensive jamo-level error analysis** decomposing failures into initial/medial/final consonant patterns
- **Scientific analysis of negative results** from hierarchical jamo-based classification (2.51% accuracy)
- **Feature-label alignment insights** demonstrating fundamental limitations of rule-based segmentation

---

## Project Structure

```
pdp_ocr/
├── feature_extractor.py           # HOG feature extraction
├── dataloader.py                  # Data loading and splitting
├── config.py                      # Centralized configuration
├── analysis.py                    # Confusion matrix and per-class analysis
├── error_case_analysis.py         # Visualization of misclassified samples
├── jamo_analysis.py               # Jamo-level error decomposition
├── jamo_classifier_train.py       # Isolated jamo classifier training
├── jamo_full_pipeline.py          # Complete jamo-based pipeline
├── romanization_mapping.py        # Romanization to jamo mapping
├── results/                       # All experimental results
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.csv
│   ├── jamo_analysis/
│   └── report_figures/
├── models/                        # Trained models
│   ├── pca.pkl
│   ├── knn.pkl
│   └── jamo/
├── ANALYSIS_REPORT.md            # Detailed error analysis
├── JAMO_ANALYSIS_REPORT.md       # Jamo-level error analysis
└── JAMO_FULL_IMPLEMENTATION_REPORT.md  # Hierarchical classification results
```

---

## Dataset

| Item | Details |
|------|---------|
| **Source** | [Handwritten Korean Characters (Extended)](https://www.kaggle.com/datasets/jkim289/handwritten-korean-characters) |
| **Classes** | 64 Korean syllables |
| **Total Images** | 134,400 (2,100 per class) |
| **Split** | 107,520 train / 26,880 test (80/20, stratified) |
| **Preprocessing** | Grayscale, 64×64 resize, normalization |
| **Augmentation** | 20× via rotation, noise, distortion, brightness/contrast |

---

## Methodology

### 1. Feature Extraction
- **HOG (Histogram of Oriented Gradients)**
  - 9 orientations, 8×8 pixel cells, 2×2 cell blocks
  - L2-Hys normalization
  - Output: 1,764-dimensional feature vector per image

### 2. Dimensionality Reduction
- **PCA**: 1,764 → 256 dimensions
- Preserves 87.4% variance
- Improves computational efficiency

### 3. Classification
- **K-Nearest Neighbors (KNN)**: k ∈ {3, 5, 7}
- **Logistic Regression**: C ∈ {0.1, 1, 10}
- **Random Forest**: estimators ∈ {25, 50, 75}
- **SVM**: Excluded due to training time (>5 hours)

### 4. Hyperparameter Tuning
- **GridSearchCV** with 3-fold cross-validation
- All experiments on Google Colab (Intel Xeon CPU @ 2.20GHz, 12GB RAM)

---

## Results

### Model Performance

| Model | Best Parameters | Accuracy | Precision | Recall | F1-Score |
|-------|----------------|----------|-----------|--------|----------|
| **KNN** | k=7 | **84.65%** | 85.29% | 84.65% | 84.74% |
| Logistic Regression | C=1.0 | 79.83% | 79.86% | 79.83% | 79.81% |
| Random Forest | n_estimators=75 | 66.13% | 66.58% | 66.13% | 65.94% |

### Key Findings

1. **Data augmentation improves performance by 5.84 percentage points** (78.83% → 84.65%)
2. **Errors are structurally driven**: Confusion occurs between visually similar characters (e.g., yeo ↔ eo, ji ↔ gi)
3. **Jamo-level error distribution is balanced**:
   - Initial consonant (초성): 37.3%
   - Medial vowel (중성): 36.2%
   - Final consonant (종성): 26.6%

### Error Analysis

**Per-Class Performance**:
- **Worst performers**: i (61%), jeong (62%), yeo (64%), ji (64%), eo (70%)
  - Characteristics: Simple structure, vertical vowels, silent initial consonants
- **Best performers**: geos (99%), bak (96%), su (96%), iss (95%)
  - Characteristics: Complex structure, final consonants, distinctive shapes

---

## Jamo-Based Hierarchical Classification

We explored decomposing characters into jamo components and classifying each separately.

### Isolated Jamo Recognition
- **Dataset**: 2,400 individual jamo images
- **Classifier**: KNN with k ∈ {3, 5, 7, 9}
- **Results**:
  - Initial consonants: **95.70% accuracy**
  - Medial vowels: **90.87% accuracy**

### Full Pipeline with Automatic Segmentation
- **Challenge**: Requires automatic segmentation of complete characters into jamo regions
- **Approach**: Rule-based horizontal/vertical splitting
- **Results on 4,500 test samples**:
  - Initial consonants: **17.60% accuracy** (↓ 78.1%p)
  - Medial vowels: **10.42% accuracy** (↓ 80.5%p)
  - Character-level: **2.51% accuracy** (vs. 84.65% baseline)

### Root Cause
**Segmentation bottleneck**: Hangul glyphs are spatially interleaved, exhibit layout variations, and form visually unified symbols. Rule-based heuristics fail to extract clean jamo regions, causing severe distribution mismatch between training (isolated jamos) and inference (segmented regions).

**Key Insight**: Feature-label alignment is critical. Linguistic structure exploitation requires learnable decomposition (e.g., attention mechanisms), not rule-based segmentation.

---

## Comparison with Deep Learning

| Approach | Model | Accuracy | Computational Cost |
|----------|-------|----------|-------------------|
| **Ours** | HOG + KNN | 84.65% | CPU-only, minutes |
| **Baseline** | [Xception CNN](https://www.kaggle.com/code/stpeteishii/korean-character-lightning-xception) | 97.66% | GPU required, hours |

**Performance Gap**: 13.01 percentage points

**Trade-offs**:
- Traditional ML: Lower accuracy, high interpretability, low computational cost
- Deep Learning: Higher accuracy, limited interpretability, high computational cost

---

## Requirements

```bash
# Core dependencies
numpy
opencv-python
scikit-image
scikit-learn
pandas
matplotlib
seaborn
joblib
jamo  # For Hangul jamo decomposition

# Install via conda
conda activate ml-env
pip install -r requirements.txt
```

---

## Usage

### 1. Extract HOG Features

```python
from feature_extractor import extract_and_save_hog_per_class

extract_and_save_hog_per_class(
    basic_root="archive/Hangul Database/Hangul Database",
    extended_root="archive/Hangul Database Extended/Hangul Database Extended",
    output_root="features",
    hog_params={
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
    }
)
```

### 2. Train and Evaluate

See `main.ipynb` for the complete training pipeline.

### 3. Run Error Analysis

```bash
# Confusion matrix and per-class accuracy
python analysis.py

# Error case visualization
python error_case_analysis.py

# Jamo-level error decomposition
python jamo_analysis.py
```

---

## Key Contributions

1. **Reasonable accuracy (84.65%) with traditional ML** in resource-limited environments
2. **Systematic error analysis**:
   - Confusion matrix analysis
   - Per-class performance breakdown
   - Jamo-level error decomposition
3. **Scientific analysis of negative results**:
   - Jamo-based hierarchical classification failure
   - Quantitative analysis of segmentation bottleneck (78-80%p degradation)
   - Feature-label alignment insights
4. **Comprehensive documentation** of methods, results, and failure modes

---

## Limitations & Future Work

### Current Limitations
- **Performance ceiling**: 13%p gap vs. deep learning
- **Hand-crafted features**: Limited ability to capture fine-grained variations (e.g., yeo vs. eo)
- **Segmentation challenges**: Rule-based approaches fail for composite writing systems

### Future Directions
1. **Hybrid approach**: Pre-trained CNN features + classical ML classifiers
2. **Alternative features**: HOG + LBP + Gabor filter fusion
3. **Jamo-aware learning**: Attention mechanisms or multi-task learning for implicit jamo decomposition

---

## Contributors

**Team: Group 2**

- Geonho Kim (20211021) - [@geonhokim1](https://github.com/geonhokim1)
- Jiseung Jeong (20211301) - [@jiseungjeong](https://github.com/jiseungjeong)
- Dahyun Kim (20221038)
- Chaerin Hwang (20221422)

**Course:** CSE36301 Machine Learning, UNIST, Fall 2025

---

## License

This project is for **educational and academic research purposes only**.  
Dataset follows original Kaggle license terms.

---

## Summary

Traditional machine learning achieves **84.65% accuracy** on Hangul OCR without deep learning, demonstrating practical utility in resource-constrained environments. Comprehensive error analysis reveals that failures are structurally driven by visual similarities in jamo compositions. The attempted jamo-based hierarchical classification, despite achieving 95%+ accuracy on isolated jamos, failed catastrophically (2.51%) due to segmentation challenges, providing valuable insights into the fundamental limitations of rule-based approaches for composite writing systems. This work emphasizes the importance of understanding **why** methods succeed or fail, beyond simply reporting accuracy.
